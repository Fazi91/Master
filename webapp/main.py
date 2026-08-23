"""FastAPI interface for evidence-grounded questions over Graph V2."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import os
import re

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "webapp" / "index.html"
load_dotenv(ROOT / ".env")

MAX_IMAGES = 2
MAX_EVIDENCE_CHUNKS = 2
FACT_MIN_SCORE = 0.14
IMAGE_MIN_SCORE = 0.35
LOCAL_ANSWER_MODEL = os.getenv(
    "LOCAL_ANSWER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
ENABLE_LOCAL_SYNTHESIS = os.getenv(
    "ENABLE_LOCAL_SYNTHESIS", "false"
).strip().lower() in {"1", "true", "yes", "on"}
MAX_SYNTHESIS_CHUNKS = 8
MAX_SYNTHESIS_CONTEXT_CHARS = 12000
NLI_VERIFIER_MODEL = os.getenv(
    "NLI_VERIFIER_MODEL", "cross-encoder/nli-deberta-v3-small"
)
NLI_ENTAILMENT_MIN = 0.60
NLI_CONTRADICTION_MAX = 0.20

SMALL_TALK = [
    (r"^(?:(?:hi|hello|hey|salam)[!,. ]*)+$",
     "Hello! How can I help you with the laboratory manual?"),
    (r"^(good morning|good afternoon|good evening)[!,. ]*$",
     "Hello! How can I help you with the laboratory manual?"),
    (r"^(how are you|how are you doing)[?!. ]*$",
     "I'm doing well, thank you. Ask me a laboratory question whenever you are ready."),
    (r"^(who are you|what are you|what is your name|what's your name)[?!. ]*$",
     "I am a laboratory assistant. I answer questions using verified evidence from the provided WHO manual."),
    (r"^(thanks|thank you|thank you very much)[!. ]*$", "You're welcome!"),
    (r"^(what can you do|help|help me|how can you help me|how should i ask)[?!. ]*$",
     "Ask a specific laboratory question. I will return a grounded answer, its exact source and a related image only when one can be verified."),
    (r"^(bye|goodbye|see you|see you later|khodafez)[?!. ]*$", "Goodbye!"),
]

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
    "could", "do", "does", "for", "from", "give", "have", "how", "i",
    "in", "information", "is", "it", "me", "of", "on", "or", "please",
    "should", "tell", "the", "this", "to", "what", "when", "where",
    "which", "who", "why", "with", "would", "you",
}
GENERIC_TERMS = {
    "answer", "explain", "help", "know", "mean", "more", "question",
    "say", "something", "thing", "understand",
}

RELATION_RULES = [
    ("TRANSMITTED_BY", ("transmit", "transmitted", "transmission", "vector", "spread by")),
    ("HAS_MEASUREMENT", ("measurement", "temperature", "duration", "how long", "speed", "rpm", "degree", "minutes", "hours")),
    ("HAS_FINDING", ("finding", "symptom", "sign", "associated with", "characterized by")),
    ("FOUND_IN", ("found in", "located in", "where is", "where are", "present in")),
    ("CAUSES", ("cause", "causes", "caused by", "responsible for")),
    ("DETECTS", ("detect", "detected", "diagnose", "diagnosis", "identify", "confirm")),
    ("USES_EQUIPMENT", ("equipment", "instrument", "device", "objective", "microscope", "used with")),
    ("USES_REAGENT", ("reagent", "stain", "solution", "chemical", "dye")),
    ("EXAMINES", ("examine", "examines", "examined", "specimen", "sample")),
]
INTENT_TERMS = {
    "equipment", "instrument", "device", "objective", "microscope",
    "reagent", "stain", "solution", "chemical", "dye", "transmit",
    "transmission", "vector", "measurement", "temperature", "duration",
    "speed", "minute", "hour", "finding", "symptom", "sign", "associate",
    "characteriz", "found", "locat", "present", "cause", "responsible",
    "detect", "diagnosis", "identify", "confirm", "examine", "specimen",
    "sample", "use",
}
RELATION_VERBS = {
    "USES_EQUIPMENT": "uses",
    "USES_REAGENT": "uses",
    "TRANSMITTED_BY": "is transmitted by",
    "HAS_MEASUREMENT": "has the specified measurement",
    "HAS_FINDING": "is associated with",
    "FOUND_IN": "is found in",
    "CAUSES": "causes",
    "DETECTS": "detects or supports the diagnosis of",
    "EXAMINES": "examines",
}


class QuestionRequest(BaseModel):
    query: str


def normalize_space(value: str) -> str:
    return " ".join((value or "").split())


def clean_answer_text(value: str) -> str:
    """Remove PDF layout markers without changing the source meaning."""
    cleaned = normalize_space(value)
    # The manual uses a standalone capital G as a rendered bullet marker.
    cleaned = re.sub(r"^(?:G\s+)+", "", cleaned)
    cleaned = re.sub(r"\.?\s+G\s+(?=[A-Z])", "; ", cleaned)
    cleaned = re.split(
        r"\s+(?:Materials and reagents|Equipment):",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    return cleaned.strip(" ;")


def normalize_token(token: str) -> str:
    token = token.lower()
    replacements = {
        "microscopic": "microscope", "microscopy": "microscope",
        "examinations": "examination", "procedures": "procedure",
        "reagents": "reagent", "specimens": "specimen",
        "measurements": "measurement", "organisms": "organism",
        "parasites": "parasite", "bacteria": "bacterium",
        "diagnostic": "diagnosis",
        "important": "importance",
        "collection": "collect",
    }
    if token in replacements:
        return replacements[token]
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> list[str]:
    terms = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if len(raw) < 3 or raw in STOPWORDS or raw in GENERIC_TERMS:
            continue
        term = normalize_token(raw)
        if term not in terms:
            terms.append(term)
    # Do not truncate evidence vocabulary. The previous 24-term limit caused
    # valid words occurring later in a Chunk (for example "methanol" and
    # "fixed") to be misclassified as unsupported by the verifier.
    return terms


def question_type(question: str) -> str:
    """Return the kind of evidence an answer must contain."""
    lowered = normalize_space(question).lower()
    if re.search(
        r"\bpurpose\b|\bused for\b|\buses? of\b|"
        r"\brespective uses?\b|\bfunction of\b|"
        r"\bhow (?:is|are|was|were) .+? used\b",
        lowered,
    ):
        return "purpose"
    if re.search(r"\bwhy\b|\breason\b", lowered):
        return "reason"
    if re.search(r"\bwhen\b|\bhow long\b|\bduration\b", lowered):
        return "time"
    # Comparison wording takes precedence over an initial "how". Otherwise
    # "How do X and Y differ ...?" is incorrectly routed as a procedure.
    if re.search(
        r"\bdiffer(?:s|ed|ence|ences|ent)?\b|\bcompare\b|"
        r"\bversus\b|\bvs\.?\b",
        lowered,
    ):
        return "comparison"
    if re.search(r"\bhow\b|\bprocedure\b|\bmethod\b|\bsteps?\b", lowered):
        return "procedure"
    if not re.search(r"\bwhat (?:is|are)\b", lowered) and re.search(
        r"\b(?:staining|preparing|collecting|examining|counting|testing)\b",
        lowered,
    ):
        return "procedure"
    if re.search(r"\bwhere\b", lowered):
        return "location"
    if re.search(r"\bwhat (?:is|are)\b|\bdefine\b|\bdefinition\b", lowered):
        return "definition"
    return "fact"


def paired_subject_terms(question: str) -> set[str]:
    """Capture contrasted modifiers such as 'thick and thin blood films'."""
    lowered = normalize_space(question).lower()
    match = re.search(
        r"\b([a-z][a-z-]+)\s+and\s+([a-z][a-z-]+)\s+"
        r"(?:blood\s+)?(?:films?|smears?|methods?|tests?|stains?|samples?)\b",
        lowered,
    )
    return {normalize_token(match.group(1)), normalize_token(match.group(2))} if match else set()


def question_facets(question: str) -> list[str]:
    """Split an explicitly compound question into independently required parts."""
    normalized = normalize_space(question).strip(" ?.!")
    # Preserve the compared subjects while turning requested dimensions into
    # independent requirements, e.g. "differ in preparation and use".
    dimension_match = re.match(
        r"^(.*?\bdiffer(?:s|ed)?\b.*?)\s+in\s+(?:their\s+)?(.+)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if dimension_match:
        prefix = dimension_match.group(1).strip()
        dimensions = re.split(
            r"\s*(?:,\s*|\band\b)\s*",
            dimension_match.group(2),
            flags=re.IGNORECASE,
        )
        dimensions = [item.strip() for item in dimensions if item.strip()]
        if len(dimensions) > 1:
            return [f"{prefix} in {dimension}" for dimension in dimensions]
    parts = re.split(
        r"(?:\s*,\s*(?:and\s+)?|\s*;\s*|\s+\band\s+)"
        r"(?=(?:why|how|what|when|where|which|who|"
        r"explain|describe|list|state|name|give|compare)\b)",
        normalized,
        flags=re.IGNORECASE,
    )
    # A leading context phrase is part of the question, not an independent
    # facet: "After staining, what ..." and "In malaria microscopy, how ...".
    if (
        len(parts) > 1
        and not re.search(
            r"\b(?:why|how|what|when|where|which|who|explain|describe|"
            r"list|state|name|give|compare)\b",
            parts[0],
            flags=re.IGNORECASE,
        )
    ):
        parts[1] = f"{parts[0]}, {parts[1]}"
        parts = parts[1:]
    facets = [part.strip(" ,;:?.!") for part in parts if part.strip(" ,;:?.!")]
    return facets if len(facets) > 1 else [normalized]


def answer_plan(question: str) -> dict[str, Any]:
    """Plan answer cardinality and stopping rules from the question form."""
    lowered = normalize_space(question).lower()
    requested_type = question_type(question)
    explicit_multi = bool(re.search(
        r"\b(?:list|enumerate|name all|what are|which are|types|kinds|ways|"
        r"methods|conditions|criteria|reasons|causes|purposes|uses|advantages|disadvantages|"
        r"differences|features|steps)\b",
        lowered,
    ))
    explanatory = bool(re.search(
        r"\b(?:explain|describe|discuss|why|how does|how do)\b",
        lowered,
    ))
    compound = len(question_facets(question)) > 1
    if compound:
        return {
            "mode": "multi", "max_claims": 24,
            "stop_on_complete": False, "required_facets": len(question_facets(question)),
        }
    if requested_type == "procedure":
        return {"mode": "procedure", "max_claims": 24, "stop_on_complete": False}
    if requested_type in {"time", "comparison"} or explicit_multi:
        return {"mode": "multi", "max_claims": 6, "stop_on_complete": False}
    if requested_type in {"purpose", "reason"}:
        return {
            "mode": "explanatory",
            "max_claims": 4 if explicit_multi else 2,
            "stop_on_complete": not explicit_multi,
        }
    if explanatory:
        return {"mode": "explanatory", "max_claims": 4, "stop_on_complete": False}
    if requested_type == "definition":
        return {"mode": "concise", "max_claims": 2, "stop_on_complete": True}
    if requested_type == "location":
        return {
            "mode": "multi" if explicit_multi else "concise",
            "max_claims": 4 if explicit_multi else 2,
            "stop_on_complete": not explicit_multi,
        }
    return {
        "mode": "multi" if explicit_multi else "concise",
        "max_claims": 6 if explicit_multi else 2,
        "stop_on_complete": not explicit_multi,
    }


def small_talk_answer(text: str) -> str | None:
    normalized = normalize_space(text).lower()
    for pattern, answer in SMALL_TALK:
        if re.fullmatch(pattern, normalized):
            return answer
    return None


def relation_intent(question: str) -> str | None:
    lowered = normalize_space(question).lower()

    # Graph facts are atomic. A compound question must go through Chunk
    # retrieval so every facet can be satisfied and jointly verified.
    if len(question_facets(question)) > 1:
        return None

    # A reagent name inside a procedure question does not mean that the user
    # is asking for a USES_REAGENT graph fact. Route procedural wording to
    # chunk retrieval, where the complete manual instruction can be returned.
    procedure_patterns = (
        r"\bhow\s+(?:do|does|to)\b",
        r"\bprocedure\b",
        r"\bmethod\b",
        r"\btechnique\b",
        r"\bprepar(?:e|ation|ing)\b",
        r"\bstaining\b",
        r"\bstain(?:ed)?\s+(?:blood|film|smear|slide|specimen|sample)\b",
    )
    explicit_reagent_question = any(re.search(pattern, lowered) for pattern in (
        r"\b(?:which|what)\s+(?:stain|reagent|solution|chemical|dye)\b",
        r"\b(?:stain|reagent|solution|chemical|dye)\s+(?:is|are|was|were)\s+used\b",
        r"\buses?\s+(?:which|what)\s+(?:stain|reagent|solution|chemical|dye)\b",
    ))
    if (any(re.search(pattern, lowered) for pattern in procedure_patterns)
            and not explicit_reagent_question):
        return None

    for relation_type, phrases in RELATION_RULES:
        if relation_type == "USES_REAGENT":
            if explicit_reagent_question:
                return relation_type
            continue
        if relation_type == "HAS_MEASUREMENT" and not re.search(
            r"\b(?:how long|what (?:is|are) (?:the )?(?:time|duration|"
            r"measurement|temperature|speed)|(?:time|duration|measurement|"
            r"temperature|speed) (?:of|for|is|are))\b",
            lowered,
        ):
            # A number or time limit embedded in a condition is not a request
            # for a graph measurement fact.
            continue
        if any(phrase in lowered for phrase in phrases):
            return relation_type
    return None


def serializable_source(row: dict[str, Any]) -> dict[str, Any]:
    chunk_id = row.get("chunk_id")
    pdf_page = row.get("pdf_page")
    source_name = normalize_space(row.get("source_name") or "")
    target_name = normalize_space(row.get("target_name") or "")
    relation_type = row.get("relation_type")
    graph_location = (
        f"(:Page {{pdf_page: {pdf_page}}})-[:HAS_CHUNK]->"
        f"(:Chunk {{id: '{chunk_id}'}})"
    )
    if source_name and target_name and relation_type:
        graph_location += (
            f"; (:Entity {{canonical_name: '{source_name}'}})"
            f"-[:{relation_type}]->"
            f"(:Entity {{canonical_name: '{target_name}'}})"
        )
    return {
        "chunk_id": chunk_id,
        "pdf_page": pdf_page,
        "printed_page": row.get("printed_page"),
        "evidence": normalize_space(row.get("evidence") or row.get("text") or "")[:1200],
        "confidence": round(float(row.get("confidence") or row.get("score") or 0.0), 4),
        "graph_location": graph_location,
    }


class GraphV2QA:
    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password), connection_timeout=10.0,
            max_connection_lifetime=300,
        )
        self._cache_lock = Lock()
        self._chunks = None
        self._word_vectorizer = None
        self._char_vectorizer = None
        self._word_matrix = None
        self._char_matrix = None
        self._model_lock = Lock()
        self._answer_tokenizer = None
        self._answer_model = None
        self._nli_model = None
        self._nli_entailment_index = None
        self._nli_contradiction_index = None

    def _run(self, query: str, **parameters: Any) -> list[dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return [dict(row) for row in session.run(query, **parameters)]

    def relation_facts(self, relation_type: str) -> list[dict[str, Any]]:
        return self._run("""
        MATCH (source:Entity)-[r]->(target:Entity)
        WHERE type(r) = $relation_type AND r.source_chunk_id IS NOT NULL
        OPTIONAL MATCH (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)
        WHERE chunk.id = r.source_chunk_id
        RETURN source.id AS source_id, source.canonical_name AS source_name,
               source.entity_type AS source_type, type(r) AS relation_type,
               target.id AS target_id, target.canonical_name AS target_name,
               target.entity_type AS target_type, r.source_chunk_id AS chunk_id,
               coalesce(r.pdf_page, page.pdf_page) AS pdf_page,
               page.printed_page AS printed_page, r.evidence_text AS evidence,
               r.confidence AS confidence, chunk.text AS text
        """, relation_type=relation_type)

    @staticmethod
    def _fact_document(row: dict[str, Any]) -> str:
        return normalize_space(
            f"{row.get('source_name', '')} {row.get('target_name', '')} "
            f"{row.get('evidence', '')}"
        )

    @staticmethod
    def _similarities(question: str, documents: list[str]) -> np.ndarray:
        if not documents:
            return np.array([], dtype="float32")
        corpus = [question, *documents]
        word = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        ).fit_transform(corpus)
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1
        ).fit_transform(corpus)
        word_scores = (word[1:] @ word[0].T).toarray().ravel()
        char_scores = (char[1:] @ char[0].T).toarray().ravel()
        return (0.55 * word_scores + 0.45 * char_scores).astype("float32")

    def ranked_facts(self, question: str, relation_type: str) -> list[dict[str, Any]]:
        facts = self.relation_facts(relation_type)
        if not facts:
            return []
        scores = self._similarities(question, [self._fact_document(row) for row in facts])
        query_terms = set(content_terms(question))
        subject_terms = query_terms - INTENT_TERMS
        ranked = []
        for row, base_score in zip(facts, scores):
            source_terms = set(content_terms(row.get("source_name") or ""))
            target_terms = set(content_terms(row.get("target_name") or ""))
            overlap = len(query_terms & source_terms) + 0.5 * len(query_terms & target_terms)
            item = dict(row)
            item["score"] = float(base_score) + min(overlap * 0.16, 0.48)
            item["subject_overlap"] = len(subject_terms & (source_terms | target_terms))
            ranked.append(item)
        ranked.sort(key=lambda row: row["score"], reverse=True)
        return ranked

    @staticmethod
    def compose_fact_answer(relation_type: str, rows: list[dict[str, Any]]) -> str:
        best_source = rows[0]["source_name"]
        targets = []
        target_keys = set()
        for row in rows:
            value = normalize_space(row.get("target_name") or "")
            key = value.lower()
            if value and key not in target_keys:
                targets.append(value)
                target_keys.add(key)
            if len(targets) == 6:
                break
        if not targets:
            return "The requested fact was not found in the provided manual."
        if len(targets) == 1:
            target_text = targets[0]
        else:
            target_text = ", ".join(targets[:-1]) + f", and {targets[-1]}"
        answer = f"According to the manual, {best_source} {RELATION_VERBS[relation_type]} {target_text}."
        if relation_type == "USES_EQUIPMENT" and len(targets) > 1:
            answer += " The exact equipment depends on the specific examination procedure."
        return answer

    def _ensure_chunk_index(self) -> None:
        if self._chunks is not None:
            return
        with self._cache_lock:
            if self._chunks is not None:
                return
            chunks = self._run("""
            MATCH (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
            RETURN chunk.id AS chunk_id, chunk.text AS text,
                   page.pdf_page AS pdf_page, page.printed_page AS printed_page,
                   collect(DISTINCT entity.canonical_name) AS entity_names
            ORDER BY chunk.id
            """)
            documents = [normalize_space(
                f"{row.get('text', '')} {' '.join(row.get('entity_names') or [])}"
            ) for row in chunks]
            self._word_vectorizer = TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), sublinear_tf=True
            )
            self._char_vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=1
            )
            self._word_matrix = self._word_vectorizer.fit_transform(documents)
            self._char_matrix = self._char_vectorizer.fit_transform(documents)
            self._chunks = chunks

    def ranked_chunks(self, question: str) -> list[dict[str, Any]]:
        """Score every Chunk; do not discard evidence before ranking."""
        self._ensure_chunk_index()
        word_query = self._word_vectorizer.transform([question])
        char_query = self._char_vectorizer.transform([question])
        semantic_scores = (
            0.55 * (self._word_matrix @ word_query.T).toarray().ravel()
            + 0.45 * (self._char_matrix @ char_query.T).toarray().ravel()
        )

        query_terms = set(content_terms(question))
        requested_type = question_type(question)
        ranked = []
        for index, semantic_score in enumerate(semantic_scores):
            row = dict(self._chunks[index])
            document = (
                f"{row.get('text', '')} {' '.join(row.get('entity_names') or [])}"
            )
            document_terms = set(content_terms(document))
            overlap = len(query_terms & document_terms)
            coverage = overlap / max(len(query_terms), 1)
            exact_phrase = normalize_space(question).lower().rstrip("?.!") in (
                normalize_space(row.get("text") or "").lower()
            )
            answerability = self._direct_answerability(
                question, row.get("text") or ""
            )
            complete_passages = self._complete_answer_passages(
                question, row.get("text") or ""
            )
            row["semantic_score"] = float(semantic_score)
            row["keyword_overlap"] = overlap
            row["keyword_coverage"] = coverage
            row["exact_phrase"] = exact_phrase
            row["answerability"] = answerability
            row["question_type"] = requested_type
            row["reference_page"] = self._is_reference_page(
                row.get("text") or ""
            )
            row["requirements_complete"] = bool(complete_passages)
            row["complete_passages"] = complete_passages
            row["score"] = (
                0.30 * float(semantic_score)
                + min(overlap * 0.08, 0.32)
                + min(coverage * 0.18, 0.18)
                + 0.90 * answerability
                + (1.20 if complete_passages else 0.0)
                + (0.08 if exact_phrase else 0.0)
            )
            ranked.append(row)

        ranked.sort(
            key=lambda row: (
                row["score"], row["keyword_coverage"], row["keyword_overlap"]
            ),
            reverse=True,
        )
        return ranked

    @staticmethod
    def _is_reference_page(text: str) -> bool:
        """Identify navigation pages that must never be answer evidence."""
        normalized = normalize_space(text)
        return bool(re.match(
            r"^(?:index|contents|table of contents)\b",
            normalized,
            flags=re.IGNORECASE,
        ))

    @staticmethod
    def _local_evidence_units(question: str, text: str) -> list[str]:
        """Create local answer units without joining unrelated paragraphs."""
        segments = GraphV2QA._evidence_segments(text)
        units = list(segments)
        units.extend(GraphV2QA._list_evidence_units(segments))
        if (
            question_type(question) in {"reason", "comparison"}
            or len(question_facets(question)) > 1
        ):
            units.extend(
                f"{segments[index]} {segments[index + 1]}"
                for index in range(len(segments) - 1)
            )
        if question_type(question) == "comparison":
            units.extend(
                " ".join(segments[index:index + 3])
                for index in range(len(segments) - 2)
            )
        return units

    @staticmethod
    def _list_evidence_units(segments: list[str]) -> list[str]:
        """Join a list lead-in to all of its payload items.

        PDF paragraph boundaries separate phrases such as ``used for:`` from
        the bullets that actually answer the question.  Treating the lead-in
        as a standalone sentence produced confident but empty answers.
        """
        units: list[str] = []
        bullet = re.compile(r"^(?:[—–-]|[•▪]|G\s+)")
        heading = re.compile(
            r"^(?:\d+(?:\.\d+)+\s+)?(?:materials and reagents|equipment|"
            r"method|procedure|principle|preparation|examination)\b",
            flags=re.IGNORECASE,
        )
        for index, lead in enumerate(segments):
            normalized_lead = normalize_space(lead)
            if not normalized_lead.endswith(":"):
                continue
            if not re.search(
                r"\b(?:used for|include|includes|following|as follows)\s*:$",
                normalized_lead,
                flags=re.IGNORECASE,
            ):
                continue
            items: list[str] = []
            for candidate in segments[index + 1:]:
                candidate = normalize_space(candidate)
                if not candidate or heading.match(candidate):
                    break
                if not bullet.match(candidate):
                    # A wrapped explanatory sentence may immediately follow
                    # the final bullet (for example, the use of Field stains).
                    # Attach it only when it repeats the final item's subject.
                    if items:
                        last_terms = content_terms(items[-1])[:2]
                        candidate_terms = set(content_terms(candidate))
                        if last_terms and all(
                            term in candidate_terms for term in last_terms
                        ):
                            items[-1] = (
                                f"{items[-1].rstrip(' .')}. {candidate}"
                            )
                    break
                cleaned = bullet.sub("", candidate, count=1).strip()
                if cleaned:
                    items.append(cleaned.rstrip(" .;"))
            if items:
                units.append(
                    f"{normalized_lead.rstrip(':')} " + "; ".join(items)
                )
        return units

    @staticmethod
    def _complete_answer_passages(question: str, text: str) -> list[str]:
        """Return only local passages satisfying all explicit requirements."""
        passages = []
        for unit in GraphV2QA._local_evidence_units(question, text):
            unit = normalize_space(unit)
            if not unit or len(unit) > 1400:
                continue
            if GraphV2QA._requirements_satisfied(question, unit):
                passages.append(unit)
        passages.sort(
            key=lambda passage: (
                -len(passage),
                GraphV2QA._direct_answerability(question, passage),
            ),
            reverse=True,
        )
        return passages[:3]

    @staticmethod
    def _direct_answerability(question: str, text: str) -> float:
        """Score whether one local passage contains the requested answer."""
        query_terms = set(content_terms(question))
        if not query_terms:
            return 0.0
        requested_type = question_type(question)
        type_patterns = {
            "purpose": re.compile(
                r"\b(?:purpose|used for|used to|serves? to|function(?:s)? as|"
                r"used (?:alone|with|in|as)|allows?|enables?|include(?:s)?)\b",
                flags=re.IGNORECASE,
            ),
            "reason": re.compile(
                r"\b(?:because|therefore|thus|hence|due to|so that|"
                r"in order to|results? in|leads? to|will give|"
                r"make it impossible|alter(?:s|ed)?|affect(?:s|ed)?|"
                r"chang(?:e|es|ed)|damage(?:s|d)?|"
                r"to (?:permit|prevent|avoid|ensure|allow))\b",
                flags=re.IGNORECASE,
            ),
            "procedure": re.compile(
                r"\b(?:method|procedure|first|then|next|finally|"
                r"\d{1,2}\.\s|prepare|add|mix|place|stain|rinse|dry|fix)\b",
                flags=re.IGNORECASE,
            ),
            "time": re.compile(
                r"\b(?:when|before|after|during|for\s+\d|minutes?|hours?|days?)\b",
                flags=re.IGNORECASE,
            ),
            "comparison": re.compile(
                r"\b(?:whereas|while|unlike|compared|difference|both)\b",
                flags=re.IGNORECASE,
            ),
            "location": re.compile(
                r"\b(?:in|inside|within|located|found|present)\b",
                flags=re.IGNORECASE,
            ),
            "definition": re.compile(
                r"\b(?:is|are|means?|defined as|refers? to|consists? of)\b",
                flags=re.IGNORECASE,
            ),
        }
        best = 0.0
        segments = GraphV2QA._evidence_segments(text)
        windows = list(segments)
        windows.extend(
            f"{segments[index]} {segments[index + 1]}"
            for index in range(len(segments) - 1)
        )
        if requested_type == "comparison":
            windows.extend(
                " ".join(segments[index:index + 3])
                for index in range(len(segments) - 2)
            )
        for passage in windows:
            passage_terms = set(content_terms(passage))
            coverage = len(query_terms & passage_terms) / len(query_terms)
            if coverage == 0:
                continue
            type_match = bool(
                type_patterns.get(requested_type, re.compile(r"."))
                .search(passage)
            )
            if requested_type == "comparison" and not type_match:
                paired_terms = paired_subject_terms(question)
                type_match = bool(
                    paired_terms
                    and paired_terms.issubset(passage_terms)
                    and re.search(
                        r"\b(?:not|used for|used to|fix(?:ed|ation)?|"
                        r"more|less|higher|lower|larger|smaller)\b",
                        passage,
                        flags=re.IGNORECASE,
                    )
                )
            polarity_match = (
                "not" not in query_terms or "not" in passage_terms
            )
            score = 0.72 * coverage
            if type_match:
                score += 0.20
            if polarity_match:
                score += 0.08
            if requested_type == "reason" and not type_match:
                score *= 0.45
            elif requested_type in {
                "purpose", "comparison", "time", "location", "definition"
            } and not type_match:
                score *= 0.55
            if "not" in query_terms and not polarity_match:
                score *= 0.55
            best = max(best, score)
        return min(best, 1.0)

    @staticmethod
    def _requirements_satisfied(question: str, passage: str) -> bool:
        """Check completeness for the relation explicitly requested."""
        facets = question_facets(question)
        if len(facets) > 1:
            # A compound question is complete only if the same evidence
            # passage answers every explicit clause, not merely one of them.
            return all(
                GraphV2QA._requirements_satisfied(facet, passage)
                for facet in facets
            )
        lowered_question = normalize_space(question).lower()
        normalized_passage = normalize_space(passage)
        # A colon-ended lead-in announces an answer but contains none of its
        # payload.  It can only be verified after the following list is joined.
        if normalized_passage.endswith(":"):
            return False
        if re.search(
            r"\b(?:characteristics?|criteria|features?|signs?)\b",
            lowered_question,
        ):
            quality_markers = re.findall(
                r"\b(?:characteristics?|criteria|features?|satisfactory|"
                r"should|must|smooth|ragged|lines?|holes?|"
                r"too\s+(?:long|thick|thin)|free from|appearance)\b",
                passage,
                flags=re.IGNORECASE,
            )
            if len({marker.lower() for marker in quality_markers}) < 2:
                return False
            subject_terms = set(content_terms(question)) & set(content_terms(passage))
            return bool(subject_terms)
        if re.search(
            r"\b(?:what are|which are|list|enumerate|name all|uses?)\b",
            lowered_question,
        ) and re.search(r"\b(?:used for|include(?:s)?)\b", normalized_passage,
                        flags=re.IGNORECASE):
            # A list answer needs payload, not only the introductory phrase.
            if ";" not in normalized_passage:
                return False
        requested_type = question_type(question)
        direct = GraphV2QA._direct_answerability(question, passage)
        passage_terms = set(content_terms(passage))
        # Relation words describe the requested answer shape; they are not
        # the subject. Require the evidence to match the remaining subject
        # anchors so an unrelated duration, purpose or definition cannot pass.
        relation_terms = {
            "purpose", "use", "uses", "used", "function", "respective",
            "reason", "cause", "why", "time", "duration", "when",
            "difference", "different", "differ", "compare", "comparison",
            "method", "procedure", "step", "steps", "explain", "describe",
            "characteristic", "characteristics", "criteria", "feature",
            "features", "sign", "signs", "identify", "indicate",
            "not",
            "each", "one",
        }
        subject_terms = set(content_terms(question)) - relation_terms
        subject_overlap = len(subject_terms & passage_terms)
        minimum_subject_overlap = 1 if len(subject_terms) <= 2 else 2
        if subject_terms and subject_overlap < minimum_subject_overlap:
            return False
        if requested_type == "procedure":
            # A reconstructed multi-step answer is complete at answer level
            # even when no individual step repeats every subject keyword.
            if len(GraphV2QA._step_numbers(passage)) >= 2:
                return True
        paired_terms = paired_subject_terms(question)
        if paired_terms and not paired_terms.issubset(passage_terms):
            return False
        if requested_type == "comparison" and paired_terms:
            lowered_question = normalize_space(question).lower()
            if re.search(r"\b(?:preparation|prepare|fixation|fixed)\b", lowered_question):
                if not re.search(
                    r"\b(?:prepare|preparation|fix|fixed|fixation|methanol)\b",
                    passage,
                    flags=re.IGNORECASE,
                ):
                    return False
                # Mentioning both films near a reagent list is not a
                # preparation comparison. Require an explicit contrast tied
                # to the paired film names.
                if paired_terms == {"thick", "thin"}:
                    thin_fixed = re.search(
                        r"(?:\bfix(?:ed)?\b.{0,60}\bthin\b|"
                        r"\bthin\b.{0,100}\bfix(?:ed)?\b)",
                        passage,
                        flags=re.IGNORECASE,
                    )
                    thick_not_fixed = re.search(
                        r"(?:\bthick\b.{0,100}\bnot\b.{0,60}\bfix(?:ed)?\b|"
                        r"\bnot\b.{0,60}\bfix(?:ed)?\b.{0,100}\bthick\b)",
                        passage,
                        flags=re.IGNORECASE,
                    )
                    if not (thin_fixed and thick_not_fixed):
                        return False
            if re.search(r"\b(?:use|uses|purpose|function)\b", lowered_question):
                if not re.search(
                    r"\b(?:used for|used to|purpose|identify|identifying|detect|detection)\b",
                    passage,
                    flags=re.IGNORECASE,
                ):
                    return False
                if "malaria" in lowered_question and not (
                    re.search(r"\bparasites?\b", passage, flags=re.IGNORECASE)
                    and re.search(
                        r"\b(?:identify|identifying|detect|detection)\b",
                        passage,
                        flags=re.IGNORECASE,
                    )
                ):
                    return False
        relation_patterns = {
            "purpose": r"\b(?:purpose|used for|used to|used (?:alone|with|in|as)|serves? to|function(?:s)? as|include(?:s)?)\b",
            "reason": r"\b(?:because|therefore|thus|hence|due to|so that|in order to|results? in|leads? to|will give|make it impossible|alter(?:s|ed)?|affect(?:s|ed)?|chang(?:e|es|ed)|damage(?:s|d)?|to (?:permit|prevent|avoid|ensure|allow))\b",
            "comparison": r"\b(?:whereas|while|unlike|compared|difference|both|not|used for|used to)\b",
            "time": r"\b(?:when|before|after|during|for\s+\d|minutes?|hours?|days?)\b",
            "location": r"\b(?:inside|within|located|found|present|occurs?)\b",
            "definition": r"\b(?:is|are|means?|defined as|refers? to|consists? of)\b",
        }
        pattern = relation_patterns.get(requested_type)
        if pattern and not re.search(pattern, passage, flags=re.IGNORECASE):
            return False
        thresholds = {
            "purpose": 0.48, "reason": 0.58, "comparison": 0.48,
            "time": 0.45, "location": 0.45, "definition": 0.42,
        }
        return direct >= thresholds.get(requested_type, 0.55)

    @staticmethod
    def _focused_answer_passages(
        question: str, text: str, limit: int = 4
    ) -> list[str]:
        """Return local passages that directly answer the question."""
        units = GraphV2QA._local_evidence_units(question, text)
        if not units:
            return []
        scored = []
        requested_type = question_type(question)
        query_terms = set(content_terms(question))
        for position, passage in enumerate(units):
            passage = normalize_space(passage)
            if not passage or len(passage) > 1400:
                continue
            passage_terms = set(content_terms(passage))
            overlap = len(query_terms & passage_terms)
            if overlap == 0:
                continue
            direct = GraphV2QA._direct_answerability(question, passage)
            similarity = float(
                GraphV2QA._similarities(question, [passage])[0]
            )
            scored.append({
                "direct": direct,
                "similarity": similarity,
                "overlap": overlap,
                "position": position,
                "passage": passage,
                "complete": GraphV2QA._requirements_satisfied(
                    question, passage
                ),
            })
        if requested_type in {"purpose", "definition", "time", "location"}:
            scored.sort(
                key=lambda item: (
                    item["complete"], -len(item["passage"]),
                    item["direct"], item["similarity"],
                ),
                reverse=True,
            )
        else:
            scored.sort(
                key=lambda item: (
                    item["complete"], item["direct"],
                    item["similarity"], item["overlap"],
                    -item["position"],
                ),
                reverse=True,
            )
        selected = []
        seen = set()
        minimum_direct = {
            "purpose": 0.48, "reason": 0.58, "comparison": 0.48,
            "time": 0.45, "location": 0.45, "definition": 0.42,
        }.get(requested_type, 0.35)
        for item in scored:
            direct = item["direct"]
            passage = item["passage"]
            if direct < minimum_direct:
                continue
            key = passage.lower()
            if key in seen:
                continue
            # Prefer the joined cause/effect window over either contained
            # fragment so the model receives the complete relation.
            if any(key in existing.lower() for existing in selected):
                continue
            selected.append(passage)
            seen.add(key)
            if len(selected) == limit:
                break
        return selected

    @staticmethod
    def chunk_is_grounded(question: str, row: dict[str, Any]) -> bool:
        query_terms = set(content_terms(question))
        entity_terms = set(content_terms(" ".join(row.get("entity_names") or [])))
        text_terms = set(content_terms(row.get("text") or ""))
        entity_overlap = len(query_terms & entity_terms)
        text_overlap = len(query_terms & text_terms)
        return entity_overlap > 0 or text_overlap >= 2

    @staticmethod
    def _step_numbers(text: str) -> list[int]:
        return [
            int(value) for value in re.findall(
                r"(?<![\d.])(\d{1,2})\.\s+", text or ""
            )
        ]

    @staticmethod
    def _procedure_chain(
        anchor: dict[str, Any], ranked_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Follow a numbered procedure across Chunk and page boundaries."""
        anchor_page = anchor.get("pdf_page")
        if not isinstance(anchor_page, int):
            return []
        window = [
            row for row in ranked_rows
            if isinstance(row.get("pdf_page"), int)
            and row["pdf_page"] >= anchor_page
        ]
        window.sort(key=lambda row: (
            row.get("pdf_page") or 0,
            row.get("chunk_id") or "",
        ))
        chain = []
        started = False
        highest_step = 0
        first_step_page = None
        for row in window:
            numbers = GraphV2QA._step_numbers(row.get("text") or "")
            if not numbers:
                continue
            page = row.get("pdf_page")
            if not started:
                # Begin only at the start of a method, not in a random table.
                if 1 not in numbers and min(numbers) > 2:
                    continue
                started = True
                first_step_page = page
            elif (
                1 in numbers and highest_step >= 2
                and isinstance(page, int)
                and isinstance(first_step_page, int)
                and page > first_step_page
            ):
                # Numbering restarted on a later page: a new method began.
                break
            if min(numbers) > highest_step + 1 and highest_step:
                continue
            chain.append(row)
            highest_step = max(highest_step, max(numbers))
        return chain

    @staticmethod
    def _evidence_segments(text: str) -> list[str]:
        cleaned = text or ""
        section_names = (
            "Principle", "Procedure", "Technique", "Method",
            "Materials and reagents", "Equipment", "Preparation",
            "Examination", "Reporting results", "Quality control",
        )
        for heading in section_names:
            cleaned = re.sub(
                rf"\s+({re.escape(heading)})\s+",
                rf".\n\1: ",
                cleaned,
                flags=re.IGNORECASE,
            )
        # PDF extraction inserts single newlines at visual line wraps. They
        # are not sentence boundaries and previously split a supported claim
        # immediately before key words such as "methanol". Preserve only
        # blank-line paragraph boundaries.
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
        cleaned = re.sub(r"\.{2,}", ".", cleaned)
        # Do not split a sentence immediately after the figure abbreviation.
        cleaned = re.sub(r"\bFig\.", "Fig§", cleaned, flags=re.IGNORECASE)
        return [
            normalize_space(part).replace("Fig§", "Fig.")
            for part in re.split(r"(?<=[.!?;])\s+|\n{2,}", cleaned)
            if normalize_space(part)
        ]

    @staticmethod
    def select_consistent_candidates(
        question: str, ranked_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Select the smallest evidence set satisfying the question."""
        grounded = [
            row for row in ranked_rows
            if not row.get("reference_page")
            and GraphV2QA.chunk_is_grounded(question, row)
        ]
        if not grounded:
            return []

        requested_type = question_type(question)

        if requested_type == "procedure":
            exact_rows = [row for row in grounded if row.get("exact_phrase")]
            anchor = exact_rows[0] if exact_rows else max(
                grounded,
                key=lambda row: (
                    row.get("answerability", 0.0),
                    row.get("score", 0.0),
                ),
            )
            chain = GraphV2QA._procedure_chain(anchor, ranked_rows)
            if chain:
                return chain

        # A locally complete passage is stronger than any combination of
        # partial passages. This decision must happen before page anchoring.
        if requested_type != "procedure":
            complete_rows = [
                row for row in grounded
                if row.get("requirements_complete")
            ]
            if complete_rows:
                complete_rows.sort(
                    key=lambda row: (
                        row.get("answerability", 0.0),
                        row.get("score", 0.0),
                        row.get("keyword_coverage", 0.0),
                    ),
                    reverse=True,
                )
                # A score-time completeness flag is only a proposal. Accept
                # it only if the same row can produce an answer that passes
                # the final facet verifier; otherwise continue searching.
                for complete_row in complete_rows:
                    proposed_answer, proposed_sources = (
                        GraphV2QA.compose_extract_answer(
                            question, [complete_row]
                        )
                    )
                    if (
                        proposed_answer and proposed_sources
                        and GraphV2QA._requirements_satisfied(
                            question, proposed_answer
                        )
                    ):
                        return [complete_row]

            # No single passage is complete: greedily build a minimal set of
            # complementary passages across the full candidate list. Do not
            # discard a correct distant page merely because another page had
            # a higher initial similarity score.
            query_terms = set(content_terms(question))
            facets = question_facets(question)
            covered_terms: set[str] = set()
            covered_facets: set[int] = set()
            selected: list[dict[str, Any]] = []
            remaining = []
            for row in grounded:
                row_text = row.get("text") or ""
                facet_passages = [
                    GraphV2QA._focused_answer_passages(
                        facet, row_text, limit=2
                    )
                    for facet in facets
                ]
                facet_coverage = {
                    index for index, passages_for_facet
                    in enumerate(facet_passages)
                    if passages_for_facet and any(
                        GraphV2QA._requirements_satisfied(facets[index], passage)
                        for passage in passages_for_facet
                    )
                }
                passages = []
                for passages_for_facet in facet_passages:
                    for passage in passages_for_facet:
                        if passage not in passages:
                            passages.append(passage)
                passage_terms = set(content_terms(" ".join(passages)))
                contribution = query_terms & passage_terms
                if passages and (contribution or facet_coverage):
                    item = dict(row)
                    item["evidence_passages"] = passages
                    item["requirement_terms"] = contribution
                    item["requirement_facets"] = facet_coverage
                    remaining.append(item)

            while remaining:
                remaining.sort(
                    key=lambda row: (
                        len(row["requirement_facets"] - covered_facets),
                        len(row["requirement_terms"] - covered_terms),
                        row.get("answerability", 0.0),
                        row.get("score", 0.0),
                    ),
                    reverse=True,
                )
                best = remaining.pop(0)
                new_terms = best["requirement_terms"] - covered_terms
                new_facets = best["requirement_facets"] - covered_facets
                if not new_terms and not new_facets:
                    break
                selected.append(best)
                covered_terms.update(new_terms)
                covered_facets.update(new_facets)
                if (
                    len(facets) > 1
                    and len(covered_facets) == len(facets)
                ):
                    break
                if (
                    len(facets) == 1 and query_terms
                    and len(covered_terms) / len(query_terms) >= 0.80
                ):
                    break
            if selected:
                return selected

        # Procedures may span consecutive Chunks. Anchor only this intent to
        # a coherent page window so steps from different methods are not mixed.
        exact_rows = [row for row in grounded if row.get("exact_phrase")]
        anchor = exact_rows[0] if exact_rows else max(
            grounded,
            key=lambda row: (
                row.get("answerability", 0.0),
                row.get("score", 0.0),
            ),
        )
        anchor_page = anchor.get("pdf_page")
        query_terms = set(content_terms(question))
        minimum_overlap = 2 if len(query_terms) >= 2 else 1

        consistent = []
        for row in grounded:
            page = row.get("pdf_page")
            same_section_window = (
                isinstance(anchor_page, int)
                and isinstance(page, int)
                and anchor_page <= page <= anchor_page + 1
            )
            same_page = page == anchor_page
            strong_term_match = row.get("keyword_overlap", 0) >= minimum_overlap
            direct_enough = row.get("answerability", 0.0) >= 0.42
            if row is anchor or (
                (same_section_window or same_page)
                and strong_term_match
                and direct_enough
            ):
                consistent.append(row)

        # If the graph has no usable page metadata, retain only the strongest
        # lexically grounded candidates. Never fall back to unrelated pages.
        if len(consistent) == 1 and not isinstance(anchor_page, int):
            consistent = [
                row for row in grounded
                if row.get("keyword_overlap", 0) >= minimum_overlap
                and row.get("score", 0.0) >= anchor.get("score", 0.0) * 0.65
            ]

        consistent.sort(
            key=lambda row: (
                row.get("answerability", 0.0),
                row.get("score", 0.0),
                row.get("keyword_coverage", 0.0),
            ),
            reverse=True,
        )
        return consistent[:8]

    @staticmethod
    def compose_numbered_procedure(
        rows: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Reconstruct numbered source steps without generating new facts."""
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                row.get("pdf_page") or 0,
                row.get("chunk_id") or "",
            ),
        )
        variants: dict[int, list[dict[str, Any]]] = {}
        step_pattern = re.compile(
            r"(?<![\d.])(\d{1,2})\.\s+(.*?)"
            r"(?=\s+\d{1,2}\.\s+|$)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        stop_heading = re.compile(
            r"\b(?:rapid|alternative|modified)\s+method\b",
            flags=re.IGNORECASE,
        )
        incomplete_ending = re.compile(
            r"\b(?:the|a|an|and|or|to|of|in|for|with|be|into|using|until|as)$",
            flags=re.IGNORECASE,
        )

        for row in ordered_rows:
            text = row.get("text") or ""
            stop = stop_heading.search(text)
            if stop:
                text = text[:stop.start()]
            for match in step_pattern.finditer(text):
                number = int(match.group(1))
                raw_step = normalize_space(match.group(2))
                if not raw_step:
                    continue
                figure_noise = len(re.findall(
                    r"\bFig\.\s*\d+\.\d+", raw_step, flags=re.IGNORECASE
                ))
                cleaned = re.sub(
                    r"\s*\(Fig\.\s*\d+\.\d+\)",
                    "",
                    raw_step,
                    flags=re.IGNORECASE,
                )
                cleaned = clean_answer_text(cleaned)
                complete = bool(re.search(r"[.!?)]$", cleaned))
                incomplete = bool(incomplete_ending.search(cleaned.rstrip(".,;:")))
                fragment_ending = bool(re.search(
                    r"(?:\(\s*Fig\.|\bFig\.)$",
                    cleaned,
                    flags=re.IGNORECASE,
                ))
                unbalanced_parenthesis = (
                    cleaned.count("(") != cleaned.count(")")
                )
                complete = (
                    complete and not incomplete and not fragment_ending
                    and not unbalanced_parenthesis
                )
                quality = (
                    (40.0 if complete else -40.0)
                    + min(len(cleaned), 500) * 0.03
                    - max(len(cleaned) - 650, 0) * 0.20
                    - figure_noise * 18.0
                )
                variants.setdefault(number, []).append({
                    "number": number,
                    "text": cleaned,
                    "row": row,
                    "quality": quality,
                    "figure_noise": figure_noise,
                    "complete": complete,
                })

        chosen = []
        for number in sorted(variants):
            candidate = max(
                variants[number], key=lambda item: item["quality"]
            )
            # Do not expose a truncated step or a step corrupted by multiple
            # interleaved figure captions. The source Chunk remains visible.
            if not candidate["complete"] or candidate["figure_noise"] >= 3:
                continue
            chosen.append(candidate)

        if len(chosen) < 2:
            return "", []

        source_rows = []
        source_number: dict[str, int] = {}
        for item in chosen:
            chunk_id = item["row"].get("chunk_id")
            if chunk_id not in source_number:
                source_number[chunk_id] = len(source_rows) + 1
                source_rows.append(item["row"])

        answer_lines = ["According to the manual's numbered procedure:"]
        included_numbers = []
        for item in chosen:
            chunk_id = item["row"].get("chunk_id")
            if chunk_id not in source_number:
                continue
            citation = source_number[chunk_id]
            included_numbers.append(item["number"])
            answer_lines.append(
                f"{item['number']}. {item['text']} [S{citation}]"
            )
        if included_numbers:
            missing = [
                number
                for number in range(
                    min(included_numbers), max(included_numbers) + 1
                )
                if number not in included_numbers
            ]
            if missing:
                labels = ", ".join(str(number) for number in missing)
                answer_lines.append(
                    "Source step(s) "
                    f"{labels} contained interleaved figure-caption OCR and "
                    "were not restated to avoid introducing unsupported text."
                )
        return "\n".join(answer_lines), source_rows

    @staticmethod
    def compose_extract_answer(
        question: str, rows: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build an extractive answer and retain the source of every sentence."""
        if not rows:
            return "", []

        query_terms = set(content_terms(question))
        facets = question_facets(question)
        lowered_question = question.lower()
        requested_type = question_type(question)
        plan = answer_plan(question)
        procedure_question = requested_type == "procedure"
        appearance_question = any(
            phrase in lowered_question
            for phrase in ("appearance", "look like", "microscopic appearance", "show")
        )
        if procedure_question:
            procedure_answer, procedure_rows = (
                GraphV2QA.compose_numbered_procedure(rows)
            )
            if procedure_answer:
                return procedure_answer, procedure_rows

        descriptive_terms = {
            "appear", "appearance", "shape", "size", "colour", "color",
            "spore", "spores", "mycelium", "filament", "filaments",
            "round", "rectangular", "oval", "branch", "branches",
            "seen", "visible", "stained", "unstained",
        }
        action_pattern = re.compile(
            r"\b(stain|prepare|add|mix|wash|dry|fix|place|transfer|"
            r"incubate|centrifuge|allow|remove|filter|heat|cool|dilute|"
            r"discard|collect|examine|read|measure|pour|rinse)\b",
            flags=re.IGNORECASE,
        )
        statement_pattern = re.compile(
            r"\b(is|are|was|were|has|have|can|may|should|must|use|uses|"
            r"appear|appears|seen|found|show|shows|examine|examined|"
            r"characterized|contains|consists|stain|stained|prepare|prepared|"
            r"add|mix|wash|dry|fix|place|transfer|incubate|centrifuge|"
            r"allow|remove|filter|heat|cool|dilute|discard|collect|read|"
            r"measure|pour|rinse)\b",
            flags=re.IGNORECASE,
        )
        header_noise = re.compile(
            r"\b(fig\.|figure|contents|materials and reagents|equipment|"
            r"principle|chapter|parasitology)\b",
            flags=re.IGNORECASE,
        )

        candidates = []
        for row in rows:
            row_text = row.get("text") or ""
            if len(facets) > 1:
                evidence_units = GraphV2QA._evidence_segments(row_text)
                evidence_units.extend(
                    GraphV2QA._list_evidence_units(evidence_units)
                )
                if any(
                    question_type(facet) == "comparison"
                    for facet in facets
                ):
                    base_units = list(evidence_units)
                    evidence_units.extend(
                        " ".join(base_units[index:index + 3])
                        for index in range(len(base_units) - 2)
                    )
                # Join only an explanatory statement to its immediately
                # following consequence. Do not create overlapping pairs for
                # list/criteria clauses, which caused duplicated answers.
                if any(question_type(facet) == "reason" for facet in facets):
                    reason_terms = set().union(*(
                        set(content_terms(facet))
                        for facet in facets
                        if question_type(facet) == "reason"
                    ))
                    evidence_units.extend(
                        f"{evidence_units[index]} {evidence_units[index + 1]}"
                        for index in range(len(evidence_units) - 1)
                        if re.search(
                            r"\b(?:because|therefore|results? in|leads? to|"
                            r"will give|make it impossible)\b",
                            evidence_units[index + 1],
                            flags=re.IGNORECASE,
                        )
                        and len(
                            reason_terms & set(content_terms(
                                f"{evidence_units[index]} "
                                f"{evidence_units[index + 1]}"
                            ))
                        ) >= min(2, len(reason_terms))
                    )
            else:
                evidence_units = GraphV2QA._local_evidence_units(
                    question, row_text
                )
            for position, sentence in enumerate(evidence_units):
                sentence = clean_answer_text(sentence)
                if not 30 <= len(sentence) <= 700:
                    continue
                if re.search(
                    r"\b(?:the|a|an|and|or|to|of|in|for|with|be|into|using|until|as)$",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    # Reject text cut off at a Chunk boundary.
                    continue
                if sentence.endswith(":"):
                    # Never expose an empty list lead-in as an answer.
                    continue
                if not statement_pattern.search(sentence):
                    continue
                sentence_terms = set(content_terms(sentence))
                overlap = len(query_terms & sentence_terms)
                normalized_sentence = normalize_space(sentence).lower().rstrip("?.!:")
                normalized_question = normalize_space(question).lower().rstrip("?.!:")
                # A section title that merely repeats the question is evidence
                # location, not an answer statement.
                if normalized_sentence == normalized_question:
                    continue
                # OCR often concatenates navigation headings. Such fragments
                # must not become answers even when they repeat the question.
                if len(header_noise.findall(sentence)) >= 3:
                    continue
                is_action = bool(action_pattern.search(sentence))
                if procedure_question and not is_action:
                    continue
                continuation = (
                    procedure_question
                    and is_action
                    and overlap == 0
                    and row.get("keyword_overlap", 0) >= 2
                    and row.get("keyword_coverage", 0.0) >= 0.40
                )
                if overlap == 0 and not continuation:
                    continue
                facet_answerability = [
                    GraphV2QA._direct_answerability(facet, sentence)
                    for facet in facets
                ]
                facet_supported = {
                    index for index, facet in enumerate(facets)
                    if GraphV2QA._requirements_satisfied(facet, sentence)
                    and len(
                        set(content_terms(facet)) & sentence_terms
                    ) >= (
                        1 if re.search(
                            r"\b(?:characteristics?|criteria|features?|signs?)\b",
                            facet,
                            flags=re.IGNORECASE,
                        ) else min(2, len(set(content_terms(facet))))
                    )
                }
                if len(facets) > 1 and not facet_supported:
                    continue
                direct_answerability = max(
                    [
                        max(score, 0.65) if index in facet_supported else score
                        for index, score in enumerate(facet_answerability)
                    ],
                    default=0.0,
                )
                minimum_direct = 0.38 if len(facets) > 1 else (
                    0.58 if requested_type == "reason" else 0.38
                )
                if direct_answerability < minimum_direct:
                    continue
                if len(facets) > 1 and facet_supported:
                    facet_overlaps = [
                        len(set(content_terms(facets[index])) & sentence_terms)
                        for index in facet_supported
                    ]
                    effective_overlap = max(facet_overlaps, default=overlap)
                    effective_term_count = max(
                        min(len(set(content_terms(facets[index]))) for index in facet_supported),
                        1,
                    )
                    similarity = max(
                        float(GraphV2QA._similarities(
                            facets[index], [sentence]
                        )[0])
                        for index in facet_supported
                    )
                else:
                    effective_overlap = overlap
                    effective_term_count = max(len(query_terms), 1)
                    similarity = float(
                        GraphV2QA._similarities(question, [sentence])[0]
                    )
                coverage = effective_overlap / effective_term_count
                score = (
                    similarity
                    + effective_overlap * 0.12
                    + coverage * 0.20
                    + min(float(row.get("score") or 0.0) * 0.08, 0.08)
                    + direct_answerability * 0.55
                )
                if procedure_question and is_action:
                    score += 0.25
                if re.search(r"\b\d+(?:\.\d+)?\b|\bpH\b|\bminutes?\b", sentence):
                    score += 0.12
                if ";" in sentence and re.search(
                    r"\b(?:what are|which are|list|enumerate|uses?)\b",
                    lowered_question,
                ):
                    # Prefer the complete joined list over a locally similar
                    # single item or an unrelated sentence containing "used".
                    score += min(sentence.count(";") * 0.22, 0.88)
                if appearance_question:
                    words = set(re.findall(r"[a-z]+", sentence.lower()))
                    description_overlap = len(words & descriptive_terms)
                    if description_overlap == 0:
                        continue
                    score += min(description_overlap * 0.10, 0.40)
                candidates.append({
                    "score": score,
                    "selection_score": (
                        score / (1.0 + len(sentence) / 240.0)
                        if requested_type == "comparison" else score
                    ),
                    "sentence": sentence,
                    "row": row,
                    "position": position,
                    "covered_terms": query_terms & sentence_terms,
                    "covered_facets": facet_supported,
                    "requirements_complete": (
                        GraphV2QA._requirements_satisfied(
                            question, sentence
                        )
                    ),
                })

        if requested_type == "comparison":
            candidates.sort(
                key=lambda item: (
                    item["requirements_complete"],
                    -len(item["sentence"]),
                    item["selection_score"],
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda item: (
                    item["requirements_complete"], item["selection_score"]
                ),
                reverse=True,
            )
        selected = []
        seen_sentences = set()
        used_chunk_ids = set()
        covered_terms: set[str] = set()
        covered_facets: set[int] = set()
        exhaustive_compound = any(re.search(
            r"\b(?:characteristics?|criteria|features?|signs?|list|all)\b",
            facet,
            flags=re.IGNORECASE,
        ) or question_type(facet) == "time" for facet in facets)
        structured_list_question = bool(re.search(
            r"\b(?:what are|which are|list|enumerate|name all|uses?)\b",
            lowered_question,
        ))
        for item in candidates:
            full_sentence_key = item["sentence"].lower()
            sentence_key = full_sentence_key[:220]
            chunk_id = item["row"].get("chunk_id")
            if item["score"] < 0.20 or sentence_key in seen_sentences:
                continue
            if any(
                full_sentence_key in chosen["sentence"].lower()
                or chosen["sentence"].lower() in full_sentence_key
                for chosen in selected
            ):
                continue
            if (
                len(facets) > 1 and not exhaustive_compound
                and item["covered_facets"].issubset(covered_facets)
            ):
                continue
            new_terms = item["covered_terms"] - covered_terms
            if (
                selected and not new_terms and not procedure_question
                and plan["mode"] == "concise"
            ):
                continue
            selected.append(item)
            seen_sentences.add(sentence_key)
            used_chunk_ids.add(chunk_id)
            covered_terms.update(item["covered_terms"])
            covered_facets.update(item["covered_facets"])
            if (
                structured_list_question
                and item["requirements_complete"]
                and ";" in item["sentence"]
            ):
                break
            if item["requirements_complete"] and plan["stop_on_complete"]:
                break
            if (
                len(facets) > 1 and not exhaustive_compound
                and len(covered_facets) == len(facets)
            ):
                break
            if len(selected) == plan["max_claims"]:
                break

        if not selected:
            return "", []

        if len(facets) > 1:
            # Present claims in the same order as the clauses in the user's
            # question, then preserve their order inside the source Chunk.
            selected.sort(key=lambda item: (
                min(item["covered_facets"]) if item["covered_facets"] else len(facets),
                item["row"].get("pdf_page") or 0,
                item["row"].get("chunk_id") or "",
                item["position"],
            ))

        # Present procedural evidence in manual order after relevance-based
        # selection so the resulting instructions remain readable.
        if procedure_question:
            selected.sort(key=lambda item: (
                item["row"].get("pdf_page") or 0,
                item["row"].get("chunk_id") or "",
                item["position"],
            ))
        elif requested_type == "time":
            selected.sort(key=lambda item: (
                item["row"].get("pdf_page") or 0,
                item["row"].get("chunk_id") or "",
                item["position"],
            ))

        source_rows = []
        source_number = {}
        for item in selected:
            chunk_id = item["row"].get("chunk_id")
            if chunk_id not in source_number:
                source_number[chunk_id] = len(source_rows) + 1
                source_rows.append(item["row"])

        answer_parts = []
        for item in selected:
            sentence = item["sentence"].rstrip()
            if sentence and sentence[-1] not in ".!?":
                sentence += "."
            citation = source_number[item["row"].get("chunk_id")]
            answer_parts.append(f"{sentence} [S{citation}]")
        return " ".join(answer_parts), source_rows

    def _ensure_answer_model(self) -> None:
        if self._answer_model is not None:
            return
        with self._model_lock:
            if self._answer_model is not None:
                return
            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_ANSWER_MODEL, use_fast=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_ANSWER_MODEL,
                dtype="auto",
                low_cpu_mem_usage=False,
            )
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
            model.eval()
            self._answer_tokenizer = tokenizer
            self._answer_model = model

    @staticmethod
    def _synthesis_context(
        question: str, rows: list[dict[str, Any]]
    ) -> str:
        parts = []
        used = 0
        requested_type = question_type(question)
        plan = answer_plan(question)
        for index, row in enumerate(rows[:MAX_SYNTHESIS_CHUNKS], 1):
            text = normalize_space(row.get("text") or "")
            passages = [text]
            if requested_type != "procedure":
                passage_limit = (
                    1 if plan["stop_on_complete"]
                    else min(int(plan["max_claims"]), 6)
                )
                passages = GraphV2QA._focused_answer_passages(
                    question, text, limit=passage_limit
                )
                text = " ".join(passages)
            if not text:
                continue
            piece = (
                f"[E{index}] chunk={row.get('chunk_id')} "
                f"pdf_page={row.get('pdf_page')}\n{text[:2400]}\n"
            )
            if used + len(piece) > MAX_SYNTHESIS_CONTEXT_CHARS:
                break
            parts.append(piece)
            used += len(piece)
            if plan["stop_on_complete"] and any(
                GraphV2QA._requirements_satisfied(question, passage)
                for passage in passages
            ):
                break
        return "\n".join(parts)

    def synthesize_answer(
        self, question: str, rows: list[dict[str, Any]]
    ) -> str:
        """Rewrite verified evidence; the verifier decides whether to use it."""
        self._ensure_answer_model()
        context = self._synthesis_context(question, rows)
        if not context:
            return ""
        system_message = (
            "You are a correctness-first laboratory evidence editor. "
            "Answer only from the supplied evidence. Do not use outside "
            "knowledge. Do not invent or change numbers, units, reagent names, "
            "organisms, equipment, durations, temperatures, pH values or "
            "procedural steps. Combine consistent evidence, remove repetition "
            "and OCR/figure-caption noise, and write clear natural English. "
            "Keep each reason attached to the exact specimen, film type, "
            "organism or procedure named in the evidence; never transfer a "
            "reason from a nearby sentence about a different subject. "
            "If the evidence does not support an answer, output exactly "
            "INSUFFICIENT_EVIDENCE."
        )
        user_message = (
            f"Question: {question}\n\nVerified evidence:\n{context}\n\n"
            "Write a complete answer. For a procedure, preserve the supported "
            "step order. Summarize and paraphrase rather than copying long "
            "passages. Do not mention evidence IDs in the answer."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = self._answer_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._answer_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=6144,
        )
        with torch.inference_mode():
            generated = self._answer_model.generate(
                **inputs,
                max_new_tokens=520,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._answer_tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        answer = self._answer_tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )
        return normalize_space(answer)

    def _ensure_nli_model(self) -> None:
        if self._nli_model is not None:
            return
        with self._model_lock:
            if self._nli_model is not None:
                return
            model = CrossEncoder(NLI_VERIFIER_MODEL, device="cpu")
            labels = {
                str(label).lower(): int(index)
                for index, label in model.model.config.id2label.items()
            }
            entailment = next(
                (index for label, index in labels.items()
                 if "entail" in label),
                None,
            )
            contradiction = next(
                (index for label, index in labels.items()
                 if "contrad" in label),
                None,
            )
            if entailment is None or contradiction is None:
                raise RuntimeError(
                    f"Unsupported NLI label mapping: {labels}"
                )
            self._nli_model = model
            self._nli_entailment_index = entailment
            self._nli_contradiction_index = contradiction

    def verify_synthesis(
        self, answer: str, evidence_rows: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Apply deterministic value gates, then claim-level NLI."""
        normalized = normalize_space(answer)
        if not normalized:
            return False, "empty generation"
        if normalized == "INSUFFICIENT_EVIDENCE":
            return False, "model reported insufficient evidence"
        if len(normalized) < 30:
            return False, "generation too short"

        evidence_text = normalize_space(
            " ".join(row.get("text") or "" for row in evidence_rows)
        )
        if not evidence_text:
            return False, "empty evidence"

        number_pattern = re.compile(
            r"(?<![A-Za-z])\d+(?:\.\d+)?%?"
        )
        source_numbers = set(number_pattern.findall(evidence_text))
        answer_numbers = set(number_pattern.findall(normalized))
        novel_numbers = sorted(answer_numbers - source_numbers)
        if novel_numbers:
            return False, f"unsupported numeric values: {novel_numbers}"

        allowed_editor_terms = {
            "according", "manual", "first", "then", "next", "finally",
            "ensure", "carefully", "procedure", "step", "steps", "method",
            "using", "before", "after", "during", "only", "avoid",
        }
        evidence_terms = set(content_terms(evidence_text))
        answer_terms = set(content_terms(normalized))
        unsupported_terms = sorted(
            answer_terms - evidence_terms - allowed_editor_terms
        )
        if (
            answer_terms
            and len(unsupported_terms) / len(answer_terms) > 0.35
        ):
            return False, (
                "too many unsupported content terms: "
                f"{unsupported_terms[:12]}"
            )

        evidence_sentences = []
        for row in evidence_rows:
            evidence_sentences.extend(
                GraphV2QA._evidence_segments(row.get("text") or "")
            )
        claims = [
            normalize_space(claim)
            for claim in re.split(
                r"(?<=[.!?])\s+|\n+|(?=\s*[-*]\s+)",
                answer,
            )
            if len(normalize_space(claim).strip("-* ")) >= 20
        ]
        if not claims or not evidence_sentences:
            return False, "no verifiable claims"

        self._ensure_nli_model()
        for claim_number, claim in enumerate(claims, 1):
            lexical_scores = GraphV2QA._similarities(
                claim, evidence_sentences
            )
            if len(lexical_scores) == 0:
                return False, f"claim {claim_number}: no evidence"
            best_indices = np.argsort(lexical_scores)[::-1][:3]
            premise = " ".join(
                evidence_sentences[int(index)]
                for index in best_indices
            )
            probabilities = self._nli_model.predict(
                [(premise, claim)],
                apply_softmax=True,
                show_progress_bar=False,
            )
            scores = np.asarray(probabilities)[0]
            entailment = float(scores[self._nli_entailment_index])
            contradiction = float(
                scores[self._nli_contradiction_index]
            )
            if (
                entailment < NLI_ENTAILMENT_MIN
                or contradiction > NLI_CONTRADICTION_MAX
            ):
                return False, (
                    f"claim {claim_number}: entailment={entailment:.3f}, "
                    f"contradiction={contradiction:.3f}"
                )
        return True, f"{len(claims)} claim(s) entailed by evidence"

    @staticmethod
    def best_display_sources(
        question: str, answer: str, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        documents = [row.get("text") or "" for row in rows]
        coverage_scores = GraphV2QA._similarities(
            f"{question} {answer}", documents
        )
        ranked = []
        for row, coverage in zip(rows, coverage_scores):
            item = dict(row)
            item["answer_coverage"] = float(coverage)
            item["display_score"] = (
                0.30 * min(float(row.get("score") or 0.0), 1.0)
                + 0.70 * float(coverage)
            )
            ranked.append(item)
        ranked.sort(
            key=lambda row: (
                row["display_score"], row["answer_coverage"]
            ),
            reverse=True,
        )
        return ranked[:MAX_EVIDENCE_CHUNKS]

    def verified_images(self, question: str, relation_type: str | None,
                        chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        rows = self._run("""
        MATCH (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)-[link:ILLUSTRATED_BY]->(image:Image)
        WHERE chunk.id IN $chunk_ids AND image.file_path IS NOT NULL
          AND coalesce(link.semantic_score, 0.0) >= $minimum_score
          AND coalesce(image.content_relevance, 'undetermined') <> 'irrelevant'
          AND coalesce(image.final_type, image.predicted_type, '') <> 'fragment_or_noise'
        RETURN DISTINCT image.id AS id, image.file_path AS file_path,
               coalesce(link.image_type, image.final_type, image.predicted_type) AS image_type,
               link.semantic_score AS confidence, chunk.id AS chunk_id,
               page.pdf_page AS pdf_page,
               image.content_relevance AS content_relevance,
               image.classification_confidence AS classification_confidence
        ORDER BY confidence DESC LIMIT $limit
        """, chunk_ids=chunk_ids, minimum_score=IMAGE_MIN_SCORE,
             limit=MAX_IMAGES)
        return rows

    def image_path(self, image_id: str) -> str | None:
        rows = self._run(
            "MATCH (image:Image {id: $id}) RETURN image.file_path AS path", id=image_id
        )
        return rows[0]["path"] if rows else None

    @staticmethod
    def response(
        kind: str, question: str, answer: str,
        sources: list[dict[str, Any]], image_rows: list[dict[str, Any]],
        chunks_scanned: int = 0, candidates_considered: int = 0,
        synthesis_mode: str = "extractive",
    ) -> dict[str, Any]:
        images = [{
            "id": row["id"], "pdf_page": row.get("pdf_page"),
            "type": row.get("image_type"),
            "content_relevance": row.get("content_relevance"),
            "classification_confidence": round(
                float(row.get("classification_confidence") or 0.0), 4
            ),
            "confidence": round(float(row.get("confidence") or 0.0), 4),
            "chunk_id": row.get("chunk_id"),
            "url": f"/image/{row['id']}",
        } for row in image_rows]
        chunk_ids = [
            str(source.get("chunk_id"))
            for source in sources
            if source.get("chunk_id")
        ]
        cypher_ids = ", ".join(
            "'" + chunk_id.replace("'", "\\'") + "'"
            for chunk_id in chunk_ids
        )
        neo4j_query = ""
        if cypher_ids:
            neo4j_query = (
                "MATCH pagePath = (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)\n"
                f"WHERE chunk.id IN [{cypher_ids}]\n"
                "OPTIONAL MATCH entityPath = (chunk)-[:MENTIONS]->(:Entity)\n"
                "OPTIONAL MATCH imagePath = (chunk)-[:ILLUSTRATED_BY]->(:Image)\n"
                "RETURN pagePath, collect(DISTINCT entityPath) AS entityPaths, "
                "collect(DISTINCT imagePath) AS imagePaths;"
            )
        return {
            "kind": kind,
            "question": question,
            "answer": answer,
            "sources": sources[:MAX_EVIDENCE_CHUNKS],
            "images": images,
            "neo4j_query": neo4j_query,
            "retrieval_summary": {
                "chunks_scanned": chunks_scanned,
                "consistent_candidates": candidates_considered,
                "sources_used": min(len(sources), MAX_EVIDENCE_CHUNKS),
                "neo4j_verification": (
                    "verified" if kind == "domain_answer" else "not_verified"
                ),
                "synthesis_mode": synthesis_mode,
                "local_model": (
                    LOCAL_ANSWER_MODEL
                    if synthesis_mode == "local_model_verified"
                    else None
                ),
            },
        }


    def answer(self, question: str) -> dict[str, Any]:
        # Neo4j supplies Chunk text, Page location, mentioned Entities,
        # explicit Entity relations and verified Chunk-to-Image links.
        chunks = self.ranked_chunks(question)
        chunks_scanned = len(chunks)
        consistent = self.select_consistent_candidates(question, chunks)
        relation_type = relation_intent(question)

        if relation_type:
            facts = self.ranked_facts(question, relation_type)
            if (facts and facts[0]["score"] >= FACT_MIN_SCORE
                    and (facts[0]["subject_overlap"] > 0 or facts[0]["score"] >= 0.28)):
                source_id = facts[0].get("source_id")
                selected = [
                    row for row in facts if row.get("source_id") == source_id
                ][:MAX_EVIDENCE_CHUNKS]
                fact_answer = self.compose_fact_answer(relation_type, selected)
                fact_evidence = normalize_space(" ".join(
                    row.get("evidence") or row.get("text") or ""
                    for row in selected
                ))
                if self._requirements_satisfied(
                    question, f"{fact_answer} {fact_evidence}"
                ):
                    sources = [serializable_source(row) for row in selected]
                    chunk_ids = [
                        row["chunk_id"] for row in selected if row.get("chunk_id")
                    ]
                    image_rows = self.verified_images(
                        question, relation_type, chunk_ids
                    )
                    return self.response(
                        "domain_answer", question, fact_answer, sources, image_rows,
                        chunks_scanned, len(consistent),
                        synthesis_mode="verified_graph_fact",
                    )

        if not consistent:
            return self.response(
                "out_of_scope", question,
                "I could not verify this question in the provided laboratory manual. Please ask a more specific laboratory question.",
                [], [], chunks_scanned, 0,
            )

        # First build the deterministic, source-only answer. Completeness is
        # evaluated on the final combined answer, because one Chunk does not
        # always contain every required facet. When the combined extract is
        # complete, generation and NLI add latency without adding evidence.
        extract_answer, extract_rows = self.compose_extract_answer(
            question, consistent
        )
        direct_complete = bool(
            extract_answer and extract_rows
            and self._requirements_satisfied(question, extract_answer)
        )
        synthesis_mode = (
            "verified_extract" if direct_complete
            else "extractive_fallback"
        )
        final_answer = extract_answer if direct_complete else ""
        answer_rows: list[dict[str, Any]] = (
            extract_rows if direct_complete else []
        )

        if not direct_complete and ENABLE_LOCAL_SYNTHESIS:
            try:
                generated_answer = self.synthesize_answer(
                    question, consistent
                )
                print(f"[SYNTHESIS] Generated: {generated_answer}")
                verified, diagnostic = self.verify_synthesis(
                    generated_answer, consistent
                )
                print(
                    f"[SYNTHESIS] Verified={verified}; "
                    f"diagnostic={diagnostic}"
                )
                if verified:
                    final_answer = generated_answer
                    answer_rows = self.best_display_sources(
                        question, final_answer, consistent
                    )
                    synthesis_mode = "local_model_verified"
            except Exception as error:
                # Model or verifier failure must never expose an unchecked answer.
                print(
                    "[SYNTHESIS] Failure: "
                    f"{type(error).__name__}: {error}"
                )

        if not final_answer:
            # Never expose a partial extract merely because generation is
            # disabled or rejected. Completeness is a hard output invariant.
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but no complete and context-consistent answer could be verified. The system will not guess.",
                [], [], chunks_scanned, len(consistent),
                synthesis_mode="verification_failed",
            )

        sources = [
            serializable_source({
                **row,
                "evidence": row.get("text"),
                "confidence": row.get("score"),
            })
            for row in answer_rows
        ]
        chunk_ids = [
            row["chunk_id"] for row in answer_rows[:MAX_EVIDENCE_CHUNKS]
            if row.get("chunk_id")
        ]
        image_rows = self.verified_images(
            question, relation_type, chunk_ids
        )
        return self.response(
            "domain_answer", question, final_answer, sources, image_rows,
            chunks_scanned, len(consistent),
            synthesis_mode=synthesis_mode,
        )



app = FastAPI(title="Laboratory Evidence Assistant")
qa_engine = None
engine_lock = Lock()


def get_engine() -> GraphV2QA:
    global qa_engine
    if qa_engine is None:
        with engine_lock:
            if qa_engine is None:
                qa_engine = GraphV2QA()
    return qa_engine


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "graph": "v2"}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, Any]:
    question = normalize_space(request.query)
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    direct_answer = small_talk_answer(question)
    if direct_answer is not None:
        return GraphV2QA.response("small_talk", question, direct_answer, [], [])
    terms = content_terms(question)
    if not terms:
        return GraphV2QA.response(
            "ambiguous", question,
            "Please ask a more specific question about the laboratory manual.", [], []
        )
    if (
        len(terms) == 1
        and len(re.findall(r"[A-Za-z0-9]+", question)) <= 2
        and not re.search(
            r"\b(?:what|why|how|when|where|which|who)\b",
            question,
            flags=re.IGNORECASE,
        )
    ):
        return GraphV2QA.response(
            "ambiguous", question,
            "Please specify what you want to know about this laboratory topic.",
            [], [],
        )
    try:
        return get_engine().answer(question)
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The evidence service is temporarily unavailable. Please try again.",
        ) from error


@app.get("/image/{image_id}")
def image(image_id: str) -> FileResponse:
    if not re.fullmatch(r"img_\d{6}", image_id):
        raise HTTPException(status_code=400, detail="Invalid image identifier")
    raw_path = get_engine().image_path(image_id)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path)

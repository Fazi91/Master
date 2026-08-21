"""FastAPI interface for evidence-grounded questions over Graph V2."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any
import os
import re

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "webapp" / "index.html"
load_dotenv(ROOT / ".env")

MAX_SOURCES = 4
MAX_IMAGES = 2
MAX_EVIDENCE_CHUNKS = 2
FALLBACK_MIN_SCORE = 0.16
FACT_MIN_SCORE = 0.14
IMAGE_MIN_SCORE = 0.35

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


def normalize_token(token: str) -> str:
    token = token.lower()
    replacements = {
        "microscopic": "microscope", "microscopy": "microscope",
        "examinations": "examination", "procedures": "procedure",
        "reagents": "reagent", "specimens": "specimen",
        "measurements": "measurement", "organisms": "organism",
        "parasites": "parasite", "bacteria": "bacterium",
        "diagnostic": "diagnosis",
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
    return terms[:24]


def small_talk_answer(text: str) -> str | None:
    normalized = normalize_space(text).lower()
    for pattern, answer in SMALL_TALK:
        if re.fullmatch(pattern, normalized):
            return answer
    return None


def relation_intent(question: str) -> str | None:
    lowered = normalize_space(question).lower()

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
            row["semantic_score"] = float(semantic_score)
            row["keyword_overlap"] = overlap
            row["keyword_coverage"] = coverage
            row["exact_phrase"] = exact_phrase
            row["score"] = (
                float(semantic_score)
                + min(overlap * 0.10, 0.40)
                + min(coverage * 0.25, 0.25)
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
    def chunk_is_grounded(question: str, row: dict[str, Any]) -> bool:
        query_terms = set(content_terms(question))
        entity_terms = set(content_terms(" ".join(row.get("entity_names") or [])))
        text_terms = set(content_terms(row.get("text") or ""))
        entity_overlap = len(query_terms & entity_terms)
        text_overlap = len(query_terms & text_terms)
        return entity_overlap > 0 or text_overlap >= 2

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
        return [
            normalize_space(part)
            for part in re.split(r"(?<=[.!?;])\s+|\n+", cleaned)
            if normalize_space(part)
        ]

    @staticmethod
    def select_consistent_candidates(
        question: str, ranked_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Keep one coherent page/section cluster instead of mixing contexts."""
        grounded = [
            row for row in ranked_rows
            if row.get("score", 0.0) >= FALLBACK_MIN_SCORE
            and GraphV2QA.chunk_is_grounded(question, row)
        ]
        if not grounded:
            return []

        exact_rows = [row for row in grounded if row.get("exact_phrase")]
        anchor = exact_rows[0] if exact_rows else grounded[0]
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
            if row is anchor or (
                (same_section_window or same_page) and strong_term_match
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
                cleaned = normalize_space(cleaned)
                complete = bool(re.search(r"[.!?)]$", cleaned))
                incomplete = bool(incomplete_ending.search(cleaned.rstrip(".,;:")))
                quality = (
                    (40.0 if complete and not incomplete else -40.0)
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
                    "complete": complete and not incomplete,
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
            if len(chosen) == 8:
                break

        if len(chosen) < 2:
            return "", []

        source_rows = []
        source_number: dict[str, int] = {}
        for item in chosen:
            chunk_id = item["row"].get("chunk_id")
            if chunk_id not in source_number:
                if len(source_rows) >= MAX_EVIDENCE_CHUNKS:
                    continue
                source_number[chunk_id] = len(source_rows) + 1
                source_rows.append(item["row"])

        answer_lines = ["According to the manual's numbered procedure:"]
        for item in chosen:
            chunk_id = item["row"].get("chunk_id")
            if chunk_id not in source_number:
                continue
            citation = source_number[chunk_id]
            answer_lines.append(
                f"{item['number']}. {item['text']} [S{citation}]"
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
        lowered_question = question.lower()
        procedure_question = any(
            phrase in lowered_question
            for phrase in (
                "how", "procedure", "method", "prepare", "stain", "examine",
                "perform", "steps", "treatment", "technique",
            )
        )
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
            for position, sentence in enumerate(
                GraphV2QA._evidence_segments(row.get("text") or "")
            ):
                if not 30 <= len(sentence) <= 700:
                    continue
                if re.search(
                    r"\b(?:the|a|an|and|or|to|of|in|for|with|be|into|using|until|as)$",
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    # Reject text cut off at a Chunk boundary.
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
                coverage = overlap / max(len(query_terms), 1)
                similarity = float(GraphV2QA._similarities(question, [sentence])[0])
                score = (
                    similarity
                    + overlap * 0.12
                    + coverage * 0.20
                    + min(float(row.get("score") or 0.0) * 0.08, 0.08)
                )
                if procedure_question and is_action:
                    score += 0.25
                if re.search(r"\b\d+(?:\.\d+)?\b|\bpH\b|\bminutes?\b", sentence):
                    score += 0.12
                if appearance_question:
                    words = set(re.findall(r"[a-z]+", sentence.lower()))
                    description_overlap = len(words & descriptive_terms)
                    if description_overlap == 0:
                        continue
                    score += min(description_overlap * 0.10, 0.40)
                candidates.append({
                    "score": score,
                    "sentence": sentence,
                    "row": row,
                    "position": position,
                })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected = []
        seen_sentences = set()
        used_chunk_ids = set()
        for item in candidates:
            sentence_key = item["sentence"].lower()[:220]
            chunk_id = item["row"].get("chunk_id")
            if item["score"] < 0.20 or sentence_key in seen_sentences:
                continue
            # The answer may use at most two mutually consistent Chunks.
            if (chunk_id not in used_chunk_ids
                    and len(used_chunk_ids) >= MAX_EVIDENCE_CHUNKS):
                continue
            selected.append(item)
            seen_sentences.add(sentence_key)
            used_chunk_ids.add(chunk_id)
            if len(selected) == 4:
                break

        if not selected:
            return "", []

        # Present procedural evidence in manual order after relevance-based
        # selection so the resulting instructions remain readable.
        if procedure_question:
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

        answer_parts = [
            f"{item['sentence']} [S{source_number[item['row'].get('chunk_id')]}]"
            for item in selected
        ]
        return " ".join(answer_parts), source_rows

    def verified_images(self, question: str, relation_type: str | None,
                        chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        rows = self._run("""
        MATCH (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)-[link:ILLUSTRATED_BY]->(image:Image)
        WHERE chunk.id IN $chunk_ids AND image.file_path IS NOT NULL
          AND coalesce(link.semantic_score, 0.0) >= $minimum_score
        RETURN DISTINCT image.id AS id, image.file_path AS file_path,
               coalesce(link.image_type, image.final_type, image.predicted_type) AS image_type,
               link.semantic_score AS confidence, chunk.id AS chunk_id,
               page.pdf_page AS pdf_page,
               coalesce(image.caption, image.context_text, image.description, "") AS caption
        ORDER BY confidence DESC LIMIT 20
        """, chunk_ids=chunk_ids, minimum_score=IMAGE_MIN_SCORE)

        query_terms = set(content_terms(question))
        verified = []
        for row in rows:
            image_terms = set(content_terms(
                f"{row.get('caption', '')} {row.get('image_type', '')}"
            ))
            # A graph link and score are necessary but not sufficient. Require
            # textual image evidence to overlap the question.
            if not image_terms or not (query_terms & image_terms):
                continue
            verified.append(row)
            if len(verified) == MAX_IMAGES:
                break
        return verified

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
    ) -> dict[str, Any]:
        images = [{
            "id": row["id"], "pdf_page": row.get("pdf_page"),
            "type": row.get("image_type"),
            "caption": normalize_space(row.get("caption") or ""),
            "confidence": round(float(row.get("confidence") or 0.0), 4),
            "chunk_id": row.get("chunk_id"),
            "url": f"/image/{row['id']}",
        } for row in image_rows]
        return {
            "kind": kind,
            "question": question,
            "answer": answer,
            "sources": sources[:MAX_EVIDENCE_CHUNKS],
            "images": images,
            "retrieval_summary": {
                "chunks_scanned": chunks_scanned,
                "consistent_candidates": candidates_considered,
                "sources_used": min(len(sources), MAX_EVIDENCE_CHUNKS),
                "neo4j_verification": (
                    "verified" if kind == "domain_answer" else "not_verified"
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
                )

        if not consistent:
            return self.response(
                "out_of_scope", question,
                "I could not verify this question in the provided laboratory manual. Please ask a more specific laboratory question.",
                [], [], chunks_scanned, 0,
            )

        extract_answer, answer_rows = self.compose_extract_answer(
            question, consistent
        )
        if not extract_answer or not answer_rows:
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but no complete and context-consistent answer could be verified. The system will not guess.",
                [], [], chunks_scanned, len(consistent),
            )

        sources = [
            serializable_source({
                **row,
                "evidence": row.get("text"),
                "confidence": row.get("score"),
            })
            for row in answer_rows[:MAX_EVIDENCE_CHUNKS]
        ]
        chunk_ids = [
            row["chunk_id"] for row in answer_rows[:MAX_EVIDENCE_CHUNKS]
            if row.get("chunk_id")
        ]
        image_rows = self.verified_images(
            question, relation_type, chunk_ids
        )
        return self.response(
            "domain_answer", question, extract_answer, sources, image_rows,
            chunks_scanned, len(consistent),
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
    if not content_terms(question):
        return GraphV2QA.response(
            "ambiguous", question,
            "Please ask a more specific question about the laboratory manual.", [], []
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


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
FALLBACK_MIN_SCORE = 0.16
FACT_MIN_SCORE = 0.14

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
    lowered = question.lower()
    for relation_type, phrases in RELATION_RULES:
        if any(phrase in lowered for phrase in phrases):
            return relation_type
    return None


def serializable_source(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row.get("chunk_id"),
        "pdf_page": row.get("pdf_page"),
        "printed_page": row.get("printed_page"),
        "evidence": normalize_space(row.get("evidence") or row.get("text") or "")[:1200],
        "confidence": round(float(row.get("confidence") or row.get("score") or 0.0), 4),
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

    def ranked_chunks(self, question: str, limit: int = 4) -> list[dict[str, Any]]:
        self._ensure_chunk_index()
        word_query = self._word_vectorizer.transform([question])
        char_query = self._char_vectorizer.transform([question])
        scores = (
            0.55 * (self._word_matrix @ word_query.T).toarray().ravel()
            + 0.45 * (self._char_matrix @ char_query.T).toarray().ravel()
        )
        output = []
        for index in np.argsort(scores)[::-1][:limit]:
            row = dict(self._chunks[int(index)])
            row["score"] = float(scores[int(index)])
            output.append(row)
        return output

    @staticmethod
    def chunk_is_grounded(question: str, row: dict[str, Any]) -> bool:
        query_terms = set(content_terms(question))
        entity_terms = set(content_terms(" ".join(row.get("entity_names") or [])))
        text_terms = set(content_terms(row.get("text") or ""))
        entity_overlap = len(query_terms & entity_terms)
        text_overlap = len(query_terms & text_terms)
        return entity_overlap > 0 or text_overlap >= 2

    @staticmethod
    def compose_extract_answer(question: str, rows: list[dict[str, Any]]) -> str:
        query_terms = set(content_terms(question))
        candidates = []
        for row in rows:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", row.get("text") or ""):
                sentence = normalize_space(sentence)
                if not 30 <= len(sentence) <= 700:
                    continue
                overlap = len(query_terms & set(content_terms(sentence)))
                similarity = float(GraphV2QA._similarities(question, [sentence])[0])
                candidates.append((similarity + overlap * 0.12, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        chosen, seen = [], set()
        for score, sentence in candidates:
            key = sentence.lower()[:180]
            if score >= 0.16 and key not in seen:
                chosen.append(sentence)
                seen.add(key)
            if len(chosen) == 2:
                break
        return " ".join(chosen)

    def verified_images(self, question: str, relation_type: str | None,
                        pages: list[int]) -> list[dict[str, Any]]:
        lowered = question.lower()
        explicitly_visual = any(term in lowered for term in ("image", "figure", "picture", "show"))
        microscopy_subject = (
            any(term in lowered for term in ("cell", "organism", "parasite", "egg", "larva", "smear", "specimen"))
            and any(term in lowered for term in ("microscope", "microscopic", "microscopy"))
        )
        if relation_type == "USES_EQUIPMENT" and not explicitly_visual:
            return []
        if not explicitly_visual and not microscopy_subject:
            return []
        allowed = ["microscopy"] if microscopy_subject else [
            "microscopy", "clinical_or_laboratory", "diagram_or_chart"
        ]
        return self._run("""
        MATCH (page:Page)-[occurrence:CONTAINS_IMAGE]->(image:Image)
        WHERE page.pdf_page IN $pages AND image.file_path IS NOT NULL
          AND coalesce(image.classification_confidence, 0.0) >= 0.65
          AND trim(coalesce(image.final_type, '')) <> ''
          AND image.final_type IN $allowed_types
        RETURN DISTINCT image.id AS id, image.file_path AS file_path,
               image.final_type AS image_type,
               image.classification_confidence AS confidence,
               page.pdf_page AS pdf_page, occurrence.page_coverage AS page_coverage
        ORDER BY confidence DESC, page_coverage DESC LIMIT $limit
        """, pages=pages, allowed_types=allowed, limit=MAX_IMAGES)

    def image_path(self, image_id: str) -> str | None:
        rows = self._run(
            "MATCH (image:Image {id: $id}) RETURN image.file_path AS path", id=image_id
        )
        return rows[0]["path"] if rows else None

    @staticmethod
    def response(kind: str, question: str, answer: str,
                 sources: list[dict[str, Any]],
                 image_rows: list[dict[str, Any]]) -> dict[str, Any]:
        images = [{
            "id": row["id"], "pdf_page": row.get("pdf_page"),
            "type": row.get("image_type"),
            "confidence": round(float(row.get("confidence") or 0.0), 4),
            "url": f"/image/{row['id']}",
        } for row in image_rows]
        return {"kind": kind, "question": question, "answer": answer,
                "sources": sources, "images": images}

    def answer(self, question: str) -> dict[str, Any]:
        relation_type = relation_intent(question)
        if relation_type:
            facts = self.ranked_facts(question, relation_type)
            if (facts and facts[0]["score"] >= FACT_MIN_SCORE
                    and (facts[0]["subject_overlap"] > 0 or facts[0]["score"] >= 0.28)):
                source_id = facts[0].get("source_id")
                selected = [row for row in facts if row.get("source_id") == source_id][:MAX_SOURCES]
                answer = self.compose_fact_answer(relation_type, selected)
                sources = [serializable_source(row) for row in selected]
                pages = list(dict.fromkeys(
                    int(row["pdf_page"]) for row in selected if row.get("pdf_page") is not None
                ))
                images = self.verified_images(question, relation_type, pages)
                return self.response("domain_answer", question, answer, sources, images)

        chunks = self.ranked_chunks(question, limit=MAX_SOURCES)
        if (not chunks or chunks[0]["score"] < FALLBACK_MIN_SCORE
                or not self.chunk_is_grounded(question, chunks[0])):
            return self.response(
                "out_of_scope", question,
                "I could not verify this question in the provided laboratory manual. Please ask a more specific laboratory question.",
                [], [],
            )
        answer = self.compose_extract_answer(question, chunks)
        if not answer:
            return self.response(
                "not_found", question,
                "Relevant material was found, but it was not sufficient to produce a reliable answer. Please make the question more specific.",
                [], [],
            )
        sources = [serializable_source({
            **row, "evidence": row.get("text"), "confidence": row.get("score")
        }) for row in chunks[:2]]
        pages = list(dict.fromkeys(
            int(row["pdf_page"]) for row in chunks[:2] if row.get("pdf_page") is not None
        ))
        images = self.verified_images(question, relation_type, pages)
        return self.response("domain_answer", question, answer, sources, images)


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

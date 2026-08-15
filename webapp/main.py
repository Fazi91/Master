from pathlib import Path
import os
import re
import threading

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel


load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "webapp" / "index.html"

SMALL_TALK = [
    (r"^(?:(?:hi|hello|hey|salam)[!,. ]*)+$",
     "Hello! How can I help you with the laboratory manual?"),
    (r"^(good morning|good afternoon|good evening)[!,. ]*$",
     "Hello! How can I help you with the laboratory manual?"),
    (r"^(how are you|how are you doing)[?!. ]*$",
     "I'm doing well, thank you. You can ask me a question about the laboratory manual."),
    (r"^(who are you|what are you|what is your name|what's your name)[?!. ]*$",
     "I am a laboratory assistant. I answer technical questions using the provided WHO laboratory manual."),
    (r"^(thanks|thank you|thank you very much)[!. ]*$",
     "You're welcome! Ask me whenever you need information from the laboratory manual."),
    (r"^(what can you do|help|help me|how can you help me|how should i ask)[?!. ]*$",
     "I can answer questions from the laboratory manual and show the supporting page, evidence and related image when available."),
    (r"^(bye|goodbye|see you|see you later|khodafez)[?!. ]*$",
     "Goodbye! Come back whenever you need help with the laboratory manual."),
]

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
    "could", "do", "does", "for", "from", "give", "have", "how", "i",
    "in", "information", "is", "it", "me", "of", "on", "or", "please",
    "should", "tell", "the", "this", "to", "used", "what", "when",
    "where", "which", "who", "why", "with", "would", "you",
}
GENERIC_ONLY = {
    "answer", "explain", "help", "know", "mean", "more", "question",
    "say", "something", "thing", "understand",
}

DOMAIN_TERMS = {
    "acid", "agar", "antibody", "antigen", "bacteria", "bacterial",
    "blood", "cell", "centrifuge", "culture", "diagnosis", "disease",
    "egg", "equipment", "examination", "faeces", "feces", "fungus",
    "haemoglobin", "hemoglobin", "infection", "laboratory", "larvae",
    "malaria", "measurement", "microscope", "microscopic", "organism",
    "parasite", "plasma", "procedure", "reagent", "sample", "serum",
    "slide", "smear", "specimen", "sputum", "stain", "staining",
    "stool", "test", "tissue", "urine", "vector", "virus", "worm",
}

SEMANTIC_MIN = 0.43
SEMANTIC_STRONG = 0.58
MAX_SOURCES = 4


class QuestionRequest(BaseModel):
    query: str


def content_terms(text: str):
    terms = [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    ]
    return list(dict.fromkeys(terms))[:16]


def route_message(text: str):
    normalized = " ".join(text.lower().split())
    for pattern, answer in SMALL_TALK:
        if re.fullmatch(pattern, normalized):
            return "small_talk", answer

    terms = content_terms(normalized)
    meaningful = [term for term in terms if term not in GENERIC_ONLY]
    if not meaningful:
        return "ambiguous", "Please ask a more specific question about the laboratory manual."

    if not (set(meaningful) & DOMAIN_TERMS):
        return (
            "guidance",
            "I can continue our conversation, but I only search the database for a "
            "specific laboratory question. For example, ask: Which equipment is used "
            "for microscopic examination?",
        )
    return "domain_candidate", None


class GraphV2Retriever:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        if not uri or not password:
            raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._lock = threading.Lock()
        self._model = None
        self._chunks = None
        self._embeddings = None

    def _load_chunks(self):
        query = """
        MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        RETURN c.id AS chunk_id, c.text AS text,
               p.pdf_page AS pdf_page, p.printed_page AS printed_page,
               collect(DISTINCT {
                   id: e.id, name: e.canonical_name, normalized: e.normalized_name,
                   type: e.entity_type
               }) AS entities
        ORDER BY c.id
        """
        with self.driver.session(database=self.database) as session:
            return [dict(row) for row in session.run(query)]

    def _ensure_semantic_index(self):
        if self._embeddings is not None:
            return
        with self._lock:
            if self._embeddings is not None:
                return
            from sentence_transformers import SentenceTransformer
            self._chunks = self._load_chunks()
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
            texts = [row["text"] for row in self._chunks]
            self._embeddings = self._model.encode(
                texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=True
            ).astype("float32")

    def retrieve(self, question: str):
        self._ensure_semantic_index()
        query_vector = self._model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0].astype("float32")
        scores = self._embeddings @ query_vector
        order = np.argsort(scores)[::-1][:12]
        terms = set(content_terms(question)) - GENERIC_ONLY

        ranked = []
        for index in order:
            row = dict(self._chunks[int(index)])
            text_tokens = set(re.findall(r"[a-z0-9]+", row["text"].lower()))
            entity_tokens = set()
            for entity in row.get("entities") or []:
                entity_tokens.update(re.findall(r"[a-z0-9]+", (entity.get("name") or "").lower()))
            lexical_hits = len(terms & text_tokens)
            entity_hits = len(terms & entity_tokens)
            row["semantic_score"] = float(scores[int(index)])
            row["lexical_hits"] = lexical_hits
            row["entity_hits"] = entity_hits
            row["validation_score"] = lexical_hits + (2 * entity_hits)
            ranked.append(row)

        if not ranked:
            return []
        top = ranked[0]
        validated = (
            top["semantic_score"] >= SEMANTIC_MIN
            and (top["validation_score"] > 0 or top["semantic_score"] >= SEMANTIC_STRONG)
        )
        if not validated:
            return []

        selected = [
            row for row in ranked
            if row["semantic_score"] >= max(SEMANTIC_MIN, top["semantic_score"] - 0.12)
            and (row["validation_score"] > 0 or row["semantic_score"] >= SEMANTIC_STRONG)
        ]
        return selected[:MAX_SOURCES]

    def images_for_pages(self, pages: list[int], limit: int = 2):
        query = """
        MATCH (p:Page)-[:CONTAINS_IMAGE]->(i:Image)
        WHERE p.pdf_page IN $pages AND i.file_path IS NOT NULL
        RETURN DISTINCT i.id AS id, i.file_path AS file_path,
               i.final_type AS final_type, p.pdf_page AS pdf_page
        ORDER BY p.pdf_page, i.id
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            return [dict(row) for row in session.run(query, pages=pages, limit=limit)]

    def image_path(self, image_id: str):
        with self.driver.session(database=self.database) as session:
            row = session.run(
                "MATCH (i:Image {id: $id}) RETURN i.file_path AS path", id=image_id
            ).single()
        return row["path"] if row else None

    def synthesize(self, question: str, rows: list):
        return extractive_fallback(question, rows)


def extractive_fallback(question: str, rows: list):
    terms = set(content_terms(question)) - GENERIC_ONLY
    candidates = []
    for row in rows:
        sentences = re.split(r"(?<=[.!?])\s+", row.get("text") or "")
        for sentence in sentences:
            if 25 <= len(sentence) <= 600:
                tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                candidates.append((len(terms & tokens), sentence.strip()))
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    answer, seen = [], set()
    for overlap, sentence in candidates:
        key = sentence.lower()[:180]
        if overlap > 0 and key not in seen:
            seen.add(key)
            answer.append(sentence)
        if len(answer) == 3:
            break
    return " ".join(answer) if answer else "Not found in the provided manual."


app = FastAPI(title="Laboratory Evidence Assistant")
retriever = None
retriever_lock = threading.Lock()


def get_retriever():
    """Create the Neo4j retriever only for an actual domain question."""
    global retriever
    if retriever is None:
        with retriever_lock:
            if retriever is None:
                retriever = GraphV2Retriever()
    return retriever


@app.get("/")
def home():
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "graph": "v2"}


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    route, direct_answer = route_message(question)
    if route != "domain_candidate":
        return {"kind": route, "question": question, "answer": direct_answer,
                "sources": [], "images": []}

    graph = get_retriever()
    rows = graph.retrieve(question)
    if not rows:
        return {
            "kind": "out_of_scope", "question": question,
            "answer": "This question is outside the scope of the provided laboratory manual.",
            "sources": [], "images": []
        }

    # Keep the answer, citation and image grounded in the same best chunk.
    rows = rows[:1]
    answer = graph.synthesize(question, rows)
    sources = [{
        "chunk_id": row["chunk_id"],
        "pdf_page": row.get("pdf_page"),
        "printed_page": row.get("printed_page"),
        "evidence": row.get("text", "")[:900],
        "semantic_score": round(row["semantic_score"], 4),
    } for row in rows]

    pages = list(dict.fromkeys(
        row["pdf_page"] for row in rows if row.get("pdf_page") is not None
    ))
    image_rows = graph.images_for_pages(pages)
    images = [{
        "id": image["id"], "pdf_page": image["pdf_page"],
        "type": image.get("final_type"), "url": f"/image/{image['id']}"
    } for image in image_rows]

    return {"kind": "domain_answer", "question": question,
            "answer": answer, "sources": sources, "images": images}


@app.get("/image/{image_id}")
def image(image_id: str):
    raw_path = get_retriever().image_path(image_id)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path)

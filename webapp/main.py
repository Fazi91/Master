"""Generic, evidence-grounded QA over Graph V2.

No corpus question, answer, topic, method, or PDF phrase is encoded here.
Every request uses the same plan/retrieve/rerank/compose/verify pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
import csv
import json
import os
import re

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from neo4j import GraphDatabase
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "webapp" / "index.html"
CHUNKS_FILE = ROOT / "data" / "graph_v2" / "chunks.csv"
load_dotenv(ROOT / ".env")

DENSE_MODEL = os.getenv("DENSE_RETRIEVER_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
GENERATOR_MODEL = os.getenv(
    "LOCAL_ANSWER_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
)
NLI_MODEL = os.getenv(
    "NLI_VERIFIER_MODEL", "cross-encoder/nli-deberta-v3-small"
)
USE_DENSE = os.getenv("USE_NEURAL_RETRIEVAL", "true").lower() in {
    "1", "true", "yes", "on"
}
TOP_FIRST_STAGE = int(os.getenv("TOP_FIRST_STAGE", "80"))
TOP_RERANK = int(os.getenv("TOP_RERANK", "24"))
MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE_CHUNKS", "10"))
MAX_DISPLAY_SOURCES = int(os.getenv("MAX_DISPLAY_SOURCES", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))
NLI_MIN = float(os.getenv("NLI_ENTAILMENT_MIN", "0.55"))
IMAGE_MIN_SCORE = float(os.getenv("IMAGE_MIN_SCORE", "0.35"))

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "is", "it", "may", "of", "on", "or", "should", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "would", "please", "explain", "describe", "tell", "me",
}


class QuestionRequest(BaseModel):
    query: str


@dataclass(frozen=True)
class Facet:
    label: str
    query: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class Plan:
    question: str
    answer_type: str
    facets: tuple[Facet, ...]


def compact(value: str) -> str:
    return " ".join((value or "").split())


def terms(value: str) -> list[str]:
    return list(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in STOPWORDS
    ))


def parse_json(value: str) -> dict[str, Any] | None:
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
    match = re.search(r"\{.*\}", value, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def integer(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class EvidenceQA:
    def __init__(self) -> None:
        self.driver = self._graph()
        self.rows = self._chunks()
        if not self.rows:
            raise RuntimeError("The corpus contains no chunks")
        self.rows.sort(key=lambda row: (
            row.get("pdf_page") or 0,
            row.get("chunk_index") or 0,
            row["chunk_id"],
        ))
        self.positions = {
            row["chunk_id"]: position for position, row in enumerate(self.rows)
        }
        self.entity_positions: dict[str, list[int]] = {}
        for position, row in enumerate(self.rows):
            for entity in row.get("entities", []):
                self.entity_positions.setdefault(entity.lower(), []).append(position)
        corpus = [self._search_text(row) for row in self.rows]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
            stop_words="english",
        )
        self.lexical_matrix = self.vectorizer.fit_transform(corpus)
        self.lock = Lock()
        self.dense = None
        self.dense_matrix = None
        self.reranker = None
        self.tokenizer = None
        self.generator = None
        self.nli = None

    @staticmethod
    def _graph():
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            return None
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver

    def _chunks(self) -> list[dict[str, Any]]:
        if self.driver:
            try:
                with self.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
                        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                        RETURN c.id AS chunk_id, c.text AS text,
                               p.pdf_page AS pdf_page,
                               p.printed_page AS printed_page,
                               coalesce(c.chunk_index_on_page, 0) AS chunk_index,
                               collect(DISTINCT e.canonical_name) AS entities
                        ORDER BY p.pdf_page, chunk_index, c.id
                        """
                    )
                    rows = [dict(record) for record in result]
                if rows:
                    return rows
            except Exception as error:
                print(f"[GRAPH] using CSV fallback: {error}")
        entity_names = {}
        with (CHUNKS_FILE.parent / "entities.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            entity_names = {
                row["entity_id"]: row["canonical_name"]
                for row in csv.DictReader(stream)
            }
        chunk_entities: dict[str, list[str]] = {}
        with (CHUNKS_FILE.parent / "rel_chunk_entity.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for mention in csv.DictReader(stream):
                name = entity_names.get(mention["entity_id"])
                if name:
                    values = chunk_entities.setdefault(mention["chunk_id"], [])
                    if name not in values:
                        values.append(name)
        with CHUNKS_FILE.open(encoding="utf-8-sig", newline="") as stream:
            return [{
                "chunk_id": row["chunk_id"],
                "text": row["chunk_text"],
                "pdf_page": integer(row.get("pdf_page")),
                "printed_page": integer(row.get("printed_page")),
                "chunk_index": integer(row.get("chunk_index_on_page")) or 0,
                "entities": chunk_entities.get(row["chunk_id"], []),
            } for row in csv.DictReader(stream)]

    @staticmethod
    def _search_text(row: dict[str, Any]) -> str:
        entities = " ".join(row.get("entities", []))
        return f"{row['text']}\n{entities}" if entities else row["text"]

    def _generator(self) -> None:
        if self.generator is not None:
            return
        with self.lock:
            if self.generator is not None:
                return
            self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
            self.generator = AutoModelForCausalLM.from_pretrained(
                GENERATOR_MODEL,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                self.generator.to("cpu")
            self.generator.eval()

    def complete(self, system: str, prompt: str, limit: int) -> str:
        self._generator()
        rendered = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt")
        device = next(self.generator.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            output = self.generator.generate(
                **encoded,
                max_new_tokens=limit,
                do_sample=False,
                repetition_penalty=1.04,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def plan(self, question: str) -> Plan:
        instruction = (
            "Build a retrieval plan without answering the question. Split only "
            "independently answerable parts, compared alternatives, requested "
            "dimensions, or explicit conditions. Preserve all entities, "
            "modifiers, relationships, technical terms, and numbers. Make each "
            "query self-contained. Do not add knowledge. Return strict JSON."
        )
        schema = {
            "answer_type": "fact|procedure|comparison|reason|list|calculation",
            "facets": [{
                "label": "short neutral label",
                "query": "self-contained retrieval query",
                "requirements": ["word or phrase from the question"],
            }],
        }
        try:
            parsed = parse_json(self.complete(
                instruction,
                f"Question: {question}\nOutput schema: "
                f"{json.dumps(schema)}",
                500,
            ))
        except Exception as error:
            print(f"[PLAN] fallback: {error}")
            parsed = None
        facets = []
        if parsed:
            for item in parsed.get("facets", [])[:8]:
                if not isinstance(item, dict):
                    continue
                query = compact(str(item.get("query") or ""))
                if not query:
                    continue
                # The facet supplies focus; the original question preserves
                # every qualifier even if the planner omits one accidentally.
                query = f"{query}. Full question: {question}"
                required = tuple(dict.fromkeys(
                    compact(str(value))
                    for value in item.get("requirements", [])
                    if compact(str(value))
                ))
                facets.append(Facet(
                    compact(str(item.get("label") or "")),
                    query,
                    required,
                ))
        if not facets:
            facets = [Facet("", question, tuple(terms(question)))]
        return Plan(
            question,
            compact(str((parsed or {}).get("answer_type") or "fact")),
            tuple(facets),
        )

    def _retrievers(self) -> None:
        with self.lock:
            if self.reranker is None:
                self.reranker = CrossEncoder(RERANK_MODEL)
            if USE_DENSE and self.dense is None:
                self.dense = SentenceTransformer(DENSE_MODEL)
                self.dense_matrix = self.dense.encode(
                    [self._search_text(row) for row in self.rows],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )

    def retrieve(self, facet: Facet) -> list[dict[str, Any]]:
        self._retrievers()
        query_vector = self.vectorizer.transform([facet.query])
        lexical = (self.lexical_matrix @ query_vector.T).toarray().ravel()
        selected = set(np.argsort(-lexical)[:TOP_FIRST_STAGE].tolist())
        if self.dense is not None:
            query_dense = self.dense.encode(
                [facet.query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            dense_scores = self.dense_matrix @ query_dense
            selected.update(
                np.argsort(-dense_scores)[:TOP_FIRST_STAGE].tolist()
            )
        candidates = [self.rows[index] for index in selected]
        scores = self.reranker.predict(
            [[facet.query, self._search_text(row)] for row in candidates],
            show_progress_bar=False,
        )
        ranked = sorted(
            ({**row, "score": float(score)}
             for row, score in zip(candidates, scores)),
            key=lambda row: row["score"],
            reverse=True,
        )[:TOP_RERANK]
        return self.expand(ranked)

    def expand(self, anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add adjacent chunks generically so boundary text is not lost."""
        result: dict[str, dict[str, Any]] = {}
        for rank, anchor in enumerate(anchors):
            position = self.positions[anchor["chunk_id"]]
            for distance in (-2, -1, 0, 1, 2):
                neighbour_position = position + distance
                if not 0 <= neighbour_position < len(self.rows):
                    continue
                row = self.rows[neighbour_position]
                page_distance = abs(
                    (row.get("pdf_page") or 0)
                    - (anchor.get("pdf_page") or 0)
                )
                if page_distance > 1:
                    continue
                score = anchor["score"] - abs(distance) * 0.35 - rank * 0.01
                old = result.get(row["chunk_id"])
                if old is None or score > old["score"]:
                    result[row["chunk_id"]] = {**row, "score": score}
            for entity in anchor.get("entities", []):
                for related_position in self.entity_positions.get(
                    entity.lower(), []
                )[:12]:
                    row = self.rows[related_position]
                    if row["chunk_id"] == anchor["chunk_id"]:
                        continue
                    score = anchor["score"] - 0.65 - rank * 0.01
                    old = result.get(row["chunk_id"])
                    if old is None or score > old["score"]:
                        result[row["chunk_id"]] = {**row, "score": score}
        return sorted(
            result.values(), key=lambda row: row["score"], reverse=True
        )

    @staticmethod
    def evidence(
        plan: Plan, groups: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        chosen: dict[str, dict[str, Any]] = {}
        quota = max(2, MAX_EVIDENCE // max(1, len(plan.facets)))
        for facet_index, candidates in enumerate(groups):
            for row in candidates[:quota]:
                item = chosen.setdefault(row["chunk_id"], dict(row))
                item.setdefault("facets", []).append(facet_index)
        pool = sorted(
            (row for group in groups for row in group),
            key=lambda row: row["score"],
            reverse=True,
        )
        for row in pool:
            chosen.setdefault(row["chunk_id"], dict(row))
            if len(chosen) >= MAX_EVIDENCE:
                break
        return sorted(chosen.values(), key=lambda row: (
            row.get("pdf_page") or 0,
            row.get("chunk_index") or 0,
        ))

    def compose(
        self, plan: Plan, evidence: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        context = "\n\n".join(
            f"[E{index}] {compact(row['text'])}"
            for index, row in enumerate(evidence, 1)
        )[:MAX_CONTEXT_CHARS]
        facets = [
            {
                "label": facet.label,
                "query": facet.query,
                "requirements": list(facet.requirements),
            }
            for facet in plan.facets
        ]
        instruction = (
            "Answer exclusively from the supplied evidence. Cover every "
            "requested facet, comparison side, dimension, and condition. "
            "Preserve conditional distinctions. Do not add outside knowledge, "
            "steps, values, explanations, or terminology. If the evidence is "
            "not complete, set complete to false. Return strict JSON with "
            "complete, answer, and claims. Cite every answer sentence as [E#]. "
            "Each claim contains the exact sentence and its evidence_ids."
        )
        schema = {
            "complete": True,
            "answer": "answer with [E#] after every sentence",
            "claims": [{
                "sentence": "one factual answer sentence",
                "evidence_ids": ["E1"],
            }],
        }
        parsed = parse_json(self.complete(
            instruction,
            f"Question: {plan.question}\n"
            f"Answer type: {plan.answer_type}\n"
            f"Required facets: {json.dumps(facets)}\n"
            f"Output schema: {json.dumps(schema)}\n\n"
            f"Evidence:\n{context}",
            1500,
        ))
        if (
            not parsed
            or parsed.get("complete") is not True
            or not compact(str(parsed.get("answer") or ""))
            or not isinstance(parsed.get("claims"), list)
        ):
            return None
        return {
            "answer": compact(str(parsed["answer"])),
            "claims": parsed["claims"],
        }

    def _nli(self) -> None:
        if self.nli is not None:
            return
        with self.lock:
            if self.nli is None:
                self.nli = CrossEncoder(NLI_MODEL)

    def verify(
        self,
        plan: Plan,
        generated: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> tuple[bool, str, list[int]]:
        pairs = []
        cited = []
        answer_without_citations = compact(re.sub(
            r"\s*\[E\d+\]", "", generated["answer"]
        ))
        for claim in generated["claims"]:
            if not isinstance(claim, dict):
                return False, "invalid claim", []
            sentence = compact(str(claim.get("sentence") or ""))
            raw_ids = claim.get("evidence_ids")
            if not sentence or not isinstance(raw_ids, list):
                return False, "claim has no evidence", []
            sentence_without_citation = compact(re.sub(
                r"\s*\[E\d+\]", "", sentence
            ))
            if sentence_without_citation not in answer_without_citations:
                return False, "claim is not present in the answer", []
            ids = []
            for raw in raw_ids:
                match = re.fullmatch(r"E?(\d+)", str(raw).strip())
                if match and 1 <= int(match.group(1)) <= len(evidence):
                    ids.append(int(match.group(1)))
            if not ids:
                return False, "invalid evidence reference", []
            premise = " ".join(evidence[index - 1]["text"] for index in ids)
            claim_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", sentence))
            source_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", premise))
            if claim_numbers - source_numbers:
                return False, "unsupported numeric value", []
            pairs.append([compact(premise), sentence])
            cited.extend(ids)
        answer_citations = [
            int(value) for value in re.findall(r"\[E(\d+)\]", generated["answer"])
        ]
        if not answer_citations:
            return False, "answer has no citations", []
        if any(not 1 <= value <= len(evidence) for value in answer_citations):
            return False, "answer has an invalid citation", []
        if not set(answer_citations).issubset(set(cited)):
            return False, "answer cites evidence not verified by a claim", []
        self._nli()
        raw = np.asarray(self.nli.predict(pairs, show_progress_bar=False))
        if raw.ndim != 2:
            return False, "unexpected NLI output", []
        probabilities = torch.softmax(torch.tensor(raw), dim=1).numpy()
        labels = [
            str(label).lower()
            for label in self.nli.model.config.id2label.values()
        ]
        entailment = next(
            (index for index, label in enumerate(labels)
             if "entail" in label),
            None,
        )
        if entailment is None:
            return False, "NLI label mapping unavailable", []
        if any(float(row[entailment]) < NLI_MIN for row in probabilities):
            return False, "claim is not entailed", []
        coverage = parse_json(self.complete(
            "Check only completeness. Does the candidate answer every part, "
            "side, requested dimension, and condition in the question? Do not "
            "judge style. Return strict JSON: complete boolean and missing list.",
            f"Question: {plan.question}\n"
            f"Candidate answer: {generated['answer']}",
            300,
        ))
        if not coverage or coverage.get("complete") is not True:
            return False, (
                f"incomplete: {(coverage or {}).get('missing', [])}"
            ), []
        return True, "verified", list(dict.fromkeys(cited))

    @staticmethod
    def source(row: dict[str, Any]) -> dict[str, Any]:
        page = row.get("pdf_page")
        chunk_id = row["chunk_id"]
        return {
            "chunk_id": chunk_id,
            "pdf_page": page,
            "printed_page": row.get("printed_page"),
            "text": row["text"],
            "evidence": row["text"],
            "confidence": round(float(row.get("score") or 0), 4),
            "graph_location": (
                f"(:Page {{pdf_page: {page}}})-[:HAS_CHUNK]->"
                f"(:Chunk {{id: '{chunk_id}'}})"
            ),
        }

    def images(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not self.driver or not chunk_ids:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
                          -[r:ILLUSTRATED_BY]->(i:Image)
                    WHERE c.id IN $ids AND i.file_path IS NOT NULL
                      AND coalesce(r.semantic_score, 0) >= $minimum
                    RETURN i.id AS id, p.pdf_page AS pdf_page,
                           c.id AS chunk_id,
                           r.semantic_score AS confidence
                    ORDER BY confidence DESC LIMIT 2
                    """,
                    ids=chunk_ids,
                    minimum=IMAGE_MIN_SCORE,
                )
                rows = [dict(record) for record in result]
            return [{
                "id": row["id"],
                "url": f"/image/{row['id']}",
                "pdf_page": row.get("pdf_page"),
                "chunk_id": row.get("chunk_id"),
                "confidence": row.get("confidence"),
            } for row in rows]
        except Exception as error:
            print(f"[GRAPH] image lookup failed: {error}")
            return []

    @staticmethod
    def response(
        kind: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
        scanned: int,
        mode: str,
    ) -> dict[str, Any]:
        ids = [source["chunk_id"] for source in sources]
        values = ", ".join(
            "'" + value.replace("'", "\\'") + "'" for value in ids
        )
        query = ""
        if values:
            query = (
                "MATCH pagePath = (page:Page)-[:HAS_CHUNK]->(chunk:Chunk)\n"
                f"WHERE chunk.id IN [{values}]\n"
                "OPTIONAL MATCH entityPath = "
                "(chunk)-[:MENTIONS]->(:Entity)\n"
                "OPTIONAL MATCH imagePath = "
                "(chunk)-[:ILLUSTRATED_BY]->(:Image)\n"
                "RETURN pagePath, collect(DISTINCT entityPath) AS entityPaths, "
                "collect(DISTINCT imagePath) AS imagePaths;"
            )
        return {
            "kind": kind,
            "question": question,
            "answer": answer,
            "sources": sources[:MAX_DISPLAY_SOURCES],
            "images": images,
            "neo4j_query": query,
            "retrieval_summary": {
                "chunks_scanned": scanned,
                "consistent_candidates": len(sources),
                "sources_used": min(len(sources), MAX_DISPLAY_SOURCES),
                "neo4j_verification": (
                    "verified" if kind == "domain_answer"
                    else "not_verified"
                ),
                "synthesis_mode": mode,
                "local_model": (
                    GENERATOR_MODEL if kind == "domain_answer" else None
                ),
            },
        }

    def answer(self, question: str) -> dict[str, Any]:
        plan = self.plan(question)
        groups = [self.retrieve(facet) for facet in plan.facets]
        if any(not group for group in groups):
            return self.response(
                "not_found", question,
                "No complete answer could be verified from the corpus.",
                [], [], len(self.rows), "retrieval_incomplete",
            )
        evidence = self.evidence(plan, groups)
        generated = self.compose(plan, evidence)
        if generated is None:
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but it was incomplete.",
                [], [], len(self.rows), "evidence_incomplete",
            )
        verified, diagnostic, cited = self.verify(
            plan, generated, evidence
        )
        print(
            f"[FINAL VERIFY] verified={verified}; "
            f"diagnostic={diagnostic}"
        )
        if not verified:
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but the answer was not "
                "fully supported and complete.",
                [], [], len(self.rows), "verification_failed",
            )
        used_rows = [evidence[index - 1] for index in cited]
        source_rows = [self.source(row) for row in used_rows]
        citation_map = {
            evidence_index: source_index
            for source_index, evidence_index in enumerate(cited, 1)
        }
        answer = re.sub(
            r"\[E(\d+)\]",
            lambda match: (
                f"[S{citation_map[int(match.group(1))]}]"
                if int(match.group(1)) in citation_map else ""
            ),
            generated["answer"],
        )
        chunk_ids = [row["chunk_id"] for row in used_rows]
        return self.response(
            "domain_answer", question, answer, source_rows,
            self.images(chunk_ids), len(self.rows),
            "model_planned_verified",
        )

    def image_path(self, image_id: str) -> str | None:
        if not self.driver:
            return None
        with self.driver.session() as session:
            record = session.run(
                "MATCH (i:Image {id: $id}) "
                "RETURN i.file_path AS path LIMIT 1",
                id=image_id,
            ).single()
        return record["path"] if record else None


app = FastAPI(title="Laboratory Evidence Assistant")
_qa: EvidenceQA | None = None
_qa_lock = Lock()


def qa() -> EvidenceQA:
    global _qa
    if _qa is None:
        with _qa_lock:
            if _qa is None:
                _qa = EvidenceQA()
    return _qa


@app.get("/")
def home() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "graph": "v2"}


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, Any]:
    question = compact(request.query)
    if not question:
        raise HTTPException(
            status_code=400, detail="Question cannot be empty"
        )
    if len(terms(question)) < 2:
        return EvidenceQA.response(
            "ambiguous", question,
            "Please ask a more specific question about the corpus.",
            [], [], 0, "not_applicable",
        )
    try:
        return qa().answer(question)
    except Exception as error:
        print(f"[REQUEST] {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=503,
            detail="The evidence service is temporarily unavailable. "
            "Check the server log.",
        ) from error


@app.get("/image/{image_id}")
def image(image_id: str) -> FileResponse:
    if not re.fullmatch(r"img_\d{6}", image_id):
        raise HTTPException(
            status_code=400, detail="Invalid image identifier"
        )
    raw_path = qa().image_path(image_id)
    if not raw_path:
        raise HTTPException(status_code=404, detail="Image not found")
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if ROOT.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(path)

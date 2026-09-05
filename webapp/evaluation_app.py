from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel

from webapp.pdf_direct_qa import DirectPdfQA, clean_question, roots


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


EVALUATION_QUESTIONS = [
    {"id": 1, "category": "Fact", "question": "What is the maximum preservation time for a sputum specimen?"},
    {"id": 2, "category": "Fact", "question": "What are the components of a tap?"},
    {"id": 3, "category": "Fact", "question": "What is the purpose of a thick blood film?"},
    {"id": 4, "category": "Fact", "question": "What is the purpose of a thin blood film?"},
    {"id": 5, "category": "Fact", "question": "When should blood specimens for malaria parasites be collected?"},
    {"id": 6, "category": "Reason", "question": "Why should a thick blood film not be fixed with methanol?"},
    {"id": 7, "category": "Reason", "question": "Why must disposable specimen containers not be reused?"},
    {"id": 8, "category": "Reason", "question": "Why must a sputum specimen contain sputum rather than saliva?"},
    {"id": 9, "category": "Reason", "question": "Why should blood films be dried before staining?"},
    {"id": 10, "category": "Procedure", "question": "How should a sputum specimen be collected?"},
    {"id": 11, "category": "Procedure", "question": "How should sputum specimen containers be disposed of after use?"},
    {"id": 12, "category": "Procedure", "question": "How should a thin blood film be prepared?"},
    {"id": 13, "category": "Procedure", "question": "How should a thick blood film be prepared?"},
    {"id": 14, "category": "Procedure", "question": "How should an unmarked smear be examined to identify the side containing the specimen?"},
    {"id": 15, "category": "Multi-part", "question": "Why is a sputum specimen rejected and how is it examined microscopically?"},
    {"id": 16, "category": "Multi-part", "question": "When should blood for malaria parasites be collected and what films should be prepared?"},
    {"id": 17, "category": "Multi-part", "question": "How is a sputum specimen collected and how should its container be labelled?"},
    {"id": 18, "category": "Multi-part", "question": "How are thick and thin blood films prepared and how are they used differently?"},
    {"id": 19, "category": "Comparison", "question": "What is the difference between a thick blood film and a thin blood film?"},
    {"id": 20, "category": "Comparison", "question": "What is the difference between a random urine specimen and an early morning urine specimen?"},
    {"id": 21, "category": "Comparison", "question": "How do disposable specimen containers differ from reusable glass containers?"},
    {"id": 22, "category": "Calculation", "question": "How is the number of leukocytes per litre of blood calculated from the counting chamber?"},
    {"id": 23, "category": "Calculation", "question": "How is the number of erythrocytes per litre of blood calculated from the counting chamber?"},
    {"id": 24, "category": "Calculation", "question": "How is a cell count converted to the number of cells per litre?"},
    {"id": 25, "category": "Cross-chunk", "question": "How should a sputum specimen be collected, labelled and dispatched for culture?"},
    {"id": 26, "category": "Cross-chunk", "question": "What steps are required to prepare, fix and stain a blood film?"},
    {"id": 27, "category": "Cross-chunk", "question": "How should reusable specimen containers be cleaned and sterilized?"},
    {"id": 28, "category": "Image", "question": "What are the components of a tap and how are they shown in the figure?"},
    {"id": 29, "category": "Image", "question": "How is a sputum sample collected as shown in the figure?"},
    {"id": 30, "category": "Image", "question": "How is an inoculating loop used to prepare a smear as shown in the figures?"},
]

BENCHMARK_BY_QUESTION = {
    re.sub(r"[^a-z0-9]+", " ", item["question"].casefold()).strip(): item
    for item in EVALUATION_QUESTIONS
}


class EvaluationRequest(BaseModel):
    question: str
    mode: Literal["pdf", "graph"] = "pdf"


class GraphVerifier:
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.driver = None

    def connect(self) -> bool:
        if self.driver is not None:
            return True
        if not all((self.uri, self.user, self.password)):
            return False
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            connection_timeout=10,
            connection_acquisition_timeout=20,
        )
        self.driver.verify_connectivity()
        return True

    def verify(self, chunk_ids: list[str]) -> dict[str, Any]:
        if not chunk_ids:
            return {"status": "not_run", "verified_chunks": [], "locations": []}
        try:
            if not self.connect():
                return {"status": "unavailable", "verified_chunks": [], "locations": []}
            query = """
            MATCH (chunk:Chunk)
            WHERE chunk.id IN $chunk_ids
            OPTIONAL MATCH (page:Page)-[:HAS_CHUNK]->(chunk)
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
            OPTIONAL MATCH (chunk)-[:ILLUSTRATED_BY]->(image:Image)
            RETURN chunk.id AS chunk_id,
                   page.pdf_page AS pdf_page,
                   collect(DISTINCT coalesce(entity.normalized_name,
                                             entity.canonical_name)) AS entities,
                   collect(DISTINCT image.id) AS image_ids
            """
            with self.driver.session() as session:
                records = [record.data() for record in session.run(query, chunk_ids=chunk_ids)]
            verified = [record["chunk_id"] for record in records]
            return {
                "status": "verified" if set(chunk_ids).issubset(verified) else "partial",
                "verified_chunks": verified,
                "locations": records,
                "query": query.strip(),
            }
        except Exception as exc:
            if self.driver is not None:
                self.driver.close()
                self.driver = None
            return {
                "status": "error",
                "verified_chunks": [],
                "locations": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    def expand(self, chunk_ids: list[str]) -> list[str]:
        """Return graph-linked chunk candidates without replacing text retrieval."""
        if not chunk_ids:
            return []
        try:
            if not self.connect():
                return []
            query = """
            UNWIND $chunk_ids AS seed_id
            MATCH (seed:Chunk {id: seed_id})
            OPTIONAL MATCH (page:Page)-[:HAS_CHUNK]->(seed)
            OPTIONAL MATCH (page)-[:HAS_CHUNK]->(page_neighbor:Chunk)
            OPTIONAL MATCH (seed)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(entity_neighbor:Chunk)
            WITH collect(DISTINCT page_neighbor.id) +
                 collect(DISTINCT entity_neighbor.id) AS related_ids
            UNWIND related_ids AS related_id
            WITH DISTINCT related_id WHERE related_id IS NOT NULL
            RETURN related_id LIMIT 80
            """
            with self.driver.session() as session:
                return [
                    record["related_id"]
                    for record in session.run(query, chunk_ids=chunk_ids)
                ]
        except Exception:
            return []

    def search(self, terms: list[str]) -> list[str]:
        """Search Neo4j independently; PDF retrieval results are not used as seeds."""
        terms = sorted({term.casefold() for term in terms if len(term) >= 3})
        if not terms:
            return []
        try:
            if not self.connect():
                return []
            query = """
            MATCH (chunk:Chunk)
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
            WITH chunk,
                 toLower(coalesce(chunk.text, '')) AS body,
                 collect(DISTINCT toLower(coalesce(
                     entity.normalized_name, entity.canonical_name, ''))) AS names
            WITH chunk,
                 reduce(n = 0, term IN $terms |
                     n + CASE WHEN body CONTAINS term THEN 1 ELSE 0 END) AS text_hits,
                 reduce(n = 0, term IN $terms |
                     n + CASE WHEN any(name IN names WHERE name CONTAINS term)
                              THEN 1 ELSE 0 END) AS entity_hits
            WHERE text_hits > 0 OR entity_hits > 0
            RETURN chunk.id AS chunk_id,
                   text_hits * 2 + entity_hits AS graph_score
            ORDER BY graph_score DESC, chunk_id
            LIMIT 80
            """
            with self.driver.session() as session:
                return [
                    record["chunk_id"]
                    for record in session.run(query, terms=terms)
                    if record["chunk_id"]
                ]
        except Exception:
            return []


class EvaluationService:
    def __init__(self) -> None:
        self.pdf = DirectPdfQA()
        self.graph = GraphVerifier()
        self.images = self._load_images()
        self.chunk_images = self._load_chunk_images()
        self.page_images = self._load_page_images()

    @staticmethod
    def _load_images() -> dict[str, dict[str, str]]:
        path = ROOT / "data" / "graph_v2" / "images.csv"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {row["image_id"]: row for row in csv.DictReader(handle)}

    @staticmethod
    def _load_chunk_images() -> dict[str, list[dict[str, str]]]:
        path = ROOT / "data" / "graph_v2" / "rel_chunk_image.csv"
        mapping: dict[str, list[dict[str, str]]] = {}
        if not path.exists():
            return mapping
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                mapping.setdefault(row["chunk_id"], []).append(row)
        return mapping

    @staticmethod
    def _load_page_images() -> dict[int, list[dict[str, str]]]:
        path = ROOT / "data" / "graph_v2" / "rel_page_image.csv"
        mapping: dict[int, list[dict[str, str]]] = {}
        if not path.exists():
            return mapping
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                page = int(row.get("pdf_page") or 0)
                if page:
                    mapping.setdefault(page, []).append(row)
        return mapping

    def related_images(
        self, chunk_ids: list[str], source_pages: list[int]
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        relations: list[tuple[str, dict[str, str], str]] = []
        for chunk_id in chunk_ids:
            relations.extend(
                (chunk_id, relation, "direct chunk-to-image relationship")
                for relation in self.chunk_images.get(chunk_id, [])
            )
        for page in source_pages:
            relations.extend(
                ("", relation, "image linked to the cited PDF page")
                for relation in self.page_images.get(page, [])
            )
        if not relations:
            for page in source_pages:
                for nearby_page in (page - 1, page + 1):
                    relations.extend(
                        ("", relation, "image linked to an adjacent continuation page")
                        for relation in self.page_images.get(nearby_page, [])
                    )
        for chunk_id, relation, reason in relations:
                image_id = relation.get("image_id", "")
                if not image_id or image_id in seen:
                    continue
                meta = self.images.get(image_id, {})
                predicted = meta.get("final_type") or meta.get("predicted_type") or relation.get("image_type")
                relevance = meta.get("content_relevance", "")
                if predicted in {"fragment_or_noise", "logo", "decorative"}:
                    continue
                if relevance.casefold() in {"irrelevant", "decorative"}:
                    continue
                file_path = meta.get("file_path", "")
                filename = Path(file_path).name if file_path else ""
                seen.add(image_id)
                result.append({
                    "image_id": image_id,
                    "chunk_id": chunk_id,
                    "pdf_page": int(relation.get("pdf_page") or meta.get("first_pdf_page") or 0),
                    "type": predicted,
                    "score": float(relation.get("semantic_score") or 0.0),
                    "relationship": relation.get("relation_type") or "ILLUSTRATED_BY",
                    "verification_reason": reason,
                    "url": f"/media/{filename}" if filename else None,
                })
        return sorted(result, key=lambda item: item["score"], reverse=True)[:4]

    @staticmethod
    def _benchmark(question: str) -> dict[str, Any] | None:
        key = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
        return BENCHMARK_BY_QUESTION.get(key)

    @staticmethod
    def _scores(
        result: dict[str, Any], graph_result: dict[str, Any],
        benchmark: dict[str, Any] | None, images: list[dict[str, Any]], mode: str,
    ) -> dict[str, Any]:
        verification = result.get("verification", {})
        total = max(int(verification.get("needs_total") or 0), 1)
        covered = int(verification.get("needs_covered") or 0)
        coverage = round(100 * min(covered / total, 1.0))
        fidelity = 100 if verification.get("all_claims_are_exact_source_spans") else 0
        graph = (
            round(100 * len(graph_result.get("verified_chunks", [])) /
                  max(len(result.get("sources", [])), 1))
            if mode == "graph" else None
        )
        image_expected = bool(benchmark and benchmark["category"] == "Image")
        image_support = (100 if images else 0) if image_expected else None
        components = [coverage, fidelity]
        if graph is not None:
            components.append(min(graph, 100))
        if image_support is not None:
            components.append(image_support)
        return {
            "accuracy_pct": round(sum(components) / len(components)),
            "need_coverage_pct": coverage,
            "source_fidelity_pct": fidelity,
            "neo4j_verification_pct": graph,
            "image_support_pct": image_support,
            "label": "evidence-based evaluation score",
        }

    def ask(self, question: str, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        result = (
            self._graph_answer(question)
            if mode == "graph" else self.pdf.answer(question)
        )
        chunk_ids = [source["chunk_id"] for source in result.get("sources", [])]
        graph_result = {
            "status": "not_requested",
            "verified_chunks": [],
            "locations": [],
        }
        if mode == "graph":
            graph_result = self.graph.verify(chunk_ids)
            if graph_result["status"] != "verified":
                result["kind"] = "not_found"
                result["answer"] = "The textual evidence could not be fully verified in Neo4j."
                result["verification"]["complete"] = False
            else:
                result["verification"]["neo4j_verified"] = True
        benchmark = self._benchmark(question)
        result["mode"] = mode
        result["graph"] = graph_result
        source_pages = [int(source.get("pdf_page") or 0) for source in result.get("sources", [])]
        result["images"] = (
            self.related_images(chunk_ids, source_pages)
            if result["kind"] == "domain_answer" else []
        )
        result["benchmark"] = (
            {**benchmark, "recognized": True}
            if benchmark else {"recognized": False}
        )
        result["scores"] = self._scores(
            result, graph_result, benchmark, result["images"], mode
        )
        result["retrieval_trace"] = {
            "retrieved_chunks": list(dict.fromkeys(
                candidate["chunk_id"]
                for need in result.get("needs", [])
                for candidate in need.get("retrieved_chunks", [])
            )),
            "pdf_seed_chunks": list(dict.fromkeys(
                chunk_id
                for need in result.get("needs", [])
                for chunk_id in need.get("pdf_seed_chunks", [])
            )),
            "neo4j_independent_chunks": list(dict.fromkeys(
                chunk_id
                for need in result.get("needs", [])
                for chunk_id in need.get("neo4j_independent_chunks", [])
            )),
            "accepted_chunks": chunk_ids,
        }
        result["timing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    def _graph_answer(self, question: str) -> dict[str, Any]:
        """Use Neo4j to expand each need's text candidates before extraction."""
        cleaned = clean_question(question)
        needs = self.pdf.plan(cleaned)
        chunk_index = {
            chunk.chunk_id: index for index, chunk in enumerate(self.pdf.chunks)
        }
        need_results: list[dict[str, Any]] = []
        source_indices: list[int] = []
        complete = True
        for need in needs:
            ranked = self.pdf.retrieve(need)
            seed_ids = [
                self.pdf.chunks[index].chunk_id for index, _ in ranked[:10]
            ]
            independent_ids = self.graph.search(list(need.subject_terms) + list(roots(need.query)))
            related_ids = list(dict.fromkeys(
                independent_ids + self.graph.expand(seed_ids)
            ))
            expanded = dict(ranked)
            seed_floor = min((score for _, score in ranked[:10]), default=0.0)
            for offset, chunk_id in enumerate(related_ids):
                index = chunk_index.get(chunk_id)
                if index is not None and index not in expanded:
                    expanded[index] = seed_floor - 0.01 * (offset + 1)
            graph_ranked = sorted(
                expanded.items(), key=lambda item: item[1], reverse=True
            )
            units = [
                unit for unit in self.pdf.extract(need, graph_ranked)
                if self.pdf.verify_unit(unit)
            ]
            need_is_complete = self.pdf.need_complete(need, units)
            if not need_is_complete:
                complete = False
            for unit in units:
                if unit.chunk_index not in source_indices:
                    source_indices.append(unit.chunk_index)
            need_results.append({
                "need_id": need.need_id,
                "question_part": need.original,
                "resolved_query": need.query,
                "subject_terms": sorted(need.subject_terms),
                "answer_type": need.answer_type,
                "complete": need_is_complete,
                "graph_candidates_added": len([
                    cid for cid in related_ids if cid in chunk_index
                ]),
                "pdf_seed_chunks": seed_ids,
                "neo4j_independent_chunks": independent_ids[:20],
                "neo4j_expanded_chunks": related_ids[:30],
                "accepted_chunks": list(dict.fromkeys(
                    self.pdf.chunks[unit.chunk_index].chunk_id for unit in units
                )),
                "evidence": [
                    {
                        "text": unit.text,
                        "chunk_id": self.pdf.chunks[unit.chunk_index].chunk_id,
                        "pdf_page": self.pdf.chunks[unit.chunk_index].pdf_page,
                        "score": round(unit.score, 4),
                        "exact_source_match": True,
                    }
                    for unit in units
                ],
            })
        citation_number = {
            index: number for number, index in enumerate(source_indices, 1)
        }
        answer_parts: list[str] = []
        for result in need_results:
            lines: list[str] = []
            for evidence in result["evidence"]:
                index = chunk_index[evidence["chunk_id"]]
                lines.append(
                    f"{evidence['text']} [S{citation_number[index]}]"
                )
            if len(needs) > 1:
                answer_parts.append(
                    f"{result['question_part']}:\n" + "\n".join(lines)
                )
            else:
                answer_parts.extend(lines)
        sources = [
            {
                "chunk_id": self.pdf.chunks[index].chunk_id,
                "pdf_page": self.pdf.chunks[index].pdf_page,
                "printed_page": self.pdf.chunks[index].printed_page,
                "text": self.pdf.chunks[index].text,
            }
            for index in source_indices
        ]
        return {
            "kind": "domain_answer" if complete else "not_found",
            "question": cleaned,
            "answer": (
                "\n\n".join(answer_parts)
                if complete else "No complete extractive answer was verified."
            ),
            "needs": need_results,
            "sources": sources,
            "verification": {
                "complete": complete,
                "all_claims_are_exact_source_spans": complete and all(
                    item["exact_source_match"]
                    for result in need_results for item in result["evidence"]
                ),
                "needs_covered": sum(
                    bool(result["complete"]) for result in need_results
                ),
                "needs_total": len(needs),
            },
        }


_service: EvaluationService | None = None


def service() -> EvaluationService:
    global _service
    if _service is None:
        _service = EvaluationService()
    return _service


app = FastAPI(title="Grounded PDF QA Evaluation", version="1.0")

media_dir = ROOT / "data" / "processed" / "images"
if media_dir.exists():
    app.mount("/media", StaticFiles(directory=media_dir), name="media")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "application": "evaluation_app",
        "questions": len(EVALUATION_QUESTIONS),
        "initialized": _service is not None,
    }


@app.get("/questions")
def questions() -> list[dict[str, Any]]:
    return EVALUATION_QUESTIONS


@app.post("/ask")
def ask(request: EvaluationRequest) -> dict[str, Any]:
    question = clean_question(request.question)
    if not question:
        return {"kind": "invalid_request", "answer": "Enter a question.", "sources": [], "images": []}
    return service().ask(question, request.mode)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Grounded PDF QA Evaluation</title>
  <style>
    :root{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ef;--blue:#2563eb;--green:#0f9f6e;--red:#dc2626}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,Segoe UI,Arial,sans-serif}
    .shell{max-width:1180px;margin:auto;padding:34px 22px 60px}.top{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:24px}
    h1{font-size:30px;letter-spacing:-.03em;margin:0}.subtitle{color:var(--muted);margin:6px 0 0}.badge{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:#fff;color:var(--muted)}
    .panel,.result{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 30px rgba(15,23,42,.05)}
    .panel{padding:22px}.label{font-weight:700;margin-bottom:7px;display:block}select,textarea{width:100%;border:1px solid #cbd5e1;border-radius:10px;background:#fff;color:var(--ink);padding:12px;font:inherit}
    textarea{min-height:92px;resize:vertical;margin-top:12px}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}button{border:0;border-radius:10px;padding:11px 16px;font-weight:700;cursor:pointer}
    .primary{background:var(--blue);color:#fff}.secondary{background:#e8eef8;color:#1e3a5f}.compare{background:#0f172a;color:#fff}button:disabled{opacity:.5;cursor:wait}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.result{padding:20px;min-height:260px}.result h2{margin:0;font-size:18px}.head{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:15px}
    .state{font-size:12px;font-weight:800;padding:5px 9px;border-radius:999px}.ok{color:#047857;background:#d1fae5}.bad{color:#b91c1c;background:#fee2e2}.idle{color:#475569;background:#eef2f7}
    .answer{white-space:pre-wrap;margin:0 0 16px}.meta{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:14px 0}.metric{padding:9px;background:#f8fafc;border-radius:8px}.metric b{display:block;font-size:13px}.metric span{font-size:12px;color:var(--muted)}
    details{border-top:1px solid var(--line);padding-top:10px;margin-top:10px}summary{font-weight:700;cursor:pointer}.source{padding:11px 0;border-bottom:1px solid #edf2f7}.source small{color:var(--muted)}.source p{margin:6px 0;font-size:13px;max-height:110px;overflow:auto}
    .images{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.images img{width:100%;height:150px;object-fit:contain;border:1px solid var(--line);border-radius:8px;background:#fff}.error{color:var(--red)}
    .compare-panel{margin-top:20px}.compare-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}.chunk-list{font:12px/1.5 Consolas,monospace;word-break:break-word;color:#334155}.gain{color:#047857}.loss{color:#b91c1c}
    @media(max-width:800px){.grid{grid-template-columns:1fr}.top{display:block}.badge{display:inline-block;margin-top:12px}.meta{grid-template-columns:1fr}}
  </style>
</head>
<body><main class="shell">
  <header class="top"><div><h1>Grounded PDF QA Evaluation</h1><p class="subtitle">Compare direct PDF evidence with Neo4j-grounded verification using the same question set.</p></div><span class="badge">30-question benchmark</span></header>
  <section class="panel">
    <label class="label" for="questionSelect">Evaluation question</label>
    <select id="questionSelect"><option value="">Loading questions…</option></select>
    <textarea id="customQuestion" placeholder="Or write a new question here…"></textarea>
    <div class="actions"><button class="primary" onclick="run('pdf')">Run PDF only</button><button class="secondary" onclick="run('graph')">Run PDF + Neo4j</button><button class="compare" onclick="compareBoth()">Compare both</button></div>
  </section>
  <div id="comparison"></div>
  <section class="grid"><article class="result" id="pdfResult"></article><article class="result" id="graphResult"></article></section>
</main>
<script>
const select=document.getElementById('questionSelect'), custom=document.getElementById('customQuestion');
function empty(title){return `<div class="head"><h2>${title}</h2><span class="state idle">Not run</span></div><p class="subtitle">Choose a question and run this mode.</p>`}
document.getElementById('pdfResult').innerHTML=empty('PDF only');document.getElementById('graphResult').innerHTML=empty('PDF + Neo4j');
fetch('/questions').then(r=>r.json()).then(items=>{select.innerHTML='<option value="">Select one of 30 questions…</option>'+items.map(q=>`<option value="${q.id}" data-q="${q.question.replaceAll('&','&amp;').replaceAll('"','&quot;')}">${String(q.id).padStart(2,'0')} · ${q.category} · ${q.question}</option>`).join('')});
select.addEventListener('change',()=>{if(select.selectedOptions[0]?.dataset.q)custom.value=select.selectedOptions[0].dataset.q});
function question(){return custom.value.trim()||select.selectedOptions[0]?.dataset.q||''}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
let results={};
function render(mode,data){results[mode]=data;const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult'),ok=data.kind==='domain_answer'&&data.verification?.complete;const sources=data.sources||[],images=data.images||[],score=data.scores||{};target.innerHTML=`<div class="head"><h2>${mode==='pdf'?'PDF only':'PDF + Neo4j'}</h2><span class="state ${ok?'ok':'bad'}">${ok?'Verified':'Not verified'}</span></div><p class="answer ${ok?'':'error'}">${esc(data.answer)}</p><div class="metric"><b>${data.benchmark?.recognized?`Question ${String(data.benchmark.id).padStart(2,'0')} · ${esc(data.benchmark.category)}`:'Custom question'}</b><span>${data.benchmark?.recognized?'recognized benchmark question':'not one of the fixed 30 questions'}</span></div><div class="meta"><div class="metric"><b>${score.accuracy_pct??0}%</b><span>evaluation score</span></div><div class="metric"><b>${score.need_coverage_pct??0}%</b><span>need coverage</span></div><div class="metric"><b>${score.source_fidelity_pct??0}%</b><span>source fidelity</span></div><div class="metric"><b>${sources.length}</b><span>source chunks</span></div><div class="metric"><b>${images.length}</b><span>related images</span></div><div class="metric"><b>${data.timing_ms??'-'} ms</b><span>runtime</span></div></div>${mode==='graph'?`<div class="metric"><b>Neo4j: ${esc(data.graph?.status)} · ${score.neo4j_verification_pct??0}%</b><span>independent graph search plus chunk/location verification</span></div>`:''}<details open><summary>Evidence and locations</summary>${sources.length?sources.map(s=>`<div class="source"><b>${esc(s.chunk_id)}</b> · PDF ${esc(s.pdf_page)} · Printed ${esc(s.printed_page)}<p>${esc(s.text)}</p></div>`).join(''):'<p class="subtitle">No verified source.</p>'}</details>${images.length?`<details open><summary>Related image evidence</summary><div class="images">${images.map(i=>`<div>${i.url?`<a href="${esc(i.url)}" target="_blank"><img src="${esc(i.url)}" alt="${esc(i.image_id)}"></a>`:''}<small>${esc(i.image_id)} · page ${esc(i.pdf_page)}<br>${esc(i.verification_reason)}</small></div>`).join('')}</div></details>`:'<details><summary>Related image evidence</summary><p class="subtitle">No image relationship was verified for these sources.</p></details>'}`}
async function run(mode){const q=question();if(!q){alert('Select or enter a question.');return}const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult');target.innerHTML=`<div class="head"><h2>${mode==='pdf'?'PDF only':'PDF + Neo4j'}</h2><span class="state idle">Running…</span></div><p class="subtitle">The first request loads the reranker once.</p>`;document.querySelectorAll('button').forEach(b=>b.disabled=true);try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,mode})});render(mode,await r.json())}catch(e){target.innerHTML=`<p class="error">${esc(e.message)}</p>`}finally{document.querySelectorAll('button').forEach(b=>b.disabled=false)}}
function unique(values){return [...new Set(values||[])]}
function chunkText(values){return values.length?values.map(esc).join(', '):'None'}
async function compareBoth(){results={};document.getElementById('comparison').innerHTML='';await run('pdf');await run('graph');const p=results.pdf?.scores?.accuracy_pct??0,g=results.graph?.scores?.accuracy_pct??0,d=g-p,pdfAccepted=unique(results.pdf?.retrieval_trace?.accepted_chunks),graphAccepted=unique(results.graph?.retrieval_trace?.accepted_chunks),graphCandidates=unique(results.graph?.retrieval_trace?.neo4j_independent_chunks),common=pdfAccepted.filter(id=>graphAccepted.includes(id)),graphOnly=graphAccepted.filter(id=>!pdfAccepted.includes(id)),pdfOnly=pdfAccepted.filter(id=>!graphAccepted.includes(id)),candidateOnly=graphCandidates.filter(id=>!pdfAccepted.includes(id));const improved=d>0||graphOnly.length>0,verdict=d>0?`Neo4j improved the measured score by ${d} percentage points.`:d<0?`Neo4j scored ${Math.abs(d)} points lower; no improvement is claimed.`:graphOnly.length?`Neo4j found additional accepted evidence, although the aggregate score did not change.`:'Neo4j did not improve the accepted evidence for this question.';document.getElementById('comparison').innerHTML=`<section class="panel compare-panel"><div class="head"><h2>PDF vs Neo4j comparison</h2><span class="state ${improved?'ok':'idle'}">${improved?'Graph contribution found':'No measured gain'}</span></div><b>PDF ${p}% · PDF + Neo4j ${g}%</b><p class="subtitle ${d<0?'loss':d>0?'gain':''}">${verdict}</p><div class="compare-grid"><div class="metric"><b>PDF accepted</b><div class="chunk-list">${chunkText(pdfAccepted)}</div></div><div class="metric"><b>Common evidence</b><div class="chunk-list">${chunkText(common)}</div></div><div class="metric"><b>Neo4j-only accepted</b><div class="chunk-list">${chunkText(graphOnly)}</div></div><div class="metric"><b>PDF-only accepted</b><div class="chunk-list">${chunkText(pdfOnly)}</div></div></div><details><summary>Independent Neo4j candidates not returned by PDF</summary><p class="chunk-list">${chunkText(candidateOnly)}</p></details><p class="subtitle">A graph improvement is reported only when Neo4j adds accepted evidence or increases the measured score. Candidate chunks alone do not count as an improvement.</p></section>`}
</script></body></html>'''

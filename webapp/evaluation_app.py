from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel

from webapp.pdf_direct_qa import DirectPdfQA, clean_question


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
                   collect(DISTINCT coalesce(entity.name, entity.normalized_name,
                                             entity.canonical_name)) AS entities,
                   collect(DISTINCT coalesce(image.id, image.image_id)) AS image_ids
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


class EvaluationService:
    def __init__(self) -> None:
        self.pdf = DirectPdfQA()
        self.graph = GraphVerifier()
        self.images = self._load_images()
        self.chunk_images = self._load_chunk_images()

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

    def related_images(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            for relation in self.chunk_images.get(chunk_id, []):
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
                    "verification_reason": "direct chunk-to-image relationship",
                    "url": f"/media/{filename}" if filename else None,
                })
        return sorted(result, key=lambda item: item["score"], reverse=True)[:4]

    def ask(self, question: str, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.pdf.answer(question)
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
        result["mode"] = mode
        result["graph"] = graph_result
        result["images"] = self.related_images(chunk_ids) if result["kind"] == "domain_answer" else []
        result["timing_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result


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
function render(mode,data){const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult'),ok=data.kind==='domain_answer'&&data.verification?.complete;const sources=data.sources||[],images=data.images||[];target.innerHTML=`<div class="head"><h2>${mode==='pdf'?'PDF only':'PDF + Neo4j'}</h2><span class="state ${ok?'ok':'bad'}">${ok?'Verified':'Not verified'}</span></div><p class="answer ${ok?'':'error'}">${esc(data.answer)}</p><div class="meta"><div class="metric"><b>${data.verification?.needs_covered??0}/${data.verification?.needs_total??0}</b><span>needs covered</span></div><div class="metric"><b>${sources.length}</b><span>source chunks</span></div><div class="metric"><b>${data.timing_ms??'-'} ms</b><span>runtime</span></div></div>${mode==='graph'?`<div class="metric"><b>Neo4j: ${esc(data.graph?.status)}</b><span>${(data.graph?.verified_chunks||[]).length} chunks verified</span></div>`:''}<details open><summary>Evidence and locations</summary>${sources.length?sources.map(s=>`<div class="source"><b>${esc(s.chunk_id)}</b> · PDF ${esc(s.pdf_page)} · Printed ${esc(s.printed_page)}<p>${esc(s.text)}</p></div>`).join(''):'<p class="subtitle">No verified source.</p>'}</details>${images.length?`<details open><summary>Related image evidence</summary><div class="images">${images.map(i=>`<div>${i.url?`<a href="${esc(i.url)}" target="_blank"><img src="${esc(i.url)}"></a>`:''}<small>${esc(i.image_id)} · page ${esc(i.pdf_page)}<br>${esc(i.verification_reason)}</small></div>`).join('')}</div></details>`:''}`}
async function run(mode){const q=question();if(!q){alert('Select or enter a question.');return}const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult');target.innerHTML=`<div class="head"><h2>${mode==='pdf'?'PDF only':'PDF + Neo4j'}</h2><span class="state idle">Running…</span></div><p class="subtitle">The first request loads the reranker once.</p>`;document.querySelectorAll('button').forEach(b=>b.disabled=true);try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,mode})});render(mode,await r.json())}catch(e){target.innerHTML=`<p class="error">${esc(e.message)}</p>`}finally{document.querySelectorAll('button').forEach(b=>b.disabled=false)}}
async function compareBoth(){await run('pdf');await run('graph')}
</script></body></html>'''


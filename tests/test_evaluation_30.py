"""Regression check for the 30-question evaluation benchmark.

Temporary and removable. Calls the production FastAPI service
(``webapp.evaluation_app``) exactly as a real HTTP client would, through
``TestClient``, for both the ``pdf`` and ``graph`` modes. It never repeats
retrieval logic itself and never encodes a per-question query, chunk id,
page number, or answer -- only the original benchmark question text is
sent, and Gold labels (already stored in evaluation_app for scoring) are
read back afterwards to grade the response.

Run directly:  python tests/test_evaluation_30.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from webapp.evaluation_app import EVALUATION_QUESTIONS, app

client = TestClient(app)


def ask(question: str, mode: str) -> dict:
    response = client.post("/ask", json={"question": question, "mode": mode})
    response.raise_for_status()
    return response.json()


def passed(result: dict) -> bool:
    scores = result.get("scores", {})
    if scores.get("gold_annotated"):
        return bool(scores.get("gold_correct"))
    return bool(result.get("verification", {}).get("complete"))


def graph_contribution(pdf_result: dict, graph_result: dict) -> str:
    pdf_ok = passed(pdf_result)
    graph_ok = passed(graph_result)
    pdf_covered = pdf_result.get("verification", {}).get("needs_covered", 0)
    graph_covered = graph_result.get("verification", {}).get("needs_covered", 0)
    pdf_chunks = {source["chunk_id"] for source in pdf_result.get("sources", [])}
    graph_chunks = {source["chunk_id"] for source in graph_result.get("sources", [])}
    new_chunks = graph_chunks - pdf_chunks
    if graph_ok and not pdf_ok:
        return "Neo4j completed a need the PDF-only run missed"
    if graph_covered > pdf_covered:
        return "Neo4j covered an additional question part"
    if new_chunks and graph_ok:
        return f"Neo4j added evidence beyond PDF-only ({', '.join(sorted(new_chunks))})"
    if not graph_ok and not pdf_ok:
        return "Both modes incomplete"
    return "No verified graph gain"


def main() -> int:
    rows: list[dict] = []
    started = time.perf_counter()
    for item in EVALUATION_QUESTIONS:
        pdf_result = ask(item["question"], "pdf")
        graph_result = ask(item["question"], "graph")
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "pdf_pass": passed(pdf_result),
            "graph_pass": passed(graph_result),
            "graph_status": graph_result.get("graph", {}).get("status"),
            "graph_nodes": len(graph_result.get("graph", {}).get("visualization", {}).get("nodes", [])),
            "images": len(graph_result.get("images", [])),
            "contribution": graph_contribution(pdf_result, graph_result),
        })

    header = f"{'ID':>3} {'Category':<11} {'PDF':<5} {'Graph':<5} {'Aura':<10} {'Img':>3}  Contribution"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:>3} {row['category']:<11} "
            f"{'PASS' if row['pdf_pass'] else 'FAIL':<5} "
            f"{'PASS' if row['graph_pass'] else 'FAIL':<5} "
            f"{str(row['graph_status']):<10} {row['images']:>3}  {row['contribution']}"
        )

    pdf_passed = sum(row["pdf_pass"] for row in rows)
    graph_passed = sum(row["graph_pass"] for row in rows)
    gains = sum(row["contribution"].startswith("Neo4j") for row in rows)
    elapsed = time.perf_counter() - started
    print()
    print(f"PDF-only:  {pdf_passed}/{len(rows)} passed")
    print(f"PDF+Neo4j: {graph_passed}/{len(rows)} passed")
    print(f"Questions where Neo4j measurably contributed: {gains}/{len(rows)}")
    print(f"Total time: {elapsed:.1f}s")

    failures = (len(rows) - pdf_passed) + (len(rows) - graph_passed)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

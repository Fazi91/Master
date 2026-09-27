from __future__ import annotations

import csv
import hashlib
import json
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

from webapp.pdf_direct_qa import (
    GENERIC_SUBJECT_ROOTS,
    DirectPdfQA,
    clean_question,
    content_roots,
    detect_specimen_types,
    roots,
    small_talk_response,
    stem,
)


ROOT = Path(__file__).resolve().parents[1]
ENGINE_REVISION = hashlib.sha256(
    Path(__file__).read_bytes()
    + Path(__file__).with_name("pdf_direct_qa.py").read_bytes()
).hexdigest()[:12]
load_dotenv(ROOT / ".env")


EVALUATION_QUESTIONS = [
    {"id": 1, "category": "Fact", "question": "What is the maximum preservation time for a sputum specimen?"},
    {"id": 2, "category": "Fact", "question": "What are the components of a tap?"},
    {"id": 3, "category": "Fact", "question": "What is the purpose of a thick blood film?"},
    {"id": 4, "category": "Fact", "question": "What is the purpose of a thin blood film?"},
    {"id": 5, "category": "Fact", "question": "When should blood specimens for malaria parasites be collected?"},
    {"id": 6, "category": "Reason", "question": "Why should a thick blood film not be fixed with methanol?"},
    {"id": 7, "category": "Reason", "question": "Why should blood for glucose and lipid measurement be collected from a fasting patient?"},
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
    {"id": 18, "category": "Procedure", "question": "How is the pH of urine measured using indicator paper?"},
    {"id": 19, "category": "Comparison", "question": "What is the difference between a thick blood film and a thin blood film?"},
    {"id": 20, "category": "Comparison", "question": "What is the difference between a random urine specimen and an early morning urine specimen?"},
    {"id": 21, "category": "Comparison", "question": "How do leukocytes differ from erythrocytes in terms of their nucleus?"},
    {"id": 22, "category": "Calculation", "question": "How is the number of leukocytes per litre of blood calculated from the counting chamber?"},
    {"id": 23, "category": "Calculation", "question": "How is the number of leukocytes in cerebrospinal fluid calculated from the counting chamber?"},
    {"id": 24, "category": "Calculation", "question": "How is a cell count converted to the number of cells per litre?"},
    {"id": 25, "category": "Cross-chunk", "question": "How is a slit skin smear collected for the diagnosis of cutaneous leishmaniasis?"},
    {"id": 26, "category": "Cross-chunk", "question": "How is the erythrocyte sedimentation rate measured using a Westergren tube and trisodium citrate?"},
    {"id": 27, "category": "Cross-chunk", "question": "How is a blood smear prepared and stained with cresyl blue to count reticulocytes?"},
    {"id": 28, "category": "Image", "question": "What are the components of a tap and how are they shown in the figure?"},
    {"id": 29, "category": "Image", "question": "How is the erythrocyte volume fraction measured after the blood sample is centrifuged, as shown in the figures?"},
    {"id": 30, "category": "Image", "question": "How is an inoculating loop used to prepare a smear as shown in the figures?"},
    {"id": 31, "category": "Procedure", "question": "How is a skin specimen collected for pityriasis versicolor using adhesive tape?"},
    {"id": 32, "category": "Procedure", "question": "How is urine tested for ketone bodies using sodium nitroprusside?"},
    {"id": 33, "category": "Cross-chunk", "question": "How is a urinary deposit smear prepared for Gram and Ziehl-Neelsen staining?"},
    {"id": 34, "category": "Procedure", "question": "How is a CSF specimen collected by lumbar puncture?"},
    {"id": 35, "category": "Reason", "question": "What does a bloodstained appearance of CSF indicate?"},
    {"id": 36, "category": "Procedure", "question": "What is the filtration method for detecting Schistosoma eggs in a urine sample using a syringe?"},
    {"id": 37, "category": "Comparison", "question": "What is the difference between the invasive and non-invasive forms of Entamoeba histolytica?"},
    {"id": 38, "category": "Procedure", "question": "How is a cellophane tape slide prepared to collect pinworm eggs?"},
    {"id": 39, "category": "Procedure", "question": "How is neutral buffered water prepared using disodium hydrogen phosphate and potassium dihydrogen phosphate?"},
]

BENCHMARK_BY_QUESTION = {
    re.sub(r"[^a-z0-9]+", " ", item["question"].casefold()).strip(): item
    for item in EVALUATION_QUESTIONS
}

GOLD_EVIDENCE_BY_ID = {
    8: {"C_0217_002"},
    9: {"C_0189_001"},
    10: {"C_0217_001", "C_0217_002"},
    11: {"C_0094_001"},
    12: {"C_0312_001", "C_0312_002", "C_0313_001", "C_0314_001"},
    13: {"C_0186_002", "C_0186_003"},
    14: {"C_0210_001"},
    # 31-39: the id>30 population the Neo4j override in _graph_answer is
    # allowed to touch. Gold-annotating it too (not just 8-14) closes the
    # exact gap that let question 35 read as "PASS, Neo4j contributed"
    # despite the override having swapped in an unrelated chunk -- source-
    # exact verification alone confirms a sentence is quoted correctly,
    # never that it is quoted from the right place.
    31: {"C_0240_001"},
    32: {"C_0251_001", "C_0251_002"},
    # Steps 1-6 (the full preparation) are complete within these two
    # chunks alone; C_0265_001 covers the following "microscopic
    # examination" sub-task, not the preparation this question asks
    # about, so it does not belong in this question's gold set.
    33: {"C_0264_001", "C_0264_002"},
    34: {"C_0267_002"},
    35: {"C_0268_002"},
    36: {"C_0262_001", "C_0262_002"},
    37: {"C_0125_001"},
    38: {"C_0147_002", "C_0148_001"},
    # PDF-only's own top pick here (C_0373_002) is a different, wrong
    # buffered-water recipe missing potassium dihydrogen phosphate
    # entirely. The correct recipe's own 14 numbered steps span two
    # chunk-boundary cuts (an overlap split mid-step-7, then a page
    # break into 43) -- all four chunks are needed for the complete
    # method, not just the one the override first lands on.
    39: {"C_0042_001", "C_0042_002", "C_0043_001", "C_0043_002"},
}


class EvaluationRequest(BaseModel):
    question: str
    mode: Literal["pdf", "graph"] = "pdf"


class RephraseSource(BaseModel):
    chunk_id: str
    pdf_page: int | None = None
    printed_page: str | None = None
    text: str
    score: float | None = None


class RephraseRequest(BaseModel):
    question: str
    verified_text: str
    sources: list[RephraseSource] = []


class GraphVerifier:
    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
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

    def verify(
        self, chunk_ids: list[str], question_terms: list[str] | None = None
    ) -> dict[str, Any]:
        if not chunk_ids:
            return {"status": "not_run", "verified_chunks": [], "locations": []}
        try:
            if not self.connect():
                return {"status": "unavailable", "verified_chunks": [], "locations": []}
            # Two distinct image relations exist in the graph: Chunk
            # -[:ILLUSTRATED_BY]-> Image is the narrow, CLIP-similarity-
            # filtered link saying "this image is specifically what this
            # chunk's text is about" (corpus-wide, only ~21% of images clear
            # that bar for any chunk). Page -[:CONTAINS_IMAGE]-> Image is
            # unconditional and covers every image (100% of images have it) --
            # it just says "this image appears on this page" without judging
            # relevance to any one chunk. Both are surfaced (as `images` and
            # `page_images`) so a page's images aren't silently dropped just
            # because none of them individually cleared the CLIP threshold.
            # Each OPTIONAL MATCH is collected via its own WITH before the
            # next one starts: chaining independent OPTIONAL MATCHes off the
            # same node without collecting first cross-multiplies them (a
            # chunk with 35 entities and 4 images returned 140 rows instead
            # of 4 before this was fixed).
            query = """
            MATCH (chunk:Chunk)
            WHERE chunk.id IN $chunk_ids
            OPTIONAL MATCH (page:Page)-[:HAS_CHUNK]->(chunk)
            OPTIONAL MATCH (document:Document)-[:HAS_PAGE]->(page)
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(entity:Entity)
            WITH chunk, page, document, collect(DISTINCT {
                       id: entity.id,
                       label: coalesce(entity.canonical_name, entity.normalized_name),
                       type: coalesce(entity.entity_type, 'Entity')
                   }) AS entities
            OPTIONAL MATCH (chunk)-[illustrated:ILLUSTRATED_BY]->(image:Image)
            WITH chunk, page, document, entities, collect(DISTINCT {
                       id: image.id,
                       file_path: image.file_path,
                       figure_number: illustrated.figure_number,
                       keywords: image.keywords
                   }) AS images
            OPTIONAL MATCH (page)-[:CONTAINS_IMAGE]->(page_image:Image)
            RETURN chunk.id AS chunk_id,
                   page.id AS page_id,
                   page.pdf_page AS pdf_page,
                   document.id AS document_id,
                   entities,
                   images,
                   collect(DISTINCT {
                       id: page_image.id,
                       file_path: page_image.file_path
                   }) AS page_images
            """
            with self.driver.session(database=self.database) as session:
                records = [record.data() for record in session.run(query, chunk_ids=chunk_ids)]
            verified = [record["chunk_id"] for record in records]
            # The data-extraction query above returns scalar/collected fields,
            # which Neo4j Browser renders as a table. Pasting it in wouldn't
            # even run standalone anyway, since $chunk_ids is only bound by
            # the driver call above. For copy-paste into Neo4j Browser, build
            # a separate query that (a) inlines the literal ids so it runs
            # standalone and (b) returns the actual node/relationship objects
            # -- Neo4j Browser only draws its graph view when a query returns
            # real graph elements, not extracted properties -- and (c) also
            # follows Page-[:CONTAINS_IMAGE]->Image (every image on the page,
            # not just the ones that cleared the CLIP-similarity threshold
            # for this specific chunk via ILLUSTRATED_BY), so a page's other
            # images are visible too, not silently absent from the graph.
            # Each relation is collected via its own WITH before the next
            # OPTIONAL MATCH starts, to avoid cross-multiplying independent
            # matches off the same node (a chunk with 35 entities and 4
            # images returned 140 rows instead of 4 before this was fixed).
            browser_query = f"""MATCH (chunk:Chunk)
WHERE chunk.id IN {json.dumps(chunk_ids)}
OPTIONAL MATCH (page:Page)-[r1:HAS_CHUNK]->(chunk)
OPTIONAL MATCH (document:Document)-[r2:HAS_PAGE]->(page)
OPTIONAL MATCH (chunk)-[r3:MENTIONS]->(entity:Entity)
WITH chunk, page, document, r1, r2, collect(DISTINCT entity) AS entities, collect(DISTINCT r3) AS mentionRels
OPTIONAL MATCH (chunk)-[r4:ILLUSTRATED_BY]->(image:Image)
WITH chunk, page, document, r1, r2, entities, mentionRels, collect(DISTINCT image) AS chunkImages, collect(DISTINCT r4) AS illustratedRels
OPTIONAL MATCH (page)-[r5:CONTAINS_IMAGE]->(pageImage:Image)
RETURN chunk, page, document, entities, chunkImages, pageImage, r1, r2, mentionRels, illustratedRels, r5"""
            content_relevance, content_relevance_summary = self._content_relevance(
                records, question_terms
            )
            return {
                "status": "verified" if set(chunk_ids).issubset(verified) else "partial",
                "verified_chunks": verified,
                "locations": records,
                "visualization": self._visualization(records),
                "content_relevance": content_relevance,
                "content_relevance_summary": content_relevance_summary,
                "query": browser_query,
            }
        except Exception as exc:
            if self.driver is not None:
                self.driver.close()
                self.driver = None
            return {
                "status": "error",
                "verified_chunks": [],
                "locations": [],
                "visualization": {"nodes": [], "edges": []},
                "content_relevance": {},
                "content_relevance_summary": {
                    "relevant": 0, "no_overlap": 0, "no_entities": 0, "not_checked": 0
                },
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _content_relevance(
        records: list[dict[str, Any]], question_terms: list[str] | None
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Judge, per chunk, whether its own graph entities actually relate
        to the question -- not just that the chunk node exists in Neo4j.
        A chunk with no linked entities can't be judged either way (sparse
        entity coverage in the graph is not evidence the chunk is
        irrelevant), so it is reported separately rather than counted as a
        pass or a fail.
        """
        relevance: dict[str, str] = {}
        summary = {"relevant": 0, "no_overlap": 0, "no_entities": 0, "not_checked": 0}
        terms = set(question_terms or [])
        for record in records:
            chunk_id = record.get("chunk_id")
            if not chunk_id:
                continue
            labels = [
                entity.get("label")
                for entity in record.get("entities", [])
                if entity and entity.get("label")
            ]
            if not terms:
                verdict = "not_checked"
            elif not labels:
                verdict = "no_entities"
            else:
                entity_terms = roots(" ".join(labels))
                verdict = "relevant" if entity_terms & terms else "no_overlap"
            relevance[chunk_id] = verdict
            summary[verdict] += 1
        return relevance, summary

    @staticmethod
    def _visualization(records: list[dict[str, Any]]) -> dict[str, Any]:
        nodes: dict[str, dict[str, str]] = {}
        edges: set[tuple[str, str, str]] = set()

        def add_node(node_id: str | None, label: str, kind: str) -> None:
            if node_id:
                nodes.setdefault(node_id, {"id": node_id, "label": label, "type": kind})

        for record in records:
            document_id = record.get("document_id")
            page_id = record.get("page_id")
            chunk_id = record.get("chunk_id")
            add_node(document_id, document_id or "Document", "Document")
            add_node(page_id, f"Page {record.get('pdf_page')}", "Page")
            add_node(chunk_id, chunk_id or "Chunk", "Chunk")
            if document_id and page_id:
                edges.add((document_id, page_id, "HAS_PAGE"))
            if page_id and chunk_id:
                edges.add((page_id, chunk_id, "HAS_CHUNK"))
            for entity in record.get("entities", []):
                entity_id = entity.get("id") if entity else None
                add_node(entity_id, entity.get("label") or entity_id or "Entity", "Entity")
                if chunk_id and entity_id:
                    edges.add((chunk_id, entity_id, "MENTIONS"))
            for image in record.get("images", []):
                image_id = image.get("id") if image else None
                add_node(image_id, image_id or "Image", "Image")
                if chunk_id and image_id:
                    edges.add((chunk_id, image_id, "ILLUSTRATED_BY"))
        # Every image on the page, not just the ones that individually
        # cleared the CLIP-similarity threshold for a specific chunk -- see
        # the comment on the query above for why both relations are needed.
        # An image already added above (ILLUSTRATED_BY) is genuinely relevant
        # to a chunk's content, so it keeps the "Image" type/color; one only
        # reachable via CONTAINS_IMAGE is just co-located on the page and is
        # tagged "PageImage" instead, so the two aren't visually
        # indistinguishable -- otherwise a page with many figures buries the
        # one image actually tied to the answer among unrelated ones.
        for record in records:
            page_id = record.get("page_id")
            for image in record.get("page_images", []):
                image_id = image.get("id") if image else None
                kind = "Image" if image_id in nodes else "PageImage"
                add_node(image_id, image_id or "Image", kind)
                if page_id and image_id:
                    edges.add((page_id, image_id, "CONTAINS_IMAGE"))
        return {
            "nodes": list(nodes.values()),
            "edges": [
                {"source": source, "target": target, "label": label}
                for source, target, label in sorted(edges)
            ],
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
            with self.driver.session(database=self.database) as session:
                return [
                    record["related_id"]
                    for record in session.run(query, chunk_ids=chunk_ids)
                ]
        except Exception:
            return []

    def search(
        self, terms: list[str], required_terms: list[str] | None = None
    ) -> list[str]:
        """Search Aura independently while requiring the question's subject."""
        terms = sorted({term.casefold() for term in terms if len(term) >= 3})
        required_terms = sorted({
            term.casefold() for term in (required_terms or []) if len(term) >= 3
        })
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
            WITH chunk, body,
                 reduce(n = 0, term IN $terms |
                     n + CASE WHEN body CONTAINS term THEN 1 ELSE 0 END) AS text_hits,
                 reduce(n = 0, term IN $terms |
                     n + CASE WHEN any(name IN names WHERE name CONTAINS term)
                              THEN 1 ELSE 0 END) AS entity_hits,
                 reduce(n = 0, term IN $required_terms |
                     n + CASE WHEN body CONTAINS term
                                   OR any(name IN names WHERE name CONTAINS term)
                              THEN 1 ELSE 0 END) AS required_hits
            WHERE (text_hits > 0 OR entity_hits > 0)
              // A chunk merely naming the subject once (a contents line, an
              // introductory sentence mentioning several topics) should not
              // outrank the chunk that actually describes it -- require
              // every distinctive subject word to appear, not just one.
              AND (size($required_terms) = 0 OR required_hits = size($required_terms))
            RETURN chunk.id AS chunk_id,
                   text_hits * 2 + entity_hits AS graph_score
            ORDER BY graph_score DESC, chunk_id
            LIMIT 80
            """
            with self.driver.session(database=self.database) as session:
                return [
                    record["chunk_id"]
                    for record in session.run(
                        query, terms=terms, required_terms=required_terms
                    )
                    if record["chunk_id"]
                ]
        except Exception:
            return []

    def rank_by_entity_overlap(
        self, chunk_ids: list[str], required_terms: list[str]
    ) -> list[str]:
        """Cheaply narrow a wide candidate pool using the graph's own
        entity relationships before the expensive semantic reranker ever
        runs on any of them. A chunk's embedding similarity reflects its
        whole text, so a chunk that is mostly about something else but
        contains one on-topic sentence can rank below several chunks that
        merely resemble the question throughout without answering it
        (observed live: the correct "purpose of a thick blood film" chunk
        ranked 8th by embedding, past the top-5 window ever inspected,
        while a graph-entity check against the same 15-candidate pool
        immediately narrowed it to the 2 chunks whose own MENTIONS
        entities actually covered every distinctive question term).
        Returns chunk_ids reordered by how many of required_terms their
        entities cover, most-covering first, ties broken by original
        order; a chunk with no entities in the graph is neither promoted
        nor dropped, just left at the back of its tier -- sparse entity
        coverage is not evidence a chunk is off-topic, only that the
        graph has less to say about it.
        """
        if not chunk_ids or not required_terms:
            return chunk_ids
        try:
            if not self.connect():
                return chunk_ids
            query = """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})
            OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
            RETURN cid AS chunk_id,
                   collect(DISTINCT coalesce(e.canonical_name, e.normalized_name)) AS names
            """
            with self.driver.session(database=self.database) as session:
                entities_by_chunk = {
                    record["chunk_id"]: record["names"]
                    for record in session.run(query, chunk_ids=chunk_ids)
                }
        except Exception:
            return chunk_ids
        required = set(required_terms)
        original_order = {cid: index for index, cid in enumerate(chunk_ids)}

        def overlap_count(cid: str) -> int:
            names = entities_by_chunk.get(cid) or []
            entity_terms = roots(" ".join(name for name in names if name))
            return len(required & entity_terms)

        return sorted(
            chunk_ids,
            key=lambda cid: (-overlap_count(cid), original_order[cid]),
        )


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
        self, chunk_ids: list[str], source_pages: list[int], answer_text: str
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        answer_fignums = set(re.findall(r"Fig(?:ure)?\.?\s*(\d+\.\d+)", answer_text, re.I))
        answer_keywords = content_roots(answer_text)
        relations: list[tuple[str, dict[str, str], str]] = []
        for chunk_id in chunk_ids:
            for relation in self.chunk_images.get(chunk_id, []):
                relations.append((chunk_id, relation, "direct chunk-to-image relationship"))
        answer_references_figure = bool(answer_fignums)
        figure_pages = {
            chunk.pdf_page
            for chunk in self.pdf.chunks
            if answer_references_figure and chunk.chunk_id in chunk_ids
        }
        for page in source_pages:
            if page not in figure_pages:
                continue
            relations.extend(
                ("", relation, "figure on the cited PDF page")
                for relation in self.page_images.get(page, [])
            )
        for chunk_id, relation, reason in relations:
                image_id = relation.get("image_id", "")
                if not image_id or image_id in seen:
                    continue
                meta = self.images.get(image_id, {})
                predicted = meta.get("final_type") or meta.get("predicted_type") or relation.get("image_type")
                relevance = meta.get("content_relevance", "")
                width = int(meta.get("pixel_width") or meta.get("width") or 0)
                height = int(meta.get("pixel_height") or meta.get("height") or 0)
                large_cited_figure = (
                    reason == "figure on the cited PDF page"
                    and width >= 300 and height >= 250
                )
                # Every image has a short set of descriptive keywords (its
                # own caption text, or, for the minority with no caption
                # anywhere, the body text immediately around it on the
                # page) -- see build_image_keywords.py. Chunk/page
                # membership, and even an exact figure-number citation, only
                # prove an image is located near the source text -- not that
                # it depicts what the *answer* actually says (a chunk can
                # cite one figure while the extracted answer sentence is
                # about an unrelated part of the same chunk). Treat a
                # figure-number match and a keyword match as two
                # independent, either-suffices signals of relevance: two
                # shared keywords is required (one is treated as
                # coincidental -- e.g. a parasite-morphology figure's
                # keywords happening to include the generic word "present",
                # which also appears in an unrelated answer sentence about
                # parasite density -- observed live). An image with no
                # stored keywords at all (no caption and no nearby text
                # could be extracted) is passed through unfiltered on this
                # check, since there is nothing to check it against either
                # way.
                figure_number = relation.get("figure_number")
                cites_matching_figure = bool(figure_number) and figure_number in answer_fignums
                image_keywords = set(meta.get("keywords", "").split())
                keyword_match = (
                    len(answer_keywords & image_keywords) >= 2 if image_keywords else True
                )
                if figure_number:
                    if not (cites_matching_figure or keyword_match):
                        continue
                elif not keyword_match:
                    continue
                if predicted in {"logo", "decorative"}:
                    continue
                if predicted == "fragment_or_noise" and not large_cited_figure:
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
                    "_file_path": file_path,
                })
        ranked = sorted(result, key=lambda item: item["score"], reverse=True)
        for item in ranked:
            item.pop("_file_path")
        return ranked

    @staticmethod
    def _benchmark(question: str) -> dict[str, Any] | None:
        key = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
        return BENCHMARK_BY_QUESTION.get(key)

    @staticmethod
    def _scores(
        result: dict[str, Any], graph_result: dict[str, Any],
        benchmark: dict[str, Any] | None, images: list[dict[str, Any]], mode: str,
    ) -> dict[str, Any]:
        accepted = {source["chunk_id"] for source in result.get("sources", [])}
        gold = GOLD_EVIDENCE_BY_ID.get(int(benchmark["id"])) if benchmark else None
        precision = recall = f1 = None
        if gold is not None:
            true_positive = len(accepted & gold)
            precision = round(100 * true_positive / max(len(accepted), 1), 1)
            recall = round(100 * true_positive / len(gold), 1)
            f1 = (
                round(2 * precision * recall / (precision + recall), 1)
                if precision + recall else 0.0
            )
        graph = (
            round(100 * len(graph_result.get("verified_chunks", [])) /
                  max(len(result.get("sources", [])), 1))
            if mode == "graph" else None
        )
        relevance_summary = graph_result.get("content_relevance_summary") or {}
        judged = relevance_summary.get("relevant", 0) + relevance_summary.get("no_overlap", 0)
        relevance_pct = (
            round(100 * relevance_summary.get("relevant", 0) / judged)
            if mode == "graph" and judged else None
        )
        return {
            "accuracy_pct": f1,
            "gold_precision_pct": precision,
            "gold_recall_pct": recall,
            "neo4j_verification_pct": graph,
            "neo4j_relevance_pct": relevance_pct,
            "neo4j_relevance_detail": relevance_summary if mode == "graph" else None,
            "gold_annotated": gold is not None,
            "gold_correct": f1 == 100.0 if f1 is not None else None,
            "source_exact": bool(result.get("verification", {}).get("complete")),
            "gold_chunks": sorted(gold) if gold is not None else [],
            "label": "gold evidence F1" if gold is not None else "not measured",
        }

    def ask(self, question: str, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        benchmark = self._benchmark(question)
        # The question text is always sent to retrieval unmodified. Gold
        # labels and the benchmark catalog are used only for post-hoc
        # scoring below, never to alter what the pipeline searches for.
        # This used to be restricted to questions added after the original
        # 30-question benchmark, since Aura's raw chunk-ranking is a simple
        # term-count heuristic that can rank an already-correct chunk below
        # an unrelated one just as easily as it can catch a genuinely wrong
        # one. That is no longer the actual safety mechanism: the override
        # below only ever fires when PDF's own answer is missing a term the
        # question names that Aura's candidate actually supplies (a real,
        # provable gap), and Aura's full candidate answer must still clear
        # the same semantic relevance floor used elsewhere in this file.
        # Those two checks are what protects an already-correct answer, not
        # which question happened to be asked -- and a free-text box means
        # a fixed question id can't be the gate anyway, since most
        # questions will have no id at all.
        allow_graph_override = True
        result = (
            self._graph_answer(question, allow_graph_override)
            if mode == "graph" else self.pdf.answer(question)
        )
        chunk_ids = [source["chunk_id"] for source in result.get("sources", [])]
        graph_result = {
            "status": "not_requested",
            "verified_chunks": [],
            "locations": [],
        }
        if mode == "graph" and result["kind"] != "small_talk":
            question_terms = sorted(set(
                term
                for need in result.get("needs", [])
                for term in need.get("distinctive_subject_terms", [])
            ))
            graph_result = self.graph.verify(chunk_ids, question_terms=question_terms)
            if graph_result["status"] != "verified":
                result["kind"] = "not_found"
                result["answer"] = "The textual evidence could not be fully verified in Neo4j."
                result["verification"]["complete"] = False
            else:
                result["verification"]["neo4j_verified"] = True
        result["question"] = question
        result["mode"] = mode
        result["graph"] = graph_result
        source_pages = [int(source.get("pdf_page") or 0) for source in result.get("sources", [])]
        result["images"] = (
            self.related_images(chunk_ids, source_pages, result.get("answer", ""))
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
        result["engine_revision"] = ENGINE_REVISION
        return result

    def _graph_answer(self, question: str, allow_override: bool = False) -> dict[str, Any]:
        """Use Neo4j to expand each need's text candidates before extraction."""
        cleaned = clean_question(question)
        canned = small_talk_response(cleaned)
        if canned is not None:
            return {
                "kind": "small_talk",
                "question": cleaned,
                "answer": canned,
                "natural_answer": None,
                "needs": [],
                "sources": [],
                "verification": {
                    "complete": True,
                    "all_claims_are_exact_source_spans": False,
                    "needs_covered": 0,
                    "needs_total": 0,
                },
            }
        needs = self.pdf.plan(cleaned)
        chunk_index = {
            chunk.chunk_id: index for index, chunk in enumerate(self.pdf.chunks)
        }
        need_results: list[dict[str, Any]] = []
        source_indices: list[int] = []
        complete = True
        for need in needs:
            ranked = self.pdf.retrieve(need)
            pdf_units = [
                unit for unit in self.pdf.extract(need, ranked)
                if self.pdf.verify_unit(unit)
            ]
            pdf_units = [
                unit for unit in self.pdf.extend_across_chunk_boundary(need, pdf_units)
                if self.pdf.verify_unit(unit)
            ]
            pdf_complete = self.pdf.need_complete(need, pdf_units)
            seed_ids = [
                self.pdf.chunks[index].chunk_id for index, _ in ranked[:10]
            ]
            distinctive_subject = sorted(
                set(need.subject_terms) - {
                    stem(term) for term in GENERIC_SUBJECT_ROOTS
                }
            )
            independent_ids = self.graph.search(
                list(need.subject_terms) + list(roots(need.query)),
                required_terms=distinctive_subject,
            )
            # Neo4j's own search is literal term-count matching, which can
            # rank a passage that merely names the subject once above the
            # chunk that actually explains it in different words, or bury
            # the right chunk outside whatever window later code inspects
            # (observed live: the correct Giemsa-stain recipe ranked 8th on
            # term count and was never reached). Embedding similarity finds
            # the same chunk by meaning even when it shares few or no terms
            # with the question, so merge it in ahead of the term-count
            # list rather than replacing it -- Neo4j's exact-terminology
            # matching still catches a specific technical term embeddings
            # alone can blur.
            semantic_ids = [
                self.pdf.chunks[index].chunk_id
                for index, _ in self.pdf.semantic_candidates(need.query)
            ]
            independent_ids = list(dict.fromkeys(semantic_ids + independent_ids))
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
            graph_units = [
                unit for unit in self.pdf.extract(need, graph_ranked)
                if self.pdf.verify_unit(unit)
            ]
            graph_units = [
                unit for unit in self.pdf.extend_across_chunk_boundary(need, graph_units)
                if self.pdf.verify_unit(unit)
            ]
            # Aura may fill a missing need, but it must not replace an already
            # complete PDF result with a weaker graph candidate.
            units = pdf_units if pdf_complete else graph_units
            # "Reason" and "comparison" answers explain or contrast in
            # whatever words the source uses, which routinely shares few
            # of the question's own surface words by design -- the term-
            # gap check below reads that as "PDF is missing something"
            # even when PDF's own answer is already the more directly
            # relevant one (e.g. it names the actual distinguishing fact,
            # while Aura's differently-worded candidate about a related
            # but different aspect just happens to supply the "missing"
            # term). Both were observed to be false positives in testing.
            # "What is the purpose of X?" asks for the same kind of
            # rationale as a "why" question and fails the override for the
            # identical reason, but the answer_type() classifier (used
            # elsewhere for reason-specific extraction narrowing that must
            # not change for this question shape) does not recognize
            # "purpose" as a reason trigger -- it is checked directly here,
            # against the question text alone, instead of widening that
            # classifier and risking a change to unrelated extraction
            # behavior for every other "reason"-classified question.
            is_purpose_question = bool(re.search(r"\bpurpose\b", need.query, re.I))
            override_eligible_type = (
                need.answer_type not in {"reason", "comparison"}
                and not is_purpose_question
            )
            if allow_override and pdf_complete and independent_ids and override_eligible_type:
                # Aura's own top hit already had to contain every
                # distinctive subject term verbatim (a stricter bar than
                # the PDF path's ~60% overlap); when the PDF's own chosen
                # chunk isn't that hit, that is real evidence the PDF path
                # anchored on the wrong section. What actually protects an
                # already-correct answer here is the term-gap and semantic-
                # relevance checks further down, not which question was
                # asked -- Aura's simple term-count ranking on its own is
                # not reliable enough to overrule an existing answer.
                pdf_chunk_ids = {
                    self.pdf.chunks[unit.chunk_index].chunk_id for unit in pdf_units
                }
                # Aura's raw ranking is pure term-counting -- a chunk that
                # only names the subject once (an intro sentence, a
                # contents line) scores the same as one that actually
                # describes it. Re-rank its top few candidates with two
                # independent, stronger signals before trusting any of
                # them over PDF: whether a required term sits in a
                # heading-like line in that chunk (the subject is what the
                # section is about, not a passing mention), and the same
                # semantic reranker the PDF path itself relies on.
                def heading_hits(chunk_id: str) -> int:
                    idx = chunk_index.get(chunk_id)
                    if idx is None or not distinctive_subject:
                        return 0
                    required = set(distinctive_subject)
                    hits = 0
                    for raw_line in self.pdf.chunks[idx].text.splitlines():
                        line = raw_line.strip()
                        tokens = line.split()
                        if 1 < len(tokens) <= 12 and not line.endswith((".", ";")):
                            hits += len(required & roots(line))
                    return hits

                # A top-5 cut straight off embedding/term rank keeps missing
                # the right chunk when its match is one on-topic sentence
                # inside an otherwise-generic chunk (observed live: the
                # correct answer ranked 8th by embedding, never reaching
                # this window). Widen the pool first, then cheaply narrow it
                # with the graph's own entity relationships -- how many of
                # the question's distinctive terms each candidate's own
                # MENTIONS entities cover -- before spending the expensive
                # reranker on only the top 5 of *that*.
                wide_pool = list(dict.fromkeys(
                    [
                        self.pdf.chunks[index].chunk_id
                        for index, _ in self.pdf.semantic_candidates(need.query, top_k=15)
                    ] + independent_ids
                ))[:15]
                ranked_pool = self.graph.rank_by_entity_overlap(
                    wide_pool, distinctive_subject
                )
                top_candidates = [
                    cid for cid in ranked_pool[:5] if cid in chunk_index
                ]
                best_id = None
                if top_candidates:
                    pairs = [
                        [need.query, self.pdf.chunks[chunk_index[cid]].text[:600]]
                        for cid in top_candidates
                    ]
                    semantic_scores = self.pdf.reranker.predict(
                        pairs, show_progress_bar=False
                    )
                    # Order candidates by the same weighted signal as
                    # before (raw relevance plus a bonus for naming the
                    # subject in a heading-like line: a chunk padded with
                    # many short reagent-list lines can rack up heading
                    # hits without being the right section, so that alone
                    # must not outrank a much stronger semantic match) --
                    # but do not simply trust whichever ranks first. Walk
                    # that order and anchor on the first candidate that is
                    # both actually new (a candidate PDF's own retrieval
                    # already settled on can't be the fix for PDF's own
                    # retrieval -- observed live: embedding similarity
                    # ranked a same-chemistry buffer recipe from an
                    # unrelated section above the correct one, and it also
                    # happened to be PDF's own wrong pick, which silently
                    # cancelled the override instead of moving on to the
                    # next candidate) and whose own relevance genuinely
                    # clears the same floor used everywhere else in this
                    # file. A candidate that fails either check is skipped
                    # outright, not patched over -- the next independently-
                    # found candidate gets the same chance.
                    for cid, score in sorted(
                        zip(top_candidates, semantic_scores),
                        key=lambda item: float(item[1]) + 0.3 * heading_hits(item[0]),
                        reverse=True,
                    ):
                        if cid not in pdf_chunk_ids and float(score) > -2:
                            best_id = cid
                            break
                if best_id is not None and best_id not in pdf_chunk_ids:
                    top_index = chunk_index.get(best_id)
                    if top_index is not None:
                        # A single chunk in isolation starves extract()'s own
                        # numbered-step continuation walk (it looks for step
                        # 8, 9, ... among the candidates it was given): a
                        # procedure that spans a chunk or page boundary --
                        # very common, since chunk cuts fall mid-sentence --
                        # would be silently truncated at whatever step the
                        # chosen chunk happens to end on. Grow a window of
                        # neighbours actually verified relevant to this
                        # query (or that continue an open numbered step)
                        # instead of an unconditional flat block of
                        # following chunks, which let extraction wander
                        # into a different, merely-nearby procedure.
                        window_indices = self.pdf.relevant_window(
                            need.query, top_index
                        )
                        # A backward neighbour and a forward neighbour the
                        # same distance from the anchor must not tie: a
                        # backward chunk is more often the general lead-in
                        # to the topic (observed live -- a "how the thick
                        # film is made" step here, two chunks before the
                        # anchor, beat the actual Giemsa recipe extract()
                        # was supposed to find, once distance alone made
                        # them score equally and stable-sort order settled
                        # the tie toward whichever came first). Score every
                        # backward chunk below every forward one so a tie
                        # in distance never lets extract() prefer it.
                        override_ranked = [
                            (
                                index,
                                1.0 - 0.01 * (index - top_index)
                                if index >= top_index
                                else -0.01 * (top_index - index)
                            )
                            for index in window_indices
                        ]
                        override_units = [
                            unit for unit in self.pdf.extract(need, override_ranked)
                            if self.pdf.verify_unit(unit)
                        ]
                        if override_units and self.pdf.need_complete(need, override_units):
                            # "Different chunk, structurally complete" is not
                            # by itself proof PDF was wrong -- a large multi-
                            # topic chunk can share the subject's vocabulary
                            # (a table row, a passing mention) without being
                            # about what the question actually asks (e.g. a
                            # Pandy-test chunk that merely mentions CSF, for
                            # a question about what a bloodstained CSF
                            # appearance means). Only trust Aura's candidate
                            # over an already-complete PDF answer when both
                            # hold: (1) PDF's own answer demonstrably misses
                            # a term the question names that Aura's answer
                            # actually supplies -- real evidence of a gap,
                            # not just a different pick -- and (2) Aura's
                            # full answer still reads as genuinely relevant
                            # to the question, not merely sharing that one
                            # term out of context.
                            pdf_covered = roots(
                                " ".join(unit.text for unit in pdf_units)
                            )
                            override_covered = roots(
                                " ".join(unit.text for unit in override_units)
                            )
                            missing_from_pdf = set(distinctive_subject) - pdf_covered
                            fills_a_real_gap = bool(missing_from_pdf & override_covered)
                            # The fragment itself -- the exact text that
                            # would be shown, not the anchor chunk's wider
                            # context -- is independent evidence PDF chose
                            # wrong even without a term-gap: PDF's own wrong
                            # chunk can already contain every distinctive
                            # word (a neighbouring paragraph, same topic)
                            # while answering a different question (observed
                            # live: both the wrong and the right "thick
                            # blood film" chunks say "thick", "blood" and
                            # "film" -- only one is actually about its
                            # purpose, and its own extracted sentence scores
                            # strongly relevant on its own). This must stay
                            # a high, fragment-only bar, not a comparison
                            # against the anchor chunk's wider context: a
                            # heading-like line elsewhere in that context
                            # can share the question's words out of context
                            # and score high regardless of whether the
                            # actual answer text is relevant (observed live:
                            # a "Boxes and jars for collecting sputum
                            # specimens" heading made an unrelated carton-
                            # folding procedure's *context* outscore PDF's
                            # own correct sputum-collection answer, even
                            # though that same carton-folding text scored
                            # very low as a fragment on its own).
                            # PDF's own already-accepted fragment is scored
                            # the same way, so the override can be required
                            # to actually beat it below -- otherwise (observed
                            # live, "purpose of a thick blood film" and "when
                            # should blood be collected") a candidate could
                            # clear the floor checks above on its own merits
                            # while still being the objectively worse of the
                            # two passages, and would replace an answer that
                            # was never shown to be inferior in the first
                            # place.
                            fragment_score, pdf_fragment_score = (
                                float(s) for s in self.pdf.reranker.predict(
                                    [
                                        [need.query, " ".join(unit.text for unit in override_units)],
                                        [need.query, " ".join(unit.text for unit in pdf_units)],
                                    ],
                                    show_progress_bar=False,
                                )
                            )
                            if fills_a_real_gap or fragment_score > 2:
                                # extract() keeps only the verified answer
                                # span, which can drop the heading/context
                                # words (e.g. the subject's own name) that
                                # told the reranker what the passage was
                                # about in the first place -- scoring only
                                # that narrower fragment then judges a
                                # correct answer out of context. Restore
                                # context from the chunk(s) the accepted
                                # text was actually drawn from -- not the
                                # window's anchor chunk, since extract()'s
                                # own walk over the window can settle on a
                                # different, less relevant chunk within it
                                # (e.g. a same-window table entry that only
                                # superficially matches). This fuller-context
                                # rescue only matters for the term-gap path
                                # above (fragment_score already cleared a
                                # high bar on its own for the other path):
                                # whichever scoring -- the narrow fragment or
                                # its own fuller context -- sees the passage
                                # as relevant is enough to clear the floor.
                                override_source_indices = sorted({
                                    unit.chunk_index for unit in override_units
                                })
                                context_scores = self.pdf.reranker.predict(
                                    [
                                        [need.query, self.pdf.chunks[i].text[:600]]
                                        for i in override_source_indices
                                    ],
                                    show_progress_bar=False,
                                )
                                override_answer_score = max(
                                    fragment_score, float(max(context_scores))
                                )
                                # Term-overlap and relevance-floor checks above
                                # can both pass while the override candidate is
                                # still about the wrong specimen (blood, urine,
                                # CSF, stool, sputum, ...) -- procedural
                                # wording repeats near-identically across
                                # specimen sections, so a fragment can score as
                                # relevant while answering a different
                                # specimen's version of the same procedure
                                # entirely (observed live: a blood-leukocyte
                                # question's override landed on a CSF-leukocyte
                                # passage). Reuse the same specimen-detection
                                # function already used for claim-level
                                # verification to veto the override whenever
                                # the question names a specimen the candidate
                                # text does not share.
                                question_specimens = detect_specimen_types(question)
                                override_specimens = detect_specimen_types(
                                    " ".join(unit.text for unit in override_units)
                                )
                                specimen_mismatch = bool(
                                    question_specimens
                                    and override_specimens
                                    and not (question_specimens & override_specimens)
                                )
                                beats_pdf_answer = override_answer_score > pdf_fragment_score
                                if (
                                    override_answer_score > -2
                                    and not specimen_mismatch
                                    and beats_pdf_answer
                                ):
                                    units = override_units
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
                "distinctive_subject_terms": distinctive_subject,
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
        # Rephrasing happens on demand via the /rephrase endpoint, not
        # here -- see the matching note in DirectPdfQA.answer().
        return {
            "kind": "domain_answer" if complete else "not_found",
            "question": cleaned,
            "answer": (
                "\n\n".join(answer_parts)
                if complete else "No complete extractive answer was verified."
            ),
            "natural_answer": None,
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


@app.post("/rephrase")
def rephrase(request: RephraseRequest) -> dict[str, Any]:
    # Separate from /ask so the fast, already-verified extractive answer
    # never has to wait on this: the local model has no usable GPU on this
    # deployment and can take tens of seconds per call.
    question = clean_question(request.question)
    if not question or not request.verified_text.strip():
        return {"natural_answer": None, "claims": []}
    natural_answer = service().pdf.rephrase(question, request.verified_text)
    if not natural_answer:
        return {"natural_answer": None, "claims": []}
    if request.sources:
        claims = service().pdf.attribute_claims_to_chunks(
            natural_answer, [source.model_dump() for source in request.sources]
        )
        _attach_claim_graph_evidence(claims)
    else:
        claims = service().pdf.verify_answer_claims(request.verified_text, natural_answer)
    _flag_specimen_mismatch(question, claims)
    return {"natural_answer": natural_answer, "claims": claims}


def _flag_specimen_mismatch(question: str, claims: list[dict[str, Any]]) -> None:
    """A second, independent hallucination-risk axis alongside the NLI
    status: does the specimen type this claim's cited source text is
    actually about (blood/urine/CSF/stool/sputum/...) match the specimen
    type the question asked about? Procedural wording (drops, centrifuge,
    ml, stain) repeats near-identically across specimen types in this
    document, which is exactly the overlap that can make word-level NLI
    call a claim "supported" even when it was answered from the wrong
    specimen's section entirely -- this flag catches that case regardless
    of the NLI verdict.
    """
    question_specimens = detect_specimen_types(question)
    if not question_specimens:
        return
    for claim in claims:
        source_text = claim.get("source_text")
        if not source_text:
            continue
        claim_specimens = detect_specimen_types(source_text)
        claim["specimen_mismatch"] = bool(claim_specimens) and not (claim_specimens & question_specimens)


def _attach_claim_graph_evidence(claims: list[dict[str, Any]]) -> None:
    """Look up the page/image the graph already links each claim's cited
    chunk to (one batched Neo4j round trip for every claim in the answer,
    not one per claim), so a claim's evidence can point at the same
    illustration or page a reader would find in the Aura graph -- reusing
    the /media/<filename> URL convention the rest of this file already
    uses for image evidence.
    """
    chunk_ids = sorted({claim["chunk_id"] for claim in claims if claim.get("chunk_id")})
    if not chunk_ids:
        for claim in claims:
            claim["image"] = None
            claim["graph_path"] = None
        return
    records = {
        record["chunk_id"]: record
        for record in service().graph.verify(chunk_ids).get("locations", [])
    }
    for claim in claims:
        record = records.get(claim.get("chunk_id"))
        if not record:
            claim["image"] = None
            claim["graph_path"] = None
            continue
        images = [img for img in record.get("images", []) if img.get("id")]
        # A citation-derived image's figure_number, or an image's own
        # descriptive keywords (see build_image_keywords.py and the longer
        # comment in related_images()), are only meaningful for *this*
        # claim if the claim's own text actually names that figure or
        # shares those keywords -- otherwise the image may belong to a
        # different topic within the same multi-topic chunk. Prefer a
        # match on either signal; an image with neither a figure_number
        # nor any stored keywords imposes no requirement and remains the
        # fallback.
        claim_text = f"{claim.get('text', '')} {claim.get('source_text', '')}"
        claim_fignums = set(re.findall(r"Fig(?:ure)?\.?\s*(\d+\.\d+)", claim_text, re.I))
        claim_keywords = content_roots(claim_text)

        def claim_matches(img: dict[str, Any]) -> bool:
            figure_number = img.get("figure_number")
            image_keywords = set((img.get("keywords") or "").split())
            if figure_number and figure_number in claim_fignums:
                return True
            if image_keywords:
                return len(claim_keywords & image_keywords) >= 2
            return not figure_number

        candidates = [img for img in images if claim_matches(img)]
        image = candidates[0] if candidates else None
        claim["image"] = (
            {
                "image_id": image["id"],
                "url": f"/media/{Path(image['file_path']).name}" if image.get("file_path") else None,
            }
            if image else None
        )
        path_parts = [f"Chunk {claim['chunk_id']}"]
        if record.get("pdf_page") is not None:
            path_parts.append(f"Page {record['pdf_page']}")
        if image:
            path_parts.append(f"Image {image['id']}")
        claim["graph_path"] = " -> ".join(path_parts)


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
    .compare-panel{margin-top:20px}.compare-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:14px}.chunk-list{font:12px/1.5 Consolas,monospace;word-break:break-word;color:#334155}.gain{color:#047857}.loss{color:#b91c1c}
    .graph{width:100%;min-height:220px;border:1px solid var(--line);border-radius:10px;background:#fbfdff}.graph text{font:12px Segoe UI,Arial,sans-serif}.graph-edge{stroke:#94a3b8;stroke-width:1.5}.graph-label{fill:#64748b;font-size:10px}.node-document{fill:#dbeafe}.node-page{fill:#dcfce7}.node-chunk{fill:#fef3c7}.node-entity{fill:#ede9fe}.node-image{fill:#ffe4e6}.node-pageimage{fill:#f1f5f9;stroke-dasharray:3,2}
    @media(max-width:800px){.grid{grid-template-columns:1fr}.top{display:block}.badge{display:inline-block;margin-top:12px}.meta{grid-template-columns:1fr}}
  </style>
</head>
<body><main class="shell">
  <header class="top"><div><h1>Grounded PDF QA Evaluation</h1><p class="subtitle">Compare direct PDF evidence with Neo4j-grounded verification using the same question set.</p></div><span class="badge" id="questionCountBadge">benchmark</span></header>
  <section class="panel">
    <label class="label" for="questionSelect">Evaluation question</label>
    <select id="questionSelect" onchange="customQuestion.value=''"><option value="">Loading questions…</option></select>
    <textarea id="customQuestion" placeholder="Or type your own question about the manual…"></textarea>
    <div class="actions"><button class="primary" onclick="run('pdf')">Run PDF only</button><button class="secondary" onclick="run('graph')">Run PDF + Neo4j</button><button class="compare" onclick="compareBoth()">Compare both</button></div>
  </section>
  <div id="comparison"></div>
  <section class="grid"><article class="result" id="pdfResult"></article><article class="result" id="graphResult"></article></section>
</main>
<script>
const select=document.getElementById('questionSelect');
function empty(title){return `<div class="head"><h2>${title}</h2><span class="state idle">Not run</span></div><p class="subtitle">Choose a question and run this mode.</p>`}
document.getElementById('pdfResult').innerHTML=empty('PDF only');document.getElementById('graphResult').innerHTML=empty('PDF + Neo4j');
fetch('/questions').then(r=>r.json()).then(items=>{select.innerHTML=`<option value="">Select one of ${items.length} questions…</option>`+items.map(q=>`<option value="${q.id}" data-q="${q.question.replaceAll('&','&amp;').replaceAll('"','&quot;')}">${String(q.id).padStart(2,'0')} · ${q.question}</option>`).join('');document.getElementById('questionCountBadge').textContent=`${items.length}-question benchmark`});
const customQuestion=document.getElementById('customQuestion');
function question(){return customQuestion.value.trim()||select.selectedOptions[0]?.dataset.q||''}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function cleanSource(text){return String(text??'').replace(/^\d+\s+Manual of basic techniques for a health laboratory\s*/i,'').trim()}
function graphMarkup(viz){const raw=viz?.nodes||[],kept=[];for(const type of ['Document','Page','Chunk','Entity','Image','PageImage']){const cap=['Entity','Image','PageImage'].includes(type)?6:20;kept.push(...raw.filter(n=>n.type===type).slice(0,cap))}if(!kept.length)return '<p class="subtitle">No graph path was returned.</p>';const ids=new Set(kept.map(n=>n.id)),edges=(viz.edges||[]).filter(e=>ids.has(e.source)&&ids.has(e.target)),columns={Document:85,Page:255,Chunk:430,Entity:620,Image:790,PageImage:960},counts={},positions={};for(const n of kept){const i=counts[n.type]||0;counts[n.type]=i+1;positions[n.id]={x:columns[n.type]||430,y:55+i*62}}const height=Math.max(220,...Object.values(positions).map(p=>p.y+45));const edgeSvg=edges.map(e=>{const a=positions[e.source],b=positions[e.target],mx=(a.x+b.x)/2,my=(a.y+b.y)/2;return `<line class="graph-edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"/><text class="graph-label" x="${mx}" y="${my-4}" text-anchor="middle">${esc(e.label)}</text>`}).join('');const nodeSvg=kept.map(n=>{const p=positions[n.id],label=String(n.label||n.id).slice(0,24);return `<g><rect class="node-${n.type.toLowerCase()}" x="${p.x-68}" y="${p.y-18}" width="136" height="36" rx="9" stroke="#94a3b8"/><text x="${p.x}" y="${p.y+4}" text-anchor="middle">${esc(label)}</text></g>`}).join('');return `<svg class="graph" viewBox="0 0 1060 ${height}" role="img" aria-label="Neo4j evidence graph">${edgeSvg}${nodeSvg}</svg><p class="subtitle">Solid pink = image directly matched to this chunk (relevant). Dashed grey = other images on the same page (context only).</p>`}
function graphStats(viz){const nodes=viz?.nodes||[],edges=viz?.edges||[];return `${nodes.length} real Aura nodes · ${edges.length} real relationships`}
let results={};
function render(mode,data){results[mode]=data;const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult'),sourceExact=data.kind==='domain_answer'&&data.verification?.complete;const sources=data.sources||[],images=data.images||[],score=data.scores||{},goldMeasured=score.gold_annotated===true,goldCorrect=score.gold_correct===true,ok=goldMeasured?goldCorrect:sourceExact,status=goldMeasured?(goldCorrect?'Gold evidence matched':'Gold evidence incomplete'):(sourceExact?'Source-exact · answer accuracy not measured':'Not verified');target.innerHTML=`<div class="head"><h2>${mode==='pdf'?'PDF only':'PDF + Neo4j'}</h2><span class="state ${ok?'ok':'bad'}">${status}</span></div><p class="answer ${sourceExact?'':'error'}">${esc(data.answer)}</p>${sourceExact&&data.kind!=='small_talk'?`<details open><summary>Rephrased answer (LLM, constrained to the verified text above)</summary><p class="subtitle" id="rephrase-${mode}">Generating…</p></details>`:''}<div class="meta"><div class="metric"><b>${sources.length}</b><span>source chunks</span></div><div class="metric"><b>${images.length}</b><span>related images</span></div><div class="metric"><b>${data.timing_ms??'-'} ms</b><span>runtime</span></div></div>${mode==='graph'?`<div class="metric"><b>Neo4j traceability: ${esc(data.graph?.status)} · ${score.neo4j_verification_pct??0}%</b><span>share of selected source chunks located in Aura; not answer accuracy</span></div><details open><summary>Neo4j evidence graph</summary><p class="subtitle">${esc(graphStats(data.graph?.visualization))} · loaded from the connected Aura database</p>${graphMarkup(data.graph?.visualization)}</details><details open><summary>Cypher executed on Aura</summary><pre class="chunk-list">${esc(data.graph?.query||'')}</pre></details>`:''}<details open><summary>Evidence and locations</summary>${sources.length?sources.map(s=>`<div class="source"><b>${esc(s.chunk_id)}</b> · PDF ${esc(s.pdf_page)} · Printed ${esc(s.printed_page)}<p>${esc(cleanSource(s.text))}</p></div>`).join(''):'<p class="subtitle">No verified source.</p>'}</details>${images.length?`<details open><summary>Related image evidence</summary><div class="images">${images.map(i=>`<div>${i.url?`<a href="${esc(i.url)}" target="_blank"><img src="${esc(i.url)}" alt="${esc(i.image_id)}"></a>`:''}<small>${esc(i.image_id)} · page ${esc(i.pdf_page)}</small></div>`).join('')}</div></details>`:'<details><summary>Related image evidence</summary><p class="subtitle">No image relationship was verified for these sources.</p></details>'}`}
const runToken={pdf:0,graph:0};
async function run(mode){const q=question();if(!q){alert('Select or enter a question.');return}const token=++runToken[mode];const target=document.getElementById(mode==='pdf'?'pdfResult':'graphResult');target.innerHTML=`<div class="head"><h2>Answer</h2><span class="state idle">Running…</span></div><p class="subtitle">The first request loads the reranker once.</p>`;document.querySelectorAll('button').forEach(b=>b.disabled=true);try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,mode})});if(runToken[mode]!==token)return;const data=await r.json();render(mode,data);const sourceExact=data.kind==='domain_answer'&&data.verification?.complete;if(sourceExact&&data.kind!=='small_talk'){const evidenceItems=(data.needs||[]).flatMap(n=>n.evidence||[]);const verifiedText=evidenceItems.map(e=>e.text).join('\n');fetch('/rephrase',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,verified_text:verifiedText,sources:evidenceItems})}).then(rr=>rr.json()).then(rd=>{if(runToken[mode]!==token)return;const el=document.getElementById(`rephrase-${mode}`);if(el)el.innerHTML=renderClaims(rd)}).catch(()=>{if(runToken[mode]!==token)return;const el=document.getElementById(`rephrase-${mode}`);if(el)el.textContent='Not available for this answer.'})}}catch(e){if(runToken[mode]===token)target.innerHTML=`<p class="error">${esc(e.message)}</p>`}finally{document.querySelectorAll('button').forEach(b=>b.disabled=false)}}
function renderClaims(rd){if(!rd.natural_answer)return 'Not available for this answer.';if(!rd.claims||!rd.claims.length)return esc(rd.natural_answer);const badge={supported:'ok',contradicted:'bad',insufficient_evidence:'idle',not_checked:'idle'};return rd.claims.map(c=>{
  // A contradicted claim is drift the model introduced, not a fact worth
  // showing in red next to everything else -- replace it with the exact
  // source wording it was checked against (shown as verified/green) instead
  // of leaving the wrong sentence on screen at all.
  if(c.status==='contradicted'&&c.source_text){
    const cite=c.chunk_id?` <small>[${esc(c.chunk_id)}${c.graph_path?` · ${esc(c.graph_path)}`:''}]</small>`:'';
    const img=c.image&&c.image.url?` <a href="${esc(c.image.url)}" target="_blank">[image]</a>`:'';
    return `<span class="state ok" style="display:inline;white-space:normal" title="Replaced: the generated sentence drifted from the source">${esc(c.source_text)}</span>${cite}${img}`;
  }
  const cls=badge[c.status]||'idle';const cite=c.chunk_id?` <small>[${esc(c.chunk_id)}${c.graph_path?` · ${esc(c.graph_path)}`:''}]</small>`:'';const img=c.image&&c.image.url?` <a href="${esc(c.image.url)}" target="_blank">[image]</a>`:'';
  // Independent of the NLI verdict: the cited source may simply be about
  // the wrong specimen (e.g. CSF instead of blood) -- wording overlap can
  // make NLI call that "supported" anyway, so this warns regardless.
  const mismatch=c.specimen_mismatch?' <small class="state bad">⚠ possibly wrong specimen type in cited source</small>':'';
  return `<span class="state ${cls}" style="display:inline;white-space:normal">${esc(c.text)}</span>${mismatch}${cite}${img}`}).join(' ')}
function unique(values){return [...new Set(values||[])]}
function chunkText(values){return values.length?values.map(esc).join(', '):'<span class="subtitle">None (same evidence as the other mode)</span>'}
async function compareBoth(){results={};document.getElementById('comparison').innerHTML='';await run('pdf');await run('graph');const p=results.pdf?.scores?.accuracy_pct,g=results.graph?.scores?.accuracy_pct,measured=p!=null&&g!=null,d=measured?g-p:null,pdfAccepted=unique(results.pdf?.retrieval_trace?.accepted_chunks),graphAccepted=unique(results.graph?.retrieval_trace?.accepted_chunks),common=pdfAccepted.filter(id=>graphAccepted.includes(id));const improved=measured&&d>0,verdict=!measured?'No Gold annotation exists for this question, so no comparison is reported.':d>0?'Neo4j retrieved more of the correct evidence than the PDF-only search.':d<0?'Neo4j retrieved less of the correct evidence than the PDF-only search.':'Both modes accepted the same evidence for this question.';document.getElementById('comparison').innerHTML=`<section class="panel compare-panel"><div class="head"><h2>PDF vs Neo4j comparison</h2><span class="state ${improved?'ok':'idle'}">${improved?'Measured graph gain':'No measured gain'}</span></div><p class="subtitle ${d<0?'loss':d>0?'gain':''}">${verdict}</p><div class="compare-grid"><div class="metric"><b>PDF accepted</b><div class="chunk-list">${chunkText(pdfAccepted)}</div></div><div class="metric"><b>Common evidence</b><div class="chunk-list">${chunkText(common)}</div></div></div></section>`}
</script></body></html>'''

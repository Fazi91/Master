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
    "LOCAL_ANSWER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
NLI_MODEL = os.getenv(
    "NLI_VERIFIER_MODEL", "cross-encoder/nli-deberta-v3-small"
)
USE_DENSE = os.getenv("USE_NEURAL_RETRIEVAL", "true").lower() in {
    "1", "true", "yes", "on"
}
TOP_FIRST_STAGE = int(os.getenv("TOP_FIRST_STAGE", "80"))
TOP_RERANK = int(os.getenv("TOP_RERANK", "24"))
MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE_CHUNKS", "6"))
MAX_DISPLAY_SOURCES = int(os.getenv("MAX_DISPLAY_SOURCES", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))
NLI_MIN = float(os.getenv("NLI_ENTAILMENT_MIN", "0.55"))
MIN_COVERAGE_SCORE = float(os.getenv("MIN_COVERAGE_SCORE", "-7.0"))
MIN_EXTRACT_SCORE = float(os.getenv("MIN_EXTRACT_SCORE", "1.0"))
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


def clean_question(value: str) -> str:
    """Remove presentation wrappers when a rendered result is pasted back."""
    value = value.strip()
    value = re.sub(
        r"^\s*(?:#{1,6}\s*)?(?:your\s+question|question)\s*[:\n]*\s*",
        "", value, flags=re.IGNORECASE,
    )
    value = re.split(
        r"\s*(?:\n|^)\s*(?:#{1,6}\s*)?"
        r"(?:answer|source location|related image evidence|"
        r"neo4j verification query)\s*[:\n]",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    # A plain-text copy may collapse headings and newlines into one line.
    value = re.split(
        r"\s+Answer\s+(?=(?:Relevant evidence|No complete answer|"
        r"According to|The |A |An ))",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return compact(value).strip()


def terms(value: str) -> list[str]:
    return list(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in STOPWORDS
    ))


def roots(value: str) -> set[str]:
    """Return lightweight morphology roots for generic action matching."""
    result = set()
    for word in re.findall(r"[a-z]+", value.lower()):
        result.add(word)
        for suffix in (
            "ization", "isation", "ations", "ation", "itions", "ition",
            "ctions", "ction", "ments", "ment", "ings", "ing", "ied",
            "ed", "es", "s",
        ):
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                base = word[:-len(suffix)]
                if suffix == "ied":
                    base += "y"
                result.add(base)
                break
    return result


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
        self.aliases: dict[str, set[str]] = {}
        for row in self.rows:
            for match in re.finditer(
                r"\b([A-Za-z][A-Za-z -]{4,70}?)\s*\(([A-Z]{2,8})\)",
                row["text"],
            ):
                long_form = compact(match.group(1)).lower()
                # PDF lines may prepend a heading; the final words nearest the
                # parentheses are the reliable long form.
                long_form = " ".join(long_form.split()[-8:])
                acronym = match.group(2).lower()
                if len(long_form.split()) >= 2:
                    self.aliases.setdefault(long_form, set()).add(acronym)
                    self.aliases.setdefault(acronym, set()).add(long_form)
        # Entity names are graph expansion keys, not passage text. Appending
        # them to a passage makes a reranker reward unrelated chunks that only
        # share a broad entity.
        corpus = [row["text"] for row in self.rows]
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

    def _expand_query(self, query: str) -> str:
        lowered = query.lower()
        additions = []
        for source, targets in self.aliases.items():
            if re.search(rf"(?<!\w){re.escape(source)}(?!\w)", lowered):
                additions.extend(sorted(targets))
        return f"{query} {' '.join(dict.fromkeys(additions))}" if additions else query

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
        """Create generic retrieval facets without an LLM or domain rules."""
        cleaned = question.strip().rstrip("?")
        pieces = [cleaned]
        # Coordinated clauses and enumerated dimensions are independent search
        # views. The complete question is retained in every view, so fragments
        # such as a bare verb or modifier never lose their subject or condition.
        for piece in re.split(r"\s*[,;]\s*|\s+\band\b\s+|\s+\bwhereas\b\s+|\s+\bwhile\b\s+", cleaned, flags=re.IGNORECASE):
            piece = compact(piece)
            piece = re.sub(
                r"^(?:and|or|whereas|while)\s+", "", piece,
                flags=re.IGNORECASE,
            )
            if piece and terms(piece):
                pieces.append(piece)
        unique_pieces = list(dict.fromkeys(pieces))[:8]
        complex_question = bool(
            re.search(r"[,;]", cleaned)
            or re.search(
                r"\b(?:compare|contrast|differ|difference|versus|vs\.?)\b",
                cleaned, re.IGNORECASE,
            )
            or re.search(
                r"\band\s+(?:what|why|how|when|where|which|who|"
                r"should|must|can|[a-z]+(?:ed|ing))\b",
                cleaned, re.IGNORECASE,
            )
        )
        if not complex_question:
            unique_pieces = [cleaned]
        lowered = question.lower()
        if re.search(r"\b(?:compare|contrast|differ|difference|versus|vs\.?)\b", lowered):
            answer_type = "comparison"
        elif re.search(r"\bwhy\b|\breason\b", lowered):
            answer_type = "reason"
        elif re.search(r"\bcalculat|\bcount|\bformula\b", lowered):
            answer_type = "calculation"
        elif re.search(r"\bhow\b|\bsteps?\b|\bprocedure\b|\bmethod\b", lowered):
            answer_type = "procedure"
        else:
            answer_type = "fact"
        facets = []
        if len(unique_pieces) == 1:
            facets = [Facet("", question, tuple(terms(question)))]
        else:
            instruction = (
                "Decompose the multi-part question into independently "
                "answerable retrieval facets. Preserve the shared subject in "
                "every facet and preserve every method, condition, comparison "
                "side, requested dimension, technical term, and number. Do "
                "not answer and do not add knowledge. Return strict JSON with "
                "facets containing label, query, and requirements."
            )
            try:
                parsed = parse_json(self.complete(
                    instruction,
                    f"Question: {question}\n"
                    "Schema: {\"facets\":[{\"label\":\"short label\","
                    "\"query\":\"self-contained query\","
                    "\"requirements\":[\"term\"]}]}",
                    500,
                ))
            except Exception as error:
                print(f"[PLAN] model fallback: {error}")
                parsed = None
            if parsed:
                planned = []
                for item in parsed.get("facets", [])[:8]:
                    if not isinstance(item, dict):
                        continue
                    query = compact(str(item.get("query") or ""))
                    if not query:
                        continue
                    if re.search(
                        r"\b(?:your question|relevant evidence|source location|"
                        r"answer mode)\b", query, re.IGNORECASE,
                    ):
                        planned = []
                        break
                    planned.append(Facet(
                        compact(str(item.get("label") or "")),
                        query,
                        tuple(compact(str(value)) for value in item.get(
                            "requirements", []
                        ) if compact(str(value))),
                    ))
                facets = planned
            if not facets:
                # Structural fallback remains generic and retains the full
                # question as context, but does not add it as a competing facet.
                facets = [Facet(
                    " ".join(piece.split()[:7]),
                    f"{piece}. Context: {question}",
                    tuple(terms(piece)),
                ) for piece in unique_pieces[1:]]
        print(f"[PLAN] type={answer_type}; facets={[facet.query for facet in facets]}")
        return Plan(question, answer_type, tuple(facets))

    def _retrievers(self) -> None:
        with self.lock:
            if self.reranker is None:
                self.reranker = CrossEncoder(RERANK_MODEL)
            if USE_DENSE and self.dense is None:
                self.dense = SentenceTransformer(DENSE_MODEL)
                self.dense_matrix = self.dense.encode(
                    [row["text"] for row in self.rows],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )

    def retrieve(self, facet: Facet) -> list[dict[str, Any]]:
        self._retrievers()
        retrieval_query = self._expand_query(facet.query)
        query_vector = self.vectorizer.transform([retrieval_query])
        lexical = (self.lexical_matrix @ query_vector.T).toarray().ravel()
        lexical_order = np.argsort(-lexical)
        selected = set(lexical_order[:TOP_FIRST_STAGE].tolist())
        dense_scores = np.zeros(len(self.rows), dtype=float)
        dense_order = np.array([], dtype=int)
        if self.dense is not None:
            query_dense = self.dense.encode(
                [retrieval_query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            dense_scores = self.dense_matrix @ query_dense
            dense_order = np.argsort(-dense_scores)
            selected.update(dense_order[:TOP_FIRST_STAGE].tolist())
        lexical_rank = {
            int(index): rank for rank, index in enumerate(lexical_order, 1)
        }
        dense_rank = {
            int(index): rank for rank, index in enumerate(dense_order, 1)
        }
        vocabulary = self.vectorizer.vocabulary_
        query_terms = [term for term in terms(retrieval_query) if term in vocabulary]
        rare_terms = sorted(
            query_terms,
            key=lambda term: self.vectorizer.idf_[vocabulary[term]],
            reverse=True,
        )[:3]
        coverage = {}
        for index in selected:
            passage_terms = set(terms(self.rows[index]["text"]))
            coverage[index] = (
                sum(term in passage_terms for term in rare_terms)
                / max(1, len(rare_terms))
            )
        # When an exact rare anchor exists in the corpus, candidates with no
        # such anchor cannot become the primary evidence merely through broad
        # semantic similarity.
        if rare_terms and any(value > 0 for value in coverage.values()):
            primary = rare_terms[0]
            primary_anchors = {
                index for index in selected
                if primary in set(terms(self.rows[index]["text"]))
            }
            anchored = (
                primary_anchors if len(primary_anchors) >= 3
                else {index for index in selected if coverage[index] > 0}
            )
            if len(anchored) >= 4:
                selected = anchored
        candidates = [self.rows[index] for index in selected]
        scores = self.reranker.predict(
            [[retrieval_query, row["text"]] for row in candidates],
            show_progress_bar=False,
        )
        ranked = []
        for row, rerank_score in zip(candidates, scores):
            index = self.positions[row["chunk_id"]]
            rrf = 1.0 / (60 + lexical_rank.get(index, 10000))
            if dense_order.size:
                rrf += 1.0 / (60 + dense_rank.get(index, 10000))
            final_score = (
                float(rerank_score)
                + 1.5 * coverage.get(index, 0.0)
                + 8.0 * rrf
                + float(lexical[index])
                + 0.25 * float(dense_scores[index])
            )
            ranked.append({
                **row,
                "score": final_score,
                "rerank_score": float(rerank_score),
                "anchor_coverage": coverage.get(index, 0.0),
            })
        ranked.sort(key=lambda row: row["score"], reverse=True)
        # One coherent section window is more useful than several unrelated
        # high-scoring pages. Multi-part questions already retrieve one anchor
        # independently for every facet.
        return self.expand(facet, ranked[:1])

    def expand(
        self, facet: Facet, anchors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add adjacent chunks generically so boundary text is not lost."""
        result: dict[str, dict[str, Any]] = {}
        for rank, anchor in enumerate(anchors):
            position = self.positions[anchor["chunk_id"]]
            for distance in (-3, -2, -1, 0, 1, 2, 3):
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
            query_tokens = set(terms(facet.query))
            for entity in anchor.get("entities", []):
                entity_tokens = set(terms(entity))
                if not entity_tokens or not entity_tokens.intersection(query_tokens):
                    continue
                for related_position in self.entity_positions.get(
                    entity.lower(), []
                )[:6]:
                    row = self.rows[related_position]
                    if row["chunk_id"] == anchor["chunk_id"]:
                        continue
                    score = anchor["score"] - 1.50 - rank * 0.01
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
        quota = max(1, MAX_EVIDENCE // max(1, len(plan.facets)))
        for facet_index, candidates in enumerate(groups):
            for row in candidates[:quota]:
                item = chosen.setdefault(row["chunk_id"], dict(row))
                item.setdefault("facets", []).append(facet_index)
                if len(chosen) >= MAX_EVIDENCE:
                    break
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

    @staticmethod
    def _units(text: str) -> list[str]:
        text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
        text = re.sub(r"\n\s*(?=\d+[.)]\s+)", "\n", text)
        parts = re.split(
            r"(?<=[.!?])\s+|\n{2,}|(?=\n\s*\d+[.)]\s+)", text
        )
        return [compact(part) for part in parts if len(compact(part)) >= 20]

    @staticmethod
    def _heading(lines: list[str], index: int) -> bool:
        line = compact(lines[index])
        if not line or len(line) > 90 or len(line.split()) > 12:
            return False
        numbered_section = bool(re.match(
            r"^\d+(?:\.\d+)+\s+[A-Za-z]", line
        ))
        if re.search(r"[.!?;]$", line) or (
            re.match(r"^(?:[-—•]|G\s|Fig\.?\s)", line, re.IGNORECASE)
            or (re.match(r"^\d", line) and not numbered_section)
        ):
            return False
        before_blank = index == 0 or not compact(lines[index - 1])
        # A final line without a following blank is usually a chunk-boundary
        # sentence fragment, not a heading.
        after_blank = index < len(lines) - 1 and not compact(lines[index + 1])
        return (
            numbered_section or (before_blank and after_blank)
        ) and bool(re.search(r"[A-Za-z]", line))

    def _best_section(
        self, facet: Facet, evidence: list[dict[str, Any]]
    ) -> tuple[float, list[tuple[int, int, str]]] | None:
        section_query = self._expand_query(facet.query)
        records = []
        headings = []
        for evidence_index, row in enumerate(evidence, 1):
            lines = row["text"].splitlines()
            for line_index, line in enumerate(lines):
                position = len(records)
                records.append((evidence_index, line_index, line))
                if self._heading(lines, line_index):
                    headings.append((position, compact(line)))
            records.append((evidence_index, len(lines), ""))
        if not headings:
            return None
        heading_passages = []
        heading_step_counts = []
        for heading_index, (position, heading) in enumerate(headings):
            next_position = (
                headings[heading_index + 1][0]
                if heading_index + 1 < len(headings) else len(records)
            )
            body_lines = [
                record[2] for record in records[position + 1:next_position]
            ]
            body = compact(" ".join(body_lines))
            heading_passages.append(f"{heading}. {body}"[:1400])
            heading_step_counts.append(sum(
                bool(re.match(r"^\s*\d+[.)]\s+", line))
                for line in body_lines
            ))
        semantic_scores = np.asarray(self.reranker.predict(
            [[section_query, passage] for passage in heading_passages],
            show_progress_bar=False,
        )).reshape(-1)
        query_roots = roots(section_query)
        lexical_scores = np.asarray([
            len(query_roots.intersection(roots(heading)))
            / max(1, len(roots(heading)))
            for _, heading in headings
        ])
        combined_scores = semantic_scores + 4.0 * lexical_scores
        if re.search(
            r"\b(?:how|steps?|procedure|method)\b", facet.query,
            re.IGNORECASE,
        ):
            structure_scores = np.asarray([
                min(3, count) for count in heading_step_counts
            ], dtype=float)
            combined_scores += structure_scores
        # If two nested headings score equally, the later, more specific one
        # is normally the useful subsection.
        combined_scores += np.arange(len(headings)) * 1e-6
        top_heading_indices = np.argsort(-combined_scores)[:5]
        print("[SECTIONS] candidates=" + str([
            (headings[int(index)][1], round(float(combined_scores[index]), 3))
            for index in top_heading_indices
        ]))
        best_at = int(np.argmax(combined_scores))
        best_score = float(combined_scores[best_at])
        if lexical_scores[best_at] == 0 and semantic_scores[best_at] < MIN_EXTRACT_SCORE:
            return None
        start = headings[best_at][0] + 1
        end = len(records)
        for position, _ in headings[best_at + 1:]:
            if position > start:
                end = position
                break
        paragraphs = []
        current = []
        current_source = None
        paragraph_index = 0

        def flush() -> None:
            nonlocal current, current_source, paragraph_index
            value = compact(" ".join(current))
            if value and len(value) >= 20 and current_source is not None:
                paragraphs.append((current_source, paragraph_index, value))
                paragraph_index += 1
            current = []
            current_source = None

        for evidence_index, _, raw_line in records[start:end]:
            line = compact(raw_line)
            if not line:
                flush()
                continue
            if current_source is not None and evidence_index != current_source:
                flush()
            if re.match(r"^\d+[.)]\s+", line) and current:
                flush()
            current_source = evidence_index
            current.append(line)
        flush()
        cleaned_paragraphs = []
        seen = set()
        for evidence_index, paragraph_index, paragraph in paragraphs:
            paragraph = re.sub(
                r"(?<=[A-Za-z])-\s+(?=[a-z])", "", paragraph
            )
            if re.match(r"^\d+\s+Manual\b", paragraph, re.IGNORECASE):
                continue
            if re.match(r"^[a-z]", paragraph) and len(paragraph) < 160:
                continue
            if re.search(
                r"\b(?:and|or|the|of|to|with|in|for)$", paragraph,
                re.IGNORECASE,
            ):
                continue
            key = re.sub(r"\W+", " ", paragraph.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            cleaned_paragraphs.append(
                (evidence_index, paragraph_index, paragraph)
            )
        paragraphs = cleaned_paragraphs
        if not paragraphs:
            return None
        return best_score, paragraphs

    def extractive_answer(
        self, plan: Plan, evidence: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Build a fast verbatim answer when every facet has strong text."""
        selected: dict[tuple[int, int], tuple[float, str]] = {}
        per_facet_limit = 5 if plan.answer_type == "procedure" else 3
        for facet in plan.facets:
            # A whole subsection is an answer only for a genuinely sequential
            # procedure.  For facts, reasons, lists and comparisons a section
            # is merely a search boundary: returning it wholesale is how a
            # nearby but irrelevant paragraph can masquerade as an answer.
            procedural_facet = (
                plan.answer_type == "procedure"
                and bool(re.search(
                    r"\b(?:how|steps?|procedure|method|prepare|collect|"
                    r"perform|carry out)\b",
                    facet.query, re.IGNORECASE,
                ))
                and not bool(re.search(
                    r"\b(?:why|compare|difference|differ|purpose|reason)\b",
                    facet.query, re.IGNORECASE,
                ))
            )
            if procedural_facet:
                section = self._best_section(facet, evidence)
                if section is not None:
                    section_score, paragraphs = section
                    section_chars = sum(len(item[2]) for item in paragraphs)
                    if section_chars <= 6000:
                        print(
                            f"[SECTION] facet={facet.label!r}; "
                            f"score={section_score:.3f}; "
                            f"paragraphs={len(paragraphs)}"
                        )
                        for evidence_index, paragraph_index, paragraph in paragraphs:
                            selected[(evidence_index, paragraph_index)] = (
                                section_score, paragraph
                            )
                        continue
            vocabulary = self.vectorizer.vocabulary_
            expanded_query = self._expand_query(facet.query)
            # The planner may append the complete question as context so a
            # short facet retains its subject during retrieval.  That context
            # must not become the sentence-selection objective, otherwise
            # every facet tends to reproduce the same broad answer.
            focus_query = re.split(
                r"\s+Context:\s*", facet.query, maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            expanded_focus = self._expand_query(focus_query)
            query_terms = [
                term for term in terms(expanded_query) if term in vocabulary
            ]
            rare_terms = sorted(
                query_terms,
                key=lambda term: self.vectorizer.idf_[vocabulary[term]],
                reverse=True,
            )[:6]

            # Rank rows before sentences.  This preserves subject context for
            # short or referential sentences that are meaningful only inside
            # the correctly retrieved subject section.
            row_scores = np.asarray(self.reranker.predict(
                [[expanded_query, row["text"]] for row in evidence],
                show_progress_bar=False,
            )).reshape(-1)
            row_coverages = []
            for row in evidence:
                row_terms = set(terms(row["text"]))
                row_coverages.append(sum(term in row_terms for term in rare_terms))
            best_row_score = float(np.max(row_scores)) if len(row_scores) else -999.0
            has_anchored_row = any(row_coverages)
            units = []
            for evidence_index, (row, row_score, anchor_coverage) in enumerate(
                zip(evidence, row_scores, row_coverages), 1
            ):
                # Broad semantic similarity alone must not pull another topic
                # into the answer when subject anchors exist in this window.
                if has_anchored_row and not anchor_coverage:
                    continue
                if float(row_score) < best_row_score - 3.0:
                    continue
                row_units = self._units(row["text"])
                for unit_index, unit in enumerate(row_units):
                    units.append((
                        evidence_index, unit_index, unit, float(row_score)
                    ))
            if not units:
                return None
            sentence_scores = np.asarray(self.reranker.predict(
                [[expanded_focus, unit] for _, _, unit, _ in units],
                show_progress_bar=False,
            )).reshape(-1)
            # Sentence relevance dominates; row relevance supplies the subject
            # context for pronouns, numbered steps and short list items.
            scores = sentence_scores + 0.15 * np.asarray([
                row_score for _, _, _, row_score in units
            ])
            order = np.argsort(-scores)
            if not len(order) or float(scores[order[0]]) < MIN_EXTRACT_SCORE:
                print(
                    f"[EXTRACT] facet={facet.label!r}; "
                    f"best={float(scores[order[0]]) if len(order) else None}; fallback=model"
                )
                return None
            kept = 0
            for unit_position in order:
                evidence_index, unit_index, unit, _ = units[int(unit_position)]
                # Do not pad an answer with low-relevance sentences merely to
                # reach the configured limit.
                if float(scores[unit_position]) < max(
                    MIN_EXTRACT_SCORE, float(scores[order[0]]) - 2.5
                ):
                    continue
                key = (evidence_index, unit_index)
                selected[key] = max(
                    selected.get(key, (-float("inf"), unit)),
                    (float(scores[unit_position]), unit),
                )
                kept += 1
                if kept >= per_facet_limit:
                    break
        ordered = sorted(selected.items(), key=lambda item: item[0])
        claims = []
        answer_parts = []
        for (evidence_index, _), (_, unit) in ordered:
            answer_parts.append(f"{unit} [E{evidence_index}]")
            claims.append({
                "sentence": unit,
                "evidence_ids": [f"E{evidence_index}"],
            })
        answer = "\n".join(answer_parts)
        coverage = np.asarray(self.reranker.predict(
            [[facet.query, answer] for facet in plan.facets],
            show_progress_bar=False,
        )).reshape(-1)
        print(f"[EXTRACT] coverage={coverage.tolist()}")
        if any(float(score) < MIN_COVERAGE_SCORE for score in coverage):
            return None
        return {"answer": answer, "claims": claims, "extractive": True}

    def extract_facets(
        self, plan: Plan, groups: list[list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Extract and verify each facet against its own evidence window."""
        global_rows: list[dict[str, Any]] = []
        global_index: dict[str, int] = {}
        answer_parts = []
        global_claims = []
        for facet, group in zip(plan.facets, groups):
            local_rows = sorted(group[:7], key=lambda row: (
                row.get("pdf_page") or 0, row.get("chunk_index") or 0
            ))
            local_plan = Plan(plan.question, plan.answer_type, (facet,))
            result = self.extractive_answer(local_plan, local_rows)
            if result is None:
                return None
            local_map = {}
            cited_local = list(dict.fromkeys(
                int(value) for value in re.findall(
                    r"\[E(\d+)\]", result["answer"]
                )
            ))
            for local_number in cited_local:
                row = local_rows[local_number - 1]
                chunk_id = row["chunk_id"]
                if chunk_id not in global_index:
                    global_rows.append(row)
                    global_index[chunk_id] = len(global_rows)
                local_map[local_number] = global_index[chunk_id]
            remap = lambda value: local_map[int(value)]
            answer = re.sub(
                r"\[E(\d+)\]",
                lambda match: f"[E{remap(match.group(1))}]",
                result["answer"],
            )
            if len(plan.facets) > 1 and facet.label:
                answer = f"{facet.label}:\n{answer}"
            answer_parts.append(answer)
            for claim in result["claims"]:
                ids = []
                for raw in claim.get("evidence_ids", []):
                    match = re.fullmatch(r"E?(\d+)", str(raw))
                    if match and int(match.group(1)) in local_map:
                        ids.append(f"E{local_map[int(match.group(1))]}")
                if ids:
                    global_claims.append({
                        "sentence": claim["sentence"],
                        "evidence_ids": ids,
                    })
        return ({
            "answer": "\n\n".join(answer_parts),
            "claims": global_claims,
            "extractive": True,
        }, global_rows)

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
            "not support a requested detail, omit that detail rather than "
            "inventing it. Return strict JSON with answer and claims. Cite "
            "every answer sentence as [E#]. "
            "Each claim contains the exact sentence and its evidence_ids."
        )
        schema = {
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
        exact_extract = bool(generated.get("extractive"))
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
            normalized_premise = compact(re.sub(
                r"(?<=[A-Za-z])-\s+(?=[a-z])", "", premise
            ))
            normalized_sentence = compact(re.sub(
                r"(?<=[A-Za-z])-\s+(?=[a-z])", "", sentence_without_citation
            ))
            if exact_extract and normalized_sentence not in normalized_premise:
                return False, "extract is not verbatim evidence", []
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
        if not exact_extract:
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
        coverage_scores = np.asarray(self.reranker.predict(
            [[facet.query, generated["answer"]] for facet in plan.facets],
            show_progress_bar=False,
        )).reshape(-1)
        print(f"[COVERAGE] scores={coverage_scores.tolist()}")
        if any(float(score) < MIN_COVERAGE_SCORE for score in coverage_scores):
            return False, "one or more requested facets are absent", []
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
                    GENERATOR_MODEL
                    if kind == "domain_answer" and mode == "model_planned_verified"
                    else None
                ),
            },
        }

    def answer(self, question: str) -> dict[str, Any]:
        plan = self.plan(question)
        groups = [self.retrieve(facet) for facet in plan.facets]
        # Facets from one question normally share a subject. Share candidate
        # windows when their query vocabulary overlaps, while retaining each
        # facet's own ranking and section selection. This prevents a generic
        # operation facet from losing the subject found by a sibling facet.
        if len(groups) > 1:
            original_groups = [list(group) for group in groups]
            for target_index, target_facet in enumerate(plan.facets):
                merged = {row["chunk_id"]: dict(row) for row in groups[target_index]}
                target_terms = set(terms(target_facet.query))
                for source_index, source_facet in enumerate(plan.facets):
                    if source_index == target_index:
                        continue
                    if len(target_terms.intersection(terms(source_facet.query))) < 2:
                        continue
                    for row in original_groups[source_index][:7]:
                        shared = {**row, "score": float(row["score"]) - 0.25}
                        old = merged.get(row["chunk_id"])
                        if old is None or shared["score"] > old["score"]:
                            merged[row["chunk_id"]] = shared
                groups[target_index] = sorted(
                    merged.values(), key=lambda row: row["score"], reverse=True
                )
        for facet, group in zip(plan.facets, groups):
            print(
                f"[RETRIEVAL] facet={facet.label!r}; top="
                f"{[(row['chunk_id'], round(row['score'], 3)) for row in group[:8]]}"
            )
        if any(not group for group in groups):
            return self.response(
                "not_found", question,
                "No complete answer could be verified from the corpus.",
                [], [], len(self.rows), "retrieval_incomplete",
            )
        extracted = self.extract_facets(plan, groups)
        if extracted is not None:
            generated, evidence = extracted
        else:
            evidence = self.evidence(plan, groups)
            print(
                "[EVIDENCE] selected="
                f"{[(row['chunk_id'], round(row['score'], 3)) for row in evidence]}"
            )
            generated = self.compose(plan, evidence)
        if generated is None:
            print("[COMPOSE] rejected: invalid or empty structured answer")
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
        mode = (
            "verified_extractive"
            if generated.get("extractive") else "model_planned_verified"
        )
        return self.response(
            "domain_answer", question, answer, source_rows,
            self.images(chunk_ids), len(self.rows),
            mode,
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
    question = clean_question(request.query)
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

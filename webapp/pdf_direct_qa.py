from __future__ import annotations

import csv
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "graph_v2" / "chunks.csv"
RERANK_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
TOP_LEXICAL = 90
TOP_RERANK = 24
TOP_CHUNKS_PER_NEED = 5
MAX_UNITS_PER_NEED = 8

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
ANAPHORA_RE = re.compile(r"\b(?:it|its|they|them|their|this|that)\b", re.I)
CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and|but|also)\s+(?=(?:why|how|when|where|what|which|who|"
    r"should|must|can|is|are|was|were|do|does|did)\b)",
    re.I,
)
QUESTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "may", "must",
    "of", "on", "or", "should", "that", "the", "their", "them", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "after", "before", "during", "through",
}
ACTION_SUFFIXES = ("ed", "ing", "ize", "ise", "ate", "fy")
GENERIC_SUBJECT_ROOTS = {
    "specimen", "container", "method", "procedure", "use",
    "shown", "figure", "purpose", "difference",
    # Operations describe what to retrieve about the subject; they are not
    # subject identity constraints.
    "prepare", "collect", "label", "dispatch", "examine", "identify",
    "fix", "stain", "clean", "sterilize", "calculate", "convert",
    "differ", "preserve", "reject",
    "not", "rather", "than", "between", "per", "number", "maximum",
}
CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|so that|in order to|to permit|to prevent|"
    r"reason|not suitable|not useful|unsuitable|due to|otherwise)\b",
    re.I,
)
PROCEDURE_RE = re.compile(
    r"^\s*(?:\d+[.)]|[a-z][.)]|[-—•]|(?:important|warning|note)\s*:)",
    re.I,
)
HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")


def compact(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_question(text: str) -> str:
    value = compact(text)
    value = re.sub(r"([?!])\s*[a-z]\s*$", r"\1", value)
    return value


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(text)]


def stem(word: str) -> str:
    value = word.casefold()
    if value.endswith("ies") and len(value) > 5:
        value = value[:-3] + "y"
    elif value.endswith(("sses", "xes", "zes", "ches", "shes")):
        value = value[:-2]
    elif value.endswith("s") and not value.endswith("ss") and len(value) > 4:
        value = value[:-1]
    for suffix in ("ization", "isation", "ation", "ments", "ment", "ingly", "edly", "ing", "ied", "ed"):
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            return value[: -len(suffix)]
    return value


def roots(text: str) -> set[str]:
    return {stem(word) for word in words(text) if word not in QUESTION_WORDS}


def subject_match(required: set[str], available: set[str]) -> bool:
    """Require the topic, without demanding every descriptive query word."""
    if not required:
        return True
    minimum = max(1, math.ceil(len(required) * 0.6))
    return len(required & available) >= minimum


def normalize_for_exact_check(text: str) -> str:
    text = re.sub(r"(?m)^\s*G\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", compact(text).casefold()).strip()


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pdf_page: int
    printed_page: str
    page_index: int
    text: str


@dataclass(frozen=True)
class Need:
    need_id: str
    original: str
    query: str
    subject_terms: frozenset[str]
    answer_type: str


@dataclass(frozen=True)
class Unit:
    chunk_index: int
    order: int
    text: str
    score: float


class AskRequest(BaseModel):
    query: str | None = None
    question: str | None = None


class DirectPdfQA:
    def __init__(self) -> None:
        self.chunks = self._load_chunks()
        corpus = [chunk.text for chunk in self.chunks]
        self.word_index = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.char_index = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        self.word_matrix = self.word_index.fit_transform(corpus)
        self.char_matrix = self.char_index.fit_transform(corpus)
        self._reranker: CrossEncoder | None = None
        self._model_lock = threading.Lock()

    @staticmethod
    def _load_chunks() -> list[Chunk]:
        rows: list[Chunk] = []
        with CHUNKS_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                text = row.get("chunk_text") or ""
                if not text.strip():
                    continue
                rows.append(Chunk(
                    chunk_id=row["chunk_id"],
                    pdf_page=int(row.get("pdf_page") or 0),
                    printed_page=row.get("printed_page") or "",
                    page_index=index,
                    text=text,
                ))
        if not rows:
            raise RuntimeError(f"No PDF chunks found in {CHUNKS_FILE}")
        return rows

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            with self._model_lock:
                if self._reranker is None:
                    self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    @staticmethod
    def answer_type(question: str) -> str:
        lowered = question.casefold()
        if re.search(r"\bcalculat\w*\b|\bformula\b|\bcomput\w*\b", lowered):
            return "calculation"
        if re.search(r"\bcompare\b|\bdifference\b|\bversus\b|\bvs\.?\b", lowered):
            return "comparison"
        if re.search(r"\bwhy\b|\breason\b", lowered):
            return "reason"
        if re.search(r"\bhow\b|\bsteps?\b|\bprocedure\b|\bmethod\b", lowered):
            return "procedure"
        return "fact"

    @staticmethod
    def subject_from_clause(clause: str) -> set[str]:
        content = [word for word in words(clause) if word not in QUESTION_WORDS]
        candidates = {
            stem(word) for word in content
            if not word.endswith(ACTION_SUFFIXES)
        }
        if not candidates:
            candidates = {stem(word) for word in content}
        return candidates

    def plan(self, question: str) -> list[Need]:
        cleaned = clean_question(question)
        clauses = [compact(part) for part in CLAUSE_SPLIT_RE.split(cleaned) if compact(part)]
        if not clauses:
            clauses = [cleaned]
        shared_subject = self.subject_from_clause(clauses[0])
        needs: list[Need] = []
        for index, clause in enumerate(clauses):
            own_subject = self.subject_from_clause(clause)
            if ANAPHORA_RE.search(clause) and shared_subject:
                subject = set(shared_subject)
                query = compact(f"{clause} {' '.join(sorted(shared_subject))}")
            else:
                subject = own_subject
                query = clause
                if index == 0 and own_subject:
                    shared_subject = set(own_subject)
            needs.append(Need(
                need_id=f"need-{index}",
                original=clause,
                query=query,
                subject_terms=frozenset(subject),
                answer_type=self.answer_type(clause),
            ))
        return needs

    def retrieve(self, need: Need) -> list[tuple[int, float]]:
        word_query = self.word_index.transform([need.query])
        char_query = self.char_index.transform([need.query])
        word_scores = (self.word_matrix @ word_query.T).toarray().ravel()
        char_scores = (self.char_matrix @ char_query.T).toarray().ravel()
        cheap_scores = 0.72 * word_scores + 0.28 * char_scores
        generic_roots = {stem(term) for term in GENERIC_SUBJECT_ROOTS}
        required_subject = set(need.subject_terms) - generic_roots
        chunk_roots = [roots(chunk.text) for chunk in self.chunks]
        figure_required = bool(re.search(r"\bfig(?:ure)?s?\b", need.query, re.I))
        minimum_figures = 2 if re.search(r"\bfigures\b", need.query, re.I) else 1
        requested_figures = set(re.findall(
            r"\bFig(?:ure)?s?\.?\s*(\d+\.\d+)", need.query, re.I
        ))
        eligible = np.asarray([
            index for index, item_roots in enumerate(chunk_roots)
            if subject_match(required_subject, item_roots)
            if not figure_required or len(re.findall(
                r"\bFig(?:ure)?\.?\s*\d", self.chunks[index].text, re.I
            )) >= minimum_figures
            if not requested_figures or requested_figures.issubset(set(re.findall(
                r"\bFig(?:ure)?\.?\s*(\d+\.\d+)",
                self.chunks[index].text,
                re.I,
            )))
        ], dtype=int)
        if not eligible.size and requested_figures:
            return []
        if not eligible.size and required_subject:
            eligible = np.asarray([
                index for index, item_roots in enumerate(chunk_roots)
                if required_subject & item_roots
            ], dtype=int)
        if not eligible.size:
            return []
        lexical_top = eligible[
            np.argsort(-cheap_scores[eligible])[:TOP_LEXICAL]
        ]
        pairs = [[need.query, self.chunks[int(index)].text] for index in lexical_top]
        semantic = np.asarray(
            self.reranker.predict(pairs, show_progress_bar=False)
        ).reshape(-1)
        semantic_order = np.argsort(-semantic)[:TOP_RERANK]
        selected: dict[int, float] = {}
        for position in semantic_order:
            index = int(lexical_top[int(position)])
            selected[index] = float(semantic[int(position)]) + float(cheap_scores[index])
        expanded = dict(selected)
        neighbor_distance = 3 if need.answer_type in {"procedure", "calculation"} else 1
        for index, score in list(selected.items())[:TOP_CHUNKS_PER_NEED]:
            for neighbor in range(index - neighbor_distance, index + neighbor_distance + 1):
                if 0 <= neighbor < len(self.chunks):
                    expanded.setdefault(neighbor, score - 0.35 * abs(neighbor - index))
        return sorted(expanded.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def units(text: str) -> list[str]:
        cleaned = text.replace("\r", "")
        cleaned = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", cleaned)
        cleaned = re.sub(r"^\s*G\s+", "— ", cleaned, flags=re.M)
        cleaned = re.sub(r"\n(?=\s*\d+[.)]\s+[A-Z])", "\n\n", cleaned)
        cleaned = re.sub(
            r"(?<=[.!?])\s+(?=\d+[.)]\s+[A-Z])", "\n\n", cleaned
        )
        blocks = re.split(r"\n\s*\n+", cleaned)
        result: list[str] = []
        for block in blocks:
            block = compact(block)
            if len(block) < 20:
                continue
            if re.match(r"^\d+\s+(?:Manual|Index)\b", block, re.I):
                continue
            if re.match(r"^Fig\.?\s*\d", block, re.I):
                continue
            if HEADING_RE.match(block) and len(block.split()) <= 12:
                continue
            protected = re.sub(r"\bFig\.", "Fig§", block, flags=re.I)
            sentences = [
                part.replace("Fig§", "Fig.")
                for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9—])", protected)
            ]
            if PROCEDURE_RE.match(block) or block.rstrip().endswith(":"):
                result.append(block)
            else:
                result.extend(part for part in map(compact, sentences) if len(part) >= 20)
        return result

    def extract(self, need: Need, ranked: list[tuple[int, float]]) -> list[Unit]:
        chunk_rank_scores = dict(ranked)
        candidates: list[tuple[int, int, str]] = []
        candidate_limit = 60 if need.answer_type in {"procedure", "calculation"} else TOP_RERANK
        for chunk_index, _chunk_score in ranked[:candidate_limit]:
            if self.chunks[chunk_index].text.rstrip().endswith("Table Table"):
                continue
            for order, text in enumerate(self.units(self.chunks[chunk_index].text)):
                candidates.append((chunk_index, order, text))
        if not candidates:
            return []
        pairs = [[need.query, text] for _, _, text in candidates]
        scores = np.asarray(
            self.reranker.predict(pairs, show_progress_bar=False)
        ).reshape(-1)
        ranked_units = sorted(
            (
                Unit(
                    chunk_index,
                    order,
                    text,
                    float(score) + 0.15 * chunk_rank_scores.get(chunk_index, 0.0),
                )
                for (chunk_index, order, text), score in zip(candidates, scores)
            ),
            key=lambda unit: unit.score,
            reverse=True,
        )
        subject_roots = set(need.subject_terms)
        generic_roots = {stem(term) for term in GENERIC_SUBJECT_ROOTS}
        required_subject = subject_roots - generic_roots
        requested_figures = set(re.findall(
            r"\bFig(?:ure)?s?\.?\s*(\d+\.\d+)", need.query, re.I
        ))
        # Procedure steps often omit the subject after the section establishes it
        # (for example, "Ask the patient..." under sputum collection).  Keep
        # subject validation at chunk level for procedures, but at unit level
        # for facts/reasons so unrelated passages still cannot leak through.
        filtered = [
            unit for unit in ranked_units
            if (
                (
                    not requested_figures
                    or bool(requested_figures & set(re.findall(
                        r"\bFig(?:ure)?\.?\s*(\d+\.\d+)",
                        self.chunks[unit.chunk_index].text,
                        re.I,
                    )))
                )
                and
                (
                    subject_match(
                        required_subject,
                        roots(self.chunks[unit.chunk_index].text),
                    )
                    if need.answer_type == "procedure"
                    else bool(required_subject & roots(unit.text))
                )
                if required_subject else
                (
                    not subject_roots
                    or subject_roots & roots(
                        self.chunks[unit.chunk_index].text
                        if need.answer_type == "procedure" else unit.text
                    )
                )
            )
        ]
        if not filtered:
            return []
        if need.answer_type == "reason":
            causal = [unit for unit in filtered if CAUSAL_RE.search(unit.text)]
            if causal:
                return [causal[0]]
        if need.answer_type == "procedure":
            numbered: list[tuple[int, Unit]] = []
            # Start inside a subject-qualified unit, then allow adjacent
            # continuation chunks whose later steps naturally omit the subject.
            for unit in ranked_units:
                match = re.match(r"^\s*(\d+)[.)]\s+", unit.text)
                if match:
                    numbered.append((int(match.group(1)), unit))
            starts = [
                unit for number, unit in numbered
                if number == 1 and unit in filtered
            ]
            if starts:
                operation_roots = roots(need.query) - subject_roots

                def section_operation_overlap(unit: Unit) -> int:
                    source = self.chunks[unit.chunk_index].text
                    heading_terms: set[str] = set()
                    for raw_line in source.splitlines():
                        line = compact(raw_line)
                        tokens = words(line)
                        if not 1 < len(tokens) <= 12 or line.endswith((".", ";")):
                            continue
                        if HEADING_RE.match(line) or not re.search(r"[,!?]", line):
                            heading_terms.update(roots(line))
                    return len(operation_roots & heading_terms)

                start = max(
                    starts,
                    key=lambda unit: (section_operation_overlap(unit), unit.score),
                )
                sequence = [start]
                expected = 2
                last_unit = start
                last_page = self.chunks[start.chunk_index].pdf_page
                while expected <= 20:
                    options = [
                        unit for number, unit in numbered
                        if number == expected
                        and last_page <= self.chunks[unit.chunk_index].pdf_page
                        <= last_page + 1
                        and (
                            (
                                unit.chunk_index == last_unit.chunk_index
                                and unit.order > last_unit.order
                            )
                            or (
                                unit.chunk_index > last_unit.chunk_index
                                and not any(
                                    other.chunk_index == unit.chunk_index
                                    and other.order < unit.order
                                    and re.match(r"^\s*\d+[.)]\s+", other.text)
                                    for other in ranked_units
                                )
                            )
                        )
                    ]
                    if not options:
                        break
                    selected_step = max(
                        options,
                        key=lambda unit: (
                            not bool(re.search(r"(?:\bFig\.?|\(|\[)\s*$", unit.text)),
                            len(unit.text),
                            unit.score,
                        ),
                    )
                    sequence.append(selected_step)
                    last_unit = selected_step
                    last_page = self.chunks[selected_step.chunk_index].pdf_page
                    expected += 1
                if len(sequence) >= 2:
                    lead_candidates = [
                        unit for unit in filtered
                        if unit not in sequence
                        and unit.chunk_index <= start.chunk_index
                        and self.chunks[unit.chunk_index].pdf_page
                            == self.chunks[start.chunk_index].pdf_page
                        and subject_roots & roots(unit.text)
                        and re.search(r"\b(?:should|must|first|before|after)\b", unit.text, re.I)
                        and len(unit.text) <= 300
                    ]
                    if lead_candidates:
                        lead = max(lead_candidates, key=lambda unit: unit.score)
                        return ([lead] + sequence)[:MAX_UNITS_PER_NEED]
                    return sequence[:MAX_UNITS_PER_NEED]
        if "component" in roots(need.query):
            component_roots = {"body", "head", "joint", "washer"}
            chunk_candidates = {unit.chunk_index for unit in filtered}
            if chunk_candidates:
                chosen_chunk = max(
                    chunk_candidates,
                    key=lambda index: (
                        len(component_roots & roots(self.chunks[index].text)),
                        chunk_rank_scores.get(index, 0.0),
                        max(
                            unit.score for unit in filtered
                            if unit.chunk_index == index
                        ),
                    ),
                )
                component_units = sorted(
                    (
                        unit for unit in ranked_units
                        if unit.chunk_index == chosen_chunk
                        and (
                            component_roots & roots(unit.text)
                            or "made up" in unit.text.casefold()
                        )
                        and len(unit.text) <= 320
                    ),
                    key=lambda unit: unit.order,
                )
                if component_units:
                    return component_units[:MAX_UNITS_PER_NEED]
        if requested_figures:
            exact_figure_units = [
                unit for unit in filtered
                if requested_figures & set(re.findall(
                    r"\bFig(?:ure)?\.?\s*(\d+\.\d+)", unit.text, re.I
                ))
            ]
            if exact_figure_units:
                filtered = exact_figure_units + [
                    unit for unit in filtered if unit not in exact_figure_units
                ]
        best = filtered[0]
        chosen = [best]
        if need.answer_type in {"procedure", "calculation"}:
            if need.answer_type == "procedure":
                structural = [
                    unit for unit in filtered
                    if PROCEDURE_RE.match(unit.text) or unit.text.rstrip().endswith(":")
                ]
                if structural:
                    best = structural[0]
            same_chunk = sorted(
                (
                    unit for unit in ranked_units
                    if unit.chunk_index == best.chunk_index
                    and abs(unit.order - best.order) <= 4
                    and (
                        need.answer_type == "calculation"
                        or PROCEDURE_RE.match(unit.text)
                        or unit.text.rstrip().endswith(":")
                    )
                ),
                key=lambda unit: unit.order,
            )
            chosen = same_chunk or chosen
        else:
            for unit in filtered[1:]:
                limit = 3 if need.answer_type == "comparison" else 4
                if len(chosen) >= limit:
                    break
                if (
                    self.chunks[unit.chunk_index].pdf_page
                    == self.chunks[best.chunk_index].pdf_page
                    and abs(unit.chunk_index - best.chunk_index) <= 1
                    and (
                        not requested_figures
                        or bool(requested_figures & set(re.findall(
                            r"\bFig(?:ure)?\.?\s*(\d+\.\d+)",
                            unit.text,
                            re.I,
                        )))
                    )
                    and unit.text not in {x.text for x in chosen}
                ):
                    chosen.append(unit)
        return chosen[:MAX_UNITS_PER_NEED]

    def verify_unit(self, unit: Unit) -> bool:
        source = normalize_for_exact_check(self.chunks[unit.chunk_index].text)
        claim = normalize_for_exact_check(unit.text)
        return bool(claim) and claim in source

    @staticmethod
    def need_complete(need: Need, units: list[Unit]) -> bool:
        if not units:
            return False
        if need.answer_type != "procedure":
            return True
        numbers = [
            int(match.group(1))
            for unit in units
            if (match := re.match(r"^\s*(\d+)[.)]\s+", unit.text))
        ]
        if numbers:
            return numbers[0] == 1 and numbers == list(range(1, len(numbers) + 1))
        has_lead_in = any(unit.text.rstrip().endswith(":") for unit in units)
        has_instruction = any(PROCEDURE_RE.match(unit.text) for unit in units)
        return has_lead_in and has_instruction

    def answer(self, question: str) -> dict[str, Any]:
        cleaned = clean_question(question)
        needs = self.plan(cleaned)
        need_results: list[dict[str, Any]] = []
        source_indices: list[int] = []
        complete = True
        for need in needs:
            ranked = self.retrieve(need)
            units = [unit for unit in self.extract(need, ranked) if self.verify_unit(unit)]
            need_is_complete = self.need_complete(need, units)
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
                "retrieved_chunks": [
                    {
                        "chunk_id": self.chunks[index].chunk_id,
                        "pdf_page": self.chunks[index].pdf_page,
                        "score": round(score, 4),
                    }
                    for index, score in ranked[:10]
                ],
                "evidence": [
                    {
                        "text": unit.text,
                        "chunk_id": self.chunks[unit.chunk_index].chunk_id,
                        "pdf_page": self.chunks[unit.chunk_index].pdf_page,
                        "score": round(unit.score, 4),
                        "exact_source_match": True,
                    }
                    for unit in units
                ],
            })
        citation_number = {index: number for number, index in enumerate(source_indices, 1)}
        answer_parts: list[str] = []
        for result in need_results:
            lines = []
            for evidence in result["evidence"]:
                index = next(
                    i for i in source_indices
                    if self.chunks[i].chunk_id == evidence["chunk_id"]
                )
                lines.append(f"{evidence['text']} [S{citation_number[index]}]")
            if len(needs) > 1:
                answer_parts.append(f"{result['question_part']}:\n" + "\n".join(lines))
            else:
                answer_parts.extend(lines)
        sources = [
            {
                "chunk_id": self.chunks[index].chunk_id,
                "pdf_page": self.chunks[index].pdf_page,
                "printed_page": self.chunks[index].printed_page,
                "text": self.chunks[index].text,
            }
            for index in source_indices
        ]
        return {
            "kind": "domain_answer" if complete else "not_found",
            "question": cleaned,
            "answer": "\n\n".join(answer_parts) if complete else "No complete extractive answer was verified.",
            "needs": need_results,
            "sources": sources,
            "verification": {
                "complete": complete,
                "all_claims_are_exact_source_spans": complete and all(
                    item["exact_source_match"]
                    for result in need_results for item in result["evidence"]
                ),
                "needs_covered": sum(bool(result["complete"]) for result in need_results),
                "needs_total": len(needs),
            },
        }


_engine: DirectPdfQA | None = None
_engine_lock = threading.Lock()


def engine() -> DirectPdfQA:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DirectPdfQA()
    return _engine


app = FastAPI(title="Direct PDF Extractive QA Prototype")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "direct_pdf_extractive",
        "initialized": _engine is not None,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    question = request.query or request.question or ""
    if not clean_question(question):
        return {
            "kind": "invalid_request",
            "answer": "A non-empty question is required.",
            "sources": [],
        }
    return engine().answer(question)

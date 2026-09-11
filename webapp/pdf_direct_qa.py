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
CONTRASTIVE_TERM_PAIRS = ({"thick", "thin"},)
SELF_LABELED_STEP_RE = re.compile(r"^\s*\d+[.)]\s*([A-Za-z]+)\s+(?:film|smear)\.\s")
MID_COLON_BULLET_RE = re.compile(r":\s*[-—•]")
FIELD_LABEL_RE = re.compile(r"^([A-Z][A-Za-z](?:[A-Za-z \-]{0,20}[A-Za-z])?):\s")
ENTITY_FIG_HEADING_RE = re.compile(
    r"^[A-Z][\w.\-' ]{2,60}\(Figs?\.?\s*\d+\.\d+(?:,\s*\d+\.\d+)*\)$"
)
# A running page header ("3. General laboratory procedures 77", "4.
# Parasitology 121") repeats the chapter number on every page of that
# chapter -- "N. " followed by a short, unpunctuated title and a trailing
# bare page number -- and is easily mistaken for numbered step "N." by any
# check that only looks at the leading digit. A real step is a sentence: it
# carries other punctuation, or ends in one.
RUNNING_HEADER_RE = re.compile(
    r"^\d{1,2}[.)]\s+[A-Za-z][A-Za-z ,\-]{0,60}?\s+\d{1,4}\s*$"
)


def promises_a_list(text: str) -> bool:
    # A lead-in sentence can end up merged with its first bullet (no
    # sentence-ending punctuation separates them), so the colon is no
    # longer the last character -- check for it followed by a bullet too.
    return bool(text.rstrip().endswith(":") or MID_COLON_BULLET_RE.search(text))


def field_label_roots(text: str) -> set[str]:
    # A structured identification block ("Size: 8-12mm.", "Shape: oval...")
    # names its own topic only in the label, not in the sentence body, so a
    # sentence-level reranker score alone can badly under-rate it against a
    # question that names that exact field.
    match = FIELD_LABEL_RE.match(text)
    return roots(match.group(1)) if match else set()
CAUSAL_RE = re.compile(
    r"\b(?:because|therefore|so that|in order to|to permit|to prevent|"
    r"reason|not suitable|not useful|unsuitable|due to|otherwise)\b",
    re.I,
)
PROCEDURE_RE = re.compile(
    # A section/subsection number ("3.4.3", "3.5.1") also starts with
    # "digit, dot" -- (?!\d) keeps it from being mistaken for a numbered
    # step "3." by requiring nothing but the step's own trailing space
    # right after the marker.
    r"^\s*(?:\d+[.)](?!\d)|[a-z][.)]|[-—•]|(?:important|warning|note)\s*:)",
    re.I,
)
HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")
LABEL_ONLY_RE = re.compile(r"\blabel(?:led|ling|s)?\b", re.I)
FIGURE_DEPICTION_RE = re.compile(
    r"\b(?:shown|illustrated|depicted|pictured|labell?ed)\b.{0,40}\b(?:figure|diagram|fig\.?)\b",
    re.I,
)
MULTISTEP_ACTION_RE = re.compile(
    r"\b(?:collect|prepare|dispatch|stain|fix|dry|clean|sterili[sz]\w*|"
    r"examine|identify|calculat\w*|convert|boil|autoclave|dispose|wash|"
    r"mount|spread|produce|process|measur\w*|centrifuge\w*)\w*\b",
    re.I,
)


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
        if re.search(r"\bcompare\b|\bdifference\b|\bdiffers?\b|\bversus\b|\bvs\.?\b", lowered):
            return "comparison"
        if re.search(r"\bwhy\b|\breason\b", lowered):
            return "reason"
        # A question that only asks about labelling names a single embedded
        # instruction, not a self-contained multi-step procedure -- unless it
        # also asks about a genuine multi-step action (collect, dispatch, ...).
        if LABEL_ONLY_RE.search(lowered) and not MULTISTEP_ACTION_RE.search(lowered):
            return "fact"
        # "How is X shown/labelled in the figure?" asks for a description of
        # an existing illustration, not a multi-step action sequence -- unless
        # it also names a genuine action (collect, prepare, ...), in which
        # case the action is the real ask and "as shown in the figure" is
        # just a qualifier.
        if FIGURE_DEPICTION_RE.search(lowered) and not MULTISTEP_ACTION_RE.search(lowered):
            return "fact"
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
        generic_roots = {stem(term) for term in GENERIC_SUBJECT_ROOTS}
        needs: list[Need] = []
        for index, clause in enumerate(clauses):
            own_subject = self.subject_from_clause(clause)
            # A later clause doesn't always use a pronoun to continue the
            # same topic ("...and what films should be prepared?" after
            # "When should blood for malaria parasites be collected"); when
            # it introduces little or no distinctive subject of its own,
            # treat it as still talking about the established one too.
            own_is_thin = len(own_subject - generic_roots) <= 1
            if index > 0 and shared_subject and ANAPHORA_RE.search(clause):
                # An explicit pronoun means the clause is entirely about
                # the earlier subject; its own words (e.g. an adverb like
                # "microscopically") are how, not what, and must not
                # become a required term that the right chunk may simply
                # phrase differently ("microscopy", not "microscopically").
                subject = set(shared_subject)
                query = compact(f"{clause} {' '.join(sorted(shared_subject))}")
            elif index > 0 and shared_subject and own_is_thin:
                subject = set(shared_subject) | own_subject
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
            # A short "Label: value." field (e.g. "Size: 8-12mm.") is a
            # complete, real statement even under the usual length floor --
            # unlike an arbitrary short fragment, its label plus closing
            # punctuation already confirm it is not a truncated artefact.
            is_short_field = bool(
                FIELD_LABEL_RE.match(block) and re.search(r"[.!?:;]$", block)
            )
            if len(block) < 20 and not is_short_field:
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
                # Drop fragments with no closing punctuation: they are almost
                # always a chunk-boundary artefact (an overlap tail cut off
                # mid-sentence), not a real standalone statement -- except an
                # "Entity name (Fig. N.NN)" sub-heading, which never carries
                # closing punctuation of its own but is real content: the one
                # place a later "Label: value." field (e.g. "Shape: oval...")
                # can find out which entry it belongs to, in identification
                # lists that cover several entries back to back.
                result.extend(
                    part for part in map(compact, sentences)
                    if (
                        len(part) >= 20
                        or (FIELD_LABEL_RE.match(part) and re.search(r"[.!?:;]$", part))
                    )
                    and (
                        re.search(r"[.!?:;]$", part)
                        or ENTITY_FIG_HEADING_RE.match(part)
                    )
                )
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
        def nearby_subject_match(unit: Unit) -> bool:
            # A sentence that omits the subject (e.g. "Secure the top and
            # label the container...") can still be about it if an adjacent
            # sentence in the same chunk already establishes that subject.
            if required_subject & roots(unit.text):
                return True
            return any(
                other.chunk_index == unit.chunk_index
                and abs(other.order - unit.order) <= 2
                and required_subject & roots(other.text)
                for other in ranked_units
            )

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
                    else nearby_subject_match(unit)
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
                block = causal[0]
                # A numbered procedure step is kept as one whole block, so
                # its causal clause can trail an unrelated leading sentence
                # (e.g. a different film's fixation step). Narrow down to
                # just the sentence(s) that actually give the reason.
                protected = re.sub(r"\bFig\.", "Fig§", block.text, flags=re.I)
                raw_sentences = [
                    part.replace("Fig§", "Fig.")
                    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9—])", protected)
                ]
                # A leading step marker ("2.") can itself get split off as
                # its own fragment; keep it glued to the sentence it heads.
                sentences: list[str] = []
                for part in raw_sentences:
                    if re.fullmatch(r"\d+[.)]", part) and sentences == []:
                        sentences.append(part)
                    elif sentences and re.fullmatch(r"\d+[.)]", sentences[-1]):
                        sentences[-1] = f"{sentences[-1]} {part}"
                    else:
                        sentences.append(part)

                def names_opposite_of_need(sentence: str) -> bool:
                    # A block can bundle steps for two contrasting items
                    # (e.g. thick and thin films). Drop a non-causal
                    # sentence only if it names the term the question did
                    # NOT ask about, not merely because it lacks a causal
                    # marker -- otherwise a genuinely relevant lead-in
                    # sentence (that just doesn't repeat every subject
                    # word) would be lost too.
                    sentence_roots = roots(sentence)
                    for pair in CONTRASTIVE_TERM_PAIRS:
                        named = pair & subject_roots
                        opposite = pair - named
                        if named and (opposite & sentence_roots) and not (named & sentence_roots):
                            return True
                    return False

                if len(sentences) > 1:
                    kept = [
                        s for s in sentences
                        if CAUSAL_RE.search(s) or not names_opposite_of_need(s)
                    ]
                    if kept != sentences:
                        return [Unit(
                            block.chunk_index, block.order,
                            " ".join(kept), block.score,
                        )]
                return [block]
        if need.answer_type == "procedure":
            numbered: list[tuple[int, Unit]] = []
            # Start inside a subject-qualified unit, then allow adjacent
            # continuation chunks whose later steps naturally omit the subject.
            for unit in ranked_units:
                match = re.match(r"^\s*(\d+)[.)]\s+", unit.text)
                if match and not RUNNING_HEADER_RE.match(unit.text.strip()):
                    numbered.append((int(match.group(1)), unit))
            number_by_unit = {unit: number for number, unit in numbered}
            distinctive_subject = subject_roots - generic_roots
            # A step 1 from a chunk that also names the item's contrastive
            # counterpart (e.g. a "thick"-film step in a chunk that covers
            # both thick and thin) is weaker evidence than one from a
            # chunk dedicated to only the asked-about item. This only ever
            # applies when the question's subject is part of a known
            # contrastive pair -- for every other question it is a no-op,
            # so a correct chunk that simply phrases some other distinctive
            # term differently (a synonym, or doesn't repeat it) is never
            # penalized for that alone.
            contrasted_subject = {
                term for pair in CONTRASTIVE_TERM_PAIRS for term in (pair & distinctive_subject)
            }

            def contrast_specificity(unit: Unit) -> tuple[int, int]:
                if not contrasted_subject:
                    return (0, 0)
                chunk_roots = roots(self.chunks[unit.chunk_index].text)
                full = 1 if distinctive_subject <= chunk_roots else 0
                pure = 1
                for pair in CONTRASTIVE_TERM_PAIRS:
                    named = pair & distinctive_subject
                    opposite = pair - named
                    if named and opposite & chunk_roots:
                        pure = 0
                return (full, pure)

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
                        # A numbered list item that happens to wrap onto a
                        # comma-free line is not a section heading.
                        if re.match(r"^\s*\d+[.)]\s", line):
                            continue
                        # A figure caption ("Fig. 9.108 Preparing the...") is
                        # short and comma-free like a heading, but it names
                        # whatever the illustration happens to show -- not
                        # the section's topic -- so it can coincidentally
                        # repeat the query's verbs for a completely
                        # unrelated procedure elsewhere in the manual.
                        if re.match(r"^\s*Fig(?:ure)?\.?\s*\d", line, re.I):
                            continue
                        if HEADING_RE.match(line) or not re.search(r"[,!?]", line):
                            heading_terms.update(roots(line))
                    return len(operation_roots & heading_terms)

                # section_operation_overlap is a real signal when a candidate
                # is at least a plausible contender (its own step 1 reads as
                # relevant on its own), but a unit the reranker has all but
                # thrown out (a very deep negative score) can still share an
                # incidental word with some unrelated heading elsewhere in
                # its chunk -- that coincidence should not be enough to make
                # a barely-plausible unit outrank a genuinely strong one.
                start = max(
                    starts,
                    key=lambda unit: (
                        *contrast_specificity(unit),
                        section_operation_overlap(unit), unit.score,
                    ),
                )
                sequence = [start]
                expected = number_by_unit[start] + 1
                last_unit = start
                last_page = self.chunks[start.chunk_index].pdf_page
                text_by_number = {number_by_unit[start]: start.text}
                while expected <= 20:
                    def restates_a_consumed_step(candidate: Unit) -> bool:
                        # An overlapping chunk boundary often restates the
                        # last few steps already consumed before its new
                        # content starts (the same sliding window that
                        # split step 7's own sentence in two) -- that is
                        # not a foreign list, just the source text repeated
                        # across the cut. A DIFFERENT step that merely
                        # reuses an already-seen number (its own unrelated
                        # "2.", "3." from some other procedure elsewhere in
                        # the manual) is not the same restatement, so the
                        # text itself -- not just the number -- must match.
                        match = re.match(r"^\s*(\d+)[.)]\s+", candidate.text)
                        if not match:
                            return False
                        seen_text = text_by_number.get(int(match.group(1)))
                        if seen_text is None:
                            return False
                        a, b = normalize_for_exact_check(candidate.text), normalize_for_exact_check(seen_text)
                        return a in b or b in a
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
                                # A chunk earning this candidate for having
                                # nothing numbered before it in the chunk is
                                # how the walk avoids jumping into the
                                # middle of some unrelated list -- an
                                # overlap restatement of an already-consumed
                                # step (see restates_a_consumed_step) is the
                                # one exception, since it is the same source
                                # text repeated across a chunk cut, not a
                                # foreign list.
                                and not any(
                                    other.chunk_index == unit.chunk_index
                                    and other.order < unit.order
                                    and re.match(r"^\s*\d+[.)]\s+", other.text)
                                    and not RUNNING_HEADER_RE.match(other.text.strip())
                                    and not restates_a_consumed_step(other)
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
                    text_by_number[expected] = selected_step.text
                    expected += 1

                def with_continuations(steps: list[Unit]) -> list[Unit]:
                    # A numbered step's instruction sometimes spills into
                    # unnumbered sentences before the next number (e.g. "Secure
                    # the top and label the container..."). Pull those back in
                    # so the step's full instruction is not dropped.
                    expanded: list[Unit] = []
                    for position, unit in enumerate(steps):
                        expanded.append(unit)
                        following = steps[position + 1] if position + 1 < len(steps) else None
                        if following is not None and following.chunk_index == unit.chunk_index:
                            expanded.extend(sorted(
                                (
                                    candidate for candidate in ranked_units
                                    if candidate.chunk_index == unit.chunk_index
                                    and unit.order < candidate.order < following.order
                                    and not re.match(r"^\s*\d+[.)]\s+", candidate.text)
                                    and len(candidate.text) <= 200
                                ),
                                key=lambda candidate: candidate.order,
                            ))
                        elif unit.text.rstrip().endswith(":"):
                            # The step promises its detail right after it in
                            # the source even though the next kept step (if
                            # any) lives in a different chunk.
                            for candidate in sorted(
                                (
                                    c for c in ranked_units
                                    if c.chunk_index == unit.chunk_index and c.order > unit.order
                                ),
                                key=lambda c: c.order,
                            ):
                                if re.match(r"^\s*\d+[.)]\s+", candidate.text):
                                    break
                                expanded.append(candidate)
                    return expanded

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
                        # A lead-in is a standalone topic sentence, not
                        # another chunk's numbered step from a differently
                        # scoped sequence (e.g. a shared preamble step that
                        # precedes a self-labeled, item-specific branch),
                        # and not itself a promise of a list that follows
                        # (e.g. "After visual inspection, report ... as:") --
                        # that belongs after the step it elaborates on, not
                        # before it.
                        and not re.match(r"^\s*\d+[.)]\s+", unit.text)
                        and not unit.text.rstrip().endswith(":")
                    ]
                    # A numbered sequence already has its own natural
                    # boundary -- it stops the moment the next expected
                    # number can't be found -- so the flat MAX_UNITS_PER_NEED
                    # cap used elsewhere to bound loosely-related sentence
                    # picks would only serve to truncate a genuinely long,
                    # fully-numbered procedure (e.g. a 14-step recipe)
                    # partway through.
                    if lead_candidates:
                        lead = max(lead_candidates, key=lambda unit: unit.score)
                        return with_continuations([lead] + sequence)
                    return with_continuations(sequence)
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
        query_roots = roots(need.query)
        if need.answer_type == "fact":
            # A terse "Label: value." field states its own topic only in
            # the label; a bare reranker score badly under-rates it against
            # sibling fields in the same structured block (e.g. a "Fibril:"
            # line elsewhere in the chunk can out-score "Shape:" simply for
            # being longer and richer). When the question names a field's
            # label explicitly, trust that literal match over the score.
            field_matches = [
                unit for unit in filtered if field_label_roots(unit.text) & query_roots
            ]
            if field_matches:
                required_subject_terms = required_subject if required_subject else subject_roots

                def anchor_distance(unit: Unit) -> int:
                    # A block can list the same field label for several
                    # entries back to back (e.g. one identification entry
                    # per species); prefer the field nearest a mention of
                    # the question's OTHER subject terms (its entity name,
                    # not the field label itself), so the right entry wins
                    # even when a wrong-entry sibling reads better in
                    # isolation to the reranker.
                    anchor_terms = required_subject_terms - field_label_roots(unit.text)
                    if not anchor_terms:
                        return 0
                    distances = [
                        abs(other.order - unit.order)
                        for other in ranked_units
                        if other.chunk_index == unit.chunk_index
                        and anchor_terms & roots(other.text)
                    ]
                    return min(distances) if distances else 999

                field_matches = sorted(
                    field_matches, key=lambda unit: (anchor_distance(unit), -unit.score)
                )
                filtered = field_matches + [
                    unit for unit in filtered if unit not in field_matches
                ]
        best = filtered[0]
        if need.answer_type == "comparison" and ranked and best.chunk_index != ranked[0][0]:
            # A comparison question is usually answered by a passage that
            # discusses both sides together; anchor on the single
            # highest-ranked chunk instead of a lone sentence elsewhere
            # that narrowly out-scored it as an isolated unit (e.g. one
            # mentioning both compared terms only as an aside example).
            same_chunk_candidate = next(
                (unit for unit in filtered[:6] if unit.chunk_index == ranked[0][0]),
                None,
            )
            if same_chunk_candidate is not None:
                best = same_chunk_candidate
        chosen = [best]
        if need.answer_type == "calculation":
            # The calculation itself is sometimes a numbered step one or
            # two chunks after a plain sentence that merely states a
            # count is "calculated" (materials/setup content sits between
            # them); anchor on the actual numbered step when one exists.
            distinctive_subject = subject_roots - generic_roots
            structural = [
                unit for unit in filtered
                if PROCEDURE_RE.match(unit.text)
                # Chunk-level matching only requires ~60% of the subject
                # terms, which lets a numbered step for a different but
                # similarly-worded calculation (e.g. leukocytes instead of
                # erythrocytes) stand in as a false anchor when the two
                # methods share most of their vocabulary.
                and distinctive_subject <= roots(self.chunks[unit.chunk_index].text)
            ]
            if structural:
                best = structural[0]
            same_chunk = sorted(
                (
                    unit for unit in ranked_units
                    if abs(unit.chunk_index - best.chunk_index) <= 1
                    and (
                        (unit.chunk_index == best.chunk_index and abs(unit.order - best.order) <= 4)
                        or (unit.chunk_index != best.chunk_index and PROCEDURE_RE.match(unit.text))
                    )
                ),
                key=lambda unit: (unit.chunk_index, unit.order),
            )
            chosen = same_chunk or chosen
        elif need.answer_type == "procedure":
            structural = [
                unit for unit in filtered
                if PROCEDURE_RE.match(unit.text) or promises_a_list(unit.text)
            ]
            if structural:
                # The chunk-level subject check above lets every unit in a
                # multi-topic chunk through (so a step that omits the
                # subject still counts) -- but that also lets an unrelated
                # numbered/bulleted list elsewhere in the same chunk win
                # here purely for having a marker. Prefer one actually
                # anchored to the subject before falling back to the
                # unanchored top scorer.
                def structural_anchor(unit: Unit) -> bool:
                    if required_subject & roots(unit.text):
                        return True
                    return any(
                        other.chunk_index == unit.chunk_index
                        and other.order == unit.order - 1
                        and required_subject & roots(other.text)
                        for other in ranked_units
                    )
                anchored = [unit for unit in structural if structural_anchor(unit)]
                if anchored:
                    best = anchored[0]
                elif not required_subject:
                    best = structural[0]
            # A colon-led promise only calls for what comes after it (the
            # list it introduces); a numbered step can have relevant
            # neighbours on either side.
            best_promises_list = promises_a_list(best.text)
            if PROCEDURE_RE.match(best.text) or best_promises_list:
                same_chunk = sorted(
                    (
                        unit for unit in ranked_units
                        if unit.chunk_index == best.chunk_index
                        and (
                            best.order <= unit.order <= best.order + 4
                            if best_promises_list
                            else abs(unit.order - best.order) <= 4
                        )
                        and (PROCEDURE_RE.match(unit.text) or promises_a_list(unit.text))
                    ),
                    key=lambda unit: unit.order,
                )
            else:
                # `best` is itself an ordinary sentence describing the
                # procedure (e.g. "The stopcock ... should be kept well
                # greased."), not a numbered/bulleted item -- a manual
                # procedure is not always a list. Walk forward collecting
                # its plain continuation sentences instead of requiring
                # each one to carry its own marker, stopping at the next
                # heading or figure caption so an unrelated section right
                # after it in the same chunk is never swept in.
                same_chunk = [best]
                for unit in sorted(
                    (
                        u for u in ranked_units
                        if u.chunk_index == best.chunk_index and u.order > best.order
                    ),
                    key=lambda u: u.order,
                ):
                    if unit.order > best.order + 6:
                        break
                    if (
                        HEADING_RE.match(unit.text)
                        or ENTITY_FIG_HEADING_RE.match(unit.text)
                        or re.match(r"^\s*Fig(?:ure)?\.?\s*\d", unit.text, re.I)
                    ):
                        break
                    same_chunk.append(unit)
            chosen = same_chunk or chosen
        else:
            fact_action_roots = roots(need.query) - subject_roots
            for unit in filtered[1:]:
                limit = 3 if need.answer_type == "comparison" else 4
                if len(chosen) >= limit:
                    break
                if (
                    self.chunks[unit.chunk_index].pdf_page
                    == self.chunks[best.chunk_index].pdf_page
                    and (
                        unit.chunk_index == best.chunk_index
                        or (
                            abs(unit.chunk_index - best.chunk_index) <= 1
                            and (
                                # A plain fact has one specific answer; an
                                # adjacent-chunk sentence is only trusted
                                # when it shares the need's own action (e.g.
                                # "label") -- a short imperative like
                                # "Secure the top and label the container"
                                # can score below a generic relevance floor
                                # yet still be exactly what was asked about.
                                need.answer_type != "fact"
                                or bool(fact_action_roots & roots(unit.text))
                            )
                        )
                    )
                    # A block can list the same field label for several
                    # entries back to back (e.g. one identification entry
                    # per species): a labelled field the question names
                    # (e.g. "Size:") scores similarly well no matter which
                    # entry it belongs to, so score alone cannot tell them
                    # apart -- require it to sit near `best`'s own entry.
                    # A field that does not match a label the question
                    # names is unaffected and keeps the plain score gate,
                    # with that gate relaxed for a nearby one that does --
                    # those terse fields are exactly the case a bare score
                    # under-rates.
                    and (
                        not (field_label_roots(unit.text) & query_roots)
                        or (
                            unit.chunk_index == best.chunk_index
                            and abs(unit.order - best.order) <= 4
                        )
                    )
                    and (
                        unit.score > -2
                        or (
                            unit.chunk_index == best.chunk_index
                            and abs(unit.order - best.order) <= 4
                            and field_label_roots(unit.text) & query_roots
                        )
                    )
                    and (
                        not requested_figures
                        or bool(requested_figures & set(re.findall(
                            r"\bFig(?:ure)?\.?\s*(\d+\.\d+)",
                            unit.text,
                            re.I,
                        )))
                    )
                    and unit.text not in {x.text for x in chosen}
                    and not any(
                        normalize_for_exact_check(unit.text) in normalize_for_exact_check(x.text)
                        or normalize_for_exact_check(x.text) in normalize_for_exact_check(unit.text)
                        for x in chosen
                    )
                ):
                    chosen.append(unit)
            # A chosen sentence that ends in ":" promises a list right after
            # it in the source; complete it from the same chunk so the
            # answer doesn't trail off before the actual content.
            completed: list[Unit] = []
            for unit in chosen:
                completed.append(unit)
                if not unit.text.rstrip().endswith(":"):
                    continue
                following = sorted(
                    (
                        candidate for candidate in ranked_units
                        if candidate.chunk_index == unit.chunk_index
                        and candidate.order > unit.order
                    ),
                    key=lambda candidate: candidate.order,
                )
                for candidate in following:
                    if not PROCEDURE_RE.match(candidate.text):
                        break
                    if candidate not in chosen and candidate not in completed:
                        completed.append(candidate)
            chosen = completed
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
            if numbers[0] == 1:
                return numbers == list(range(1, len(numbers) + 1))
            # A step can validly start above 1 when it names, in its own
            # text, the specific item the question asked about (a shared
            # procedure branching per item, e.g. "5. Thick film. ...").
            label_match = SELF_LABELED_STEP_RE.match(units[0].text)
            if label_match and stem(label_match.group(1)) in {
                stem(term) for term in need.subject_terms
            }:
                return numbers == list(range(numbers[0], numbers[0] + len(numbers)))
            return False
        has_lead_in = any(promises_a_list(unit.text) for unit in units)
        has_instruction = any(PROCEDURE_RE.match(unit.text) for unit in units)
        if has_lead_in and has_instruction:
            return True
        # A procedure is not always written as a numbered or bulleted list
        # or a colon lead-in -- it can be an ordinary paragraph (e.g. "The
        # stopcock ... should be kept well greased. To grease ..., apply
        # ..."). Extraction's own heading-bounded gather (see extract()'s
        # plain-prose fallback) already establishes such a passage as one
        # coherent, complete block; two or more contiguous sentences from
        # the same source chunk is that shape's signature.
        return len(units) >= 2 and len({unit.chunk_index for unit in units}) == 1

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

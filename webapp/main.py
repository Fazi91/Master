"""Generic, evidence-grounded QA over Graph V2.

No corpus question, answer, topic, method, or PDF phrase is encoded here.
Every request uses the same hierarchical agentic pipeline: a conversational
router; immutable question-contract planning (RequestedItem /
InformationNeed / QuestionContract); staged, subject-verified retrieval;
authoritative-scope and obligation discovery; four-state evidence coverage
(uncovered -> candidate -> supported -> verified); a verification-to-
retrieval feedback loop; and generation that is only ever allowed to run
once every requirement for a need is fully supported.

Architecture reference: "Hierarchical Agentic RAG", Revision 4, approved
subject to two mandatory implementation corrections:
  Correction A - hard generation precondition. answer_need_with_recovery()
    is never called for a need with any RequestedItem/DiscoveredObligation
    below "supported". Explicit runtime checks enforce this immediately
    before every generator call that composes a NeedAnswer from evidence.
  Correction B - safe subject matching. subject_matches() only ever passes
    through one of five paths (identical entity id; identical normalized
    canonical name; a corpus-derived acronym/long-form alias; overlap of
    discriminative non-generic core terms with compatible entity type; or
    inheritance of an already-specific subject from a confirmed scope or a
    coherent evidence chain). Entity-type equality alone is never enough,
    and a generic head term shared by many sections can never establish
    subject identity by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal
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
IMAGES_FILE = ROOT / "data" / "graph_v2" / "images.csv"
REL_CHUNK_IMAGE_FILE = ROOT / "data" / "graph_v2" / "rel_chunk_image.csv"
REL_PAGE_IMAGE_FILE = ROOT / "data" / "graph_v2" / "rel_page_image.csv"
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


IMAGE_MIN_SCORE = float(os.getenv("IMAGE_MIN_SCORE", "0.15"))
MAX_DISPLAY_IMAGES = int(os.getenv("MAX_DISPLAY_IMAGES", "4"))


IMAGE_MIN_CLASSIFICATION_CONFIDENCE = float(
    os.getenv("IMAGE_MIN_CLASSIFICATION_CONFIDENCE", "0.5")
)


RELEVANT_IMAGE_TYPES = frozenset({
    "microscopy", "clinical_or_laboratory", "diagram_or_chart",
})


_FIGURE_CITATION_RE = re.compile(
    r"\bFig(?:ure)?s?\.?\s*\d+(?:\.\d+)?\b", re.IGNORECASE
)


CONTRACT_REPAIR_PASSES = 2
RETRIEVAL_STAGES = 3
GENERIC_DISPERSION_FLOOR = 4
GENERIC_DISPERSION_FRACTION = 0.15
NEIGHBOR_WINDOW = (-3, -2, -1, 0, 1, 2, 3)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "is", "it", "may", "of", "on", "or", "should", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
    "would", "please", "explain", "describe", "tell", "me",
}


GENERIC_QUERY_WORDS = {
    "difference", "differences", "compare", "comparison", "contrast",
    "similarity", "similarities", "reason", "reasons", "purpose",
    "purposes",
}


NON_IMPERATIVE_OPENERS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "he", "she",
    "they", "we", "you", "i", "there", "if", "when", "while", "note",
    "caution", "important", "warning", "figure", "table", "fig",
    "each", "some", "many", "most", "all", "no", "one", "two", "three",
}


_SOCIAL_FILLER = r"(?:\s+(?:there|folks|everyone|everybody|all|team|guys))?"
SOCIAL_GREETING_RE = re.compile(
    r"^(?:hi+|hello+|hey+|yo|greetings|good\s+(?:morning|afternoon|evening|day))"
    + _SOCIAL_FILLER +
    r"\s*[!.,]*$",
    re.IGNORECASE,
)
SOCIAL_THANKS_RE = re.compile(
    r"^(?:thanks?|thank\s+you)(?:\s+(?:very|so)\s+much|\s+a\s+lot)?"
    r"\s*[!.,]*$"
    r"|^(?:thx|cheers|much\s+appreciated|appreciate\s+it)\s*[!.,]*$",
    re.IGNORECASE,
)
SOCIAL_FAREWELL_RE = re.compile(
    r"^(?:bye+|goodbye|see\s+you(?:\s+later)?|farewell|take\s+care)"
    + _SOCIAL_FILLER +
    r"\s*[!.,]*$",
    re.IGNORECASE,
)
SOCIAL_LEAD_RE = re.compile(
    r"^\s*(?:hi+|hello+|hey+|yo|greetings|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"thanks?(?:\s+you)?|thank\s+you|thx|cheers)"
    + _SOCIAL_FILLER +
    r"\b[\s,!.]*",
    re.IGNORECASE,
)


class QuestionRequest(BaseModel):
    query: str


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

    value = re.split(
        r"\s+Answer\s+(?=(?:Relevant evidence|No complete answer|"
        r"According to|The |A |An ))",
        value, maxsplit=1, flags=re.IGNORECASE,
    )[0]
    return compact(value).strip()


_NEGATION_HYPHEN_RE = re.compile(r"\b(non|un)-([a-z]+)\b", re.IGNORECASE)


def _collapse_negation_hyphens(value: str) -> str:
    return _NEGATION_HYPHEN_RE.sub(lambda m: m.group(1) + m.group(2), value)


def terms(value: str) -> list[str]:
    value = _collapse_negation_hyphens(value)
    return list(dict.fromkeys(
        token for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in STOPWORDS
    ))


def roots(value: str) -> set[str]:
    """Return lightweight morphology roots for generic action matching."""
    value = _collapse_negation_hyphens(value)
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


def terms_overlap_morphologically(a: frozenset[str] | set[str], b: set[str]) -> bool:
    """Correction 19: the same lightweight, corpus-agnostic morphological
    equivalence already used to widen retrieval (`roots()`'s suffix
    stripping, plus a shared-prefix check for longer words - the same two
    schemes `_morphological_variants` combines) - reused here so
    RequestedItem/DiscoveredObligation coverage-matching recognises
    "examined" against "examination", or "microscopically" against
    "microscopic", rather than only a byte-for-byte identical inflection.
    An exact-term evidence sentence can otherwise be a truncated/rejected
    fragment while the genuinely correct sentence uses a different
    inflection and is wrongly scored as not covering the request. Purely
    structural English morphology, never a corpus-specific synonym."""
    if a & b:
        return True
    if roots(" ".join(a)) & roots(" ".join(b)):
        return True
    long_a = {word[:6] for word in a if len(word) >= 7}
    long_b = {word[:6] for word in b if len(word) >= 7}
    return bool(long_a & long_b)


def terms_covered_morphologically(
    required: frozenset[str] | set[str], candidate: set[str]
) -> bool:
    """Require meaningful operation coverage, not one shared token."""
    wanted = set(required)
    if not wanted:
        return True
    candidate_roots = roots(" ".join(candidate))
    matched = sum(
        1 for word in wanted
        if word in candidate or roots(word) & candidate_roots
    )
    minimum = len(wanted) if len(wanted) <= 2 else max(2, len(wanted) - 1)
    return matched >= minimum


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


def dehyphenate(text: str) -> str:
    """Join a PDF line-break hyphenation without touching character counts
    anywhere offsets still need to line up (callers that need offset-
    accurate text must dehyphenate only *after* computing spans)."""
    return re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", text)


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


_ABBREVIATION_DOT_RE = re.compile(
    r"\b(?:Fig(?:ure)?s?|Table|e\.g|i\.e|etc|vs|cf|No|approx|Dr|Mr|Mrs)\.",
    re.IGNORECASE,
)


def _mask_abbreviation_dots(text: str) -> str:
    """Replace the trailing '.' of a protected abbreviation with a single
    non-period sentinel character (same length, so any character offset
    computed against the masked text is still valid against the original).
    """
    return _ABBREVIATION_DOT_RE.sub(lambda m: m.group(0)[:-1] + "\x00", text)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Offset-accurate sentence splitting against the *original* row text.

    Unlike `_units()` (which cleans text before splitting and therefore
    cannot be reconciled with character offsets), this keeps every span's
    start/end aligned with `row["text"]` so it can be intersected with a
    `MentionRecord`'s own `start_char`/`end_char_exclusive` (Correction 10,
    Correction 12 tier A). Cosmetic cleanup (dehyphenation) is applied only
    to the returned text, never to the offsets.

    The boundary search itself runs against an abbreviation-masked copy of
    `text` (same length, so offsets still line up 1:1) so an inline
    "Fig. 5.18" or "e.g." reference is never mistaken for a sentence end;
    the returned substrings are still sliced from the original, un-masked
    text.
    """
    search_text = _mask_abbreviation_dots(text)
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(search_text):
        end = match.start()
        if end > start:
            spans.append((start, end, text[start:end]))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    result = [
        (s, e, _strip_interleaved_captions(dehyphenate(compact(u))))
        for s, e, u in spans
        if len(compact(u)) >= 8
    ]


    result = [
        (s, e, u) for s, e, u in result
        if not _is_layout_artifact(u) and (
            EvidenceQA._classify_obligation_kind(u) is not None
            or (
                not _STANDALONE_CAPTION_RE.match(u)
                and not _PAGE_HEADER_RE.match(u)
            )
        )
    ]


    if result:
        last_start, last_end, _ = result[-1]
        raw_last = text[last_start:last_end].rstrip()
        truncated = (
            raw_last.endswith(",")
            or raw_last.endswith("-")
            or raw_last.count("(") > raw_last.count(")")
            or bool(re.search(
                r"\b(?:a|an|the|and|or|of|to|into|from|with|by|for|in|on)$",
                raw_last, re.IGNORECASE,
            ))
        )
        if truncated:
            result = result[:-1]
    return result


_CLAUSE_BOUNDARY_RE = re.compile(
    r"\s*;\s*"
    r"|\s*,\s+(?=(?:and|or|but|whereas|while)\b)"
    r"|\s+(?:and|or|but|whereas|while)\s+",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|[a-zA-Z]{1,10}(?:/[a-zA-Z]{1,10})?)\b"
)
_CONDITION_CUE_RE = re.compile(r"\bif\b|\bunless\b|\bin\s+case\b", re.IGNORECASE)
_CONDITION_SPAN_RE = re.compile(
    r"(?:\bif\b|\bunless\b|\bin\s+case\b).{3,150}?(?:,|\bthen\b)",
    re.IGNORECASE | re.DOTALL,
)
_REASON_CUE_RE = re.compile(r"\bwhy\b|\bbecause\b|\breasons?\b", re.IGNORECASE)


_CAUSAL_ENTAILMENT_RE = re.compile(
    r"\bbecause\b|\bsince\b|\bdue to\b|\bowing to\b|\bas a result\b|"
    r"\btherefore\b|\bthus\b|\bhence\b|\bso that\b|\bleads? to\b|\bcauses?\b|"
    r"\bresults? in\b|"
    r"\b(?:will|would|does|do|did)\s+not\s+be\b|"
    r"\b(?:is|are|was|were)\s+not\s+(?:suitable|useful|acceptable|valid)\b|"
    r"\bcannot\s+be\s+(?:used|accepted)\b|"
    r"\b(?:should|must)\s+not\s+be\b|"
    r"\b(?:is|are|was|were)\s+(?:rejected|unsuitable|unacceptable|invalid)\b",
    re.IGNORECASE,
)


_DIMENSION_CUE_RE = re.compile(
    r"\bdimensions?\b|\baspects?\b|\bfactors?\b|\bcriteria\b|"
    r"\bin\s+terms\s+of\b|\bwith\s+regard\s+to\b|\bwith\s+respect\s+to\b",
    re.IGNORECASE,
)
_UNIT_CUE_RE = re.compile(
    r"\bunits?\b|\bmeasured\s+in\b|\bexpressed\s+in\b", re.IGNORECASE
)
_OUTPUT_CUE_RE = re.compile(
    r"\breport(?:ed|ing)?\b|\brecord(?:ed|ing)?\b|\bexpress(?:ed)?\s+as\b|"
    r"\bresults?\s+(?:are|is)\s+(?:reported|expressed|recorded)\b",
    re.IGNORECASE,
)
_EXCEPTION_CUE_RE = re.compile(
    r"\bexcept\b|\bexceptions?\b|\bunless\b|\bother\s+than\b", re.IGNORECASE
)
_THRESHOLD_CUE_RE = re.compile(
    r"\bgreater\s+than\b|\bless\s+than\b|\bat\s+least\b|\bat\s+most\b|"
    r"\bexceeds?\b|\bbelow\b|\babove\b|\bthreshold\b|\bcut[- ]?offs?\b|"
    r"[<>]=?",
    re.IGNORECASE,
)


_ANAPHORA_RE = re.compile(
    r"\b(?:it|this|that|these|those|they|its|their|them)\b", re.IGNORECASE
)


_PASSIVE_ACTION_VERB_RE = re.compile(
    r"\b(?:be|is|are|was|were|been|being)\s+([a-z]+ed)\b", re.IGNORECASE
)


_ILLUSTRATIVE_EXAMPLE_RE = re.compile(
    r"\bfor example\b|\be\.g\.,?|\bsuch as\b|\bfor instance\b|\bincluding\b",
    re.IGNORECASE,
)


_LIST_LEAD_IN_RE = re.compile(r":\s*$")
_LIST_REQUEST_RE = re.compile(
    r"\b(?:components?|parts?|types?|items?|features?|consists?\s+of|"
    r"made\s+up\s+of)\b",
    re.IGNORECASE,
)
_TEMPORAL_REQUEST_RE = re.compile(r"^\s*when\b", re.IGNORECASE)
_TEMPORAL_ANSWER_RE = re.compile(
    r"\b(?:before|after|during|while|until|within|immediately|early|late|"
    r"at\s+the\s+(?:time|height|start|end)|when|once|daily|weekly|"
    r"morning|evening|hours?|days?|minutes?)\b",
    re.IGNORECASE,
)


_INTERLEAVED_CAPTION_RE = re.compile(
    r"\b(?:Fig(?:ure)?|Table)s?\.?\s*\d+(?:\.\d+)?\s+"
    r"(?:[A-Z][A-Za-z'\-]*\s+){1,7}[A-Z][A-Za-z'\-]*"
    r"(?=\s+[a-z])"
)

_TRAILING_FIGURE_TEXT_RE = re.compile(
    r"\s+Fig(?:ure)?s?\.?\s*\d+(?:\.\d+)?(?:\s+[^.;:]*)?$",
    re.IGNORECASE,
)


def _strip_interleaved_captions(text: str) -> str:


    return compact(_INTERLEAVED_CAPTION_RE.sub(" ", text))


def _is_layout_artifact(text: str) -> bool:
    value = compact(text)
    if not value:
        return True
    if re.match(r"^(?:Fig(?:ure)?s?|Table)\.?\s*\d", value, re.IGNORECASE):
        return True
    if re.match(
        r"^\d+\s+Manual of basic techniques for a health laboratory\b",
        value, re.IGNORECASE,
    ):
        return True
    # A mojibake dash at the beginning of a line is still a list marker,
    # not a layout artifact.  It is classified and normalized downstream.
    return False


_STANDALONE_CAPTION_RE = re.compile(
    r"^\s*(?:Fig(?:ure)?|Table)s?\.?\s*\d+(?:\.\d+)?"
    r"(?:\s+[A-Z][A-Za-z'\-]*)*\s*$"
)


_PAGE_HEADER_RE = re.compile(r"^[A-Z0-9][A-Z0-9 .,'\-&/]{6,80}$")


_STEP_RE = re.compile(r"^\s*(\d+)\s*[.)]\s+")
_ROMAN_STEP_RE = re.compile(r"^\s*\(?([ivxIVX]{1,5})\)?[.)]\s+")
_LETTER_STEP_RE = re.compile(r"^\s*\(?([a-zA-Z])\)?[.)]\s+")
_DASH_BULLET_RE = re.compile(r"^\s*(?:[-•*–—]|â)\s+")
_WARNING_RE = re.compile(
    r"^\s*(important|warning|caution|note|exception)s?\s*[:\-]",
    re.IGNORECASE,
)
_CONDITION_BRANCH_RE = re.compile(


    r"\bif\b(?:[^.!?]|\.(?=\d)){3,220}?(?:,\s*(?:then\b)?|\bthen\b)",
    re.IGNORECASE,
)
_FORMULA_RE = re.compile(
    r"[=×]\s*\d|\d\s*[=×]|\d+(?:\.\d+)?\s*%"


    r"|\b\d+(?:\.\d+)?[a-zA-Z°]{1,4}\b"


    r"|\b\d+(?:\.\d+)?\s*[a-zA-Z]{1,4}/[a-zA-Z]{1,4}\b"
)
_TABLE_ROW_RE = re.compile(r"(?:\S+[ \t]{2,}){2,}\S+|(?:\d+(?:\.\d+)?[ \t]+){2,}\d")


def clause_spans(question: str) -> list[tuple[int, int, str]]:
    """Split the ORIGINAL question into semantic clauses with exact
    character spans (Correction 6: every RequestedItem maps to an exact
    span of the original question, never a synthesized paraphrase, and
    never one item per content word). A fragment left with fewer than two
    content terms after a split (a bare connective) is folded back into its
    neighbour rather than becoming its own clause."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(question):
        end = match.start()
        if end > start:
            spans.append((start, end, question[start:end]))
        start = match.end()
    if start < len(question):
        spans.append((start, len(question), question[start:]))
    merged: list[tuple[int, int, str]] = []
    for s, e, text in spans:
        if merged and len(terms(text)) < 2:
            prev_s, _, _ = merged[-1]
            merged[-1] = (prev_s, e, question[prev_s:e])
        else:
            merged.append((s, e, text))
    return [
        (s, e, compact(question[s:e])) for s, e, _ in merged
        if compact(question[s:e])
    ]


@dataclass(frozen=True)
class RequestedItem:
    """One explicitly requested piece of information inside a need, mapped
    to an EXACT character span of the ORIGINAL question - never a
    synthesized paraphrase, and never one item per content word. `kind`
    distinguishes the semantic role a span plays in the contract, and is
    genuinely populated by `plan()` from the clause's own generic shape
    (condition/reason/comparison-side/quantity cues already used
    elsewhere for `contract.conditions`/`reasons`/`comparison_sides`/
    `quantities`) rather than always defaulting to "clause" - and is
    actually consumed downstream by the answer-type-specific obligation
    policy (`_obligation_required`). Every item carries: a unique `key`,
    the owning `need_id`, its `kind`, the exact `description` source text
    and character `span`, a `normalized` canonicalized form, and a
    `required` flag; its coverage state (uncovered/candidate/supported/
    verified) is tracked externally, per `key`, by the need's
    `CoverageMap` - never stored as an unverified plain string."""

    key: str
    description: str
    span: tuple[int, int]
    terms: frozenset[str] = frozenset()
    kind: Literal[
        "clause", "condition", "reason", "comparison_side", "dimension",
        "unit", "output", "exception", "quantity", "contextual",
    ] = "clause"
    need_id: str = ""
    required: bool = True
    normalized: str = ""


@dataclass(frozen=True)
class InformationNeed:
    """One independently answerable facet of the question."""

    label: str
    query: str
    requirements: tuple[str, ...]
    retrieval_query: str
    kind: str = "fact"
    requested_items: tuple[RequestedItem, ...] = ()
    subject_terms: frozenset[str] = frozenset()
    subject_entity_ids: frozenset[str] = frozenset()
    need_id: str = ""
    operation_terms: frozenset[str] = frozenset()
    context_terms: frozenset[str] = frozenset()


@dataclass(frozen=True)
class QuestionContract:
    """Immutable, validated decomposition of the whole question. Beyond the
    per-need breakdown, this also carries the structural facts Correction 6
    asks for at the whole-question level: the subject shared across every
    need, and any conditions/reasons/comparison sides/quantities/ordering
    requirement detected directly from the question's own wording (never
    corpus content - these are generic sentence-structure cues, the same
    category as STOPWORDS)."""

    question: str
    answer_type: str
    needs: tuple[InformationNeed, ...]
    shared_subject_terms: frozenset[str] = frozenset()
    conditions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    comparison_sides: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()
    quantities: tuple[str, ...] = ()
    required_ordering: bool = False


@dataclass(frozen=True)
class EntityInfo:
    canonical_name: str
    normalized_name: str
    entity_type: str


@dataclass(frozen=True)
class MentionRecord:
    """One entity mention inside one chunk, with chunk-relative character
    offsets. Neo4j and the Graph V2 CSV fallback both populate this same
    shape (Correction 1), so subject resolution behaves identically
    regardless of which loading path produced it."""

    entity_id: str
    canonical_name: str
    normalized_name: str
    entity_type: str
    mention_text: str
    start_char: int
    end_char_exclusive: int


@dataclass(frozen=True)
class LocalSubject:
    """The subject an evidence unit carries, resolved through the four-tier
    precedence Correction 3 specifies: (A) an entity mention overlapping the
    unit's own span; (B) the nearest enclosing heading, when its own core
    terms are non-generic; (C) continuity from the coherent run of units
    examined so far; (D) unresolved, when a heading/topic boundary is
    crossed and neither A nor B applies - inheritance stops rather than
    guessing."""

    resolved_terms: frozenset[str]
    resolved_entity_ids: frozenset[str]
    resolution_source: Literal[
        "sentence", "heading", "scope", "body", "chain_context", "lead_in", "none"
    ]

    @property
    def is_resolved(self) -> bool:
        return bool(self.resolved_terms or self.resolved_entity_ids)


@dataclass(frozen=True)
class DiscoveredObligation:
    """A requirement found only while reading evidence (an additional step,
    branch, warning, or formula), never asked for up front and never
    authored - purely a record of what the confirmed authoritative source
    scope itself turned out to contain (Correction 6). `kind` records which
    generic textual pattern produced it; `required` separates a genuinely
    binding obligation (a numbered step, a condition/action branch, a
    formula the calculation needs) from a merely supplementary one (an
    optional background note) so optional background can never substitute
    for missing required coverage."""

    key: str
    description: str
    terms: frozenset[str]
    discovered_in_chunk_id: str
    kind: Literal[
        "step", "bullet", "condition_branch", "warning", "formula",
        "comparison_dimension", "table_row", "threshold", "output",
    ] = "step"
    required: bool = True


@dataclass(frozen=True)
class CandidateRegion:
    """A contiguous run of chunks considered together as one retrieval unit
    before any individual sentence is chosen from it - the window
    `_select_authoritative_region()` scores and chooses between, instead of
    isolated sentences being pulled straight out of independently ranked
    chunks."""

    region_id: str
    need_label: str
    chunk_ids: tuple[str, ...]
    start_position: int
    end_position: int
    heading: str
    score: float


@dataclass(frozen=True)
class AuthoritativeScope:
    """The single region selected as the authoritative source for a need
    (Correction 4): the heading-bounded span that obligation discovery
    reads from, and that local-subject heading-inheritance (tier B) and
    chain continuity (tier C) are both anchored to."""

    scope_id: str
    region_id: str
    heading: str
    subject_terms: frozenset[str]
    start_position: int
    end_position: int


@dataclass(frozen=True)
class EvidenceUnit:
    """One sentence/line-level span considered as evidence, carrying its
    own resolved local subject, its offset back into the source chunk, and
    an evidence role. Only "primary" and "required_conditional" units may
    reach the generator by default (Correction 9/10) - "optional_background"
    units are tracked but excluded unless explicitly requested."""

    evidence_index: int
    unit_index: int
    chunk_id: str
    start_char: int
    end_char: int
    text: str
    local_subject: LocalSubject
    row_score: float
    unit_score: float
    role: Literal["primary", "required_conditional", "optional_background"]
    covered_item_ids: frozenset[str] = frozenset()
    covered_obligation_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class EvidenceChain:
    """A subject-coherent, ordered run of EvidenceUnits assembled as the
    evidence for one InformationNeed (Correction 7) - materialized and
    inspectable, and the object answer assembly/citation actually walks,
    rather than a loose dict of sentence scores."""

    chain_id: str
    need_label: str
    units: tuple[EvidenceUnit, ...]
    chain_type: Literal["contiguous", "distributed"]
    source_chunk_ids: tuple[str, ...]
    region_id: str
    scope_id: str
    local_subjects: tuple[LocalSubject, ...]
    covered_item_ids: frozenset[str]
    covered_obligation_ids: frozenset[str]
    continuation_state: Literal["closed", "open", "boundary"]
    validation_status: Literal["candidate", "validated", "rejected"]
    rejection_reason: str | None = None


class CoverageEntry:
    """Mutable per-item coverage state: uncovered -> candidate -> supported
    -> verified (Revision 3's four states, Correction A's precondition)."""

    STATES = ("uncovered", "candidate", "supported", "verified")
    __slots__ = ("key", "state", "evidence_ids")

    def __init__(self, key: str) -> None:
        self.key = key
        self.state = "uncovered"
        self.evidence_ids: list[int] = []

    def advance(self, state: str, evidence_id: int | None = None) -> None:
        if self.STATES.index(state) > self.STATES.index(self.state):
            self.state = state
        if evidence_id is not None and evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)


class CoverageMap:
    """Tracks every RequestedItem and DiscoveredObligation for one need."""

    def __init__(self, need: InformationNeed) -> None:
        self.need = need
        self.entries: dict[str, CoverageEntry] = {
            item.key: CoverageEntry(item.key) for item in need.requested_items
        }
        self.obligations: dict[str, DiscoveredObligation] = {}

    def add_obligation(self, obligation: DiscoveredObligation) -> None:
        if obligation.key not in self.entries:
            self.entries[obligation.key] = CoverageEntry(obligation.key)
            self.obligations[obligation.key] = obligation

    def mark(
        self, key: str, state: str, evidence_id: int | None = None
    ) -> None:
        self.entries.setdefault(key, CoverageEntry(key)).advance(
            state, evidence_id
        )

    def required_keys(self) -> list[str]:
        """Every RequestedItem key whose own `required` flag is True, plus
        every obligation key whose `required` flag is True (Correction 8:
        optional background must never be used to hide missing primary
        evidence, so an optional obligation - or an optional, purely
        supplementary RequestedItem such as a tracked-but-non-gating
        quantity mention - is tracked but never gates completeness)."""
        item_required = {
            item.key: item.required for item in self.need.requested_items
        }
        return [
            key for key in self.entries
            if (key in self.obligations and self.obligations[key].required)
            or (key not in self.obligations and item_required.get(key, True))
        ]

    def all_verified(self) -> bool:
        """Correction 8's strict completion gate: a question is complete
        only once every required item/obligation has actually passed
        independent verification, not merely been matched by a candidate
        sentence."""
        required = self.required_keys()
        return all(
            self.entries[key].state == "verified" for key in required
        ) if required else True

    def all_supported(self) -> bool:
        """Correction A's hard precondition: every REQUIRED tracked
        item/obligation must be at least 'supported'. An empty coverage map
        (a need with no explicit requested items and no required discovered
        obligations) is vacuously satisfied - there is nothing left to
        recover. An optional (non-required) obligation never gates this."""
        required = self.required_keys()
        return all(
            self.entries[key].state in ("supported", "verified")
            for key in required
        ) if required else True

    def cited_evidence_ids(self) -> list[int]:
        ids: list[int] = []
        for entry in self.entries.values():
            for evidence_id in entry.evidence_ids:
                if evidence_id not in ids:
                    ids.append(evidence_id)
        return ids


def all_need_requirements_supported(
    need: InformationNeed, coverage_map: CoverageMap, answer_type: str = "",
) -> bool:
    """Correction A: the single predicate every generator call must assert
    immediately before it runs. Never call the generator/composer for a
    need unless this returns True.

    Correction 19 / Blocker 3: term-overlap "supported" coverage on the
    base RequestedItem clause alone is never sufficient for a
    "calculation" need - that would accept a passage that only discusses
    the same equipment/subject itself without containing any of the
    actual computation. A calculation additionally
    requires that at least one of the SAME generic obligation kinds
    `_ANSWER_TYPE_OBLIGATION_KINDS["calculation"]` already declares
    (formula/threshold/branch/step/bullet/table_row/output) was actually
    discovered in the confirmed scope and reached "supported" - purely a
    structural completeness rule, never a numeric threshold change and
    never keyed on any corpus-specific word."""
    if coverage_map.need is not need or not coverage_map.all_supported():
        return False
    if answer_type == "procedure":
        procedure_kinds = EvidenceQA._ANSWER_TYPE_OBLIGATION_KINDS["procedure"]
        has_procedure_obligation = any(
            obligation.kind in procedure_kinds
            and coverage_map.entries[key].state in ("supported", "verified")
            for key, obligation in coverage_map.obligations.items()
        )
        if not has_procedure_obligation:
            return False
    if answer_type == "calculation":
        calculation_kinds = EvidenceQA._ANSWER_TYPE_OBLIGATION_KINDS["calculation"]
        has_calculation_obligation = any(
            obligation.kind in calculation_kinds
            and coverage_map.entries[key].state in ("supported", "verified")
            for key, obligation in coverage_map.obligations.items()
        )
        if not has_calculation_obligation:
            return False
    return True


_RequestKind = Literal["social_only", "mixed", "ambiguous", "corpus_question"]


def classify_request(question: str) -> tuple[_RequestKind, str]:
    """Return (kind, effective_question). `effective_question` is the text
    the rest of the pipeline should see - unchanged for "corpus_question"
    and "ambiguous", and with the social wrapper stripped for "mixed"."""
    stripped = compact(question)
    if not stripped:
        return "ambiguous", stripped
    if SOCIAL_GREETING_RE.match(stripped) or SOCIAL_THANKS_RE.match(stripped) \
            or SOCIAL_FAREWELL_RE.match(stripped):
        return "social_only", stripped
    remainder = compact(SOCIAL_LEAD_RE.sub("", stripped, count=1))
    if remainder and remainder != stripped and len(terms(remainder)) >= 2:
        return "mixed", remainder
    if len(terms(stripped)) < 2:
        return "ambiguous", stripped
    return "corpus_question", stripped


def lightweight_social_response(question: str) -> dict[str, Any]:
    if SOCIAL_FAREWELL_RE.match(question):
        text = "Goodbye - come back any time you have a question about the manual."
    elif SOCIAL_THANKS_RE.match(question):
        text = "You're welcome. Let me know if you have another question about the manual."
    else:
        text = (
            "Hello. Ask me a question about the laboratory manual and I "
            "will answer strictly from its evidence."
        )
    return EvidenceQA.response(
        "conversational", question, text, [], [], 0, "not_applicable"
    )


def clarification_response(question: str) -> dict[str, Any]:
    return EvidenceQA.response(
        "ambiguous", question,
        "Please ask a more specific question about the corpus.",
        [], [], 0, "not_applicable",
    )


class EvidenceQA:
    def __init__(self) -> None:
        self.request_lock = RLock()
        self.corpus_source = "csv"
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


                long_form = " ".join(long_form.split()[-8:])
                acronym = match.group(2).lower()
                if len(long_form.split()) >= 2:
                    self.aliases.setdefault(long_form, set()).add(acronym)
                    self.aliases.setdefault(acronym, set()).add(long_form)


        self.entity_index: dict[str, EntityInfo] = {}
        for row in self.rows:
            for mention in row.get("mentions", []):
                if mention.entity_id not in self.entity_index:
                    self.entity_index[mention.entity_id] = EntityInfo(
                        mention.canonical_name,
                        mention.normalized_name,
                        mention.entity_type,
                    )


        self.chunk_heading: dict[str, str] = {}
        current_heading = ""
        for row in self.rows:
            lines = row["text"].splitlines()
            own_heading = None
            for line_index, line in enumerate(lines):
                if self._heading(lines, line_index):
                    own_heading = compact(line)
            if own_heading:
                current_heading = own_heading
            self.chunk_heading[row["chunk_id"]] = current_heading


        self.term_headings: dict[str, set[str]] = {}
        for row in self.rows:
            heading = self.chunk_heading.get(row["chunk_id"], "")
            if not heading:
                continue
            row_terms = set(terms(row["text"]))
            for mention in row.get("mentions", []):
                row_terms.add(mention.normalized_name.lower())
            for term in row_terms:
                self.term_headings.setdefault(term, set()).add(heading)
        distinct_headings = len({h for h in self.chunk_heading.values() if h})
        self.generic_floor = max(
            GENERIC_DISPERSION_FLOOR,
            round(GENERIC_DISPERSION_FRACTION * distinct_headings),
        )


        corpus = [row["text"] for row in self.rows]
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
            stop_words="english",
        )
        self.lexical_matrix = self.vectorizer.fit_transform(corpus)


        self.root_index: dict[str, set[str]] = {}
        for vocab_word in self.vectorizer.vocabulary_:
            if " " in vocab_word or not vocab_word.isalpha() or len(vocab_word) < 4:
                continue
            for key in roots(vocab_word):
                self.root_index.setdefault(f"s:{key}", set()).add(vocab_word)
            if len(vocab_word) >= 7:
                self.root_index.setdefault(f"p:{vocab_word[:6]}", set()).add(vocab_word)
        self.lock = Lock()
        self.dense = None
        self.dense_matrix = None
        self.reranker = None
        self.tokenizer = None
        self.generator = None
        self.nli = None
        self.retrieval_cache: dict[tuple[Any, ...], tuple[dict[str, Any], ...]] = {}


        self.last_chains: dict[str, EvidenceChain] = {}


        self.last_coverage_maps: dict[str, CoverageMap] = {}


        self._load_images()

    @staticmethod
    def _graph():
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            return None
        driver = None
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            return driver
        except Exception as error:
            if driver is not None:
                driver.close()
            print(f"[GRAPH] unavailable, using CSV fallback: {error}")
            return None

    @staticmethod
    def _mention_record(raw: dict[str, Any]) -> MentionRecord | None:
        entity_id = raw.get("entity_id")
        if not entity_id:
            return None
        return MentionRecord(
            entity_id=str(entity_id),
            canonical_name=str(raw.get("canonical_name") or ""),
            normalized_name=str(
                raw.get("normalized_name") or raw.get("canonical_name") or ""
            ).lower(),
            entity_type=str(raw.get("entity_type") or ""),
            mention_text=str(raw.get("mention_text") or ""),
            start_char=integer(raw.get("start_char")) or 0,
            end_char_exclusive=integer(raw.get("end_char_exclusive")) or 0,
        )

    def _chunks(self) -> list[dict[str, Any]]:
        if self.driver:
            try:
                with self.driver.session() as session:


                    result = session.run(
                        """
                        MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
                        OPTIONAL MATCH (c)-[m:MENTIONS]->(e:Entity)
                        RETURN c.id AS chunk_id, c.text AS text,
                               p.pdf_page AS pdf_page,
                               p.printed_page AS printed_page,
                               coalesce(c.chunk_index_on_page, 0) AS chunk_index,
                               collect(DISTINCT e.canonical_name) AS entities,
                               collect(DISTINCT CASE WHEN e IS NULL THEN NULL
                                   ELSE {
                                       entity_id: e.id,
                                       canonical_name: e.canonical_name,
                                       normalized_name: e.normalized_name,
                                       entity_type: e.entity_type,
                                       mention_text: m.text,
                                       start_char: m.start_char,
                                       end_char_exclusive: m.end_char_exclusive
                                   } END) AS mentions
                        ORDER BY p.pdf_page, chunk_index, c.id
                        """
                    )
                    rows = []
                    for record in result:
                        row = dict(record)
                        row["entities"] = [
                            name for name in row.get("entities", []) if name
                        ]
                        row["mentions"] = [
                            record for raw in (row.get("mentions") or [])
                            if raw is not None
                            and (record := self._mention_record(raw)) is not None
                        ]
                        rows.append(row)
                if rows:
                    self.corpus_source = "neo4j"
                    return rows
            except Exception as error:
                print(f"[GRAPH] using CSV fallback: {error}")
        entity_info: dict[str, EntityInfo] = {}
        with (CHUNKS_FILE.parent / "entities.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                entity_info[row["entity_id"]] = EntityInfo(
                    row.get("canonical_name") or "",
                    (row.get("normalized_name") or row.get("canonical_name") or "").lower(),
                    row.get("entity_type") or "",
                )
        entity_names = {
            entity_id: info.canonical_name
            for entity_id, info in entity_info.items()
        }
        chunk_entities: dict[str, list[str]] = {}
        chunk_mentions: dict[str, list[MentionRecord]] = {}
        with (CHUNKS_FILE.parent / "rel_chunk_entity.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            for mention in csv.DictReader(stream):
                info = entity_info.get(mention["entity_id"])
                name = entity_names.get(mention["entity_id"])
                if name:
                    values = chunk_entities.setdefault(mention["chunk_id"], [])
                    if name not in values:
                        values.append(name)
                if info is not None:
                    chunk_mentions.setdefault(mention["chunk_id"], []).append(
                        MentionRecord(
                            entity_id=mention["entity_id"],
                            canonical_name=info.canonical_name,
                            normalized_name=info.normalized_name,
                            entity_type=info.entity_type,
                            mention_text=mention.get("mention_text") or "",
                            start_char=integer(mention.get("start_char")) or 0,
                            end_char_exclusive=integer(
                                mention.get("end_char_exclusive")
                            ) or 0,
                        )
                    )
        with CHUNKS_FILE.open(encoding="utf-8-sig", newline="") as stream:
            return [{
                "chunk_id": row["chunk_id"],
                "text": row["chunk_text"],
                "pdf_page": integer(row.get("pdf_page")),
                "printed_page": integer(row.get("printed_page")),
                "chunk_index": integer(row.get("chunk_index_on_page")) or 0,
                "entities": chunk_entities.get(row["chunk_id"], []),
                "mentions": chunk_mentions.get(row["chunk_id"], []),
            } for row in csv.DictReader(stream)]

    def _load_images(self) -> None:
        """Part 2: load the verified, generic Graph V2 image-relation CSVs
        (never modified, never assumed - the exact columns below were
        confirmed directly from the repository's own
        `data/graph_v2/images.csv` / `rel_chunk_image.csv` /
        `rel_page_image.csv`). Any file that is missing (a corpus with no
        image pipeline run) leaves the corresponding table empty rather
        than raising - image retrieval is optional, never load-bearing for
        the text QA path."""
        self.images_by_id: dict[str, dict[str, Any]] = {}
        self.chunk_images: dict[str, list[dict[str, Any]]] = {}
        self.page_images: dict[int, list[dict[str, Any]]] = {}
        try:
            with IMAGES_FILE.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    image_id = row.get("image_id")
                    if not image_id:
                        continue
                    confidence = row.get("classification_confidence")
                    self.images_by_id[image_id] = {
                        "file_path": row.get("file_path") or "",
                        "predicted_type": (row.get("predicted_type") or "").lower(),
                        "classification_confidence": (
                            float(confidence) if confidence not in (None, "") else None
                        ),
                        "classification_status": row.get("classification_status") or "",
                        "review_status": row.get("review_status") or "",
                        "final_type": (row.get("final_type") or "").lower(),
                        "content_relevance": row.get("content_relevance") or "",
                        "first_pdf_page": integer(row.get("first_pdf_page")),
                    }
        except (FileNotFoundError, OSError, csv.Error) as error:
            print(f"[IMAGES] images.csv unavailable: {error}")
        try:
            with REL_CHUNK_IMAGE_FILE.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    chunk_id = row.get("chunk_id")
                    image_id = row.get("image_id")
                    if not chunk_id or not image_id:
                        continue
                    try:
                        score = float(row.get("semantic_score") or 0.0)
                    except ValueError:
                        score = 0.0
                    self.chunk_images.setdefault(chunk_id, []).append({
                        "image_id": image_id,
                        "pdf_page": integer(row.get("pdf_page")),
                        "semantic_score": score,
                        "image_type": (row.get("image_type") or "").lower(),
                    })
        except (FileNotFoundError, OSError, csv.Error) as error:
            print(f"[IMAGES] rel_chunk_image.csv unavailable: {error}")
        try:
            with REL_PAGE_IMAGE_FILE.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    image_id = row.get("image_id")
                    pdf_page = integer(row.get("pdf_page"))
                    if not image_id or pdf_page is None:
                        continue
                    try:
                        coverage = float(row.get("page_coverage") or 0.0)
                    except ValueError:
                        coverage = 0.0
                    self.page_images.setdefault(pdf_page, []).append({
                        "image_id": image_id,
                        "page_coverage": coverage,
                    })
        except (FileNotFoundError, OSError, csv.Error) as error:
            print(f"[IMAGES] rel_page_image.csv unavailable: {error}")
        self._row_by_chunk: dict[str, dict[str, Any]] = {
            row["chunk_id"]: row for row in self.rows
        }
        self._printed_page_by_pdf_page: dict[int, int] = {}
        for row in self.rows:
            pdf_page, printed_page = row.get("pdf_page"), row.get("printed_page")
            if pdf_page is not None and printed_page is not None:
                self._printed_page_by_pdf_page.setdefault(pdf_page, printed_page)

    def _caption_for(self, chunk_id: str) -> str | None:
        """A generic, structural caption extraction: the corpus carries no
        dedicated caption field (confirmed from the repository's own
        CSVs) - captions appear only as inline OCR'd "Fig. N.N ..."
        fragments inside ordinary chunk text, so this opportunistically
        pulls that fragment out using the same generic citation shape
        used for page-link corroboration. Returns None (never an
        invented placeholder) when the chunk carries no such fragment."""
        row = self._row_by_chunk.get(chunk_id)
        if row is None:
            return None
        text = row.get("text", "")
        match = re.search(
            r"(?m)^\s*Fig(?:ure)?s?\.?\s*\d+(?:\.\d+)?\b[^\n]*",
            text, re.IGNORECASE,
        )
        if match:
            caption = compact(match.group(0))
            second = re.search(
                r"\s+Fig(?:ure)?s?\.?\s*\d+(?:\.\d+)?\b",
                caption[1:], re.IGNORECASE,
            )
            if second:
                caption = caption[:second.start() + 1]
            return caption[:200] or None
        match = _FIGURE_CITATION_RE.search(text)
        if not match:
            return None
        return compact(text[match.start():match.start() + 140]) or None

    def images(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Part 2: verified image retrieval for the exact, already-cited
        evidence chunk ids an answer was built from. Text determines the
        answer first - images are strictly supplementary, and this is
        only ever called AFTER a text answer has been fully verified
        (see the `answer()` call site). Tries live Neo4j first when
        configured (mirroring `_chunks()`'s own graph-then-CSV-fallback
        pattern); falls back to the verified Graph V2 relation CSVs
        loaded once at process start by `_load_images()` - never re-read
        per request, and never a CLIP/image model."""
        if not chunk_ids:
            return []
        if self.driver:
            try:
                rows = self._images_from_graph(chunk_ids)
                if rows:
                    return rows
            except Exception as error:
                print(f"[GRAPH] image lookup failed, using CSV fallback: {error}")
        return self._images_from_csv(chunk_ids)

    def _images_from_graph(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (p:Page)-[:HAS_CHUNK]->(c:Chunk)
                      -[r:ILLUSTRATED_BY]->(i:Image)
                WHERE c.id IN $ids AND i.file_path IS NOT NULL
                  AND coalesce(r.semantic_score, 0) >= $minimum
                RETURN i.id AS id, i.file_path AS file_path,
                       p.pdf_page AS pdf_page,
                       p.printed_page AS printed_page,
                       c.id AS chunk_id,
                       r.semantic_score AS confidence,
                       r.image_type AS image_type
                ORDER BY confidence DESC LIMIT {MAX_DISPLAY_IMAGES}
                """,
                ids=chunk_ids,
                minimum=IMAGE_MIN_SCORE,
            )
            rows = [dict(record) for record in result]
        verified: dict[str, dict[str, Any]] = {}
        for row in rows:
            image_id = str(row.get("id") or "")
            info = self.images_by_id.get(image_id, {})
            file_path = row.get("file_path") or info.get("file_path")
            if not file_path:
                continue
            classifier_type = (
                info.get("final_type") or row.get("image_type")
                or info.get("predicted_type") or ""
            ).lower()
            if classifier_type not in RELEVANT_IMAGE_TYPES:
                continue
            raw_path = Path(str(file_path))
            path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            try:
                path = path.resolve()
            except OSError:
                continue
            if ROOT.resolve() not in path.parents or not path.is_file():
                continue
            confidence = float(row.get("confidence") or 0.0)
            current = verified.get(image_id)
            if current is not None and current["confidence"] >= confidence:
                continue
            verified[image_id] = {
                "id": image_id,
                "url": f"/image/{image_id}",
                "pdf_page": row.get("pdf_page"),
                "printed_page": row.get("printed_page"),
                "chunk_id": row.get("chunk_id"),
                "confidence": confidence,
                "type": info.get("final_type") or "figure",
                "caption": self._caption_for(row.get("chunk_id") or ""),
                "verification_reason": "directly linked to the cited evidence chunk",
            }
        return sorted(
            verified.values(), key=lambda item: item["confidence"], reverse=True
        )[:MAX_DISPLAY_IMAGES]

    def _images_from_csv(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not self.chunk_images and not self.page_images:
            return []
        candidates: dict[str, dict[str, Any]] = {}

        def consider(
            image_id: str, chunk_id: str, pdf_page: int | None,
            rank_score: float, reason: str, image_type_hint: str | None,
            require_classification_confidence: bool = False,
        ) -> None:
            info = self.images_by_id.get(image_id)
            if not info or not info.get("file_path"):


                return
            raw_path = Path(str(info["file_path"]))
            path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            try:
                path = path.resolve()
            except OSError:
                return
            if ROOT.resolve() not in path.parents or not path.is_file():
                return
            effective_type = (
                info.get("final_type") or image_type_hint
                or info.get("predicted_type") or ""
            )
            if effective_type not in RELEVANT_IMAGE_TYPES:


                return
            if require_classification_confidence:
                confidence = info.get("classification_confidence")
                if (
                    confidence is None
                    or confidence < IMAGE_MIN_CLASSIFICATION_CONFIDENCE
                ):


                    return
            existing = candidates.get(image_id)
            if existing is not None and existing["confidence"] >= rank_score:


                return
            candidates[image_id] = {
                "id": image_id,
                "url": f"/image/{image_id}",
                "pdf_page": pdf_page,
                "printed_page": self._printed_page_by_pdf_page.get(pdf_page),
                "chunk_id": chunk_id,
                "confidence": round(float(rank_score), 4),
                "type": info.get("final_type") or "figure",
                "caption": self._caption_for(chunk_id),
                "verification_reason": reason,
            }


        for chunk_id in chunk_ids:
            for rel in self.chunk_images.get(chunk_id, []):
                if rel["semantic_score"] < IMAGE_MIN_SCORE:
                    continue
                consider(
                    rel["image_id"], chunk_id, rel.get("pdf_page"),
                    rel["semantic_score"],
                    "directly linked to the cited evidence chunk",
                    rel.get("image_type"),
                )


        ranked = sorted(
            candidates.values(), key=lambda row: row["confidence"], reverse=True
        )
        return ranked[:MAX_DISPLAY_IMAGES]

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

    def _morphological_variants(self, query: str) -> str:
        """Correction 16: generic morphological expansion (see
        `self.root_index`) - for each word actually in the query, collect
        whatever ACTUAL corpus vocabulary word(s) share its generic
        suffix-stripped root or fixed-length prefix truncation, so a
        candidate-generation pass over this text can find a passage using
        a different inflection of the same word (Correction 16). Never a
        hardcoded synonym/topic list - purely derived from ordinary
        English morphology and whatever corpus is loaded. Deliberately
        returned SEPARATELY from `_expand_query` and never folded into the
        query text used for scoring/reranking/dense-encoding: a rare,
        OCR-glued corpus token that happens to share a root/prefix would
        otherwise dominate a TF-IDF query vector purely by its own high
        IDF weight - this is used only to WIDEN the first-stage retrieval
        candidate POOL in `retrieve()`, never to alter how any candidate
        already found by the original query is scored or ranked."""
        variants: list[str] = []
        for word in re.findall(r"[a-z]{3,}", query.lower()):
            keys = {f"s:{key}" for key in roots(word)}
            if len(word) >= 7:
                keys.add(f"p:{word[:6]}")
            for key in keys:
                for vocab_word in self.root_index.get(key, ()):
                    if vocab_word != word:
                        variants.append(vocab_word)
        return " ".join(dict.fromkeys(variants[:24]))

    def _reformulate_query(self, need: InformationNeed) -> str:
        """Correction 17 / Blocker 1: a generic semantic reformulation of a
        need's own retrieval query, used only as the LAST-resort retrieval
        escalation step when the original query's own dense+lexical+rerank
        pipeline (see `retrieve()`) still returns nothing even at maximum
        widen. Built purely from the corpus's own morphological vocabulary
        (`_morphological_variants`, the same generic root/prefix expansion
        used to widen the candidate pool) plus the need's own already-
        grounded subject terms and requirements - never a synonym or rule
        authored for any specific sample question, and never a corpus
        string that was not already part of the need itself."""
        base = self._expand_query(need.retrieval_query)
        variants = self._morphological_variants(base)
        extra_terms = " ".join(sorted(need.subject_terms | set(need.requirements)))
        reformulated = compact(
            " ".join(part for part in (base, variants, extra_terms) if part)
        )
        return reformulated or need.retrieval_query

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


    def is_generic_subject_term(self, term: str) -> bool:
        return len(self.term_headings.get(term, ())) >= self.generic_floor

    def core_terms(self, text: str) -> set[str]:
        return {t for t in terms(text) if not self.is_generic_subject_term(t)}

    def resolve_local_subject(
        self,
        row: dict[str, Any],
        start_char: int,
        end_char: int,
        chain_subject: LocalSubject | None,
    ) -> LocalSubject:
        """Four-tier precedence local-subject resolution (Correction 3)."""
        mentions: list[MentionRecord] = row.get("mentions", [])
        overlapping = [
            m for m in mentions
            if m.start_char < end_char and m.end_char_exclusive > start_char
        ]
        if overlapping:

            resolved_terms: set[str] = set()
            resolved_entity_ids: set[str] = set()
            for mention in overlapping:
                resolved_entity_ids.add(mention.entity_id)
                resolved_terms.update(self.core_terms(mention.canonical_name))
                resolved_terms.add(mention.normalized_name.lower())


            resolved_terms.update(self.core_terms(row["text"][start_char:end_char]))
            return LocalSubject(
                frozenset(resolved_terms), frozenset(resolved_entity_ids),
                "sentence",
            )


        own_text = row["text"][start_char:end_char]
        if _LIST_LEAD_IN_RE.search(own_text.rstrip()):
            lead_in_terms = self.core_terms(own_text)
            if lead_in_terms:
                return LocalSubject(
                    frozenset(lead_in_terms), frozenset(), "lead_in",
                )


        heading = self.chunk_heading.get(row["chunk_id"], "")
        heading_core = self.core_terms(heading)
        if heading_core:
            return LocalSubject(frozenset(heading_core), frozenset(), "heading")


        if chain_subject is not None and chain_subject.is_resolved:
            return LocalSubject(
                chain_subject.resolved_terms,
                chain_subject.resolved_entity_ids,
                "chain_context",
            )


        return LocalSubject(frozenset(), frozenset(), "none")

    def subject_matches(
        self, local_subject: LocalSubject, need: InformationNeed
    ) -> bool:
        """Correction B: exactly five allowed paths. Entity-type equality
        is never, by itself, one of them."""
        if not need.subject_entity_ids and not need.subject_terms:


            return True

        if local_subject.resolved_entity_ids & need.subject_entity_ids:
            return True

        if local_subject.resolved_entity_ids and need.subject_entity_ids:
            need_names = {
                self.entity_index[eid].normalized_name.lower()
                for eid in need.subject_entity_ids if eid in self.entity_index
            }
            local_names = {
                self.entity_index[eid].normalized_name.lower()
                for eid in local_subject.resolved_entity_ids
                if eid in self.entity_index
            }
            if need_names & local_names:
                return True

        for need_term in need.subject_terms:
            for local_term in local_subject.resolved_terms:
                if (
                    local_term in self.aliases.get(need_term, set())
                    or need_term in self.aliases.get(local_term, set())
                ):
                    return True


        non_generic_need_terms = {
            t for t in need.subject_terms if not self.is_generic_subject_term(t)
        }


        entity_grounded_terms = {
            term
            for eid in need.subject_entity_ids if eid in self.entity_index
            for term in (
                set(terms(self.entity_index[eid].canonical_name))
                | {self.entity_index[eid].normalized_name.lower()}
            )
        } & non_generic_need_terms
        if entity_grounded_terms:


            most_specific = min(
                len(self.term_headings.get(t, ())) for t in entity_grounded_terms
            )
            need_core = {
                t for t in entity_grounded_terms
                if len(self.term_headings.get(t, ())) == most_specific
            }
        else:
            need_core = non_generic_need_terms
        local_core = {
            t for t in local_subject.resolved_terms
            if not self.is_generic_subject_term(t)
        }
        if need_core & local_core:
            need_types = {
                self.entity_index[e].entity_type for e in need.subject_entity_ids
                if e in self.entity_index
            }
            local_types = {
                self.entity_index[e].entity_type
                for e in local_subject.resolved_entity_ids
                if e in self.entity_index
            }
            if not need_types or not local_types or (need_types & local_types):
                return True


        return False

    def _non_illustrative_terms(
        self, paragraph: str, candidate_terms: set[str]
    ) -> set[str]:
        """Correction 28 / Defect 4: drop a candidate subject-enrichment
        term when every sentence of the paragraph that mentions it is
        itself an illustrative-example sentence ("For example, one kind
        of X..., while another kind of Y..."). A whole-section
        procedure extraction otherwise lets ANY paragraph mentioning the
        need's subject anywhere borrow it as the section's own subject,
        even when that paragraph is really a generic multi-example list
        that happens to name the subject once in passing - exactly the
        "accepted a generic section merely because lexical coverage was
        satisfied" failure class. A term that appears in at least one
        NON-illustrative sentence of the paragraph is kept unchanged -
        this never weakens the Correction 19 fix that lets a genuine
        condition/subject stated in ordinary paragraph body text (not an
        example list) still enrich the section's subject."""
        if not candidate_terms:
            return set()


        genuine: set[str] = set()
        for _, _, sentence in sentence_spans(paragraph):
            if _ILLUSTRATIVE_EXAMPLE_RE.search(sentence):
                continue
            genuine |= candidate_terms & set(terms(sentence))
        return genuine

    def ground_need_subjects(self, contract: QuestionContract) -> QuestionContract:
        """Correction 12: populate each need's subject once, from the
        question and the loaded corpus - never hand-authored."""

        def explicit_entities(text: str) -> set[str]:
            normalized = compact(text).casefold()
            text_roots = roots(normalized)
            matches: list[tuple[int, str]] = []
            for entity_id, info in self.entity_index.items():
                names = {
                    compact(info.canonical_name).casefold(),
                    compact(info.normalized_name).casefold(),
                }
                best = max(
                    (
                        name for name in names
                        if name and (
                            re.search(
                                rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])",
                                normalized,
                            )
                            or roots(name).issubset(text_roots)
                        )
                    ),
                    key=len,
                    default="",
                )
                if best:
                    matches.append((len(best), entity_id))
            if not matches:
                return set()
            longest = max(length for length, _ in matches)
            return {
                entity_id for length, entity_id in matches
                if length >= max(3, longest // 2)
            }

        grounded = []
        shared_entity_ids = explicit_entities(contract.question)
        shared_entity_terms: set[str] = set()
        for entity_id in shared_entity_ids:
            info = self.entity_index[entity_id]
            shared_entity_terms.update(self.core_terms(info.canonical_name))
            shared_entity_terms.update(self.core_terms(info.normalized_name))
        for need in contract.needs:
            focus = re.split(
                r"\s+Context:\s*", need.query, maxsplit=1, flags=re.IGNORECASE
            )[0]
            full_terms = self.core_terms(focus)
            entity_ids = explicit_entities(focus)
            candidate_terms: set[str] = set()
            for entity_id in entity_ids:
                info = self.entity_index[entity_id]
                candidate_terms.update(self.core_terms(info.canonical_name))
                candidate_terms.update(self.core_terms(info.normalized_name))
            passive_actions = {
                match.lower() for match in _PASSIVE_ACTION_VERB_RE.findall(focus)
            }
            if (
                not entity_ids and _ANAPHORA_RE.search(focus)
                and shared_entity_terms
            ):
                candidate_terms = set(shared_entity_terms)
                entity_ids = set(shared_entity_ids)
            elif not candidate_terms:
                candidate_terms = full_terms - passive_actions
            remaining_terms = (full_terms - candidate_terms) | passive_actions
            # In passive questions the participle is the requested action.
            # Keep surrounding nouns/adverbials as context instead of letting
            # one broad word satisfy operation matching by itself.
            relation_terms = passive_actions or remaining_terms
            context_terms = full_terms - relation_terms
            grounded.append(replace(
                need,
                subject_terms=frozenset(candidate_terms),
                subject_entity_ids=frozenset(entity_ids),
                operation_terms=frozenset(relation_terms),
                context_terms=frozenset(context_terms),
            ))
        grounded_shared = set().union(*(
            set(need.subject_terms) for need in grounded
        )) if grounded else set()
        return replace(
            contract,
            needs=tuple(grounded),
            shared_subject_terms=frozenset(grounded_shared),
        )

    def plan(self, question: str) -> QuestionContract:
        """Create generic retrieval needs without an LLM or domain rules,
        decomposing the question into semantic CLAUSES - never one item per
        content word - each mapped to an exact character span of the
        ORIGINAL question (Correction 6). There is no fixed maximum number
        of needs: every clause the question actually contains gets one."""
        lowered = question.lower()
        if re.search(r"\b(?:compare|contrast|differ|difference|versus|vs\.?)\b", lowered):
            answer_type = "comparison"
        elif _REASON_CUE_RE.search(lowered):
            answer_type = "reason"
        elif re.search(


            r"\b(?:calculat\w*|comput\w*|deriv\w*|formula|formulae|"
            r"multipl\w*|divid\w*|subtract\w*|add\s+up|sum\s+of|"
            r"percentage\s+of|ratio\s+of|how\s+many\s+times)\b",
            lowered,
        ):
            answer_type = "calculation"
        elif re.search(r"\bhow\b|\bsteps?\b|\bprocedure\b|\bmethod\b", lowered):
            answer_type = "procedure"
        else:
            answer_type = "fact"

        all_clauses = clause_spans(question)
        shared_subject_terms = self.core_terms(question)

        def condition_text(clause: str) -> str:


            match = _CONDITION_SPAN_RE.search(clause)
            if match:
                return compact(match.group(0).rstrip(",").rstrip())
            return compact(clause)

        conditions = tuple(dict.fromkeys(
            condition_text(text) for _, _, text in all_clauses
            if _CONDITION_CUE_RE.search(text)
        ))
        reasons = tuple(dict.fromkeys(
            compact(text) for _, _, text in all_clauses
            if _REASON_CUE_RE.search(text)
        ))
        comparison_sides: tuple[str, ...] = ()
        if answer_type == "comparison":
            sides = re.split(
                r"\s+(?:and|versus|vs\.?)\s+", question, flags=re.IGNORECASE
            )
            comparison_sides = tuple(
                compact(side) for side in sides
                if compact(side) and terms(side)
            )
        dimensions = tuple(dict.fromkeys(
            compact(text) for _, _, text in all_clauses
            if _DIMENSION_CUE_RE.search(text)
        ))
        quantities = tuple(dict.fromkeys(_QUANTITY_RE.findall(question)))
        required_ordering = answer_type == "procedure"


        complex_question = bool(
            re.search(r"[,;]", question)
            or answer_type == "comparison"
            or re.search(
                r"\band\s+(?:what|why|how|when|where|which|who|"
                r"should|must|can|[a-z]+(?:ed|ing))\b",
                question, re.IGNORECASE,
            )
        )
        clauses = all_clauses if complex_question else [
            (0, len(question), compact(question))
        ]

        needs: list[InformationNeed] = []
        if len(clauses) == 1:
            start, end, text = clauses[0]
            needs = [InformationNeed(
                "", question, tuple(sorted(shared_subject_terms)), question,
                answer_type,
                (RequestedItem(
                    "clause-0", text, (start, end),
                    frozenset(shared_subject_terms), "clause",
                ),),
            )]
        else:
            for index, (start, end, text) in enumerate(clauses):
                contextual_query = f"{text}. Context: {question}"
                needs.append(InformationNeed(
                    " ".join(text.split()[:7]),
                    contextual_query,
                    tuple(terms(text)),
                    contextual_query,
                    answer_type,
                    (RequestedItem(
                        f"clause-{index}", text, (start, end),
                        frozenset(terms(text)), "clause",
                    ),),
                ))


        finalized: list[InformationNeed] = []
        for need_index, need in enumerate(needs):
            need_id = f"need-{need_index}"
            items = []
            for item in need.requested_items:
                start, end = item.span
                clause_text = question[start:end]
                kind = item.kind
                if kind == "clause":
                    condition_match = _CONDITION_SPAN_RE.search(clause_text)
                    if condition_match and len(condition_match.group(0)) >= 0.5 * max(1, len(clause_text)):
                        kind = "condition"
                    elif answer_type == "comparison" and _DIMENSION_CUE_RE.search(clause_text):


                        kind = "dimension"
                    elif answer_type == "comparison" and any(
                        compact(clause_text) == side
                        or compact(clause_text) in side
                        or side in compact(clause_text)
                        for side in comparison_sides
                    ):
                        kind = "comparison_side"
                    elif _REASON_CUE_RE.search(clause_text):
                        kind = "reason"
                    elif _EXCEPTION_CUE_RE.search(clause_text):
                        kind = "exception"
                    elif _UNIT_CUE_RE.search(clause_text):
                        kind = "unit"
                    elif _OUTPUT_CUE_RE.search(clause_text):
                        kind = "output"
                items.append(replace(
                    item, need_id=need_id, kind=kind,
                    normalized=compact(item.description).lower(),
                ))
            finalized.append(replace(
                need, need_id=need_id, requested_items=tuple(items),
            ))
        needs = finalized


        for quantity in quantities:
            for need_index, need in enumerate(needs):
                item_spans = [item.span for item in need.requested_items]
                start = min((s for s, _ in item_spans), default=0)
                end = max((e for _, e in item_spans), default=len(question))
                q_start = question.find(quantity, start, end)
                if q_start < 0:
                    continue
                needs[need_index] = replace(need, requested_items=need.requested_items + (
                    RequestedItem(
                        f"quantity-{need.need_id}-{len(need.requested_items)}",
                        quantity, (q_start, q_start + len(quantity)),
                        frozenset(terms(quantity)), "quantity", need.need_id,
                        False, quantity.lower(),
                    ),
                ))
                break
        print(f"[PLAN] type={answer_type}; needs={[need.query for need in needs]}")
        contract = QuestionContract(
            question, answer_type, tuple(needs),
            frozenset(shared_subject_terms), conditions, reasons,
            comparison_sides, dimensions, quantities, required_ordering,
        )
        return self.ground_need_subjects(contract)

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

    def retrieve(
        self, need: InformationNeed, widen: int = 0, exhaustive: bool = False
    ) -> list[dict[str, Any]]:
        cache_key = (
            need.retrieval_query,
            tuple(sorted(need.subject_terms)),
            tuple(sorted(need.operation_terms)),
            widen,
            exhaustive,
        )
        cached = self.retrieval_cache.get(cache_key)
        if cached is not None:
            return [dict(row) for row in cached]
        self._retrievers()
        retrieval_query = self._expand_query(need.retrieval_query)
        query_vector = self.vectorizer.transform([retrieval_query])
        lexical = (self.lexical_matrix @ query_vector.T).toarray().ravel()
        lexical_order = np.argsort(-lexical)


        first_stage = (
            len(self.rows) if exhaustive
            else min(len(self.rows), TOP_FIRST_STAGE + widen * TOP_FIRST_STAGE)
        )
        selected = set(lexical_order[:first_stage].tolist())


        variants = self._morphological_variants(retrieval_query)
        if variants:
            variant_vector = self.vectorizer.transform([variants])
            variant_lexical = (self.lexical_matrix @ variant_vector.T).toarray().ravel()
            variant_order = np.argsort(-variant_lexical)
            selected.update(
                int(index) for index in variant_order[:first_stage]
                if variant_lexical[index] > 0
            )
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
            selected.update(dense_order[:first_stage].tolist())
        lexical_rank = {
            int(index): rank for rank, index in enumerate(lexical_order, 1)
        }
        dense_rank = {
            int(index): rank for rank, index in enumerate(dense_order, 1)
        }
        vocabulary = self.vectorizer.vocabulary_
        query_terms = [
            term for term in terms(retrieval_query)
            if term in vocabulary and term not in GENERIC_QUERY_WORDS
        ]
        rare_terms = sorted(
            query_terms,
            key=lambda term: self.vectorizer.idf_[vocabulary[term]],
            reverse=True,
        )[:3]
        coverage = {}
        subject_coverage = {}
        operation_coverage = {}
        subject_roots = roots(" ".join(need.subject_terms))
        operation_roots = roots(" ".join(need.operation_terms))
        for index in selected:
            passage_terms = set(terms(self.rows[index]["text"]))
            passage_roots = roots(self.rows[index]["text"])
            coverage[index] = (
                sum(term in passage_terms for term in rare_terms)
                / max(1, len(rare_terms))
            )
            subject_coverage[index] = (
                len(subject_roots & passage_roots) / len(subject_roots)
                if subject_roots else 0.0
            )
            operation_coverage[index] = (
                sum(
                    terms_overlap_morphologically(
                        {term}, passage_terms
                    )
                    for term in need.operation_terms
                ) / len(need.operation_terms)
                if need.operation_terms else 0.0
            )


        # `TOP_RERANK` existed as configuration but was never consumed: the
        # CrossEncoder was receiving the entire lexical+dense+variant union
        # (and all 767 rows during exhaustive fallback).  Rank that union
        # cheaply first, then apply the expensive CrossEncoder to the actual
        # rerank pool.  Exhaustive still scores every corpus row in the first
        # stage; it does not mean cross-encoding every row repeatedly.
        def preliminary_score(index: int) -> float:
            rrf = 1.0 / (60 + lexical_rank.get(index, 10000))
            if dense_order.size:
                rrf += 1.0 / (60 + dense_rank.get(index, 10000))
            return (
                1.5 * coverage.get(index, 0.0)
                + 1.5 * subject_coverage.get(index, 0.0)
                + 1.5 * operation_coverage.get(index, 0.0)
                + 8.0 * rrf
                + float(lexical[index])
                + 0.25 * float(dense_scores[index])
            )

        rerank_limit = min(len(selected), TOP_RERANK)
        selected = set(sorted(
            selected, key=preliminary_score, reverse=True,
        )[:rerank_limit])
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
                + 1.5 * subject_coverage.get(index, 0.0)
                + 1.5 * operation_coverage.get(index, 0.0)
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


        positive_ranked = [row for row in ranked if row["score"] >= 0]
        if not positive_ranked:
            print(
                f"[RETRIEVE] need={need.label!r}: no candidate cleared the "
                f"zero-score floor (best={ranked[0]['score'] if ranked else None})"
            )
            return []


        if exhaustive:
            anchor_count = min(len(positive_ranked), 32)
        else:
            anchor_count = min(len(positive_ranked), 6 + widen * 4)
        result = self.expand(need, positive_ranked[:anchor_count])
        self.retrieval_cache[cache_key] = tuple(dict(row) for row in result)
        return result

    def expand(
        self, need: InformationNeed, anchors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add adjacent chunks generically so boundary text is not lost."""
        requirement_terms: set[str] = set()
        for requirement in need.requirements:
            requirement_terms.update(terms(requirement))

        def on_topic(text: str) -> bool:


            return not requirement_terms or bool(
                requirement_terms.intersection(terms(text))
            )

        result: dict[str, dict[str, Any]] = {}
        for rank, anchor in enumerate(anchors):
            position = self.positions[anchor["chunk_id"]]
            for distance in NEIGHBOR_WINDOW:
                neighbour_position = position + distance
                if not 0 <= neighbour_position < len(self.rows):
                    continue
                row = self.rows[neighbour_position]
                page_distance = abs(
                    (row.get("pdf_page") or 0)
                    - (anchor.get("pdf_page") or 0)
                )


                if page_distance > max(1, abs(distance)):
                    continue
                if distance != 0 and not on_topic(row["text"]):
                    continue
                score = anchor["score"] - abs(distance) * 0.35 - rank * 0.01
                old = result.get(row["chunk_id"])
                if old is None or score > old["score"]:
                    result[row["chunk_id"]] = {**row, "score": score}
            query_tokens = set(terms(need.query))
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
                    if not on_topic(row["text"]):
                        continue
                    score = anchor["score"] - 1.50 - rank * 0.01
                    old = result.get(row["chunk_id"])
                    if old is None or score > old["score"]:
                        result[row["chunk_id"]] = {**row, "score": score}
        return sorted(
            result.values(), key=lambda row: row["score"], reverse=True
        )

    def evidence(
        self, contract: QuestionContract, groups: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:


        if contract.answer_type == "procedure":
            reordered_groups = []
            for need_index, candidates in enumerate(groups):
                need = contract.needs[need_index] if need_index < len(contract.needs) else None
                subject_terms = need.subject_terms if need is not None else frozenset()
                if not subject_terms:
                    reordered_groups.append(candidates)
                    continue
                on_subject = []
                off_subject = []
                for row in candidates:
                    row_terms = set(terms(row.get("text") or ""))
                    (on_subject if subject_terms & row_terms else off_subject).append(row)
                reordered_groups.append(on_subject + off_subject)
            groups = reordered_groups
        chosen: dict[str, dict[str, Any]] = {}


        unbounded = contract.answer_type in ("procedure", "calculation")
        evidence_cap = None if unbounded else MAX_EVIDENCE
        quota = (
            None if unbounded
            else max(1, MAX_EVIDENCE // max(1, len(contract.needs)))
        )
        for need_index, candidates in enumerate(groups):
            for row in candidates[:quota]:
                item = chosen.setdefault(row["chunk_id"], dict(row))
                item.setdefault("needs", []).append(need_index)
                if evidence_cap is not None and len(chosen) >= evidence_cap:
                    break


        pool: list[tuple[float, dict[str, Any]]] = []
        for candidates in groups:
            if not candidates:
                continue
            raw_scores = [float(row["score"]) for row in candidates]
            low, high = min(raw_scores), max(raw_scores)
            spread = high - low
            for row, raw_score in zip(candidates, raw_scores):
                normalized = (
                    (raw_score - low) / spread if spread > 1e-9 else 1.0
                )
                pool.append((normalized, row))
        pool.sort(key=lambda item: item[0], reverse=True)
        if contract.answer_type == "procedure":


            def continuity_gap(row: dict[str, Any]) -> int:
                page = row.get("pdf_page")
                index = row.get("chunk_index") or 0
                gaps = [
                    abs(index - (existing.get("chunk_index") or 0))
                    for existing in chosen.values()
                    if existing.get("pdf_page") == page
                ]
                return min(gaps) if gaps else 1000
            pool.sort(key=lambda item: (continuity_gap(item[1]), -item[0]))
        for _, row in pool:
            chosen.setdefault(row["chunk_id"], dict(row))
            if evidence_cap is not None and len(chosen) >= evidence_cap:
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


        after_blank = index < len(lines) - 1 and not compact(lines[index + 1])
        return (
            numbered_section or (before_blank and after_blank)
        ) and bool(re.search(r"[A-Za-z]", line))


    def _build_candidate_regions(
        self, need: InformationNeed, evidence: list[dict[str, Any]]
    ) -> list[CandidateRegion]:
        """Group retrieved rows into contiguous document-order windows
        before any individual sentence is chosen from them."""
        ordered = sorted(evidence, key=lambda row: self.positions[row["chunk_id"]])
        max_gap = max(NEIGHBOR_WINDOW) if NEIGHBOR_WINDOW else 1
        regions: list[CandidateRegion] = []
        current: list[dict[str, Any]] = []

        def flush() -> None:
            if not current:
                return
            chunk_ids = tuple(row["chunk_id"] for row in current)
            start = self.positions[current[0]["chunk_id"]]
            end = self.positions[current[-1]["chunk_id"]]
            heading = self.chunk_heading.get(current[0]["chunk_id"], "")
            score = float(np.mean([float(row.get("score", 0.0)) for row in current]))
            regions.append(CandidateRegion(
                f"region-{len(regions)}", need.label, chunk_ids, start, end,
                heading, score,
            ))

        previous_position: int | None = None
        for row in ordered:
            position = self.positions[row["chunk_id"]]
            if previous_position is not None and position - previous_position > max_gap:
                flush()
                current = []
            current.append(row)
            previous_position = position
        flush()
        return regions

    def _select_authoritative_region(
        self, need: InformationNeed, regions: list[CandidateRegion]
    ) -> tuple[CandidateRegion, AuthoritativeScope] | None:
        """Correction 4: choose the authoritative region using subject,
        heading, and score - an adjacent region whose own heading names a
        different, specific (non-generic) subject is rejected even when it
        scores higher, in favour of one whose heading is compatible."""
        if not regions:
            return None

        def region_subject_ok(region: CandidateRegion) -> bool:
            heading_core = self.core_terms(region.heading)
            local_subject = (
                LocalSubject(frozenset(heading_core), frozenset(), "heading")
                if heading_core
                else LocalSubject(frozenset(), frozenset(), "none")
            )
            if self.subject_matches(local_subject, need):
                return True


            region_terms: set[str] = set()
            for chunk_id in region.chunk_ids:
                position = self.positions.get(chunk_id)
                if position is not None:
                    region_terms |= set(terms(self.rows[position].get("text", "")))
            body_subject = LocalSubject(frozenset(region_terms), frozenset(), "body")
            return self.subject_matches(body_subject, need)

        matching = [region for region in regions if region_subject_ok(region)]
        pool = matching if matching else regions
        passages = []
        subject_roots = roots(" ".join(need.subject_terms))
        operation_roots = roots(" ".join(need.operation_terms))
        for region in pool:
            rows = [
                self.rows[self.positions[chunk_id]]["text"]
                for chunk_id in region.chunk_ids
                if chunk_id in self.positions
            ]
            focused_units: list[str] = []
            for row_text in rows:
                for _, _, unit in sentence_spans(row_text):
                    unit_roots = roots(unit)
                    subject_ok = not subject_roots or bool(subject_roots & unit_roots)
                    operation_ok = (
                        not operation_roots
                        or terms_covered_morphologically(
                            need.operation_terms, set(terms(unit))
                        )
                    )
                    if subject_ok and operation_ok:
                        focused_units.append(unit)
            focused = " ".join(focused_units[:6])
            if not focused:
                focused = " ".join(rows)[:1800]
            passages.append(compact(f"{region.heading}. {focused}")[:2400])
        semantic = np.asarray(self.reranker.predict(
            [[need.query, passage] for passage in passages],
            show_progress_bar=False,
        )).reshape(-1)
        subject_terms = {
            term for term in need.subject_terms
            if not self.is_generic_subject_term(term)
        }
        operation_terms = set(need.operation_terms)

        def region_rank(index: int) -> float:
            passage_terms = set(terms(passages[index]))
            heading_terms = self.core_terms(pool[index].heading)
            subject_coverage = (
                len(subject_terms & passage_terms) / len(subject_terms)
                if subject_terms else 0.0
            )
            operation_coverage = (
                sum(
                    terms_overlap_morphologically(
                        {term}, set(terms(passages[index]))
                    )
                    for term in operation_terms
                ) / max(1, len(operation_terms))
            )
            heading_subject_coverage = (
                len(heading_terms & subject_terms) / len(subject_terms)
                if subject_terms else 0.0
            )
            heading_operation_coverage = (
                sum(
                    terms_overlap_morphologically({term}, set(heading_terms))
                    for term in operation_terms
                ) / max(1, len(operation_terms))
            )
            requested_terms = subject_terms | operation_terms
            heading_precision = (
                len(heading_terms & requested_terms) / len(heading_terms)
                if heading_terms else 0.0
            )
            return (
                float(semantic[index])
                + 3.0 * subject_coverage
                + 2.0 * operation_coverage
                + 2.0 * heading_subject_coverage
                + 2.0 * heading_operation_coverage
                + 1.5 * heading_precision
                + 0.10 * pool[index].score
            )

        best_index = max(range(len(pool)), key=region_rank)
        best = pool[best_index]
        scope = AuthoritativeScope(
            f"scope-{best.region_id}", best.region_id, best.heading,
            frozenset(self.core_terms(best.heading)), best.start_position,
            best.end_position,
        )
        return best, scope


    @staticmethod
    def _classify_obligation_kind(text: str) -> str | None:
        if _STEP_RE.match(text) or _ROMAN_STEP_RE.match(text):
            return "step"
        if _LETTER_STEP_RE.match(text):
            return "step"
        if _DASH_BULLET_RE.match(text):
            return "bullet"
        if _WARNING_RE.match(text):
            return "warning"
        if re.search(
            r"\b(?:must|should)\s+not\b|^\s*(?:never|do\s+not)\b",
            text, re.IGNORECASE,
        ):
            return "warning"
        if _CONDITION_BRANCH_RE.search(text):
            return "condition_branch"
        if _THRESHOLD_CUE_RE.search(text):


            return "threshold"
        if _FORMULA_RE.search(text):
            return "formula"
        if _OUTPUT_CUE_RE.search(text):


            return "output"
        if _TABLE_ROW_RE.search(text):
            return "table_row"
        words = text.split()
        first_word = words[0].lower().strip(".,;:()") if words else ""
        if (
            len(words) >= 3
            and first_word
            and first_word not in NON_IMPERATIVE_OPENERS
            and first_word not in STOPWORDS
            and text[:1].isupper()
            and not first_word.endswith("s")
            and not re.match(r"^(?:is|are|was|were|has|have|had)$", first_word)
        ):


            rest_words = [w for w in words[1:] if any(c.isalpha() for c in w)]
            lowercase_rest = sum(1 for w in rest_words if w[:1].islower())
            title_like = bool(rest_words) and lowercase_rest < 0.5 * len(rest_words)
            no_terminal_punctuation = not text.rstrip().endswith((".", "!", "?", ":"))
            if title_like and no_terminal_punctuation:
                return None
            return "step"
        return None


    _ANSWER_TYPE_OBLIGATION_KINDS: dict[str, frozenset[str]] = {
        "fact": frozenset(),
        "reason": frozenset({"condition_branch"}),
        "procedure": frozenset({"step", "bullet", "condition_branch", "warning"}),
        "comparison": frozenset({"comparison_dimension"}),
        "calculation": frozenset({
            "condition_branch", "formula", "table_row", "threshold", "output",
        }),
    }

    def _operation_entailed(
        self, item_terms: frozenset[str], subject_terms: frozenset[str],
        text: str, operation_terms: frozenset[str] = frozenset(),
    ) -> bool:
        """Correction 45 / item 1 (generalized relation/operation
        entailment): the single shared test for whether a candidate
        sentence actually entails the operation/relation a RequestedItem
        asks for, not merely its subject - used by BOTH
        `_obligation_required` and `accept()`'s per-item coverage check,
        so the same rule applies regardless of obligation kind or answer
        type, never limited to condition_branch alone. Strips the need's
        own grounded SUBJECT vocabulary (`subject_terms`) from the item's
        full term set: whenever the item asks for something beyond its
        own subject, that operation/relation content - not the bare
        subject - must be morphologically present in the text, so subject
        overlap alone can never satisfy an operation/relation item. A
        clause that is nothing but its own subject (e.g. a bare "What is
        X?") falls back to requiring the subject itself, since there is
        nothing more specific to ask for. Purely generic, question-
        derived vocabulary and morphological matching - never a
        corpus-specific word."""
        check_terms = operation_terms or (item_terms - subject_terms) or item_terms
        if not check_terms:
            return True
        return terms_covered_morphologically(check_terms, set(terms(text)))

    def _obligation_required(
        self, kind: str, answer_type: str, text: str, need: InformationNeed,
    ) -> bool:
        """The answer-type-specific obligation policy, additionally
        consuming the need's own genuinely-populated RequestedItem.kind
        (Correction 16): a need that explicitly requested a CONDITION
        (kind="condition") must still resolve any condition_branch
        obligation it discovers even for an otherwise non-procedural
        answer type, since the condition itself was asked for, not merely
        encountered in passing.

        Correction 45 / item 1 (generalized relation/operation
        entailment): the operation/relation check Correction 41 applied
        only inside the condition_branch branch below now gates EVERY
        obligation kind alike, via the shared `_operation_entailed`
        helper - an obligation sentence that is merely on-subject is not
        enough; whenever the need has a condition-kind RequestedItem, any
        discovered obligation (step, bullet, warning, threshold, formula,
        output, table_row, or condition_branch alike) must also entail
        that SAME requested operation/relation before it can count as
        required, not merely some other instruction that happens to apply
        under the same condition (e.g. a staining instruction is not an
        answer to how a specimen is SENT to the laboratory)."""
        condition_items = [
            item for item in need.requested_items if item.kind == "condition"
        ]
        if kind == "bullet" and _LIST_REQUEST_RE.search(need.query):
            return True
        if kind == "condition_branch" and condition_items:
            if re.match(r"^\s*notes?\s*[:\-]", text, re.IGNORECASE):
                return False
            return True
        allowed = self._ANSWER_TYPE_OBLIGATION_KINDS.get(
            answer_type, self._ANSWER_TYPE_OBLIGATION_KINDS["fact"]
        )
        if kind not in allowed:
            return False
        if kind == "warning" and re.match(r"^\s*notes?\s*[:\-]", text, re.IGNORECASE):
            return False
        return True

    @staticmethod
    def _obligation_key(kind: str, chunk_id: str, start_char: int) -> str:


        return f"{kind}:{chunk_id}:{start_char}"

    def _discover_comparison_dimension_obligations(
        self, rows: list[dict[str, Any]], contract: QuestionContract
    ) -> list[DiscoveredObligation]:
        """Correction 6/11: a comparison's declared dimensions must each
        actually be found under a matching heading within the confirmed
        scope. Kept separate from per-sentence step/bullet/warning/etc.
        discovery (done inline, per-need, only for sentences already
        confirmed on-subject - see the sentence-scan loop below) because a
        dimension obligation is anchored to a HEADING match, not a
        sentence shape, and is not gated by any one need's own subject."""
        obligations: list[DiscoveredObligation] = []
        dimension_terms = self.core_terms(" ".join(contract.dimensions))
        if contract.answer_type != "comparison" or not dimension_terms:
            return obligations
        for row in rows:
            heading = self.chunk_heading.get(row["chunk_id"], "")
            if dimension_terms & self.core_terms(heading):
                obligations.append(DiscoveredObligation(
                    self._obligation_key("comparison_dimension", row["chunk_id"], 0),
                    heading, frozenset(terms(heading)), row["chunk_id"],
                    "comparison_dimension", True,
                ))
        return obligations

    @staticmethod
    def _classify_role(
        text: str, obligation_kind: str | None
    ) -> Literal["primary", "required_conditional", "optional_background"]:
        """Correction 9/10: only primary/required-conditional evidence may
        reach the generator by default."""
        if obligation_kind in ("warning", "condition_branch"):
            return "required_conditional"
        if _CONDITION_BRANCH_RE.search(text) or _WARNING_RE.match(text):
            return "required_conditional"
        if re.search(
            r"\bfor example\b|\bin general\b|\bhistorically\b|\bbackground\b",
            text, re.IGNORECASE,
        ):
            return "optional_background"
        return "primary"

    def _best_section(
        self, need: InformationNeed, evidence: list[dict[str, Any]]
    ) -> tuple[float, list[tuple[int, int, str]], str] | None:
        section_query = self._expand_query(need.query)
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
        operation_matches = np.asarray([
            not need.operation_terms or terms_covered_morphologically(
                need.operation_terms, set(terms(passage))
            )
            for passage in heading_passages
        ], dtype=bool)
        query_roots = roots(section_query)
        lexical_scores = np.asarray([
            len(query_roots.intersection(roots(heading)))
            / max(1, len(roots(heading)))
            for _, heading in headings
        ])
        subject_roots = roots(" ".join(need.subject_terms))
        subject_matches = np.asarray([
            not subject_roots or bool(subject_roots & roots(passage))
            for passage in heading_passages
        ], dtype=bool)
        eligible = operation_matches & subject_matches
        if not eligible.any():
            eligible = operation_matches if operation_matches.any() else subject_matches
        if not eligible.any():
            eligible = np.ones(len(headings), dtype=bool)
        eligible_indices = np.flatnonzero(eligible)
        semantic_scores = np.full(len(headings), -np.inf, dtype=float)
        predicted = np.asarray(self.reranker.predict(
            [[section_query, heading_passages[int(index)]]
             for index in eligible_indices],
            show_progress_bar=False,
        )).reshape(-1)
        semantic_scores[eligible_indices] = predicted
        combined_scores = semantic_scores + 4.0 * lexical_scores
        combined_scores = np.where(eligible, combined_scores, -np.inf)
        if need.kind == "procedure" and re.search(
            r"\b(?:how|steps?|procedure|method)\b", need.query,
            re.IGNORECASE,
        ):
            structure_scores = np.asarray([
                min(3, count) for count in heading_step_counts
            ], dtype=float)
            combined_scores += structure_scores


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
        nested_heading_positions: set[int] = set()
        for position, following_heading in headings[best_at + 1:]:
            if position <= start:
                continue
            heading_core = self.core_terms(following_heading)
            heading_subject = LocalSubject(
                frozenset(heading_core), frozenset(), "heading"
            )
            continues_subject = self.subject_matches(heading_subject, need)
            method_subheading = bool(re.match(
                r"^(?:using|boiling|heating|cooling|mixing|adding|removing|"
                r"washing|cleaning|staining|counting|calculating|preparing)\b",
                following_heading, re.IGNORECASE,
            ))
            if continues_subject or method_subheading:
                nested_heading_positions.add(position)
                continue
            if (
                following_heading.rstrip().endswith(":")
                and self._classify_obligation_kind(following_heading) is not None
            ):
                nested_heading_positions.add(position)
                continue
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

        for record_position in range(start, end):
            evidence_index, _, raw_line = records[record_position]
            if record_position in nested_heading_positions:
                flush()
                continue
            line = compact(raw_line)
            line = _TRAILING_FIGURE_TEXT_RE.sub("", line).strip()
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
            paragraph = re.sub(r"^\s*â\s+", "— ", paragraph)


            paragraph = _strip_interleaved_captions(paragraph)
            if re.match(r"^\d+\s+Manual\b", paragraph, re.IGNORECASE):
                continue
            if (
                re.match(r"^[a-z]", paragraph)
                and len(paragraph) < 160
                and not headings[best_at][1].rstrip().endswith(":")
            ):
                continue
            if re.search(
                r"\b(?:and|or|the|of|to|with|in|for)$", paragraph,
                re.IGNORECASE,
            ):
                continue


            if (
                self._classify_obligation_kind(paragraph) is None
                and (
                    _STANDALONE_CAPTION_RE.match(paragraph)
                    or _PAGE_HEADER_RE.match(paragraph)
                )
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


        selected_heading = headings[best_at][1]
        if (
            selected_heading.rstrip().endswith(":")
            and (
                not need.operation_terms
                or terms_covered_morphologically(
                    need.operation_terms, set(terms(selected_heading))
                )
            )
        ):
            heading_source = records[headings[best_at][0]][0]
            paragraphs.insert(0, (heading_source, -1, selected_heading))
        return best_score, paragraphs, selected_heading

    def _narrow_authoritative_rows(
        self, need: InformationNeed, rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Correction 4/6: a CandidateRegion is only a retrieval-level
        grouping of position-adjacent chunks and can still span more than
        one heading/subject (e.g. several distinct specimen-collection
        subsections that happen to sit next to each other in the
        document). The "confirmed authoritative source scope" that
        obligation discovery and evidence extraction must read from is the
        subset of the region whose OWN heading is subject-compatible with
        the need - an adjacent chunk under a different, specific heading
        is excluded even though it was close enough in document order to
        enter the same region. Rows with no detectable heading of their
        own are kept (their subject is judged later, per-sentence)."""
        def row_heading_ok(row: dict[str, Any]) -> bool:
            heading = self.chunk_heading.get(row["chunk_id"], "")
            heading_core = self.core_terms(heading)
            if not heading_core:
                return True
            local_subject = LocalSubject(
                frozenset(heading_core), frozenset(), "heading"
            )
            return self.subject_matches(local_subject, need)

        matching = [row for row in rows if row_heading_ok(row)]
        return matching if matching else rows

    def _chain_is_coherent(
        self, chain: EvidenceChain
    ) -> tuple[bool, str | None]:
        """Correction 35 / Defect 6: each unit in a DISTRIBUTED chain
        (pulled from separate, non-adjacent chunks) already passed
        `subject_matches()` against the need INDIVIDUALLY at accept()
        time - but nothing about several units each independently, and
        separately, matching the need's subject guarantees the units are
        subject-consistent WITH EACH OTHER, genuinely in document order,
        or add up to anything more than a collection of merely topically
        adjacent fragments stitched into one incoherent answer (e.g. a
        table-of-contents line, an unrelated equipment-layout caption, and
        a genuine procedure sentence, each vaguely "on topic" on its own).
        Require, for a multi-unit distributed chain: the source positions
        are non-decreasing (a genuine ordered run, never a shuffled
        grab-bag), and at least one non-generic discriminative term
        (`is_generic_subject_term`, already corpus-wide and never a new
        word) is shared by EVERY unit's own local subject, not just each
        one individually against the need. A CONTIGUOUS chain (already
        positionally adjacent - the strongest coherence signal there is)
        or a single-unit chain is exempt; this only tightens the weaker,
        riskier distributed case."""
        if chain.chain_type != "distributed" or len(chain.units) <= 1:
            return True, None
        positions = [self.positions[unit.chunk_id] for unit in chain.units]
        if any(b < a for a, b in zip(positions, positions[1:])):
            return False, "distributed evidence units are not in document order"
        term_sets = [
            {t for t in subject.resolved_terms if not self.is_generic_subject_term(t)}
            for subject in chain.local_subjects
        ]
        if any(not terms for terms in term_sets):
            return False, "a distributed evidence unit has no discriminative subject of its own"
        if not set.intersection(*term_sets):
            return False, "distributed evidence units share no common discriminative subject"
        return True, None

    def _assemble_chain(
        self,
        need: InformationNeed,
        units: list[EvidenceUnit],
        scope: AuthoritativeScope,
        coverage_map: CoverageMap,
    ) -> EvidenceChain:
        """Correction 7: materialize the actual, inspectable EvidenceChain
        an answer's citations are drawn from - not a loose dict of scores."""
        ordered_units = tuple(sorted(
            units, key=lambda unit: (unit.evidence_index, unit.start_char)
        ))
        source_chunk_ids = tuple(dict.fromkeys(
            unit.chunk_id for unit in ordered_units
        ))
        positions = [self.positions[chunk_id] for chunk_id in source_chunk_ids]
        max_gap = max(NEIGHBOR_WINDOW) if NEIGHBOR_WINDOW else 1
        contiguous = all(
            b - a <= max_gap for a, b in zip(positions, positions[1:])
        ) if len(positions) > 1 else True
        covered_item_ids = frozenset().union(
            *(unit.covered_item_ids for unit in ordered_units)
        ) if ordered_units else frozenset()
        covered_obligation_ids = frozenset().union(
            *(unit.covered_obligation_ids for unit in ordered_units)
        ) if ordered_units else frozenset()
        remaining_required = [
            key for key in coverage_map.required_keys()
            if coverage_map.entries[key].state not in ("supported", "verified")
        ]
        validated = coverage_map.all_supported()
        chain = EvidenceChain(
            chain_id=f"chain-{need.label or 'need'}-{scope.scope_id}",
            need_label=need.label,
            units=ordered_units,
            chain_type="contiguous" if contiguous else "distributed",
            source_chunk_ids=source_chunk_ids,
            region_id=scope.region_id,
            scope_id=scope.scope_id,
            local_subjects=tuple(unit.local_subject for unit in ordered_units),
            covered_item_ids=covered_item_ids,
            covered_obligation_ids=covered_obligation_ids,
            continuation_state="open" if remaining_required else "closed",
            validation_status="validated" if validated else "rejected",
            rejection_reason=None if validated else f"unresolved: {remaining_required}",
        )
        if validated:


            coherent, reason = self._chain_is_coherent(chain)
            if not coherent:
                chain = replace(
                    chain, validation_status="rejected", rejection_reason=reason,
                )
        return chain

    def _extract_need_evidence(
        self,
        need: InformationNeed,
        evidence: list[dict[str, Any]],
        contract: QuestionContract,
        per_need_limit: int,
    ) -> tuple[CoverageMap, list[EvidenceUnit], AuthoritativeScope | None]:
        """Core Correction 4/6/8 pipeline for one need, shared by the
        extractive path and the generator's precondition/context-building
        path: build CandidateRegions, select the authoritative scope,
        discover its obligations, extract subject-verified EvidenceUnits (a
        whole section for a genuine procedure, per-sentence otherwise), and
        - only via the strongest identity signal - recover distributed
        evidence when the primary scope still leaves something required
        uncovered."""
        coverage_map = CoverageMap(need)
        regions = self._build_candidate_regions(need, evidence)
        selection = self._select_authoritative_region(need, regions)
        if selection is None:
            return coverage_map, [], None
        region, scope = selection
        region_rows_indexed = [
            (index, row) for index, row in enumerate(evidence, 1)
            if row["chunk_id"] in region.chunk_ids
        ]
        authoritative_rows = self._narrow_authoritative_rows(
            need, [row for _, row in region_rows_indexed]
        )
        authoritative_chunk_ids = frozenset(
            row["chunk_id"] for row in authoritative_rows
        )
        scope_rows_indexed = [
            (index, row) for index, row in region_rows_indexed
            if row["chunk_id"] in authoritative_chunk_ids
        ]
        scope_rows = [row for _, row in scope_rows_indexed]
        if contract.answer_type in ("procedure", "calculation"):
            scope_rows_indexed = list(enumerate(evidence, 1))
            scope_rows = [row for _, row in scope_rows_indexed]
        if scope_rows and len(authoritative_chunk_ids) < len(region.chunk_ids):


            narrowed_positions = [
                self.positions[row["chunk_id"]] for row in scope_rows
            ]
            scope = replace(
                scope,
                start_position=min(narrowed_positions),
                end_position=max(narrowed_positions),
            )

        for item in need.requested_items:
            coverage_map.mark(item.key, "candidate")

        accepted: list[EvidenceUnit] = []
        scope_semantically_matched = False


        need_focus = re.split(
            r"\s+Context:\s*", need.query, maxsplit=1, flags=re.IGNORECASE,
        )[0]
        is_reason_need = bool(_REASON_CUE_RE.search(need_focus))

        def accept(
            evidence_index: int, unit_index: int, chunk_id: str,
            start_char: int, end_char: int, unit_text: str,
            row_score: float, unit_score: float,
            local_subject: LocalSubject, kind: str | None, role: str,
            obligation_keys: frozenset[str] | None = None,
        ) -> None:

            if (
                _TEMPORAL_REQUEST_RE.search(need_focus)
                and not _TEMPORAL_ANSWER_RE.search(unit_text)
            ):
                return


            covered_items = frozenset(
                item.key for item in need.requested_items
                if not item.terms or (
                    (
                        terms_overlap_morphologically(
                            item.terms, set(terms(unit_text)),
                        )
                        if is_reason_need
                        else (
                            bool(
                                _LIST_REQUEST_RE.search(need.query)
                                and _LIST_REQUEST_RE.search(unit_text)
                            )
                            or self._operation_entailed(
                                item.terms, need.subject_terms, unit_text,
                                need.operation_terms,
                            )
                        )
                    )
                    and (not is_reason_need or _CAUSAL_ENTAILMENT_RE.search(unit_text))
                )
            )
            if obligation_keys is not None:


                covered_obligations = frozenset(
                    key for key in obligation_keys if key in coverage_map.obligations
                )
            else:
                covered_obligations = frozenset()
                if kind is not None:
                    obligation_key = self._obligation_key(kind, chunk_id, start_char)
                    if obligation_key in coverage_map.obligations:
                        covered_obligations = frozenset({obligation_key})
            required_obligations = frozenset(
                key for key in covered_obligations
                if coverage_map.obligations[key].required
            )
            if (
                not covered_items
                and required_obligations
                and contract.answer_type in ("procedure", "calculation")
                and (
                    scope_semantically_matched
                    or (
                        need.operation_terms
                        and terms_covered_morphologically(
                            need.operation_terms,
                            set(terms(scope.heading)),
                        )
                    )
                )
            ):
                covered_items = frozenset(
                    item.key for item in need.requested_items if item.required
                )
            if not covered_items and not required_obligations:
                return
            for item_key in covered_items:
                coverage_map.mark(item_key, "supported", evidence_index)
            for obligation_key in covered_obligations:
                coverage_map.mark(obligation_key, "supported", evidence_index)
            accepted.append(EvidenceUnit(
                evidence_index, unit_index, chunk_id, start_char, end_char,
                unit_text, local_subject, row_score, unit_score, role,
                covered_items, covered_obligations,
            ))


        # Planning already classified the question. Do not maintain a second
        # verb whitelist here: it silently diverted valid procedures whose
        # action verb was absent from that list into sentence-only extraction.
        procedural_need = contract.answer_type in ("procedure", "calculation")
        section_used = False
        if procedural_need and scope_rows:
            section = self._best_section(need, scope_rows)
            if section is not None:
                section_score, paragraphs, selected_heading = section
                scope_semantically_matched = section_score >= MIN_EXTRACT_SCORE
                scope = replace(
                    scope,
                    heading=selected_heading,
                    subject_terms=frozenset(self.core_terms(selected_heading)),
                )
                section_chars = sum(len(item[2]) for item in paragraphs)
                if section_chars <= 6000:
                    print(
                        f"[SECTION] need={need.label!r}; "
                        f"score={section_score:.3f}; "
                        f"paragraphs={len(paragraphs)}"
                    )


                    heading_core = self.core_terms(selected_heading)
                    # A section heading may deliberately use a generic head
                    # noun while its body names the precise subject (for
                    # example, a generic list lead-in followed by subject-
                    # specific bullets).  Ground the scope from the heading
                    # plus only question-subject terms actually present in
                    # the selected section.  This remains question-derived
                    # and scope-local; it does not borrow entities from other
                    # retrieved chunks.
                    section_text_terms = set(terms(" ".join(
                        paragraph for _, _, paragraph in paragraphs
                    )))
                    scoped_subject_terms = heading_core | {
                        subject_term for subject_term in need.subject_terms
                        if terms_overlap_morphologically(
                            {subject_term}, section_text_terms
                        )
                    }
                    section_subject = (
                        LocalSubject(
                            frozenset(scoped_subject_terms), frozenset(), "scope"
                        )
                        if scoped_subject_terms
                        else LocalSubject(frozenset(), frozenset(), "none")
                    )


                    chain_subject: LocalSubject | None = section_subject
                    for local_index, paragraph_index, paragraph in paragraphs:
                        if local_index - 1 >= len(scope_rows_indexed):
                            continue
                        global_index, row = scope_rows_indexed[local_index - 1]


                        own_terms = self._non_illustrative_terms(
                            paragraph, self.core_terms(paragraph),
                        )
                        matching_own_terms = own_terms & set(need.subject_terms)
                        subject_types = {
                            self.entity_index[entity_id].entity_type
                            for entity_id in need.subject_entity_ids
                            if entity_id in self.entity_index
                        }
                        paragraph_lower = paragraph.casefold()
                        competing_mentions = [
                            mention for mention in row.get("mentions", [])
                            if mention.entity_type in subject_types
                            and compact(mention.mention_text).casefold() in paragraph_lower
                            and not (
                                set(terms(mention.mention_text)) & set(need.subject_terms)
                                or mention.entity_id in need.subject_entity_ids
                            )
                        ]
                        if matching_own_terms:
                            own_subject = LocalSubject(
                                frozenset(matching_own_terms), frozenset(), "sentence",
                            )
                            # A generic word from the lead-in must not replace
                            # the more specific subject already grounded from
                            # the selected scope.  Update the chain only when
                            # the paragraph's own subject independently matches
                            # the requested subject.
                            if self.subject_matches(own_subject, need):
                                paragraph_subject = own_subject
                                chain_subject = paragraph_subject
                            elif chain_subject is not None and chain_subject.is_resolved:
                                paragraph_subject = chain_subject
                            else:
                                continue
                        elif competing_mentions:
                            chain_subject = None
                            continue
                        elif chain_subject is not None and chain_subject.is_resolved:
                            paragraph_subject = chain_subject
                        else:
                            paragraph_subject = section_subject


                        if not self.subject_matches(paragraph_subject, need):
                            continue
                        paragraph_obligation_keys: set[str] = set()
                        paragraph_kind: str | None = None
                        for sent_start, _sent_end, sentence in sentence_spans(paragraph):
                            sent_kind = self._classify_obligation_kind(sentence)
                            if sent_kind is None:
                                continue
                            if paragraph_kind is None:
                                paragraph_kind = sent_kind
                            required = self._obligation_required(
                                sent_kind, contract.answer_type, sentence, need,
                            )


                            key = self._obligation_key(
                                sent_kind, row["chunk_id"],
                                paragraph_index * 1_000_000 + sent_start,
                            )
                            coverage_map.add_obligation(DiscoveredObligation(
                                key, sentence[:160], frozenset(terms(sentence)),
                                row["chunk_id"], sent_kind, required,
                            ))
                            coverage_map.mark(key, "candidate")
                            if required:
                                paragraph_obligation_keys.add(key)
                        # OCR or an upstream graph import can corrupt or drop
                        # the visible dash while preserving the list topology.
                        # Under an operation-matched colon lead-in, every
                        # following content paragraph inside the selected
                        # heading scope is therefore a list obligation even
                        # when no marker survives for the regex classifier.
                        if (
                            paragraph_index >= 0
                            and selected_heading.rstrip().endswith(":")
                            and not paragraph_obligation_keys
                        ):
                            fallback_kind = (
                                self._classify_obligation_kind(paragraph)
                                or "bullet"
                            )
                            # The colon-terminated heading supplies the
                            # requested operation to each child item.  The
                            # children may legitimately express only the
                            # concrete actions (for example, destroy, clean,
                            # sterilize, or reuse) without repeating the
                            # heading's relation verb.  Since _best_section
                            # selected this heading only after operation
                            # matching, every non-empty child is a required
                            # list obligation.
                            fallback_required = True
                            fallback_key = self._obligation_key(
                                fallback_kind, row["chunk_id"],
                                paragraph_index * 1_000_000,
                            )
                            coverage_map.add_obligation(DiscoveredObligation(
                                fallback_key, paragraph[:160],
                                frozenset(terms(paragraph)), row["chunk_id"],
                                fallback_kind, fallback_required,
                            ))
                            coverage_map.mark(fallback_key, "candidate")
                            if fallback_required:
                                paragraph_obligation_keys.add(fallback_key)
                            if paragraph_kind is None:
                                paragraph_kind = fallback_kind
                        if paragraph_kind is None:
                            paragraph_kind = self._classify_obligation_kind(paragraph)
                        role = self._classify_role(paragraph, paragraph_kind)
                        if role == "optional_background":


                            role = "primary"
                        accept(
                            global_index, paragraph_index, row["chunk_id"],
                            0, len(paragraph), paragraph, section_score,
                            section_score, paragraph_subject, paragraph_kind, role,
                            obligation_keys=frozenset(paragraph_obligation_keys),
                        )
                    section_used = True

        if not section_used:
            for obligation in self._discover_comparison_dimension_obligations(
                scope_rows, contract
            ):
                coverage_map.add_obligation(obligation)
                coverage_map.mark(obligation.key, "candidate")
            vocabulary = self.vectorizer.vocabulary_
            expanded_query = self._expand_query(need.query)


            focus_query = re.split(
                r"\s+Context:\s*", need.query, maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            expanded_focus = self._expand_query(focus_query)
            query_terms = [
                term for term in terms(expanded_query)
                if term in vocabulary and term not in GENERIC_QUERY_WORDS
            ]
            rare_terms = sorted(
                query_terms,
                key=lambda term: self.vectorizer.idf_[vocabulary[term]],
                reverse=True,
            )[:6]
            row_score_by_chunk: dict[str, float] = {}
            if scope_rows:
                arr = np.asarray(self.reranker.predict(
                    [[expanded_query, row["text"]] for row in scope_rows],
                    show_progress_bar=False,
                )).reshape(-1)
                for row, score in zip(scope_rows, arr):
                    row_score_by_chunk[row["chunk_id"]] = float(score)
            coverage_by_chunk = {
                row["chunk_id"]: sum(
                    term in set(terms(row["text"])) for term in rare_terms
                )
                for row in scope_rows
            }
            best_row_score = (
                max(row_score_by_chunk.values()) if row_score_by_chunk else -999.0
            )
            has_anchored_row = any(coverage_by_chunk.values())

            raw_candidates = []
            chain_subject: LocalSubject | None = None
            last_heading: str | None = None
            for global_index, row in scope_rows_indexed:
                row_score = row_score_by_chunk.get(row["chunk_id"], -999.0)
                if has_anchored_row and not coverage_by_chunk.get(row["chunk_id"], 0):
                    continue
                if row_score < best_row_score - 3.0:
                    continue
                heading = self.chunk_heading.get(row["chunk_id"], "")
                if last_heading is not None and heading != last_heading:


                    if self.core_terms(heading):
                        chain_subject = None
                last_heading = heading
                for unit_index, (start_char, end_char, unit) in enumerate(
                    sentence_spans(row["text"])
                ):
                    local_subject = self.resolve_local_subject(
                        row, start_char, end_char, chain_subject
                    )
                    if local_subject.is_resolved:
                        chain_subject = local_subject
                    if not self.subject_matches(local_subject, need):
                        continue
                    kind = self._classify_obligation_kind(unit)
                    role = self._classify_role(unit, kind)
                    if kind is not None:


                        required = self._obligation_required(
                            kind, contract.answer_type, unit, need,
                        )
                        obligation_key = self._obligation_key(
                            kind, row["chunk_id"], start_char
                        )
                        coverage_map.add_obligation(DiscoveredObligation(
                            obligation_key, unit[:160], frozenset(terms(unit)),
                            row["chunk_id"], kind, required,
                        ))
                        coverage_map.mark(obligation_key, "candidate")
                    raw_candidates.append((
                        global_index, unit_index, row["chunk_id"], start_char,
                        end_char, unit, row_score, kind, role, local_subject,
                    ))


            obligation_candidates = [
                c for c in raw_candidates
                if c[7] is not None and c[8] != "optional_background"
                and self._obligation_required(c[7], contract.answer_type, c[5], need)
            ]


            causal_candidates = [
                c for c in raw_candidates
                if is_reason_need and c[8] != "optional_background"
                and c not in obligation_candidates
                and _CAUSAL_ENTAILMENT_RE.search(c[5])
            ]
            other_candidates = [
                c for c in raw_candidates
                if c[8] != "optional_background" and not (
                    c[7] is not None
                    and self._obligation_required(c[7], contract.answer_type, c[5], need)
                )
                and c not in causal_candidates
            ]
            if is_reason_need and causal_candidates:
                other_candidates = []
            elif _TEMPORAL_REQUEST_RE.search(need_focus):
                other_candidates = [
                    c for c in other_candidates if _TEMPORAL_ANSWER_RE.search(c[5])
                ]
            if other_candidates:
                focus_scores = np.asarray(self.reranker.predict(
                    [[expanded_focus, c[5]] for c in other_candidates],
                    show_progress_bar=False,
                )).reshape(-1)
                combined = focus_scores + 0.15 * np.asarray(
                    [c[6] for c in other_candidates]
                )
                order = np.argsort(-combined)
                if len(order):
                    top_score = float(combined[order[0]])
                    kept = 0
                    for position in order:
                        if per_need_limit is not None and kept >= per_need_limit:
                            break
                        if float(combined[position]) < max(
                            MIN_EXTRACT_SCORE, top_score - 2.5
                        ):
                            continue
                        c = other_candidates[int(position)]
                        accept(
                            c[0], c[1], c[2], c[3], c[4], c[5], c[6],
                            float(combined[position]), c[9], c[7], c[8],
                        )
                        kept += 1
            for c in obligation_candidates:
                accept(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[6], c[9], c[7], c[8])
            for c in causal_candidates:
                accept(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[6], c[9], c[7], c[8])


        if not coverage_map.all_supported():
            outside = [
                (index, row) for index, row in enumerate(evidence, 1)
                if row["chunk_id"] not in authoritative_chunk_ids
            ]
            for global_index, row in outside:
                if coverage_map.all_supported():
                    break
                for unit_index, (start_char, end_char, unit) in enumerate(
                    sentence_spans(row["text"])
                ):
                    local_subject = self.resolve_local_subject(
                        row, start_char, end_char, None
                    )
                    strong_identity_signal = bool(
                        local_subject.resolved_entity_ids & need.subject_entity_ids
                    ) or (
                        local_subject.resolution_source in ("heading", "sentence")
                        and self.subject_matches(local_subject, need)
                    )


                    if not strong_identity_signal:
                        continue
                    kind = self._classify_obligation_kind(unit)
                    role = self._classify_role(unit, kind)
                    if role == "optional_background":
                        continue
                    accept(
                        global_index, unit_index, row["chunk_id"], start_char,
                        end_char, unit, 0.0, 0.0, local_subject, kind, role,
                    )

        return coverage_map, accepted, scope

    def extractive_answer(
        self, contract: QuestionContract, evidence: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Build a fast verbatim answer when every need has a validated
        EvidenceChain. Revision 4 / Correction 3, 4, 6, 7, 10, 12, B: every
        candidate sentence is only ever admitted once its own LocalSubject
        (resolved through the four-tier precedence) matches the need's
        grounded subject through one of `subject_matches()`'s five allowed
        paths - this is the direct fix for the cross-topic contamination
        failure class (a generically-true sentence from an unrelated
        section can no longer enter an answer merely by sharing a broad
        word like "specimen" or "collection")."""
        selected: dict[tuple[int, int], tuple[float, str]] = {}


        per_need_limit = (
            None if contract.answer_type in ("procedure", "calculation") else 3
        )
        chains: dict[str, EvidenceChain] = {}
        coverage_maps: dict[str, CoverageMap] = {}
        for need in contract.needs:
            coverage_map, accepted, scope = self._extract_need_evidence(
                need, evidence, contract, per_need_limit
            )
            if scope is None or not accepted:
                print(f"[EXTRACT] need={need.label!r}: no candidate region/evidence")
                return None


            if not all_need_requirements_supported(need, coverage_map, contract.answer_type):
                unresolved = [
                    key for key in coverage_map.required_keys()
                    if coverage_map.entries[key].state not in ("supported", "verified")
                ]
                print(f"[COVERAGE] need={need.label!r} unresolved: {unresolved}")
                return None
            chain = self._assemble_chain(need, accepted, scope, coverage_map)
            if chain.validation_status != "validated":
                print(
                    f"[COVERAGE] need={need.label!r} chain rejected: "
                    f"{chain.rejection_reason}"
                )
                return None
            chains[need.label] = chain
            coverage_maps[need.label] = coverage_map
            for unit in chain.units:
                if unit.role == "optional_background":
                    continue
                key = (unit.evidence_index, unit.unit_index)
                existing = selected.get(key)
                if existing is None or unit.unit_score > existing[0]:
                    selected[key] = (unit.unit_score, unit.text)
        self.last_chains = chains
        self.last_coverage_maps = coverage_maps
        ordered = sorted(selected.items(), key=lambda item: item[0])


        normalized_by_key = {
            key: compact(dehyphenate(unit)).casefold()
            for key, (_, unit) in ordered
        }


        chunk_id_by_key: dict[tuple[int, int], str] = {}
        for key, _unused in ordered:
            evidence_index, _unit_index = key
            if 1 <= evidence_index <= len(evidence):
                chunk_id_by_key[key] = evidence[evidence_index - 1]["chunk_id"]
        fragment_keys: set[tuple[int, int]] = set()
        for key, normalized in normalized_by_key.items():
            if not normalized or len(normalized) < 8:
                continue
            key_position = self.positions.get(chunk_id_by_key.get(key, ""))
            for other_key, other_normalized in normalized_by_key.items():
                if other_key == key:
                    continue
                if not (
                    other_normalized != normalized
                    and len(other_normalized) > len(normalized)
                    and normalized in other_normalized
                ):
                    continue
                key_chunk = chunk_id_by_key.get(key)
                other_chunk = chunk_id_by_key.get(other_key)
                if (
                    key_chunk is None or other_chunk is None
                    or key_chunk == other_chunk
                ):
                    continue
                other_position = self.positions.get(other_chunk)
                if (
                    key_position is not None and other_position is not None
                    and abs(key_position - other_position) <= 1
                ):
                    fragment_keys.add(key)
                    break
        claims = []
        answer_parts = []
        seen_normalized_units: set[str] = set()
        for key, (_, unit) in ordered:
            if key in fragment_keys:
                continue
            evidence_index, _unit_index = key


            normalized = compact(dehyphenate(unit)).casefold()
            if normalized in seen_normalized_units:
                continue
            seen_normalized_units.add(normalized)
            answer_parts.append(f"{unit} [E{evidence_index}]")
            claims.append({
                "sentence": unit,
                "evidence_ids": [f"E{evidence_index}"],
            })
        answer = "\n".join(answer_parts)
        coverage = np.asarray(self.reranker.predict(
            [[need.query, answer] for need in contract.needs],
            show_progress_bar=False,
        )).reshape(-1)
        print(f"[EXTRACT] coverage={coverage.tolist()}")
        if any(float(score) < MIN_COVERAGE_SCORE for score in coverage):
            return None


        return {
            "answer": answer, "claims": claims, "extractive": True,
            "chains": chains, "coverage_maps": coverage_maps,
        }

    def extract_needs(
        self, contract: QuestionContract, groups: list[list[dict[str, Any]]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """Extract and verify each need against its own evidence window.
        Revision 4 / Correction A: if any need's extraction is unresolved
        (`extractive_answer` returns None because its coverage map is not
        fully supported), this returns None for the *whole* contract -
        never a partial answer that silently drops the unresolved need."""
        global_rows: list[dict[str, Any]] = []
        global_index: dict[str, int] = {}
        answer_parts = []
        global_claims = []
        global_chains: dict[str, EvidenceChain] = {}
        global_coverage_maps: dict[str, CoverageMap] = {}
        for need, group in zip(contract.needs, groups):
            local_contract = QuestionContract(
                contract.question, contract.answer_type, (need,)
            )
            page_order = lambda row: (
                row.get("pdf_page") or 0, row.get("chunk_index") or 0
            )


            row_window_limit = (
                None if contract.answer_type in ("procedure", "calculation")
                else 7
            )
            local_rows = sorted(group[:row_window_limit], key=page_order)
            result = self.extractive_answer(local_contract, local_rows)
            if result is None:
                reformulated = self._reformulate_query(need)
                alt_need = (
                    need if reformulated == need.retrieval_query
                    else replace(need, retrieval_query=reformulated)
                )
                reformulated_pool = self.retrieve(
                    alt_need, widen=RETRIEVAL_STAGES - 1, exhaustive=True
                )
                original_pool = (
                    self.retrieve(need, widen=RETRIEVAL_STAGES - 1, exhaustive=True)
                    if alt_need is not need else reformulated_pool
                )
                merged_pool: dict[str, dict[str, Any]] = {}
                for row in [*original_pool, *reformulated_pool]:
                    existing = merged_pool.get(row["chunk_id"])
                    if existing is None or row["score"] > existing["score"]:
                        merged_pool[row["chunk_id"]] = row
                if merged_pool:
                    exhaustive_rows = sorted(
                        sorted(
                            merged_pool.values(),
                            key=lambda row: row["score"], reverse=True,
                        )[:row_window_limit],
                        key=page_order,
                    )
                    result = self.extractive_answer(local_contract, exhaustive_rows)
                    if result is not None:
                        local_rows = exhaustive_rows
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
            if len(contract.needs) > 1 and need.label:
                answer = f"{need.label}:\n{answer}"
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


            local_chain = result.get("chains", {}).get(need.label)
            if local_chain is not None:
                remapped_units = tuple(
                    replace(unit, evidence_index=local_map[unit.evidence_index])
                    for unit in local_chain.units
                    if unit.evidence_index in local_map
                )
                if remapped_units:
                    global_chains[need.label] = replace(
                        local_chain, units=remapped_units
                    )


            local_coverage_map = result.get("coverage_maps", {}).get(need.label)
            if local_coverage_map is not None:
                global_coverage_maps[need.label] = local_coverage_map
        self.last_chains = global_chains
        self.last_coverage_maps = global_coverage_maps
        return ({
            "answer": "\n\n".join(answer_parts),
            "claims": global_claims,
            "extractive": True,
            "chains": global_chains,
            "coverage_maps": global_coverage_maps,
        }, global_rows)

    def _pool_coverage_and_chain(
        self,
        need: InformationNeed,
        evidence: list[dict[str, Any]],
        contract: QuestionContract,
    ) -> tuple[CoverageMap, EvidenceChain | None]:
        """Build a CoverageMap AND a materialized EvidenceChain for `need`
        over a whole evidence pool, using the exact same region-selection,
        obligation-discovery, and subject-verified admission the extractive
        path uses (`_extract_need_evidence`), so the generator's
        precondition (Correction A) and its context (Correction 9/10)
        reflect the same notion of "supported" and "primary evidence" the
        extractive path enforces."""


        per_need_limit = (
            None if contract.answer_type in ("procedure", "calculation") else 5
        )
        coverage_map, accepted, scope = self._extract_need_evidence(
            need, evidence, contract, per_need_limit
        )
        if scope is None or not accepted:
            return coverage_map, None
        return coverage_map, self._assemble_chain(need, accepted, scope, coverage_map)

    def _pool_need_resolved(
        self, need: InformationNeed, evidence: list[dict[str, Any]],
        contract: QuestionContract,
    ) -> tuple[CoverageMap, bool]:
        """Correction 34 / Defect 4 (escalation gate must not stop early on
        a wrong section): `all_need_requirements_supported()` alone only
        checks discrete term-overlap coverage of RequestedItems/
        DiscoveredObligations - the exact same coverage state a merely
        on-topic but WRONG section can already satisfy (Correction 29 had
        to add a separate, additive "coverage score must not be negative"
        semantic gate to the final extractive/verify paths for exactly
        this reason). But `answer()`'s own widen/reformulate ESCALATION
        LADDER decided whether to keep searching using ONLY that discrete
        check, so it could declare a need "resolved" and stop widening the
        moment a wrong section happened to satisfy bag-of-words coverage -
        never reaching the deeper retrieval level where the genuinely
        correct section (already reachable by the existing widen/
        exhaustive/reformulate machinery) would have been found. Folding
        the SAME non-negative-coverage semantic check into the escalation
        decision itself - never a new constant, never a change to
        `MIN_COVERAGE_SCORE` itself - closes that gap generically, for
        every answer type, not only procedures."""
        coverage_map, chain = self._pool_coverage_and_chain(need, evidence, contract)
        if not all_need_requirements_supported(need, coverage_map, contract.answer_type):
            return coverage_map, False
        if chain is None or not chain.units:
            return coverage_map, False
        if chain.validation_status != "validated":
            return coverage_map, False
        return coverage_map, True

    def _diagnose_need_failure(
        self, need: InformationNeed, contract: QuestionContract,
        evidence: list[dict[str, Any]] | None = None,
    ) -> str:
        """Blocker 1: classify exactly which generic pipeline stage a
        failed InformationNeed broke down at - retrieval, scope_selection,
        subject_validation, extraction, coverage, or verification - by
        actually re-running the same stages `answer()` runs, never by a
        sample-specific rule and never by lowering any global threshold.
        The retrieval stage itself already exhausts the full escalation
        ladder Blocker 1 asks for: semantic reformulation
        (`_reformulate_query`), dense retrieval, lexical retrieval,
        CrossEncoder reranking, heading/entity matching (`subject_matches`/
        `expand`'s entity-position lookup), neighbour expansion (`expand`),
        and a true exhaustive full-corpus scan (`retrieve(exhaustive=True)`)
        - all already exercised below or inside `retrieve()`/`expand()`
        themselves."""
        if evidence is None:
            evidence = self.retrieve(need, widen=RETRIEVAL_STAGES - 1)
            if not evidence:
                reformulated = self._reformulate_query(need)
                alt_need = (
                    need if reformulated == need.retrieval_query
                    else replace(need, retrieval_query=reformulated)
                )
                evidence = self.retrieve(
                    alt_need, widen=RETRIEVAL_STAGES - 1, exhaustive=True
                )
        if not evidence:
            return "retrieval"
        per_need_limit = (
            None if contract.answer_type in ("procedure", "calculation") else 5
        )
        coverage_map, accepted, scope = self._extract_need_evidence(
            need, evidence, contract, per_need_limit,
        )
        if scope is None:
            return "scope_selection"
        if not accepted:
            return "subject_validation"
        if not all_need_requirements_supported(need, coverage_map, contract.answer_type):
            required = coverage_map.required_keys()
            covered_any = any(
                coverage_map.entries[key].state != "uncovered"
                for key in required
            )
            return "coverage" if covered_any else "extraction"
        return "verification"

    def compose(
        self, contract: QuestionContract, evidence: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Generator (LLM) path - only ever reached once every need has a
        validated EvidenceChain (Correction A). Immediately before the
        generator call this asserts every one of Correction 10's guards:
        every RequestedItem/required DiscoveredObligation is supported,
        each need has a validated chain, no continuation is broken, every
        evidence unit carries the correct subject, and only primary/
        required-conditional evidence is supplied - the context below is
        built from exactly that filtered chain material, never the raw
        chunk text, so the generator cannot see (and so cannot leak in) any
        excluded optional-background sentence."""
        chains: dict[str, EvidenceChain] = {}
        coverage_maps: dict[str, CoverageMap] = {}
        for need in contract.needs:
            coverage_map, chain = self._pool_coverage_and_chain(need, evidence, contract)

            if chain is None or chain.validation_status != "validated":
                return None
            if chain.continuation_state == "open":
                return None
            if any(unit.role == "optional_background" for unit in chain.units):
                return None
            if not all(
                not unit.local_subject.is_resolved
                or self.subject_matches(unit.local_subject, need)
                for unit in chain.units
            ):
                return None
            if not all_need_requirements_supported(
                need, coverage_map, contract.answer_type
            ):
                return None
            chains[need.label] = chain
            coverage_maps[need.label] = coverage_map
        self.last_chains = chains
        self.last_coverage_maps = coverage_maps


        unit_pool = sorted(
            (
                unit for chain in chains.values() for unit in chain.units
                if unit.role != "optional_background"
            ),
            key=lambda unit: (unit.evidence_index, unit.start_char),
        )
        seen_keys: set[tuple[int, int]] = set()
        context_lines = []
        for unit in unit_pool:
            key = (unit.evidence_index, unit.start_char)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            context_lines.append(f"[E{unit.evidence_index}] {unit.text}")
        context = "\n".join(context_lines)
        if len(context) > MAX_CONTEXT_CHARS:


            print(
                f"[compose] verified context ({len(context)} chars) "
                f"exceeds MAX_CONTEXT_CHARS ({MAX_CONTEXT_CHARS}); "
                f"refusing to truncate required evidence."
            )
            return None
        needs_payload = [
            {
                "label": need.label,
                "query": need.query,
                "requirements": list(need.requirements),
            }
            for need in contract.needs
        ]
        if contract.answer_type == "procedure":


            type_instruction = (
                " This is a procedure: reproduce every step from the "
                "evidence in its exact original order, without skipping, "
                "merging, reordering, or summarizing any step."
            )
        elif contract.answer_type in ("fact", "reason"):
            type_instruction = (
                " This is a fact/reason question: answer with the single "
                "most precise sentence that directly supports each need; "
                "do not include surrounding background or introductory text."
            )
        else:
            type_instruction = ""
        instruction = (
            "Answer exclusively from the supplied evidence. Cover every "
            "requested need, comparison side, dimension, and condition. "
            "Preserve conditional distinctions. Do not add outside knowledge, "
            "steps, values, explanations, or terminology. If the evidence is "
            "not support a requested detail, omit that detail rather than "
            "inventing it. Return strict JSON with answer and claims. Cite "
            "every answer sentence as [E#]. "
            "Each claim contains the exact sentence and its evidence_ids."
            + type_instruction
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
            f"Question: {contract.question}\n"
            f"Answer type: {contract.answer_type}\n"
            f"Required needs: {json.dumps(needs_payload)}\n"
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
            "chains": chains,
        }

    def _nli(self) -> None:
        if self.nli is not None:
            return
        with self.lock:
            if self.nli is None:
                self.nli = CrossEncoder(NLI_MODEL)

    _QUALIFIER_TERMS = {
        "always", "never", "only", "must", "should", "cannot", "except",
        "unless", "immediately", "before", "after", "within",
    }

    def _claim_subject_consistent(
        self,
        ids: list[int],
        sentence: str,
        evidence: list[dict[str, Any]],
        contract: QuestionContract,
    ) -> bool:
        """Correction 11: a verifier must not accept a claim merely because
        it is copied verbatim from SOME nearby chunk - the cited chunk's
        own local subject (at the span the claim actually corresponds to)
        must still match the need the claim is answering.

        The cited unit's LocalSubject was already resolved once, correctly,
        during extraction (Correction 3's four-tier precedence, including
        heading-inheritance for a whole accepted procedure paragraph that
        does not itself repeat the subject noun in every sentence). Re-
        deriving it here from scratch over a freshly re-split single
        sentence - with no inherited chain/heading context at all - can
        find a different, narrower, merely-incidental entity mention
        inside that one sentence (e.g. an instrument or reagent the step
        refers to) and wrongly treat it as a competing subject. So this
        first checks the actual chain unit's own resolved LocalSubject for
        the cited evidence index, and only falls back to a fresh
        re-derivation when no such unit is tracked (e.g. a generated,
        non-extractive answer whose claim sentence has no 1:1 unit)."""
        sentence_terms = set(terms(sentence))
        best_need = max(
            contract.needs,
            key=lambda need: len(need.subject_terms & sentence_terms),
            default=None,
        )
        if best_need is None or (
            not best_need.subject_terms and not best_need.subject_entity_ids
        ):
            return True
        chain = self.last_chains.get(best_need.label) if self.last_chains else None
        chain_units_by_index: dict[int, list[LocalSubject]] = {}
        if chain is not None:
            for unit in chain.units:
                chain_units_by_index.setdefault(unit.evidence_index, []).append(
                    unit.local_subject
                )
        for index in ids:
            for local_subject in chain_units_by_index.get(index, ()):
                if self.subject_matches(local_subject, best_need):
                    return True
            if index in chain_units_by_index:


                continue
            row = evidence[index - 1]
            best_span = None
            best_overlap = -1
            for start, end, unit in sentence_spans(row["text"]):
                overlap = len(set(terms(unit)) & sentence_terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_span = (start, end)
            if best_span is None:
                continue
            local_subject = self.resolve_local_subject(row, best_span[0], best_span[1], None)
            if self.subject_matches(local_subject, best_need):
                return True
        return False

    def verify(
        self,
        contract: QuestionContract,
        generated: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> tuple[bool, str, list[int]]:
        """Correction 11: independent checks for claim-to-evidence
        entailment, subject consistency, contradiction/unsupported terms,
        numbers, units, conditions and qualifiers, completeness against the
        QuestionContract, and (for a required-ordering answer type)
        citation ordering."""
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


            if not self._claim_subject_consistent(ids, sentence, evidence, contract):
                return False, "claim's cited evidence has the wrong subject", []
            claim_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", sentence))
            source_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", premise))
            if claim_numbers - source_numbers:
                return False, "unsupported numeric value", []
            claim_units = set(re.findall(r"\b\d+(?:\.\d+)?\s*([a-zA-Z]{1,6})\b", sentence))
            source_units = set(re.findall(r"\b\d+(?:\.\d+)?\s*([a-zA-Z]{1,6})\b", premise))
            if claim_units - source_units - STOPWORDS:
                return False, "unsupported unit", []
            claim_qualifiers = {
                term for term in terms(sentence) if term in self._QUALIFIER_TERMS
            }
            premise_terms = set(terms(premise))
            if claim_qualifiers - premise_terms:
                return False, "unsupported condition or qualifier", []
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
        if contract.required_ordering:


            positions = [
                self.positions[evidence[value - 1]["chunk_id"]]
                for value in answer_citations
            ]
            if positions != sorted(positions):
                return False, "answer citations are out of document order", []
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
            [[need.query, generated["answer"]] for need in contract.needs],
            show_progress_bar=False,
        )).reshape(-1)
        print(f"[COVERAGE] scores={coverage_scores.tolist()}")
        if any(float(score) < MIN_COVERAGE_SCORE for score in coverage_scores):
            return False, "one or more requested needs are absent", []


        cited_unique = list(dict.fromkeys(cited))
        cited_set = set(cited_unique)

        # Independently re-check the assembled answer against every need.
        # A valid citation and a matching subject are insufficient when the
        # cited evidence does not also express the requested operation.
        for need in contract.needs:
            chain = self.last_chains.get(need.label)
            if chain is None or not chain.units:
                return False, "a requested need has no evidence chain", []
            cited_units = [
                unit for unit in chain.units
                if unit.evidence_index in cited_set
            ]
            if not cited_units:
                return False, "a requested need has no cited evidence", []
            covered = frozenset().union(*(
                unit.covered_item_ids | unit.covered_obligation_ids
                for unit in cited_units
            ))
            required_items = {
                item.key for item in need.requested_items if item.required
            }
            if not required_items.issubset(covered):
                return False, "a requested item is absent from cited evidence", []
            if need.operation_terms:
                operation_text = " ".join(
                    unit.text for unit in cited_units
                )
                if not terms_covered_morphologically(
                    need.operation_terms, set(terms(operation_text))
                ):
                    return False, "cited evidence does not entail the requested operation", []

        answer_terms = set(terms(generated["answer"]))
        for side in contract.comparison_sides:
            side_terms = self.core_terms(side)
            if side_terms and not (side_terms & answer_terms):
                return False, f"answer is missing a comparison side: {side!r}", []
        cited_text = " ".join(evidence[index - 1]["text"] for index in cited_unique)
        for quantity in contract.quantities:
            if quantity in cited_text and quantity not in generated["answer"]:
                return False, f"answer is missing a required quantity: {quantity!r}", []
        return True, "verified", cited_unique

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

    @staticmethod
    def response(
        kind: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        images: list[dict[str, Any]],
        scanned: int,
        mode: str,
        neo4j_verified: bool = False,
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
                    "verified" if kind == "domain_answer" and neo4j_verified
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
        with self.request_lock:
            return self._answer_locked(question)

    def _answer_locked(self, question: str) -> dict[str, Any]:
        contract = self.plan(question)
        groups = [self.retrieve(need) for need in contract.needs]


        for attempt in range(1, RETRIEVAL_STAGES):
            if all(groups):
                break
            for index, need in enumerate(contract.needs):
                if not groups[index]:
                    groups[index] = self.retrieve(need, widen=attempt)


        for index, need in enumerate(contract.needs):
            if groups[index]:
                continue
            reformulated = self._reformulate_query(need)
            alt_need = (
                need if reformulated == need.retrieval_query
                else replace(need, retrieval_query=reformulated)
            )
            groups[index] = self.retrieve(
                alt_need, widen=RETRIEVAL_STAGES - 1, exhaustive=True
            )
            if groups[index]:
                print(
                    f"[RETRIEVE] need={need.label!r}: recovered via "
                    "semantic reformulation + exhaustive scan"
                )


        if len(groups) > 1:
            original_groups = [list(group) for group in groups]
            for target_index, target_need in enumerate(contract.needs):
                merged = {row["chunk_id"]: dict(row) for row in groups[target_index]}
                target_terms = set(terms(target_need.query))
                for source_index, source_need in enumerate(contract.needs):
                    if source_index == target_index:
                        continue
                    if len(target_terms.intersection(terms(source_need.query))) < 2:
                        continue
                    for row in original_groups[source_index][:7]:
                        shared = {**row, "score": float(row["score"]) - 0.25}
                        old = merged.get(row["chunk_id"])
                        if old is None or shared["score"] > old["score"]:
                            merged[row["chunk_id"]] = shared
                groups[target_index] = sorted(
                    merged.values(), key=lambda row: row["score"], reverse=True
                )
        for need, group in zip(contract.needs, groups):
            print(
                f"[RETRIEVAL] need={need.label!r}; top="
                f"{[(row['chunk_id'], round(row['score'], 3)) for row in group[:8]]}"
            )
        if any(not group for group in groups):
            for need, group in zip(contract.needs, groups):
                if not group:
                    print(
                        f"[DIAGNOSE] need={need.label!r}: stage=retrieval "
                        "(no candidate cleared the zero-score floor even "
                        "after semantic reformulation and an exhaustive "
                        "full-corpus scan)"
                    )
            return self.response(
                "not_found", question,
                "No complete answer could be verified from the corpus.",
                [], [], len(self.rows), "retrieval_incomplete",
            )
        extracted = self.extract_needs(contract, groups)
        if extracted is not None:
            generated, evidence = extracted
        else:
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but it was incomplete.",
                [], [], len(self.rows), "evidence_incomplete",
            )
        if generated is None:
            print("[COMPOSE] rejected: invalid or empty structured answer")
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but it was incomplete.",
                [], [], len(self.rows), "evidence_incomplete",
            )
        verified, diagnostic, cited = self.verify(
            contract, generated, evidence
        )
        print(
            f"[FINAL VERIFY] verified={verified}; "
            f"diagnostic={diagnostic}"
        )
        if not verified:
            for need in contract.needs:
                print(
                    f"[DIAGNOSE] need={need.label!r}: stage=verification "
                    f"({diagnostic})"
                )
            return self.response(
                "not_found", question,
                "Relevant evidence was found, but the answer was not "
                "fully supported and complete.",
                [], [], len(self.rows), "verification_failed",
            )


        cited_set = set(cited)
        incomplete_needs = []
        for need in contract.needs:
            chain = self.last_chains.get(need.label)
            if chain is None:
                incomplete_needs.append(need.label or need.query)
                continue
            verified_ids: frozenset[str] = frozenset()
            for unit in chain.units:
                if unit.evidence_index in cited_set:
                    verified_ids |= unit.covered_item_ids | unit.covered_obligation_ids


            coverage_map = self.last_coverage_maps.get(need.label)
            if coverage_map is not None:
                for key in verified_ids:
                    coverage_map.mark(key, "verified")


            required_item_keys = {
                item.key for item in need.requested_items if item.required
            }
            if not required_item_keys.issubset(verified_ids):
                incomplete_needs.append(need.label or need.query)
                continue
            required_obligation_ids = frozenset(
                key for key in chain.covered_obligation_ids
                if coverage_map is None
                or key not in coverage_map.obligations
                or coverage_map.obligations[key].required
            )
            if not required_obligation_ids.issubset(verified_ids):
                incomplete_needs.append(need.label or need.query)
                continue
            if coverage_map is not None and not coverage_map.all_verified():
                incomplete_needs.append(need.label or need.query)
        if incomplete_needs:
            print(f"[COVERAGE] verified-completeness failed for: {incomplete_needs}")
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
            mode, neo4j_verified=self.corpus_source == "neo4j",
        )

    def image_path(self, image_id: str) -> str | None:
        if self.driver:
            try:
                with self.driver.session() as session:
                    record = session.run(
                        "MATCH (i:Image {id: $id}) "
                        "RETURN i.file_path AS path LIMIT 1",
                        id=image_id,
                    ).single()
                if record and record["path"]:
                    raw = Path(str(record["path"]))
                    candidate = raw if raw.is_absolute() else ROOT / raw
                    candidate = candidate.resolve()
                    if ROOT.resolve() in candidate.parents and candidate.is_file():
                        return str(candidate)
            except Exception as error:
                print(f"[GRAPH] image path lookup failed, using CSV fallback: {error}")
        info = self.images_by_id.get(image_id)
        return info.get("file_path") or None if info else None


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
    return {
        "status": "ok",
        "graph": "v2",
        "evidence_service": "initialized" if _qa is not None else "not_initialized",
    }


@app.post("/ask")
def ask(request: QuestionRequest) -> dict[str, Any]:
    question = clean_question(request.query)
    if not question:
        raise HTTPException(
            status_code=400, detail="Question cannot be empty"
        )


    kind, effective_question = classify_request(question)
    if kind == "social_only":
        return lightweight_social_response(effective_question)
    if kind == "ambiguous":
        return clarification_response(effective_question)
    try:
        return qa().answer(effective_question)
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

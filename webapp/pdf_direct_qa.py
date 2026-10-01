from __future__ import annotations

import csv
import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "graph_v2" / "chunks.csv"
RERANK_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
GENERATOR_MODEL = os.getenv(
    "LOCAL_ANSWER_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
)
CONTRADICTION_NOISE_FLOOR = float(os.getenv("NLI_CONTRADICTION_FLOOR", "0.02"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
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
    # The stemmer only strips a literal "-ation"/"-ization" suffix, so it
    # does not equate a verb already listed above with its own noun form
    # (stem("examine") stays "examine", but stem("examination") becomes
    # "examin") -- without these, a question phrased with the noun form
    # ("the examination of...", "sample preparation") would wrongly treat
    # that operation word as the question's actual topic.
    "examination", "preparation", "identification", "calculation",
    "sterilization", "collection",
    "not", "rather", "than", "between", "per", "number", "maximum",
}
CONTRASTIVE_TERM_PAIRS = ({"thick", "thin"},)
SYNONYM_SIMILARITY = 0.745  # see DirectPdfQA.paraphrased
# A descriptive detail needs a closer match to count as kept: "nasal" for
# "nose" (0.92) is the same detail, "parasites" for "malaria" is not.
DETAIL_SIMILARITY = 0.85
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
NUMBERED_STEP_LINE_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
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


# A free-text box can be asked plain conversational small talk that has no
# business going through retrieval at all -- there is no "source chunk"
# for "hi". Matched by exact text after normalization (casefold, strip
# trailing punctuation) so a real question that merely contains one of
# these words ("How is a thick blood film prepared?") is never caught by
# accident. More pairs are expected to be added here over time.
SMALL_TALK_RESPONSES: dict[str, str] = {
    "hi": "Hi! Ask me anything about the laboratory manual.",
    "hello": "Hello! Ask me anything about the laboratory manual.",
    "hey": "Hey! Ask me anything about the laboratory manual.",
    "bye": "Goodbye!",
    "goodbye": "Goodbye!",
    "how are you": "I'm just a document assistant, but I'm ready to help -- ask me anything about the laboratory manual.",
    "how are you today": "I'm just a document assistant, but I'm ready to help -- ask me anything about the laboratory manual.",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
}


def small_talk_response(question: str) -> str | None:
    normalized = re.sub(r"[?!.]+$", "", question.strip().casefold()).strip()
    return SMALL_TALK_RESPONSES.get(normalized)


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


def content_roots(text: str) -> set[str]:
    """`roots()` minus the same question-framing/generic-action words the
    rest of this file already excludes from a need's subject (GENERIC_
    SUBJECT_ROOTS) -- the words that describe the shape of the ask ("what
    treatment", "how should", "for the purpose of") rather than its actual
    topic. Used to measure whether an answer's own text actually covers
    the meaningful words of an arbitrary, unseen question -- a free-text
    box can be asked anything, so this has to work from the question's own
    words alone, never a per-question list."""
    generic = {stem(term) for term in GENERIC_SUBJECT_ROOTS}
    return roots(text) - generic


# The specimen a technique is performed on is a hard, unambiguous
# constraint this document repeats across every section (Part III's own
# structure is literally "examination of urine, cerebrospinal fluid and
# blood") -- yet a claim can still get attributed to a chunk about the
# wrong specimen, because the surrounding procedural vocabulary (drops,
# centrifuge, ml, stain) is nearly identical across specimen types, which
# is exactly the kind of overlap that fools word-level NLI. This check is
# a second, independent axis from the NLI status: it flags a mismatch
# between the specimen named in the question and the specimen named in
# the claim's own cited source text, regardless of whether NLI called
# that claim supported or contradicted.
SPECIMEN_TYPES = {
    "blood": {"blood"},
    "urine": {"urine", "urinary"},
    "cerebrospinal fluid (csf)": {"csf", "cerebrospinal"},
    "stool": {"stool", "stools", "faeces", "feces", "faecal", "fecal"},
    "sputum": {"sputum"},
    "serum": {"serum"},
    "plasma": {"plasma"},
}


def detect_specimen_types(text: str) -> set[str]:
    """Which of this document's specimen categories a piece of text names,
    by literal word match -- general across the whole corpus (these are
    the manual's own top-level specimen categories, not anything tied to
    one question), and used only as a same/different signal, not as a
    retrieval mechanism of its own."""
    lowered = text.casefold()
    return {
        label for label, words in SPECIMEN_TYPES.items()
        if any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)
    }


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_UNIT_ALIASES = {
    "g": "g", "gram": "g", "grams": "g", "gm": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "minute": "min", "minutes": "min", "min": "min", "mins": "min",
    "hour": "hr", "hours": "hr", "hr": "hr", "hrs": "hr",
    "second": "sec", "seconds": "sec", "sec": "sec", "secs": "sec",
    "%": "%",
}
NUMBER_WITH_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)*)\s*(" + "|".join(sorted(_UNIT_ALIASES, key=len, reverse=True)) + r")\b",
    re.I,
)


def _number_unit_pairs(text: str) -> set[tuple[str, str]]:
    pairs = set()
    for match in NUMBER_WITH_UNIT_RE.finditer(text):
        value = match.group(1).replace(",", "")
        unit = _UNIT_ALIASES.get(match.group(2).lower())
        if unit:
            pairs.add((value, unit))
    return pairs


RUNNING_HEADER_RE = re.compile(r"^\s*(?:\d+\.\s+)?[A-Z][^.;:]*\s\d{1,3}\s*$")
REFERENCE_PREFIX_RE = re.compile(
    r"\b(?:fig(?:ure)?s?|tables?|sections?|no|page|reagent)\.?$", re.I
)


def clean_answer_text(answer: str) -> str:
    """Present an already-verified extractive answer as readable text with
    no model involved: drop the [S#] source tags, page running headers and
    figure captions glued onto the end of a step, and repair the PDF's
    undecodable glyphs (U+FFFD) from their surrounding characters. Wording
    is never changed, so nothing can be added, dropped or reordered."""
    lines: list[str] = []
    for raw_line in answer.splitlines():
        line = re.sub(r"\s*\[S\d+\]", "", raw_line).strip()
        if not line or RUNNING_HEADER_RE.match(line):
            continue
        if re.match(r"^\d+\s+Manual of basic techniques for a health laboratory\b", line, re.I):
            continue
        line = re.sub(r"(?<=[.!?])\s+Fig(?:ure)?\.?\s*\d+\.\d+\s+[A-Z][^.()]*$", "", line)
        # This PDF's font maps the multiplication sign to the yen glyph.
        line = line.replace("¥", "×")
        line = re.sub(r"(?<=\d)�(?=\d)", "–", line)
        line = re.sub(r"(?<=[\d)])\s*�\s*(?=[\d(])", " × ", line)
        line = re.sub(r"(?<=[A-Za-z])�(?=[A-Za-z])", "’", line)
        line = re.sub(r"^�\s*", "• ", line)
        line = re.sub(r"�\s*(?=\d)", "× ", line)
        line = line.replace("�", "\"")
        lines.append(" ".join(line.split()))
    return "\n".join(lines)


# (label, pattern that marks it in the source, pattern that keeps it in a
# rewrite). The words that set a step's conditions matter more than any
# other wording: "before" vs "after", alternatives ("or") vs a sequence,
# a negation, a condition, a limit.
CONDITION_WORDS: list[tuple[str, str, str]] = [
    ("before", r"\b(?:before|prior to)\b", r"\b(?:before|prior to|preceding)\b"),
    ("after", r"\b(?:after|following)\b", r"\b(?:after|following|once)\b"),
    # Only a real choice between options ("either ... or", "..., or ...",
    # "alternatively"); a plain "X or Y" noun pair ("nose or pharynx") is
    # not a condition and rewording it is not a change of meaning.
    ("or (alternatives)", r"\beither\b|,\s*or\b|\balternatively\b", r"\b(?:or|either|alternatively|alternative)\b"),
    ("not/never/without", r"\b(?:not|never|no|without|cannot|nor)\b", r"\b(?:not|never|no|without|cannot|nor|avoid|none)\b"),
    ("if/unless", r"\b(?:if|unless)\b", r"\b(?:if|unless|when|whenever|provided|in case)\b"),
    ("until", r"\b(?:until|till)\b", r"\b(?:until|till)\b"),
    ("at least", r"\b(?:at least|not less than|minimum)\b", r"\b(?:at least|not less than|no less than|minimum)\b"),
    ("at most", r"\b(?:at most|not more than|no more than|maximum|up to)\b", r"\b(?:at most|not more than|no more than|maximum|up to)\b"),
]


def _normalize_for_conditions(text: str) -> str:
    text = text.casefold().replace("n't", " not")
    return re.sub(r"\bno\.\s*\d", " ", text)  # "reagent no. 28" is not a negation


DEFINES_TERM_RE = re.compile(
    r"\b(?:is|are)\s+(?:called|known as|defined as|termed)\s+(?:the\s+|an?\s+)?([^.(;,]+)",
    re.I,
)


def introduces_other_concept(text: str, query: str) -> bool:
    """True when a sentence defines a term the question did not ask about
    ("The number of erythrocytes in 1 litre ... is called the erythrocyte
    number concentration") -- in a manual that is how a new section opens,
    so it marks where the current topic ends."""
    match = DEFINES_TERM_RE.search(text)
    if not match:
        return False
    defined = roots(match.group(1))
    return bool(defined) and not defined <= roots(query)


GENERIC_ENTITY_KEYS = {
    stem(word) for word in (
        "sample", "specimen", "test", "material", "solution", "method",
        "result", "patient", "procedure", "examination", "technique",
    )
}


def entity_display(name: str) -> str:
    """An entity name as extraction left it can carry a list-bullet glyph
    rendered as a leading "G" ("G Westergren ESR tubes") or a leading
    article ("The sample"); strip both for display and matching."""
    text = re.sub(r"^G\s+(?=[A-Z0-9])", "", name.strip())
    return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I).strip()


def american_spelling(text: str) -> str:
    """centre/litre/metre/fibre/theatre (and compounds such as millilitre)
    -> -er; no other "-re" word is touched ("culture", "measure")."""
    return re.sub(r"\b(\w*?(?:cent|lit|met|fib|theat))re(s?)\b", r"\1er\2", text)


def entity_key(name: str) -> str:
    """One key per real-world entity: case, bullets, articles, British/
    American spelling and singular/plural removed, and "sample of blood"
    read as "blood sample" -- so the graph's near-duplicate entity names
    collapse to one for matching and display."""
    text = entity_display(name).casefold()
    text = american_spelling(text)
    of_phrase = re.fullmatch(r"(\w+) of (\w+)", text)
    if of_phrase:
        text = f"{of_phrase.group(2)} {of_phrase.group(1)}"
    return " ".join(stem(word) for word in re.findall(r"[a-z0-9]+", text))


def entity_is_generic(name: str) -> bool:
    """Too generic to count as an entity: an empty name, a generic word
    ("sample"), a very short one, or a single word that is really a
    qualifier ("well" as in "mix well", "early") or a lab action
    ("Collecting" -- the action checks already cover it)."""
    key = entity_key(name)
    if not key:
        return True
    if " " in key:
        return False
    return key in GENERIC_ENTITY_KEYS or len(key) < 4 or key in _ACTION_BY_ROOT or key in {
        stem(word) for group in QUALIFIER_GROUPS for word in group
    } | {stem(word) for word in _EXTRA_QUALIFIERS}


# ---------------------------------------------------------------------------
# Fact layer: the graph's own record of what each source sentence states,
# so an LLM rewrite can be checked against Neo4j instead of re-read text.
# Built into Neo4j by scripts/extract_semantic_relations.py --facts as
# (:Chunk)-[:STATES]->(:Fact)-[:INVOLVES]->(:Entity), (:Fact)-[:NEXT]->(:Fact).
# ---------------------------------------------------------------------------

# Qualifiers that change how or when a step is done ("cough deeply",
# "early morning", "mix gently"); words in one group are interchangeable.
QUALIFIER_GROUPS: list[set[str]] = [
    {"well", "thoroughly"}, {"gently", "softly", "lightly"},
    {"quickly", "rapidly", "fast", "promptly"}, {"slowly", "gradually"},
    {"immediately", "promptly", "straightaway", "instantly"},
    {"firmly", "tightly", "securely"}, {"carefully", "cautiously"},
    {"completely", "fully", "entirely", "thoroughly"}, {"usually", "normally", "typically", "generally", "commonly"},
    {"approximately", "roughly", "about", "around"}, {"always", "invariably"},
    {"especially", "specifically", "particularly"}, {"exclusively", "solely", "only"},
    {"deeply"}, {"early"}, {"directly"}, {"freshly"}, {"daily"}, {"separately"},
    {"just", "only", "merely", "solely"}, {"exactly", "precisely"}, {"almost", "nearly", "mostly", "most"},
    {"sufficient", "enough", "adequate"},
]
_NOT_QUALIFIERS = {
    "only", "apply", "supply", "reply", "assembly", "family", "rely", "fly",
    "july", "belly", "jelly", "lily", "oily", "likely", "anomaly", "butterfly",
    "friendly", "ugly", "holy", "silly", "firstly", "secondly", "finally",
}
_EXTRA_QUALIFIERS = {"well", "fast", "always", "about", "around", "just", "almost", "nearly", "most"}
LAB_ACTIONS: list[set[str]] = [
    {"collect", "gather", "obtain", "take", "taken", "took"}, {"place", "put", "set", "position"},
    {"add", "pour"}, {"mix", "stir"}, {"shake", "agitate"}, {"boil"}, {"pour", "empty"},
    {"fill"}, {"centrifuge", "spin", "spun"}, {"examine", "inspect", "look", "observe", "check"},
    {"check", "ensure", "verify", "confirm"},
    {"count"}, {"leave", "left", "allow", "let", "stand"}, {"remove", "withdraw"}, {"draw", "drawn", "drew"},
    {"filter"}, {"spread", "smear"}, {"dry", "dried"}, {"fix"}, {"stain"}, {"wash", "rinse"},
    {"clean"}, {"sterilize", "autoclave"}, {"heat", "warm"}, {"cool"}, {"cover", "plug", "close"},
    {"label", "mark"}, {"record", "note", "write", "written", "wrote"}, {"report"}, {"read"}, {"measure"},
    {"read", "check"},
    {"dilute"}, {"incubate"}, {"discard", "dispose"}, {"store", "keep", "kept"}, {"transfer"},
    {"pipette"}, {"cough"}, {"expectorate", "spit"}, {"ask", "instruct"},
    {"prepare", "make", "made"}, {"use"}, {"send", "sent", "dispatch"}, {"wait"}, {"cut"}, {"insert"},
]
# A word in several groups ("check" = examine, and "check that" = ensure)
# accepts the synonyms of all of them.
_ACTION_BY_ROOT: dict[str, frozenset[str]] = {}
for _group in LAB_ACTIONS:
    for _word in _group:
        _ACTION_BY_ROOT[stem(_word)] = _ACTION_BY_ROOT.get(stem(_word), frozenset()) | {stem(w) for w in _group}
_QUALIFIER_GROUP: dict[str, frozenset[str]] = {}
for _group in QUALIFIER_GROUPS:
    for _word in _group:
        _QUALIFIER_GROUP[_word] = _QUALIFIER_GROUP.get(_word, frozenset()) | frozenset(_group)

# Fixed phrases whose verb is not a lab action ("care should be taken").
_IDIOMS_RE = re.compile(
    r"\b(?:care\s+(?:should|must)\s+be\s+taken|take\s+care|taken?\s+into\s+account|takes?\s+place)\b",
    re.I,
)


def fact_qualifiers(text: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", text.casefold().replace("at once", "immediately"))
    return {
        token for token in tokens
        if token in _EXTRA_QUALIFIERS
        or (token.endswith("ly") and len(token) >= 5 and token not in _NOT_QUALIFIERS)
    }


_ACTION_WORD = {stem(word): word for group in LAB_ACTIONS for word in sorted(group)}


def _all_stems(text: str) -> set[str]:
    # Every word, including ones roots() drops as question-framing words
    # ("ensure", "check") -- for actions those are the content.
    return {stem(word) for word in words(text)}


def fact_actions(text: str) -> set[str]:
    return {root for root in _all_stems(_IDIOMS_RE.sub(" ", text)) if root in _ACTION_BY_ROOT}


def fact_quantities(text: str) -> set[str]:
    """Quantities stated in the text as "value unit" (or a bare value used
    as a quantity, e.g. "divide by 100"), skipping step numbers and figure,
    table, section and reagent references."""
    quantities = {f"{value} {unit}" for value, unit in _number_unit_pairs(text)}
    unit_spans = [match.span() for match in NUMBER_WITH_UNIT_RE.finditer(text)]
    for match in NUMBER_RE.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in unit_spans):
            continue
        prefix = text[:match.start()].rstrip()
        if not prefix or re.search(r"(?:^|[.\n])\s*$", prefix) or match.group(0).count(".") >= 2:
            continue
        if REFERENCE_PREFIX_RE.search(prefix):
            continue
        quantities.add(match.group(0).replace(",", ""))
    return quantities


def fact_conditions(text: str) -> set[str]:
    normalized = _normalize_for_conditions(text)
    return {label for label, in_source, _kept in CONDITION_WORDS if re.search(in_source, normalized)}


# Nouns that make an "at/in/on ..." phrase a time, not a place ("at the
# height of an episode of fever", "in the morning").
TIME_WORDS = {
    "height", "peak", "morning", "evening", "night", "day", "days", "hour", "hours", "minute",
    "minutes", "onset", "end", "beginning", "start", "stage", "phase", "episode", "time", "period",
    "week", "weeks", "month", "months", "year", "moment", "interval", "afternoon", "noon", "midnight",
}

# Prepositions that attach a role to the action, and the role they mark.
ROLE_PREPOSITIONS = {
    "into": "destination", "onto": "destination",
    "with": "material", "containing": "material",
    "for": "purpose", "from": "source",
    "in": "place", "inside": "place", "on": "place", "at": "place", "under": "place",
}
_ROLE_STOP = {
    "the", "a", "an", "each", "some", "any", "this", "that", "these", "those",
    "its", "their", "his", "her", "all", "both", "another", "other", "such", "his", "your",
}
# "for this purpose", "in this way": the phrase names no real target.
_ROLE_FILLER = {stem(word) for word in ("purpose", "reason", "way", "case", "example", "instance", "manner", "task")}
_ROLE_BREAK = set(ROLE_PREPOSITIONS) | {"of", "either", "neither", "both", "whether", "and", "or", "then", "until", "before", "after", "if", "unless", "to", "by", "which", "that", "as"}


def fact_roles(text: str) -> set[str]:
    """What the action is done into/with/for/from/in: "role=head", the head
    being the last word of the phrase after the preposition ("into the
    container" -> "destination=container", "for culture of Mycobacterium
    tuberculosis" -> "purpose=tuberculosis"). Phrases carrying a number are
    left to the quantity check."""
    roles: set[str] = set()
    normalized = american_spelling(text.casefold())
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*|[,.;:()]", normalized)
    for index, token in enumerate(tokens):
        role = ROLE_PREPOSITIONS.get(token)
        if not role:
            continue
        following_words = [t for t in tokens[index + 1:index + 3] if re.match(r"[a-z]", t) and t not in _ROLE_STOP]
        if token == "for" and following_words and following_words[0].endswith("ing") and len(following_words[0]) >= 6:
            # The purpose is the action ("for preparing the film"), not its
            # object -- "to maintain the film" keeps "film" but not the purpose.
            roles.add(f"purpose={following_words[0]}")
            continue
        phrase: list[str] = []
        for following in tokens[index + 1:index + 8]:
            if following == ",":
                continue  # "a wide-mouthed, screw-topped jar"
            if not re.match(r"[a-z0-9]", following) or following in _ROLE_BREAK:
                break
            if following in _ROLE_STOP:
                continue
            phrase.append(following)
        if not phrase or any(ch.isdigit() for word in phrase for ch in word):
            continue
        head = stem(phrase[-1])
        if role == "place" and phrase[-1] in TIME_WORDS:
            role = "time"
        if len(head) >= 3 and head not in _ROLE_FILLER:
            roles.add(f"{role}={phrase[-1]}")
    return roles


def fact_attributes(text: str, entity_names: list[str]) -> set[str]:
    """Descriptive words directly in front of an entity the sentence names
    ("liquid frothy saliva" -> "liquid|saliva"): dropping one ("frothy
    saliva") changes what is described."""
    attributes: set[str] = set()
    lowered = text.casefold()
    action_roots = set(_ACTION_BY_ROOT)
    for name in entity_names:
        display = entity_display(name).casefold()
        if not display:
            continue
        for match in re.finditer(rf"\b{re.escape(display)}\b", lowered):
            before = re.findall(r"[a-z]+|[^a-z\s]", lowered[max(0, match.start() - 40):match.start()])
            for word in reversed(before[-2:]):
                # Only words directly in front of the name, stopping at the
                # first one that is not a description ("culture of X": "of").
                if not (
                    len(word) >= 4 and word.isalpha() and word not in _ROLE_STOP
                    and word not in _ROLE_BREAK and not word.endswith("ly")
                    and stem(word) not in action_roots and word not in QUESTION_WORDS
                    and stem(word) not in set(entity_key(display).split())
                ):
                    break
                attributes.add(f"{word}|{display}")
    return attributes


CLAIM_SIMILARITY = 0.78  # see DirectPdfQA.claim_covered


def fact_claims(text: str) -> list[str]:
    """The separate statements a source sentence makes (split at sentence
    ends, semicolons and dashes): "Wait for 5 minutes before reading --
    a positive result may be obvious before this time." holds two, and a
    rewrite that keeps only the first has left information out."""
    body = re.sub(r"^\s*\d+[.)]\s+", "", text.strip())
    parts = re.split(r"(?<=[.:!?])\s+(?=[A-Z(])|(?<=;)\s+|\s+[—–]\s+", body)
    return [part.strip() for part in parts if len(words(part)) >= 3]


def extract_fact_frame(text: str) -> dict[str, list[str]]:
    """What one sentence states, as the graph stores it: its lab actions,
    the qualifiers on them, its quantities, its condition words and the
    roles attached to the action (destination, material, purpose, source,
    place). Attributes need the sentence's entities and are added by the
    graph build (fact_attributes)."""
    return {
        "actions": sorted(fact_actions(text)),
        "qualifiers": sorted(fact_qualifiers(text)),
        "quantities": sorted(fact_quantities(text)),
        "conditions": sorted(fact_conditions(text)),
        "roles": sorted(fact_roles(text)),
        "claims": fact_claims(text),
    }


def fact_id(chunk_id: str, text: str) -> str:
    return hashlib.sha1(f"{chunk_id}|{normalize_for_exact_check(text)}".encode("utf-8")).hexdigest()[:20]


def _loose(root: str) -> str:
    # The stemmer keeps a final "e" on some words and not their inflections
    # ("culture" vs "culturing" -> "cultur"); compare without it.
    return root[:-1] if len(root) > 4 and root.endswith("e") else root


def _context_roots(text: str, word: str, width: int = 3) -> set[str]:
    """Roots of the words around the first form of `word` in `text`."""
    tokens = re.findall(r"[a-z]+", text.casefold())
    target = _loose(stem(word))
    for index, token in enumerate(tokens):
        if _loose(stem(token)) == target or token.startswith(word[:5]):
            window = tokens[max(0, index - width):index] + tokens[index + 1:index + 1 + width]
            return {_loose(stem(t)) for t in window if len(t) >= 3}
    return set()


_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
    "should", "must", "may", "might", "can", "could", "will", "would", "shall", "need", "needs",
}


# Place/time/sequence adverbs: "as described below" -- a word in front of
# one of these describes nothing ("described" is not a detail of "below").
_ADVERBS = {
    "below", "above", "here", "there", "later", "earlier", "again", "first", "then", "now",
    "once", "twice", "too", "also", "only", "even", "still", "back", "away", "out", "off",
    "up", "down", "over", "under", "beforehand", "afterwards", "overnight", "together", "instead",
}


_PRONOUNS = {"which", "that", "who", "whom", "it", "they", "this", "these", "those", "he", "she", "we", "you"}
_COMMON_VERBS = {
    verb + ending
    for verb in ("contain", "include", "require", "allow", "show", "give", "cause", "become",
                 "remain", "provide", "produce", "indicate", "mean", "need", "help", "make",
                 "take", "appear", "seem", "occur", "depend", "consist", "represent", "prevent",
                 "avoid", "ensure")
    for ending in ("", "s")
}


def _technical_terms(text: str) -> list[list[str]]:
    """Runs of three or more consecutive content words -- the manual's
    compound technical terms ("leukocyte type number fractions",
    "sodium nitroprusside crystals") -- as lists of loose stems."""
    terms: list[list[str]] = []
    run: list[str] = []
    for token in re.findall(r"[a-z]+(?:-[a-z]+)*|[^a-z\s]", text.casefold()) + ["."]:
        if (
            token[0].isalpha() and len(token) >= 3 and token not in _ROLE_STOP
            and token not in _ROLE_BREAK and token not in _AUXILIARIES and token not in _ADVERBS
            and token not in _PRONOUNS and token not in QUESTION_WORDS
            and token not in _COMMON_VERBS and not token.endswith(("ed", "ing"))
            and stem(token) not in _ACTION_BY_ROOT
        ):
            run.append(_loose(stem(token)))
            continue
        if len(run) >= 3:
            terms.append(run)
        run = []
    return terms


def _term_intact(term: list[str], generated_sequence: list[str]) -> bool:
    """The term's words appear in order and together -- or as "<head> of
    <rest>" ("crystals of sodium nitroprusside")."""
    size = len(term)
    head_first = [term[-1], "of", *term[:-1]]
    for start in range(len(generated_sequence)):
        if generated_sequence[start:start + size] == term:
            return True
        if generated_sequence[start:start + size + 1] == head_first:
            return True
    return False


# Words that only measure the noun after them ("a sufficient number of
# crystals" = "enough crystals"): never the thing a description is about.
_MEASURE_WORDS = {
    "number", "numbers", "amount", "amounts", "quantity", "quantities", "volume", "volumes",
    "portion", "part", "parts", "piece", "pieces", "series", "range", "kind", "kinds", "type", "types",
    "sort", "sorts", "lot", "lots", "variety",
}


def _snippet(text: str, word: str, width: int = 3) -> str:
    """The source words around `word` ("at the height of an episode")."""
    tokens = re.findall(r"\S+", text)
    for index, token in enumerate(tokens):
        if token.casefold().strip(".,;:()").startswith(word.casefold()[:5]):
            return " ".join(tokens[max(0, index - width):index + width + 1])
    return word


def _premodifiers(text: str) -> dict[str, str]:
    """Descriptive words directly in front of another content word, i.e.
    the modifiers inside a noun phrase ("liquid frothy saliva" -> liquid,
    frothy; "thick film" -> thick), found without needing the noun to be an
    entity in the graph. Verbs, qualifiers and function words are skipped."""
    tokens = re.findall(r"[a-z]+(?:-[a-z]+)*|[^a-z\s]", text.casefold())
    modifiers: dict[str, str] = {}
    for index, (word, following) in enumerate(zip(tokens, tokens[1:])):
        if not (word[0].isalpha() and following[0].isalpha()):
            continue
        # "stains, which contain essential dyes": after a relative or
        # personal pronoun comes the verb, not a description.
        if (index and tokens[index - 1] in _PRONOUNS) or word in _COMMON_VERBS:
            continue
        # "Blood specimens should ...": a word before a verb is the subject,
        # not a description; generic nouns ("specimen") describe nothing.
        if following in _AUXILIARIES or following in _ADVERBS or stem(word) in GENERIC_ENTITY_KEYS:
            continue
        if (
            len(word) >= 4 and len(following) >= 3
            and word not in _ROLE_STOP and word not in _ROLE_BREAK
            and following not in _ROLE_STOP and following not in _ROLE_BREAK
            and word not in QUESTION_WORDS and not word.endswith(("ly", "ing"))
            and stem(word) not in _ACTION_BY_ROOT
            and word not in _EXTRA_QUALIFIERS
        ):
            before = tokens[index - 1] if index and tokens[index - 1] in {"most", "more", "less", "least", "very"} else ""
            modifiers[word] = " ".join(part for part in (before, word, following) if part)
    return modifiers


def compare_with_fact(
    fact: dict[str, Any],
    generated: str,
    paraphrased: Callable[..., bool] | None = None,
    list_items: str = "",
    covered: Callable[[str, str], bool] | None = None,
    similarity: Callable[[str, str], float] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Neo4j's verdict on an LLM sentence, compared in both directions with
    the Fact of its source sentence. Returns four lists of reasons --
    (meaning changed, unsupported claim, information dropped, review) --
    all empty when the graph supports the sentence. A difference in
    wording alone is never any of them.

    Meaning changed (source -> answer, something replaced):
      - a quantity replaced by another, or one the graph does not record;
      - a condition word gone (before/after, alternatives, negation,
        if/unless, until, at least/at most) -- for a sentence introducing a
        list, the LLM's rewrite of the list items (`list_items`) counts, so
        "either:" followed by "... or ..." keeps the choice;
      - a qualifier replaced by another ("deeply" -> "forcefully").
    Unsupported claim (answer -> source): a statement of the LLM sentence
    that the source sentence does not make, in words or in meaning.
    Information dropped (source -> answer, something gone): a whole
    statement (claim), a quantity, a qualifier ("Just before" -> "Before"),
    or a descriptive detail ("liquid frothy saliva" -> "frothy saliva").
    Review (the graph cannot tell a synonym from a change): the lab action
    replaced by a different one (only the closest replacement is named:
    "expectorate -> cough", not every new verb), a role of the action
    (destination, material, purpose, source, place) or an entity it
    involves no longer there.
    `paraphrased(term, text)` and `covered(claim, text)` are meaning-
    similarity checks that clear synonyms; `similarity(a, b)` ranks which
    new action replaced a dropped one."""
    def kept(term: str) -> bool:
        return bool(paraphrased and paraphrased(term, generated))

    changed: list[str] = []
    unsupported: list[str] = []
    dropped: list[str] = []
    review: list[str] = []
    source = fact.get("text") or ""

    recorded_quantities = set(fact.get("quantities") or [])
    generated_quantities = fact_quantities(generated)
    missing = sorted(recorded_quantities - generated_quantities)
    added = sorted(generated_quantities - recorded_quantities)
    if missing and added:
        changed.append(f"quantity changed: {', '.join(missing)} -> {', '.join(added)}")
    elif added:
        changed.append("quantity not in the source: " + ", ".join(added))
    elif missing:
        dropped.append("quantity left out: " + ", ".join(missing))

    lost_claims = [claim for claim in fact.get("claims") or [] if covered and not covered(claim, generated)]
    generated_conditions = _normalize_for_conditions(f"{generated} {list_items}")
    # A condition word inside a statement that was left out entirely is
    # reported with that statement (information dropped), not again here.
    kept_source = source
    for claim in lost_claims:
        kept_source = kept_source.replace(claim, " ")
    source_conditions = _normalize_for_conditions(kept_source)
    missing_conditions = [
        label for label, in_source, kept_pattern in CONDITION_WORDS
        if label in (fact.get("conditions") or [])
        and len(re.findall(kept_pattern, generated_conditions))
        < max(1, len(re.findall(in_source, source_conditions)))
    ]
    if missing_conditions:
        changed.append("condition dropped: " + ", ".join(missing_conditions))

    # Qualifiers are compared statement by statement: each source statement
    # with the LLM statement that says the same thing, so a qualifier of
    # one statement ("dry thoroughly") is never matched against another's
    # ("gently fanning"). Word forms count as the same qualifier
    # ("urgently" / "urgent results", "gentle heat" / "gently").
    statements = fact_claims(generated) or [generated]

    def aligned(claim: str) -> str:
        if len(statements) == 1:
            return statements[0]
        if similarity:
            return max(statements, key=lambda statement: similarity(claim, statement))
        claim_roots = roots(claim)
        return max(statements, key=lambda statement: len(claim_roots & roots(statement)))

    def has_qualifier(qualifier: str, qualifiers: set[str], text: str, own: str) -> bool:
        tokens = set(re.findall(r"[a-z]+", text.casefold()))
        if _QUALIFIER_GROUP.get(qualifier, frozenset({qualifier})) & (qualifiers | tokens):
            return True
        # "urgently" is kept by "urgent results" -- but "deeply" is not kept
        # by the "deep" of "a deep breath" that was already there.
        own_tokens = set(re.findall(r"[a-z]+", own.casefold()))
        forms = {qualifier[:-2], qualifier[:-1] + "e"} if qualifier.endswith("ly") else set()
        return bool((forms - own_tokens) & tokens)

    source_claims = [claim for claim in (fact.get("claims") or [source]) if claim not in lost_claims] or [source]
    for claim in source_claims:
        statement = aligned(claim)
        claim_qualifiers = fact_qualifiers(claim)
        statement_qualifiers = fact_qualifiers(statement)
        missing_qualifiers = sorted(q for q in claim_qualifiers if not has_qualifier(q, statement_qualifiers, statement, claim))
        added_qualifiers = sorted(q for q in statement_qualifiers if not has_qualifier(q, claim_qualifiers, claim, statement))
        if missing_qualifiers and added_qualifiers:
            changed.append(f"'{', '.join(missing_qualifiers)}' changed to '{', '.join(added_qualifiers)}' in \"{claim}\"")
        elif missing_qualifiers:
            dropped.append(f"'{', '.join(missing_qualifiers)}' left out of \"{claim}\"")
        elif added_qualifiers:
            review.append(f"'{', '.join(added_qualifiers)}' added to \"{claim}\"")

    for claim in lost_claims:
        dropped.append(f"left out: '{claim}'")

    if covered and source:
        for statement in fact_claims(generated):
            if not covered(statement, f"{source} {list_items}"):
                unsupported.append(f"'{statement}'")

    generated_roots = {
        _loose(root)
        for root in roots(american_spelling(re.sub(r"['’]s\b", "", generated.casefold())))
    }
    generated_tokens = set(re.findall(r"[a-z0-9]+", generated.casefold()))

    def present(word: str) -> bool:
        parts = re.findall(r"[a-z0-9]+", word.casefold())
        return bool(parts) and all(part in generated_tokens for part in parts)

    missing_entities = sorted(
        entity_display(name) for name in fact.get("entities") or []
        if not entity_is_generic(name)
        and (key_words := {_loose(w) for w in entity_key(name).split()}) and not key_words <= generated_roots
        and not present(entity_display(name)) and not kept(entity_display(name))
    )
    # One change reported once: words already inside a reported lost
    # statement or a missing entity are not reported again.
    reported = {_loose(word) for name in missing_entities for word in entity_key(name).split()}
    reported |= {_loose(root) for claim in lost_claims for root in roots(claim)}

    # A detail counts as kept by a synonym only where the LLM put a word
    # of its own ("appropriate preparation" -> "proper preparation"), not
    # where it simply left the word out ("liquid frothy saliva" -> "frothy
    # saliva" still resembles the source because "frothy saliva" remains).
    source_words = {_loose(stem(word)) for word in re.findall(r"[a-z0-9]+", source.casefold())}
    detail_context = dict(_premodifiers(source))
    for attribute in fact.get("attributes") or []:
        word, entity = attribute.split("|", 1)
        if stem(word) not in GENERIC_ENTITY_KEYS:
            detail_context.setdefault(word, f"{word} {entity}")
    details = sorted(
        word for word, context in detail_context.items()
        if (loose := _loose(stem(word))) not in generated_roots and not present(word)
        and not (_QUALIFIER_GROUP.get(word, frozenset()) & generated_tokens)
        and loose not in reported
        and not (paraphrased and (
            paraphrased(context, generated, DETAIL_SIMILARITY, source_words)
            or paraphrased(word, generated, DETAIL_SIMILARITY, source_words)
        ))
    )
    if details:
        dropped.append("detail left out: " + ", ".join(details))

    # The noun a description belongs to must survive too: "gentle heat" ->
    # "warm light" keeps the idea of gentleness but changes what is used;
    # "small drop" -> "small amount" generalises the thing. Likewise every
    # part of a hyphenated compound ("heat-fixed" -> "fixed permanently"
    # loses the heat). A synonym in its place ("specimen" -> "sample")
    # clears it; a different word is reported as a change of meaning.
    def replacement_for(term: str, threshold: float = SYNONYM_SIMILARITY) -> bool:
        return bool(paraphrased and paraphrased(term, generated, threshold, source_words))

    for word, context in _premodifiers(source).items():
        head = context.split()[-1]
        # The phrase is judged as a whole ("gentle heat" vs "warm light"):
        # a single new word can resemble the head ("warm" ~ "heat") while
        # the phrase says something else.
        if (
            len(head) >= 3 and stem(head) not in GENERIC_ENTITY_KEYS and head not in QUESTION_WORDS
            and head not in _MEASURE_WORDS
            and _loose(stem(head)) not in generated_roots and not present(head)
            and _loose(stem(head)) not in reported
            and (not replacement_for(context, DETAIL_SIMILARITY) or re.search(
                rf"\b{re.escape(word)}\s+(?:{'|'.join(sorted(_MEASURE_WORDS))})\b", generated.casefold()
            ))
        ):
            generalised = re.search(
                rf"\b{re.escape(word)}\s+({'|'.join(sorted(_MEASURE_WORDS))})\b", generated.casefold()
            )
            if generalised:
                dropped.append(f"'{context}' generalised to '{word} {generalised.group(1)}'")
            else:
                changed.append(f"'{context}' no longer refers to '{head}'")
    for compound in sorted(set(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", source.casefold()))):
        parts = compound.split("-")
        # A part is kept in any form ("fixed" by "fix", "heat" by "heating").
        lost = [
            part for part in parts
            if len(part) >= 3 and _loose(stem(part)) not in generated_roots and not any(
                token.startswith(part) or (len(token) >= 3 and part.startswith(token))
                for token in generated_tokens
            )
        ]
        if lost and len(lost) < len(parts) and not present(compound) and not replacement_for(compound, DETAIL_SIMILARITY):
            changed.append(f"'{compound}' lost '{', '.join(lost)}'")

    recorded_actions = fact.get("actions") or []
    allowed = {root for action in recorded_actions for root in _ACTION_BY_ROOT.get(action, {action})}
    generated_stems = {_loose(root) for root in _all_stems(generated)}
    dropped_actions = [
        _ACTION_WORD.get(action, action) for action in recorded_actions
        if not ({_loose(r) for r in _ACTION_BY_ROOT.get(action, frozenset({action}))} & generated_stems)
        and not kept(_ACTION_WORD.get(action, action))
    ]
    new_actions = [_ACTION_WORD.get(action, action) for action in sorted(fact_actions(generated) - allowed)]
    for action in dropped_actions:
        if not new_actions:
            break
        # The replacement is the new verb standing where the old one stood
        # ("to expectorate directly into" -> "to cough directly into", not
        # the "filled" of "filled with 25 ml"): most shared neighbouring
        # words, then meaning similarity as a tie-break.
        source_context = _context_roots(source, action)
        replacement = max(
            new_actions,
            key=lambda candidate: (
                len(source_context & _context_roots(generated, candidate)),
                similarity(action, candidate) if similarity else 0.0,
            ),
        )
        review.append(f"action changed: {action} -> {replacement}")

    # A compound technical term whose words are all still there but no
    # longer in order ("leukocyte type number fractions" -> "the number of
    # leukocyte types fractions") has become a different, garbled term.
    generated_sequence = [
        "of" if token == "of" else _loose(stem(token))
        for token in re.findall(r"[a-z]+(?:-[a-z]+)*", generated.casefold())
        if token == "of" or (token not in _ROLE_STOP and len(token) >= 3)
    ]
    def together(term: list[str]) -> bool:
        width = len(term) + 2
        return any(
            set(term) <= set(generated_sequence[start:start + width])
            for start in range(max(1, len(generated_sequence) - width + 1))
        )

    for term in _technical_terms(source):
        if together(term) and not _term_intact(term, generated_sequence):
            words_in_order = [w for w in re.findall(r"[a-z]+(?:-[a-z]+)*", source.casefold()) if _loose(stem(w)) in term]
            review.append(f"term reworded: '{' '.join(words_in_order[:len(term)])}'")
            break

    # A list item ("— estimating the number of thrombocytes") rewritten
    # into a longer statement with new content words has gained a claim.
    if re.match(r"^\s*[-—•]", source):
        source_roots = {_loose(root) for root in content_roots(source)}
        added_words = sorted({
            _loose(root) for root in content_roots(generated)
            if _loose(root) not in source_roots and not any(ch.isdigit() for ch in root)
        })
        if len(added_words) >= 3 and len(added_words) >= 0.5 * max(1, len(source_roots)):
            review.append(
                "list item rewritten as a separate statement -- its link to the list is lost "
                "and wording was added: " + ", ".join(added_words)
            )

    lost_roles = [
        role for role in fact.get("roles") or []
        if (head := _loose(stem(role.split("=", 1)[1]))) not in generated_roots
        and head not in reported and not kept(role.split("=", 1)[1])
    ]
    def role_kind(role: str) -> str:
        kind, head = role.split("=", 1)
        return "time" if kind == "place" and head in TIME_WORDS else kind

    lost_times = [role for role in lost_roles if role_kind(role) == "time"]
    lost_others = [role for role in lost_roles if role_kind(role) != "time"]
    if lost_times:
        dropped.append("time detail left out: " + ", ".join(
            f"\"{_snippet(source, role.split('=', 1)[1])}\"" for role in lost_times
        ))
    if lost_others:
        review.append("no longer kept: " + ", ".join(
            f"{role_kind(role)} \"{_snippet(source, role.split('=', 1)[1])}\"" for role in lost_others
        ))
    if missing_entities:
        dropped.append("no longer named: " + ", ".join(missing_entities[:4]))
    return changed, unsupported, dropped, review


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
        self._generator = None
        self._generator_tokenizer = None
        self._embedder: SentenceTransformer | None = None
        self._chunk_embeddings: np.ndarray | None = None
        self._model_lock = threading.Lock()
        self._rewrite_lock = threading.Lock()
        self._rewrite_cache: dict[str, str] = {}
        self._term_embeddings: dict[str, np.ndarray] = {}

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

    def _ensure_generator(self) -> None:
        if self._generator is not None:
            return
        with self._model_lock:
            if self._generator is not None:
                return
            if not torch.cuda.is_available():
                # PyTorch's CPU thread pool defaults to half the logical
                # core count on some setups; this machine's generation
                # speed is CPU-bound (no usable GPU), so give it every
                # thread the machine actually has.
                torch.set_num_threads(os.cpu_count() or 4)
            self._generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
            generator = AutoModelForCausalLM.from_pretrained(
                GENERATOR_MODEL,
                torch_dtype="auto",
                device_map="auto" if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                generator.to("cpu")
            generator.eval()
            self._generator = generator

    def claim_covered(self, claim: str, text: str) -> bool:
        """Does `text` still say `claim` (one statement of a source
        sentence)? Yes when it keeps most of the claim's content words, or
        when one of its sentences means the same (local embedding model).
        Measured on real rewrites here: kept statements score 0.88-1.00
        (e.g. "check that a sufficient amount of sputum has been produced"
        vs "ensure that a sufficient volume ... has been collected" 0.89),
        dropped ones 0.47-0.65 ("a positive result may be obvious before
        this time" 0.65) -- CLAIM_SIMILARITY sits between."""
        claim_roots = {r for r in content_roots(claim) if not any(ch.isdigit() for ch in r)}
        text_roots = {_loose(r) for r in roots(text)}
        if claim_roots and len({_loose(r) for r in claim_roots} & text_roots) >= 0.8 * len(claim_roots):
            return True
        parts = [p for p in re.split(r"(?<=[.;!?])\s+|\s+[—–]\s+", text) if len(words(p)) >= 3] + [text]
        self._ensure_embedder()
        missing = [t for t in [claim, *parts] if t not in self._term_embeddings]
        if missing:
            vectors = self._embedder.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self._term_embeddings.update(zip(missing, vectors))
        target = self._term_embeddings[claim]
        return max(float(target @ self._term_embeddings[p]) for p in parts) >= CLAIM_SIMILARITY

    def term_similarity(self, a: str, b: str) -> float:
        self._ensure_embedder()
        missing = [t for t in (a, b) if t not in self._term_embeddings]
        if missing:
            vectors = self._embedder.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self._term_embeddings.update(zip(missing, vectors))
        return float(self._term_embeddings[a] @ self._term_embeddings[b])

    def paraphrased(
        self, term: str, text: str, threshold: float = SYNONYM_SIMILARITY,
        source_words: set[str] | None = None,
    ) -> bool:
        """Does `text` say `term` in other words? The local embedding model
        compares the term with every 1-3 word phrase of the text. Measured on
        this manual's wording, real synonyms mostly score above ~0.75
        (toilet/lavatory 0.75, nasal/nose 0.92, 60 grams/60g 0.92) and real
        changes mostly below (keep/use 0.72, expectorate/cough 0.58) -- but
        the ranges overlap (thick film/thin film 0.86), so a known
        contrastive counterpart in the text always counts as a change."""
        term_words = set(re.findall(r"[a-z]+", term.casefold()))
        text_words = set(re.findall(r"[a-z]+", text.casefold()))
        for pair in CONTRASTIVE_TERM_PAIRS:
            for word in pair & term_words:
                if (pair - {word}) & text_words and word not in text_words:
                    return False
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
        grams = sorted({
            " ".join(tokens[i:i + n])
            for n in (1, 2, 3) for i in range(len(tokens) - n + 1)
            if any(len(w) >= 3 and w not in _ROLE_STOP and w not in _ROLE_BREAK for w in tokens[i:i + n])
            and (source_words is None or any(_loose(stem(w)) not in source_words for w in tokens[i:i + n]))
        })
        if not grams:
            return False
        self._ensure_embedder()
        missing = [t for t in [term.casefold(), *grams] if t not in self._term_embeddings]
        if missing:
            vectors = self._embedder.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            self._term_embeddings.update(zip(missing, vectors))
        target = self._term_embeddings[term.casefold()]
        return max(float(target @ self._term_embeddings[g]) for g in grams) >= threshold

    def _ensure_embedder(self) -> None:
        if self._chunk_embeddings is not None:
            return
        with self._model_lock:
            if self._chunk_embeddings is not None:
                return
            embedder = SentenceTransformer(EMBED_MODEL)
            # Corpus is small (a few hundred chunks) -- brute-force cosine
            # similarity over a plain in-memory matrix is exact and, at
            # this size, effectively instant, so there is no need for an
            # approximate-search index (FAISS et al.) built for corpora
            # orders of magnitude larger than this one.
            embeddings = embedder.encode(
                [chunk.text for chunk in self.chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._embedder = embedder
            self._chunk_embeddings = np.asarray(embeddings)

    def semantic_candidates(self, query: str, top_k: int = 8) -> list[tuple[int, float]]:
        """Rank chunks by embedding similarity to the query -- a meaning-
        based alternative to literal keyword matching, so a chunk that
        answers the question in different words (or whose one shared
        keyword is buried in an unrelated passage) is not simply invisible
        to the search the way it is to substring/term-count matching."""
        self._ensure_embedder()
        query_vec = self._embedder.encode(
            [EMBED_QUERY_PREFIX + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._chunk_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(index), float(scores[index])) for index in top_indices]

    def relevant_window(
        self,
        query: str,
        anchor_index: int,
        max_radius_forward: int = 10,
        max_radius_backward: int = 2,
    ) -> list[int]:
        """Grow a verified chunk into a window of neighbours actually worth
        extracting from, instead of handing extract() a flat, unconditional
        block of however-many following chunks (which let it wander into a
        different, merely-nearby procedure -- observed live: asked to
        prepare Giemsa stain, it walked past the Giemsa steps into a later,
        unrelated staining method that also happened to start a numbered
        list). Step outward one chunk at a time in each direction; a
        neighbour is included, and expansion continues past it, if either
        it is itself still relevant to the query, or the chunk just
        confirmed contains an open numbered step -- chunk boundaries
        routinely cut a procedure step in half, and its continuation often
        carries none of the question's own vocabulary for the reranker to
        recognise. Expansion in that direction stops the first time neither
        holds.

        The two directions are not symmetric. Forward keeps the wider
        radius a multi-chunk procedure has always needed here. Backward is
        kept short on purpose: a document routinely opens a topic with a
        general definition/overview before its specific procedure or
        finding (observed live: walking back from a chunk describing what
        a positive CATT reaction looks like reached, a few chunks earlier,
        a plain "CATT is a serological test used to diagnose..." sentence
        -- itself relevant enough to individually clear the floor, but it
        then outcompeted the anchor's own, actually-requested content in
        extract()'s selection). A short leash still catches a step split
        across a page break without reopening that door.
        """
        window = {anchor_index}
        for direction, max_radius in ((1, max_radius_forward), (-1, max_radius_backward)):
            index = anchor_index
            for _ in range(max_radius):
                neighbour = index + direction
                if neighbour < 0 or neighbour >= len(self.chunks):
                    break
                if not NUMBERED_STEP_LINE_RE.search(self.chunks[index].text):
                    score = float(self.reranker.predict(
                        [[query, self.chunks[neighbour].text[:600]]],
                        show_progress_bar=False,
                    )[0])
                    if score <= -2:
                        break
                window.add(neighbour)
                index = neighbour
        return sorted(window)


    def rephrase_units(self, question: str, units: list[dict[str, str]]) -> list[dict[str, Any]]:
        """The LLM's answer: each verified unit of the extractive answer
        rewritten on its own, so nothing can be dropped or reordered.
        `units` are the answer's evidence items ({"text", "chunk_id"});
        returns [] if the generator is unavailable. No judgement is made
        here -- the LLM output is judged against Neo4j (see
        compare_with_fact) by the caller. Rewrites are cached per unit, so
        the PDF-only and PDF + Neo4j views of one question show the same
        LLM sentences from a single generation."""
        try:
            self._ensure_generator()
        except Exception:
            return []
        items: list[dict[str, Any]] = []
        for unit in units:
            text = (unit.get("text") or "").strip()
            if not text or RUNNING_HEADER_RE.match(text):
                continue
            marker = re.match(r"^\s*(\d+[.)])\s+", text)
            items.append({
                "chunk_id": unit.get("chunk_id"),
                "source_raw": text,
                "marker": f"{marker.group(1)} " if marker else "",
                "source": text[marker.end():] if marker else text,
                "llm": None,
                "status": "unchanged",
                "problems": [],
                "graph_fact": None,
            })
        to_rewrite = [
            index for index, item in enumerate(items)
            if len(words(item["source"])) >= 6
            and not re.match(r"^\s*Fig(?:ure)?\.?\s*\d", item["source"], re.I)
        ]
        rewritten = self._rewrite_units([items[i]["source"] for i in to_rewrite])
        for index, candidate in zip(to_rewrite, rewritten):
            if candidate:
                items[index]["llm"] = candidate
                items[index]["status"] = "unchecked"
        return items

    def _rewrite_units(self, bodies: list[str]) -> list[str | None]:
        # One lock around lookup + generation: when both answer panels ask
        # at once, the second waits and then reuses the first's rewrites
        # instead of running the CPU-bound model a second time.
        with self._rewrite_lock:
            pending = list(dict.fromkeys(body for body in bodies if body not in self._rewrite_cache))
            if pending:
                for body, text in zip(pending, self._generate_rewrites(pending)):
                    if text:
                        self._rewrite_cache[body] = text
            return [self._rewrite_cache.get(body) for body in bodies]

    def _generate_rewrites(self, bodies: list[str]) -> list[str | None]:
        if not bodies:
            return []
        # The question is deliberately not shown: given it, the model
        # pulled the question's own nouns into unrelated steps ("a tube
        # containing 2.0 ml of trisodium citrate").
        system = (
            "Rewrite the given text from a laboratory manual in your own "
            "clear, fluent, natural English so it is easy to read. Keep "
            "every fact, number, unit, reagent, condition and the order of "
            "actions exactly as they are. Do not add, remove, summarise or "
            "explain anything. Reply with the rewritten text only."
        )
        tokenizer = self._generator_tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": body},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for body in bodies
        ]
        device = next(self._generator.parameters()).device
        results: list[str | None] = []
        batch_size = 8
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            batch_bodies = bodies[start:start + batch_size]
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            longest = max(
                len(tokenizer(body, add_special_tokens=False)["input_ids"])
                for body in batch_bodies
            )
            try:
                with torch.inference_mode():
                    output = self._generator.generate(
                        **encoded,
                        max_new_tokens=min(400, longest * 2 + 40),
                        do_sample=False,
                        repetition_penalty=1.04,
                        pad_token_id=tokenizer.pad_token_id,
                    )
            except Exception:
                results.extend([None] * len(batch_prompts))
                continue
            prompt_length = encoded["input_ids"].shape[1]
            for row in output:
                text = tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
                results.append(" ".join(text.split()) or None)
        return results

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
            protected = re.sub(r"\bq\.s\.", "q§s§", protected, flags=re.I)
            sentences = [
                part.replace("Fig§", "Fig.").replace("q§s§", "q.s.")
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
                protected = re.sub(r"\bq\.s\.", "q§s§", protected, flags=re.I)
                raw_sentences = [
                    part.replace("Fig§", "Fig.").replace("q§s§", "q.s.")
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
                    # A chunk-overlap boundary can cut a step's own sentence
                    # short (the same-chunk copy ends mid-sentence, a
                    # neighbouring chunk has the complete continuation) --
                    # that must still win on completeness regardless of
                    # which one is "closer", so drop any candidate that is
                    # only a strict prefix of another before ranking by
                    # position at all.
                    def is_strict_prefix_of_another(unit: Unit) -> bool:
                        normalized = normalize_for_exact_check(unit.text)
                        return any(
                            other is not unit
                            and normalized != normalize_for_exact_check(other.text)
                            and normalize_for_exact_check(other.text).startswith(normalized)
                            for other in options
                        )
                    options = [
                        unit for unit in options
                        if not is_strict_prefix_of_another(unit)
                    ]
                    # A chunk can hold two independent numbered lists that
                    # both restart at 1 (e.g. "Collection of samples" and,
                    # further down the same page, "Performing the test"):
                    # once past step 2, both lists have their own genuine
                    # "3.", and ranking by text length alone can pick the
                    # wrong list's step just for reading longer. The right
                    # step is the one immediately after the last one in the
                    # source; that beats raw length whenever both live in
                    # the same chunk (a genuine cross-chunk completion, the
                    # case just above, no longer competes on length at all
                    # by this point).
                    def order_distance(unit: Unit) -> int:
                        if unit.chunk_index != last_unit.chunk_index:
                            return 999
                        return abs(unit.order - last_unit.order)

                    selected_step = max(
                        options,
                        key=lambda unit: (
                            not bool(re.search(r"(?:\bFig\.?|\(|\[)\s*$", unit.text)),
                            -order_distance(unit),
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
                        elif following is None and distinctive_subject:
                            # Sentences right after the final step, in the
                            # same chunk, often finish the procedure without
                            # being numbered (e.g. "Examine at least 100
                            # erythrocytes. Keep a careful count of ...
                            # reticulocytes."). Take the contiguous run up
                            # to the last on-topic sentence within a short
                            # window -- an off-topic sentence in between
                            # still belongs to that run -- but stop where the
                            # text starts defining a different concept
                            # ("X is called Y"), which marks the next section
                            # of the manual rather than more of this one.
                            window = sorted(
                                (
                                    c for c in ranked_units
                                    if c.chunk_index == unit.chunk_index
                                    and unit.order < c.order <= unit.order + 6
                                ),
                                key=lambda c: c.order,
                            )
                            run: list[Unit] = []
                            hit_boundary = False
                            for candidate in window:
                                if (
                                    re.match(r"^\s*\d+[.)]\s+", candidate.text)
                                    or re.match(r"^\s*Fig(?:ure)?\.?\s*\d", candidate.text, re.I)
                                    or introduces_other_concept(candidate.text, need.query)
                                    or candidate.text.rstrip().endswith(":")
                                ):
                                    hit_boundary = True
                                    break
                                run.append(candidate)
                            # Keep up to the last on-topic sentence. When a
                            # boundary (next numbered item, figure caption,
                            # a new concept being defined, or a sentence
                            # opening its own ":" sub-list) closes the tail
                            # right after the final step, the one or two
                            # sentences before it finish that step even when
                            # they don't repeat the subject ("Let the slide
                            # dry completely in the air.").
                            last_on_topic = max(
                                (
                                    position for position, candidate in enumerate(run)
                                    if distinctive_subject & roots(candidate.text)
                                ),
                                default=-1,
                            )
                            keep = last_on_topic + 1
                            if hit_boundary:
                                keep = max(keep, min(2, len(run)))
                            expanded.extend(run[:keep])
                    return expanded

                if len(sequence) >= 2:
                    lead_candidates = [
                        unit for unit in filtered
                        if unit not in sequence
                        # A lead-in must precede step 1 in the source; a
                        # sentence from after the last step is a trailing
                        # remark (handled by with_continuations), and
                        # promoting it to the front both misorders the
                        # answer and repeats it at the end.
                        and (unit.chunk_index, unit.order) < (start.chunk_index, start.order)
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

                def anchor_distance(unit: Unit) -> tuple[int, int]:
                    # A block can list the same field label for several
                    # entries back to back (e.g. one identification entry
                    # per species); prefer the field nearest a mention of
                    # the question's OTHER subject terms (its entity name,
                    # not the field label itself), so the right entry wins
                    # even when a wrong-entry sibling reads better in
                    # isolation to the reranker. A dedicated "Entity name
                    # (Fig. N.NN)" heading naming the entity is a far
                    # stronger anchor than an ordinary sentence that merely
                    # mentions it in passing (e.g. one entry's description
                    # comparing itself to another: "Similar to the eggs of
                    # Clonorchis sinensis...") -- that kind of cross-
                    # reference can sit closer, by raw distance, than the
                    # entity's own heading is to its own fields, so a
                    # heading anchor always outranks a plain-text one
                    # regardless of which is numerically closer.
                    anchor_terms = required_subject_terms - field_label_roots(unit.text)
                    if not anchor_terms:
                        return (1, 0)
                    heading_distances = [
                        abs(other.order - unit.order)
                        for other in ranked_units
                        if other.chunk_index == unit.chunk_index
                        and anchor_terms & roots(other.text)
                        and ENTITY_FIG_HEADING_RE.match(other.text.strip())
                    ]
                    if heading_distances:
                        return (0, min(heading_distances))
                    any_distances = [
                        abs(other.order - unit.order)
                        for other in ranked_units
                        if other.chunk_index == unit.chunk_index
                        and anchor_terms & roots(other.text)
                    ]
                    return (1, min(any_distances) if any_distances else 999)

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

    def extend_across_chunk_boundary(self, need: Need, units: list[Unit]) -> list[Unit]:
        """Check the chunk immediately before and after the answer's own
        chunks for a genuine continuation, so a real answer split across a
        chunk cut is never silently dropped -- purely additive (nothing
        already selected is ever removed or replaced). Sharing a subject
        word is not enough to cross into a neighbour (a whole chapter can
        repeat the same word without being the same answer); the neighbour
        must itself read as relevant to the question via the same semantic
        reranker used everywhere else, past the same relevance floor used
        elsewhere in this file. Chains at most two chunks each way.
        """
        if not units:
            return units
        extended = list(units)
        for direction in (1, -1):
            chunk_idx = (extended[-1] if direction == 1 else extended[0]).chunk_index
            for _ in range(2):
                neighbor_idx = chunk_idx + direction
                if not 0 <= neighbor_idx < len(self.chunks):
                    break
                neighbor_texts = self.units(self.chunks[neighbor_idx].text)
                if not neighbor_texts:
                    break
                probe_text = neighbor_texts[0] if direction == 1 else neighbor_texts[-1]
                # An overlapping chunk boundary restates content right at
                # the edge (the same sliding window seen elsewhere in this
                # file); that is not new material, and including it as
                # well can corrupt a strict check downstream (e.g. two
                # copies of numbered step 1 breaking need_complete's
                # sequence check) -- stop rather than duplicate.
                normalized_probe = normalize_for_exact_check(probe_text)
                # A chunk-overlap boundary can cut a sentence off mid-way
                # (the same sliding window seen elsewhere in this file):
                # the already-included unit is then a strict prefix of
                # this neighbour's fuller version of that same sentence,
                # not new material -- replace the truncated copy with the
                # complete one instead of tacking on a near-duplicate.
                replaced_truncated = False
                already_have_fuller = False
                for index, existing in enumerate(extended):
                    normalized_existing = normalize_for_exact_check(existing.text)
                    if normalized_existing == normalized_probe:
                        already_have_fuller = True
                        break
                    if normalized_probe and normalized_probe in normalized_existing:
                        # probe is fully contained in what we already have
                        # (a prefix, a suffix -- e.g. a heading-plus-fact
                        # unit already carries this same fact on its own --
                        # or a middle excerpt of it): nothing new here.
                        already_have_fuller = True
                        break
                    if normalized_existing and normalized_existing in normalized_probe:
                        if normalized_probe.startswith(normalized_existing):
                            extended[index] = Unit(
                                existing.chunk_index, existing.order, probe_text, existing.score,
                            )
                            replaced_truncated = True
                        else:
                            # existing appears somewhere within probe but
                            # not as its opening (a suffix or middle
                            # excerpt): probe still isn't new material.
                            already_have_fuller = True
                        break
                if replaced_truncated:
                    chunk_idx = neighbor_idx
                    continue
                if already_have_fuller:
                    break
                # A neighbouring section commonly restarts its own step
                # numbering from 1 (a distinct sub-procedure, e.g.
                # "collect the specimen" before "centrifuge it"): a
                # candidate step number that collides with one already in
                # the answer is a different procedure's step, not this
                # one's continuation, even when it scores well on its own.
                probe_number = re.match(r"^\s*(\d+)[.)]\s+", probe_text)
                if probe_number and any(
                    re.match(r"^\s*(\d+)[.)]\s+", existing.text)
                    and re.match(r"^\s*(\d+)[.)]\s+", existing.text).group(1)
                        == probe_number.group(1)
                    for existing in extended
                ):
                    break
                score = float(self.reranker.predict(
                    [[need.query, probe_text]], show_progress_bar=False
                )[0])
                if score <= -2:
                    break
                order = 0 if direction == 1 else len(neighbor_texts) - 1
                candidate = Unit(neighbor_idx, order, probe_text, score)
                if direction == 1:
                    extended.append(candidate)
                else:
                    extended.insert(0, candidate)
                chunk_idx = neighbor_idx
        return extended

    @staticmethod
    def blocks(text: str) -> list[str]:
        """The chunk's paragraphs in source order, headings included --
        the same cleaning units() applies, without its filtering."""
        cleaned = text.replace("\r", "")
        cleaned = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", cleaned)
        cleaned = re.sub(r"^\s*G\s+", "— ", cleaned, flags=re.M)
        cleaned = re.sub(r"\n(?=\s*\d+[.)]\s+[A-Z])", "\n\n", cleaned)
        cleaned = re.sub(r"(?<=[.!?])\s+(?=\d+[.)]\s+[A-Z])", "\n\n", cleaned)
        return [block for block in map(compact, re.split(r"\n\s*\n+", cleaned)) if block]

    def complete_section(self, need: Need, units: list[Unit]) -> list[Unit]:
        """Give a procedure answer the structure of the section it comes
        from. units() drops short unpunctuated lines, so sub-headings that
        separate alternative methods ("Using an autoclave" / "Boiling in
        detergent" / "Using formaldehyde solution or cresol") vanished and
        the methods read as one continuous sequence, while a later method
        with no numbered steps of its own was never reached. Three
        additions, all exact source text:
          - before step 1: the short heading/intro lines leading into it,
            back to the heading that names the question's subject;
          - between selected units: any sub-heading that sits between them;
          - after the last unit, when the next thing in the source is a
            sub-heading: keep going heading by heading, into the next chunk
            if the section continues there, while each heading's section is
            about the question's subject (by shared subject words or the
            reranker), and stop at the first unrelated one, a numbered
            section heading, a restarted numbered list, or a new concept
            being defined.
        """
        if need.answer_type != "procedure" or not units:
            return units
        subject = set(need.subject_terms) - {stem(term) for term in GENERIC_SUBJECT_ROOTS}
        if not subject:
            return units

        def is_noise(block: str) -> bool:
            return bool(
                re.match(r"^\d+\s+(?:Manual|Index)\b", block, re.I)
                or RUNNING_HEADER_RE.match(block)
                or re.match(r"^Fig(?:ure)?\.?\s*\d", block, re.I)
            )

        def is_section_number_heading(block: str) -> bool:
            return bool(re.match(r"^\d+(?:\.\d+)+\s+\S", block))

        def is_heading(block: str) -> bool:
            return (
                len(words(block)) <= 12
                and not re.search(r"[.!?:;]$", block)
                and not re.match(r"^\s*\d+[.)]\s", block)
                and not re.match(r"^[—•\-]", block)
                and not is_noise(block)
            )

        def heading_text(block: str) -> str:
            text = re.sub(r"^\d+(?:\.\d+)+\s+", "", block)
            return re.sub(r"(?<=[a-z])\d{1,2}$", "", text)

        def locate(unit: Unit, chunk_blocks: list[str]) -> int | None:
            # The block containing the unit; only if none does, a block that
            # makes up most of the unit (a unit spanning a block break) --
            # never a short heading that merely occurs inside the unit's
            # text ("Method" inside "This is the best method.").
            target = normalize_for_exact_check(unit.text)
            if not target:
                return None
            normalized_blocks = [normalize_for_exact_check(block) for block in chunk_blocks]
            for index, normalized in enumerate(normalized_blocks):
                if target in normalized:
                    return index
            for index, normalized in enumerate(normalized_blocks):
                if normalized and normalized in target and len(normalized) >= 0.5 * len(target):
                    return index
            return None

        def section_is_related(chunk_blocks: list[str], index: int) -> bool:
            window = [chunk_blocks[index]]
            for block in chunk_blocks[index + 1:]:
                if is_heading(block) or len(window) >= 3:
                    break
                if not is_noise(block):
                    window.append(block)
            if subject & roots(" ".join(window)):
                return True
            score = float(self.reranker.predict(
                [[need.query, " ".join(window)]], show_progress_bar=False
            )[0])
            return score > 0

        step_numbers = [
            int(match.group(1)) for unit in units
            if (match := re.match(r"^\s*(\d+)[.)]\s+", unit.text))
        ]
        max_step = max(step_numbers, default=0)

        def make(unit: Unit, text: str, chunk_index: int | None = None) -> Unit:
            return Unit(unit.chunk_index if chunk_index is None else chunk_index, unit.order, text, unit.score)

        result: list[Unit] = []
        # Before step 1.
        first = units[0]
        if re.match(r"^\s*1[.)]\s", first.text):
            chunk_blocks = self.blocks(self.chunks[first.chunk_index].text)
            position = locate(first, chunk_blocks)
            lead: list[str] = []
            index = (position if position is not None else 0) - 1
            while position is not None and index >= 0 and len(lead) < 4:
                block = chunk_blocks[index]
                index -= 1
                if is_noise(block):
                    continue
                if is_section_number_heading(block) or re.match(r"^\s*\d+[.)]\s", block):
                    break
                if is_heading(block):
                    lead.insert(0, heading_text(block))
                    if subject & roots(block):
                        break
                    continue
                if len(words(block)) <= 15 and block.endswith("."):
                    lead.insert(0, block)
                    continue
                break
            result.extend(make(first, text) for text in lead)
        # Sub-headings between selected units of the same chunk.
        for position, unit in enumerate(units):
            result.append(unit)
            following = units[position + 1] if position + 1 < len(units) else None
            if following is None or following.chunk_index != unit.chunk_index:
                continue
            chunk_blocks = self.blocks(self.chunks[unit.chunk_index].text)
            start, end = locate(unit, chunk_blocks), locate(following, chunk_blocks)
            if start is None or end is None or end <= start + 1:
                continue
            for block in chunk_blocks[start + 1:end]:
                if is_heading(block) or (
                    is_section_number_heading(block) and len(words(block)) <= 12
                ):
                    result.append(make(unit, heading_text(block)))
        # After the last unit, section by section.
        last = units[-1]
        chunk_index = last.chunk_index
        chunk_blocks = self.blocks(self.chunks[chunk_index].text)
        position = locate(last, chunk_blocks)
        if position is not None:
            existing = " ".join(normalize_for_exact_check(unit.text) for unit in result)
            index = position + 1
            while index < len(chunk_blocks) and is_noise(chunk_blocks[index]):
                index += 1
            if (
                index < len(chunk_blocks)
                and is_heading(chunk_blocks[index])
                and section_is_related(chunk_blocks, index)
            ):
                added = 0
                crossed = False
                while added < 12:
                    if index >= len(chunk_blocks):
                        if crossed or chunk_index + 1 >= len(self.chunks):
                            break
                        chunk_index += 1
                        crossed = True
                        chunk_blocks = self.blocks(self.chunks[chunk_index].text)
                        index = 0
                        continue
                    block = chunk_blocks[index]
                    index += 1
                    if is_noise(block):
                        continue
                    # Headings are judged as section boundaries first, before
                    # the overlap check below -- a short heading's text can
                    # occur inside an earlier answer sentence and must not
                    # be skipped past as if already included.
                    if is_section_number_heading(block):
                        break
                    if is_heading(block):
                        if normalize_for_exact_check(heading_text(block)) in {
                            normalize_for_exact_check(unit.text) for unit in result
                        }:
                            continue  # repeated in the next chunk's overlap
                        if not section_is_related(chunk_blocks, index - 1):
                            break
                        result.append(make(last, heading_text(block), chunk_index))
                        added += 1
                        continue
                    # Chunk-overlap text already in the answer.
                    if normalize_for_exact_check(block) in existing:
                        continue
                    # Only the next step of the same list may follow; any
                    # other number is a different list, and a gap would
                    # break need_complete()'s 1..n check and blank the answer.
                    step = re.match(r"^\s*(\d+)[.)]\s+", block)
                    if step:
                        if int(step.group(1)) != max_step + 1:
                            break
                        max_step += 1
                    if introduces_other_concept(block, need.query):
                        break
                    result.append(make(last, block, chunk_index))
                    existing += " " + normalize_for_exact_check(block)
                    added += 1
        return result

    def extract_verified(self, need: Need, ranked: list[tuple[int, float]]) -> list[Unit]:
        units = [unit for unit in self.extract(need, ranked) if self.verify_unit(unit)]
        units = [
            unit for unit in self.extend_across_chunk_boundary(need, units)
            if self.verify_unit(unit)
        ]
        return [unit for unit in self.complete_section(need, units) if self.verify_unit(unit)]

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
                    "question_term_coverage": 1.0,
                    "question_terms_missing": [],
                },
            }
        needs = self.plan(cleaned)
        need_results: list[dict[str, Any]] = []
        source_indices: list[int] = []
        complete = True
        for need in needs:
            ranked = self.retrieve(need)
            units = self.extract_verified(need, ranked)
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
        answer_text = "\n\n".join(answer_parts) if complete else ""
        question_terms = content_roots(cleaned)
        answer_terms = content_roots(answer_text)
        question_term_coverage = (
            round(len(question_terms & answer_terms) / len(question_terms), 2)
            if question_terms else 1.0
        )
        # Rephrasing is a separate, on-demand call (see the /rephrase
        # endpoint) rather than done here: the LLM step is CPU-bound and
        # can take tens of seconds on hardware with no usable GPU, and
        # answer() itself needs to stay fast so the verified extractive
        # answer -- already fully correct on its own -- appears
        # immediately instead of waiting on a slow rewrite of it.
        return {
            "kind": "domain_answer" if complete else "not_found",
            "question": cleaned,
            "answer": answer_text if complete else "No complete extractive answer was verified.",
            "natural_answer": None,
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
                "question_term_coverage": question_term_coverage,
                "question_terms_missing": sorted(question_terms - answer_terms),
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

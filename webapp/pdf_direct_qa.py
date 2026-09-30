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
NLI_MODEL = os.getenv(
    "NLI_VERIFIER_MODEL", "cross-encoder/nli-deberta-v3-small"
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


def numbers_are_grounded(generated: str, source: str) -> bool:
    """The one class of hallucination a fluency-only rephrase could still
    introduce is inventing or altering a quantity (a reagent amount, a
    time, a temperature) -- exactly the detail that matters most in a
    laboratory procedure. Reject the rephrase outright if it states any
    number the verified source text does not itself contain; a step
    number ("1.", "2.") is exempt since the source's own numbered list
    supplies those independently of this check.

    A bare digit-only check is not enough: a source with several reagent
    amounts (e.g. "Phenol red crystals 0.1g" next to "Distilled water
    10ml") already contains the digit "10" somewhere, so a rephrase that
    misquotes the phenol red amount as "10 grams" would pass a check that
    only asks whether "10" appears anywhere in the source. Whenever a
    number carries a recognizable unit, require that exact (number, unit)
    pair -- not just the bare digit -- to appear in the source; only
    numbers without an attached unit (step numbers, section references)
    fall back to the bare-digit check.
    """
    source_pairs = _number_unit_pairs(source)
    unit_claimed_spans: list[tuple[int, int]] = []
    for match in NUMBER_WITH_UNIT_RE.finditer(generated):
        unit_claimed_spans.append(match.span())
        value = match.group(1).replace(",", "")
        unit = _UNIT_ALIASES.get(match.group(2).lower())
        if unit and (value, unit) not in source_pairs:
            return False
    source_numbers = set(NUMBER_RE.findall(source))
    for match in NUMBER_RE.finditer(generated):
        if any(start <= match.start() and match.end() <= end for start, end in unit_claimed_spans):
            continue  # already checked above as a number+unit pair
        value = match.group(0)
        if value in source_numbers:
            continue
        prefix = generated[:match.start()].rstrip()
        if re.search(r"(?:^|[.\n])\s*$", prefix) or not prefix:
            continue  # a step-number heading a new sentence, not a claimed quantity
        return False
    return True


RUNNING_HEADER_RE = re.compile(r"^\s*(?:\d+\.\s+)?[A-Z][^.;:]*\s\d{1,3}\s*$")
REFERENCE_PREFIX_RE = re.compile(
    r"\b(?:fig(?:ure)?s?|tables?|sections?|no|page|reagent)\.?$", re.I
)


def strip_running_headers(text: str) -> str:
    """Drop page running-header lines (e.g. "9. Haematology 293") that
    extraction occasionally keeps as a unit of their own."""
    return "\n".join(
        line for line in text.splitlines() if not RUNNING_HEADER_RE.match(line)
    ).strip()


def drop_unsupported_sentences(generated: str, source: str, question: str) -> str:
    """Remove generated sentences whose content words mostly do not occur
    in the verified source (or the question itself) -- a small model can
    append a plausible-sounding general-knowledge fact that no number or
    NLI check catches (observed: "Reticulocytes can be identified by their
    larger size..." added to a stain procedure that never says so).
    Numbered lines are renumbered afterwards so a dropped step leaves no
    gap."""
    allowed = roots(source) | roots(question)
    kept_lines: list[str] = []
    for line in generated.splitlines():
        marker = re.match(r"^\s*\d+[.)]\s+", line)
        body = line[marker.end():] if marker else line
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+(?=\S)", body):
            # Quantities are checked separately (numbers_are_grounded /
            # missing_quantity_sentences) and tokenize inconsistently
            # ("15cm" vs "15 cm"), so only words are judged here.
            sentence_roots = {r for r in content_roots(sentence) if not any(ch.isdigit() for ch in r)}
            if len(sentence_roots) >= 4 and len(sentence_roots & allowed) < 0.6 * len(sentence_roots):
                continue
            kept.append(sentence)
        if body.strip() and not " ".join(kept).strip():
            continue
        kept_lines.append((marker.group(0) if marker else "") + " ".join(kept))
    step = 0
    renumbered: list[str] = []
    for line in kept_lines:
        if re.match(r"^\s*\d+[.)]\s+", line):
            step += 1
            line = re.sub(r"^\s*\d+([.)])", lambda m: f"{step}{m.group(1)}", line, count=1)
        renumbered.append(line)
    return "\n".join(renumbered).strip()


def missing_quantity_sentences(generated: str, source: str) -> list[str]:
    """The reverse of numbers_are_grounded(): return each source sentence
    stating a quantity (a number+unit pair, or a bare number used as a
    value such as "divide by 100") that the generated text no longer
    contains. Step numbers, figure/table/section/reagent references, and
    running headers are not quantities and are ignored."""
    generated_pairs = _number_unit_pairs(generated)
    generated_numbers = set(NUMBER_RE.findall(generated))
    missing: list[str] = []
    for line in source.splitlines():
        if RUNNING_HEADER_RE.match(line):
            continue
        for sentence in re.split(r"(?<=[.;!?])\s+(?=\S)", line):
            unit_spans: list[tuple[int, int]] = []
            dropped = False
            for match in NUMBER_WITH_UNIT_RE.finditer(sentence):
                unit_spans.append(match.span())
                value = match.group(1).replace(",", "")
                unit = _UNIT_ALIASES.get(match.group(2).lower())
                if unit and (value, unit) not in generated_pairs:
                    dropped = True
            for match in NUMBER_RE.finditer(sentence):
                if dropped:
                    break
                if any(s <= match.start() and match.end() <= e for s, e in unit_spans):
                    continue
                value = match.group(0)
                prefix = sentence[:match.start()].rstrip()
                if not prefix or value.count(".") >= 2:
                    continue
                if REFERENCE_PREFIX_RE.search(prefix):
                    continue
                if value not in generated_numbers:
                    dropped = True
            if dropped and sentence.strip() not in missing:
                missing.append(sentence.strip())
    return missing


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
        self._nli: CrossEncoder | None = None
        self._embedder: SentenceTransformer | None = None
        self._chunk_embeddings: np.ndarray | None = None
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

    def _ensure_nli(self) -> None:
        if self._nli is not None:
            return
        with self._model_lock:
            if self._nli is None:
                self._nli = CrossEncoder(NLI_MODEL)

    def _claim_nli_status(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Judge one claim (hypothesis) against one piece of source text
        (premise): does the source actually support this claim, contradict
        it, or say nothing either way?

        This model, measured on real rephrases here, calls almost every
        full-paragraph premise/hypothesis pair "neutral" (0.95+) even for a
        clearly faithful restatement -- it was trained on short, single-
        clause sentence pairs, not paraphrases of a multi-clause technical
        passage, so requiring either probability to clear an absolute bar
        rejects good claims along with bad ones. With neutral this
        dominant, entailment and contradiction both often sit near zero for
        a genuinely faithful claim too (observed as low as 0.0005 and
        0.0008 respectively) -- noise at that scale, not signal, so
        comparing them directly flips on a coin toss. A real drift
        (observed: 0.05 contradiction on a CATT/reticulocyte mix-up) clears
        that noise floor by roughly an order of magnitude, so only treat
        one side outscoring the other as meaningful once it is also
        clearly above the floor faithful claims sit at; otherwise the
        claim is neither confirmed nor refuted by this premise.

        If the NLI model itself is unavailable, this reports "not_checked"
        rather than guessing.
        """
        try:
            self._ensure_nli()
        except Exception:
            return {"status": "not_checked", "entailment_score": None, "contradiction_score": None}
        raw = np.asarray(
            self._nli.predict([[premise, hypothesis]], show_progress_bar=False)
        )
        if raw.ndim != 2:
            return {"status": "not_checked", "entailment_score": None, "contradiction_score": None}
        probabilities = torch.softmax(torch.tensor(raw), dim=1).numpy()[0]
        labels = [
            str(label).lower() for label in self._nli.model.config.id2label.values()
        ]
        entailment_index = next(
            (index for index, label in enumerate(labels) if "entail" in label),
            None,
        )
        contradiction_index = next(
            (index for index, label in enumerate(labels) if "contra" in label),
            None,
        )
        if entailment_index is None or contradiction_index is None:
            return {"status": "not_checked", "entailment_score": None, "contradiction_score": None}
        entailment_score = float(probabilities[entailment_index])
        contradiction_score = float(probabilities[contradiction_index])
        if contradiction_score > entailment_score and contradiction_score > CONTRADICTION_NOISE_FLOOR:
            status = "contradicted"
        elif entailment_score > contradiction_score and entailment_score > CONTRADICTION_NOISE_FLOOR:
            status = "supported"
        else:
            status = "insufficient_evidence"
        return {
            "status": status,
            "entailment_score": entailment_score,
            "contradiction_score": contradiction_score,
        }

    def _rephrase_is_entailed(self, source: str, generated: str) -> bool:
        """Whole-answer gate used by rephrase() itself: reject only when
        the generated text as a whole reads as contradicted by the
        verified source (see _claim_nli_status for why an absolute NLI
        threshold doesn't work on this model). A missing/unusable NLI
        model fails open here, same as before -- rephrase already has
        numbers_are_grounded as a first line of defence.

        The NLI cross-encoder has a hard 512-token limit on the combined
        premise+hypothesis; past that it silently truncates one or both
        instead of erroring. For a long, multi-step source (observed live:
        a 14-step, ~700-token procedure) this cuts off the later steps
        before scoring, so the model judges the generated text against an
        incomplete premise and can call a fully faithful, complete
        rephrase "contradicted" simply because it can no longer see the
        source material for the steps near the end. Fail open in that
        case, the same way an unavailable NLI model already does here --
        numbers_are_grounded has already checked the generated text
        against the untruncated source before this runs.
        """
        try:
            self._ensure_nli()
        except Exception:
            return True
        combined_length = len(
            self._nli.tokenizer(source, generated, add_special_tokens=True)["input_ids"]
        )
        if combined_length > self._nli.tokenizer.model_max_length:
            return True
        return self._claim_nli_status(source, generated)["status"] != "contradicted"

    @staticmethod
    def split_answer_claims(text: str) -> list[dict[str, Any]]:
        """Split a free-form generated answer into sentence-level claims,
        each with its exact character offset in the original text, using
        the same abbreviation-protected sentence-boundary rule as extract()
        (see its Fig./q.s. handling) so "Fig. 5" and "q.s. water" are never
        mistaken for sentence ends.
        """
        protected = re.sub(r"\bFig\.", "Fig§", text, flags=re.I)
        protected = re.sub(r"\bq\.s\.", "q§s§", protected, flags=re.I)
        matches = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9—])", protected))
        segment_bounds = [0] + [match.end() for match in matches]
        segment_ends = [match.start() for match in matches] + [len(text)]
        claims: list[dict[str, Any]] = []
        for start, end in zip(segment_bounds, segment_ends):
            segment = text[start:end]
            stripped = segment.strip()
            if not stripped:
                continue
            offset = start + segment.find(stripped)
            claims.append({"text": stripped, "start": offset, "end": offset + len(stripped)})
        # A bare step marker ("1.", "2)") is not itself a claim -- the same
        # split point extract() protects against with its own sentences==[]
        # handling. Glue it onto the claim that follows so "1." and "Weigh
        # out 3.76g..." are judged and cited together, not as two
        # unrelated fragments.
        merged: list[dict[str, Any]] = []
        pending_marker: dict[str, Any] | None = None
        for claim in claims:
            if re.fullmatch(r"\d+[.)]", claim["text"]):
                pending_marker = claim
                continue
            if pending_marker is not None:
                claim = {
                    "text": f"{pending_marker['text']} {claim['text']}",
                    "start": pending_marker["start"],
                    "end": claim["end"],
                }
                pending_marker = None
            merged.append(claim)
        if pending_marker is not None:
            merged.append(pending_marker)
        return merged

    def verify_answer_claims(self, verified_text: str, generated: str) -> list[dict[str, Any]]:
        """Localize hallucination detection to individual sentences of a
        generated (rephrased) answer, instead of the whole-answer gate
        rephrase() applies. Each claim is checked independently against
        the same verified source text, so one drifting sentence inside an
        otherwise faithful answer is identified by name -- not hidden
        inside a whole-answer PASS/FAIL.
        """
        claims = self.split_answer_claims(generated)
        for claim in claims:
            claim.update(self._claim_nli_status(verified_text, claim["text"]))
        return claims

    def attribute_claims_to_chunks(
        self, generated: str, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Same per-claim NLI check as verify_answer_claims, but scored
        against each individual source chunk in turn instead of one merged
        blob of verified text -- so each claim can be traced back to the
        one chunk it actually agrees or disagrees with, not just to the
        answer as a whole. A chunk that actively supports the claim always
        wins (highest entailment among those); a claim is only reported as
        contradicted when none of the candidate chunks support it. This
        priority matters because the candidates here are small individual
        units (a single extracted step, not a whole chunk) -- and this NLI
        model, measured on short-premise/short-hypothesis pairs, tends to
        call an unrelated short step "contradicted" rather than "neutral"
        (the noise-floor behaviour documented on _claim_nli_status was
        calibrated on long-premise pairs, a different regime). Without this
        priority, a claim genuinely supported by one unit would lose to a
        spurious contradiction against a completely different, unrelated
        step from the same procedure. When no chunk clears the noise floor
        either way the claim stays insufficient_evidence and carries no
        chunk attribution.
        """
        claims = self.split_answer_claims(generated)
        for claim in claims:
            claim_roots = content_roots(claim["text"])
            best: dict[str, Any] | None = None
            best_score = -1.0
            fallback: dict[str, Any] | None = None
            fallback_score = -1.0
            # A candidate the NLI call rated "insufficient_evidence" is
            # still tracked by its lexical overlap with the claim -- a
            # near-identical paraphrase of *this* candidate can score
            # insufficient_evidence on this NLI model even though it is
            # obviously the claim's real source (observed live: "Place the
            # containers in the autoclave..." against its own, almost
            # word-for-word source scored 0.0005/0.0002, both under the
            # noise floor), while a completely unrelated candidate
            # elsewhere in the same answer scores a confident false
            # "contradicted" and would otherwise win the fallback below by
            # default. Prefer the lexically closest insufficient_evidence
            # candidate over a contradicted one unless the contradicted
            # candidate is itself at least as lexically close -- a genuine
            # contradiction (a real relational reversal, wrong specimen,
            # etc.) usually still shares most of the claim's words.
            closest_insufficient: dict[str, Any] | None = None
            closest_insufficient_overlap = -1
            for chunk in chunks:
                text = chunk.get("text") or ""
                if not text.strip():
                    continue
                result = self._claim_nli_status(text, claim["text"])
                overlap = len(claim_roots & content_roots(text)) if claim_roots else 0
                if result["status"] == "insufficient_evidence":
                    if overlap > closest_insufficient_overlap:
                        closest_insufficient_overlap = overlap
                        closest_insufficient = {
                            "chunk_id": chunk.get("chunk_id"),
                            "pdf_page": chunk.get("pdf_page"),
                            "printed_page": chunk.get("printed_page"),
                            "source_text": text,
                        }
                    continue
                attributed = {
                    **result,
                    "chunk_id": chunk.get("chunk_id"),
                    "pdf_page": chunk.get("pdf_page"),
                    "printed_page": chunk.get("printed_page"),
                    # The exact source wording this claim was judged against --
                    # for a contradicted claim, this is what the answer should
                    # have said instead, so the UI can show the grounded fact
                    # in place of the drifted one rather than just flagging it.
                    "source_text": text,
                    "_overlap": overlap,
                }
                if result["status"] == "supported":
                    entailment = result["entailment_score"] or 0.0
                    if entailment > best_score:
                        best_score = entailment
                        best = attributed
                else:
                    contradiction = result["contradiction_score"] or 0.0
                    if contradiction > fallback_score:
                        fallback_score = contradiction
                        fallback = attributed
            if best is None and fallback is not None and closest_insufficient is not None:
                if closest_insufficient_overlap > fallback.get("_overlap", -1):
                    fallback = None
            if best is None:
                best = fallback
            if best is not None:
                best.pop("_overlap", None)
                claim.update(best)
            elif closest_insufficient is not None:
                claim.update({
                    "status": "insufficient_evidence",
                    "entailment_score": None,
                    "contradiction_score": None,
                    **closest_insufficient,
                })
            else:
                claim.update({
                    "status": "insufficient_evidence",
                    "entailment_score": None,
                    "contradiction_score": None,
                    "chunk_id": None,
                    "pdf_page": None,
                    "printed_page": None,
                    "source_text": None,
                })
        return claims

    def rephrase(self, question: str, verified_text: str) -> str | None:
        """Restate an already exact-span-verified extractive answer in
        fluent language -- the model's only job is wording, not content: it
        is handed nothing but text that has already passed verify_unit(),
        and told explicitly not to add anything beyond it. This is
        deliberately not open-ended RAG generation from raw chunks, which
        would give the model room to introduce a fact the source never
        stated; confined to rephrasing already-verified text, it has none.
        Returns None (falls back to the extractive answer) if generation
        is unavailable or the output fails the post-hoc number check.
        """
        if not verified_text.strip():
            return None
        try:
            self._ensure_generator()
        except Exception:
            return None
        system = (
            "You restate already-verified laboratory manual text in clear, "
            "natural English. Use only the facts given to you. Do not add "
            "any number, quantity, reagent, or step that is not already in "
            "the given text. Do not answer from general knowledge. If the "
            "source describes numbered steps, restate every step exactly "
            "once, in the same order as the source, without skipping, "
            "merging, repeating, or renumbering any of them. If a fact in "
            "the source is conditional (e.g. \"if X, then Y; with Z, then "
            "W\"), keep it conditional in your restatement instead of "
            "stating only one branch as if it were the only rule."
        )
        prompt = (
            f"Question: {question}\n\n"
            f"Verified source text:\n{verified_text}\n\n"
            "Restate this as a clear, natural answer to the question, "
            "using only facts present in the source text above."
        )
        # A fluent restatement that silently drops a quantity (a time
        # limit, a dilution, "divide by 100") is worse than the plain
        # extractive text, because it reads as complete. Every quantity in
        # the verified text must survive; one retry names the dropped
        # sentences explicitly, and if they are still missing the complete
        # verified text itself is returned instead of a partial rewrite.
        fallback = strip_running_headers(verified_text)
        generated = self._generate_rephrase(system, prompt, verified_text)
        if generated is None:
            return fallback
        missing = missing_quantity_sentences(generated, verified_text)
        if missing:
            retry_prompt = (
                prompt
                + "\n\nYour answer must also include every one of these facts "
                "from the source, in their original place in the sequence:\n"
                + "\n".join(f"- {sentence}" for sentence in missing)
            )
            generated = self._generate_rephrase(system, retry_prompt, verified_text)
            if generated is None or missing_quantity_sentences(generated, verified_text):
                return fallback
        return generated

    def _generate_rephrase(self, system: str, prompt: str, verified_text: str) -> str | None:
        rendered = self._generator_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._generator_tokenizer(rendered, return_tensors="pt")
        device = next(self._generator.parameters()).device
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        # A fixed 420-token budget is enough for a short answer but not for
        # a long, multi-step procedure (observed live: 7-14 step answers
        # either had steps silently dropped, or the model's own output
        # started repeating/reordering steps as it ran short on room to
        # restate everything, sometimes truncating mid-sentence badly
        # enough to fail the post-hoc checks below and return None
        # entirely). Scale the budget to the actual verified text -- a
        # faithful restatement is rarely shorter than the source and is
        # sometimes longer, so double the source's own token count, with
        # the original 420 as a floor for short answers and a cap so one
        # unusually long chunk can't make generation run away.
        source_tokens = len(
            self._generator_tokenizer(verified_text, add_special_tokens=False)["input_ids"]
        )
        max_new_tokens = max(420, min(1600, source_tokens * 2))
        with torch.inference_mode():
            output = self._generator.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.04,
                pad_token_id=self._generator_tokenizer.eos_token_id,
            )
        generated = self._generator_tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        generated = drop_unsupported_sentences(generated, verified_text, prompt)
        if not generated or not numbers_are_grounded(generated, verified_text):
            return None
        if not self._rephrase_is_entailed(verified_text, generated):
            return None
        return generated

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

    def extract_verified(self, need: Need, ranked: list[tuple[int, float]]) -> list[Unit]:
        units = [unit for unit in self.extract(need, ranked) if self.verify_unit(unit)]
        return [
            unit for unit in self.extend_across_chunk_boundary(need, units)
            if self.verify_unit(unit)
        ]

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

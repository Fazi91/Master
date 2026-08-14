import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import Counter
from pathlib import Path

MODEL_NAME = "urchade/gliner_large_bio-v0.1"
DEFAULT_THRESHOLD = 0.35

INDEX_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\s+)?Index(?:\s+\d+)?\b",
    re.IGNORECASE,
)

ENTITY_LABELS = {
    "procedure": "PROCEDURE",
    "specimen": "SPECIMEN",
    "reagent": "REAGENT",
    "laboratory equipment": "EQUIPMENT",
    "organism": "ORGANISM",
    "disease": "DISEASE",
    "clinical finding": "FINDING",
    "anatomical site": "ANATOMICAL_SITE",
    "disease vector": "VECTOR",
    "cell": "CELL",
}

MIN_SEMANTIC_PDF_PAGE = 13

GENERIC_ENTITY_NAMES = {
    "procedure",
    "procedures",
    "reagent",
    "reagents",
    "salt",
    "each reagent",
    "reagent no",
    "certain diseases",
    "invasive form",
    "non-invasive form",
    "folds",
    "laboratory procedures",
    "eggs",
    "specimen",
    "larvae",
    "chemical test",
    "initial infection",
    "species",
    "smears",
    "standard method",
    "diseases",
    "organisms",
    "test",
    "these tests",
    "certain haemorrhagic disorders",
    "well cut",
    "screening purposes",
    "screening test",
    "materials and reagents",
    "instrument",
    "instruments",
    "equipment",
    "apparatus",
    "basic equipment",
    "clinical materials",
    "few instruments",
    "reagents and equipment",
    "diagnosis",
    "examination procedures",
    "examinations",
    "laboratory examinations",
    "techniques",
    "these cells",
    "sinks",
    "specimens",
    "laboratory",
    "laboratories",
    "method",
    "examination",
    "certain techniques",
    "these procedures",
    "object",
    "all specimens",
    "each numbered specimen",
    "general laboratory procedures",
    "laboratory tests",
    "new procedures",
    "this method",
    "analysis",
    "sample",
    "blood spills",
    "g stirrups",
}

GENERIC_PROCEDURE_NAMES = {
    "haemoglobin",
    "centring",
    "flattening",
    "heating",
    "lubricate",
    "refocus",
    "rinse",
    "rinse thoroughly",
    "rinsed and cleaned",
    "sterilize",
    "detection",
}

GENERIC_EQUIPMENT_NAMES = {
    "box",
    "can",
    "file",
    "fire",
    "flame",
    "glass",
    "stop",
    "support",
    "the instrument",
    "these cartons",
}

MICROSCOPE_PART_NAMES = {
    "body tube",
    "diaphragm",
    "eyepiece",
    "iris diaphragm",
    "lens",
    "lenses",
    "ocular",
    "capillary bore",
    "hub cavity",
    "lid",
    "side arm",
    "stem",
}

LABORATORY_PLACE_NAMES = {
    "laboratory",
    "laboratories",
    "health laboratory",
    "medical laboratory",
    "peripheral laboratory",
    "peripheral laboratories",
    "small laboratory",
    "small laboratories",
}

CELL_TERMS = {
    "cell",
    "cells",
    "erythrocyte",
    "erythrocytes",
    "leukocyte",
    "leukocytes",
    "lymphocyte",
    "lymphocytes",
    "thrombocyte",
    "thrombocytes",
    "flame cell",
    "flame cells",
}

CANONICAL_ALIASES = {
    ("PROCEDURE", "culture1"): "culture",
    ("CELL", "erythrocytes"): "erythrocyte",
    ("CELL", "lymphocytes"): "lymphocyte",
    ("CELL", "red blood cell"): "erythrocyte",
    ("CELL", "white cell"): "leukocyte",
    ("EQUIPMENT", "generators"): "generator",
    ("EQUIPMENT", "inverters"): "inverter",
    ("EQUIPMENT", "microscopes"): "microscope",
    ("SPECIMEN", "sputum specimens"): "sputum",
    ("SPECIMEN", "urine specimens"): "urine",
}

INVALID_EQUIPMENT_PATTERN = re.compile(
    r"""
    (?:
        \blaborator(?:y|ies)\b |
        \bhealth\s+laboratory\b |
        \bclinical\s+laboratory\b |
        \bmedical\s+laboratories\b |
        \bcurrent\s+network\b |
        \bhaematology\s+area\b |
        \brecord-keeping\s+area\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

INVALID_EQUIPMENT_NAMES = {
    "si",
    "bed for patients",
}

PROCEDURAL_MEASUREMENT_CUE = re.compile(
    r"\b(?:add|apply|centrifuge|dilute|filter|fix|heat|incubate|"
    r"measure|mix|stain|transfer|wash|weigh)\w*\b",
    re.IGNORECASE,
)

CONVERSION_TABLE_MARKERS = (
    "Traditional unit Conversion factors",
    "Conversion factors and examples",
)

TABLE_FRAGMENT_PATTERN = re.compile(
    r"\b(?:"
    r"erythrocyte number no|"
    r"leukocyte number no|"
    r"leukocyte type number|"
    r"thrombocyte number no|"
    r"mean erythrocyte|"
    r"differential leukocyte|"
    r"reticulocyte count"
    r")\b",
    re.IGNORECASE,
)

REAGENT_CONTEXT_PATTERN = re.compile(
    r"\b(?:"
    r"reagent|reagents|reagent no|"
    r"stain|staining|buffer|"
    r"prepare|preparing|preparation|"
    r"add|mix|dilute|fix"
    r")\b",
    re.IGNORECASE,
)

DIMENSION_PATTERN = re.compile(
    r"""
    (?<![\w.])
    \d+(?:\.\d+)?\s*(?:m|cm|mm)
    \s*(?:×|x|X|¥)\s*
    \d+(?:\.\d+)?\s*(?:m|cm|mm)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ENTITY_FIELDS = [
    "entity_id",
    "canonical_name",
    "normalized_name",
    "entity_type",
]

MENTION_FIELDS = [
    "mention_id",
    "chunk_id",
    "entity_id",
    "page_id",
    "pdf_page",
    "printed_page",
    "mention_text",
    "start_char",
    "end_char_exclusive",
    "extraction_method",
    "confidence",
]

RELATION_FIELDS = [
    "relation_id",
    "source_entity_id",
    "relation_type",
    "target_entity_id",
    "source_chunk_id",
    "source_page_id",
    "pdf_page",
    "printed_page",
    "evidence_text",
    "extraction_method",
    "confidence",
]

EXCLUDED_REGION_REQUIRED_FIELDS = {
    "region_id",
    "page_id",
    "pdf_page",
    "start_char",
    "end_char_exclusive",
    "region_type",
    "reviewed",
    "text_sha256",
}

# Only these source-relation-target combinations are allowed.
ALLOWED_RELATIONS = {
    ("PROCEDURE", "USES_REAGENT", "REAGENT"),
    ("PROCEDURE", "USES_EQUIPMENT", "EQUIPMENT"),
    ("PROCEDURE", "EXAMINES", "SPECIMEN"),
    ("PROCEDURE", "DETECTS", "ORGANISM"),
    ("PROCEDURE", "DETECTS", "DISEASE"),
    ("PROCEDURE", "DETECTS", "FINDING"),
    ("ORGANISM", "CAUSES", "DISEASE"),
    ("DISEASE", "HAS_FINDING", "FINDING"),
    ("DISEASE", "TRANSMITTED_BY", "VECTOR"),
    ("ORGANISM", "TRANSMITTED_BY", "VECTOR"),
    ("ORGANISM", "FOUND_IN", "SPECIMEN"),
    ("ORGANISM", "FOUND_IN", "ANATOMICAL_SITE"),
    ("CELL", "FOUND_IN", "SPECIMEN"),
    ("CELL", "FOUND_IN", "ANATOMICAL_SITE"),
    ("PROCEDURE", "HAS_MEASUREMENT", "MEASUREMENT"),
}

# Relation cues must occur in the same sentence as both entities.
RELATION_RULES = [
    {
        "source_types": {"PROCEDURE"},
        "target_types": {"REAGENT"},
        "relation_type": "USES_REAGENT",
        "pattern": re.compile(
            r"\b(?:use|using|used|add|adding|stain(?:ed|ing)?|"
            r"fix(?:ed|ing)?|mix(?:ed|ing)?|dilute(?:d|ing)?|"
            r"placed\s+in)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.90,
    },
    {
        "source_types": {"PROCEDURE"},
        "target_types": {"EQUIPMENT"},
        "relation_type": "USES_EQUIPMENT",
        "pattern": re.compile(
            r"\b(?:use|using|used|with|under|place(?:d)? in|"
            r"examine(?:d)? with)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.88,
    },
    {
        "source_types": {"PROCEDURE"},
        "target_types": {"SPECIMEN"},
        "relation_type": "EXAMINES",
        "pattern": re.compile(
            r"\b(?:examine|examines|examined|examination of|"
            r"test(?:ed|ing)?|analyse|analyze|analysis of)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.92,
    },
    {
        "source_types": {"PROCEDURE"},
        "target_types": {"ORGANISM", "DISEASE", "FINDING"},
        "relation_type": "DETECTS",
        "pattern": re.compile(
            r"\b(?:detect|detects|detected|detection of|identify|"
            r"identifies|identified|diagnose|diagnosis of|demonstrate)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.93,
    },
    {
        "source_types": {"ORGANISM", "CELL"},
        "target_types": {"DISEASE"},
        "relation_type": "CAUSES",
        "pattern": re.compile(
            r"\b(?:cause|causes|caused by|causative agent|"
            r"responsible for|produces?)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.94,
    },
    {
        "source_types": {"DISEASE"},
        "target_types": {"FINDING"},
        "relation_type": "HAS_FINDING",
        "pattern": re.compile(
            r"\b(?:symptom|symptoms|sign|signs|characterized by|"
            r"associated with|presents? with|manifestation)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.89,
    },
    {
        "source_types": {"DISEASE", "ORGANISM"},
        "target_types": {"VECTOR"},
        "relation_type": "TRANSMITTED_BY",
        "pattern": re.compile(
            r"\b(?:transmitted by|transmission by|spread by|vector)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.96,
    },
    {
        "source_types": {"ORGANISM"},
        "target_types": {"SPECIMEN", "ANATOMICAL_SITE"},
        "relation_type": "FOUND_IN",
        "pattern": re.compile(
            r"\b(?:found in|present in|occurs? in|seen in|"
            r"observed in|isolated from|recovered from)\b",
            re.IGNORECASE,
        ),
        "confidence": 0.92,
    },
]

MEASUREMENT_PATTERN = re.compile(
    r"""
    (?<![\w.])
    (?:
        \d+(?:\.\d+)?
        \s*(?:-|–|—|to)\s*
        \d+(?:\.\d+)?
        |
        \d+(?:\.\d+)?
    )
    \s*
    (?:
        % |
        °C |
        µm | μm | um |
        mm | cm | m |
        µl | μl | ul | ml | l |
        mg | g | kg |
        min(?:ute)?s? |
        sec(?:ond)?s? |
        hours? |
        days?
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

PH_PATTERN = re.compile(
    r"\bpH\s*\d+(?:\.\d+)?"
    r"(?:\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?)?",
    re.IGNORECASE,
)

SENTENCE_PATTERN = re.compile(r"[^.!?]+(?:[.!?]+|$)")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build semantic graph CSV files from chunks.csv."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/graph_v2/chunks.csv"),
        help="Path to chunks.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/graph_v2"),
        help="Directory for the three output CSV files",
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=Path("data/graph_v2/pages.csv"),
        help="Path to pages.csv used to validate exclusion offsets",
    )
    parser.add_argument(
        "--excluded-regions",
        type=Path,
        default=Path("data/graph_v2/excluded_regions.csv"),
        help="Path to the audited table exclusion mask",
    )
    parser.add_argument(
        "--audit-mask",
        action="store_true",
        help=(
            "Validate and approve the table mask, remove masked mentions "
            "from existing semantic CSVs, and rebuild relations without "
            "running GLiNER"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum GLiNER entity confidence",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a new extraction and replace existing semantic CSV files",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first input chunk",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of input chunks to process in this run",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=2,
        help="Number of complete sentences processed together by GLiNER",
    )
    return parser.parse_args()


def normalize_name(text):
    value = text.strip().lower()
    value = value.replace("μ", "µ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\r\n.,;:()[]{}")
    return value


def normalize_entity_name(text, entity_type):
    value = normalize_name(text)

    if entity_type == "MEASUREMENT":
        value = re.sub(
            r"(?<=\d)\s*(°c|µm|um|mm|cm|ml|ul|µl|mg|kg|g|l|m|%)\b",
            r" \1",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\s*(?:¥|x)\s*", " × ", value)
        value = re.sub(r"\s+", " ", value).strip()

    value = CANONICAL_ALIASES.get(
        (entity_type, value),
        value,
    )

    return value


def corrected_entity_type(mention_text, predicted_type):
    normalized = normalize_name(mention_text)

    # GLiNER occasionally labels this organism as a disease. Keep the
    # correction deterministic so restored mentions are cleaned as well.
    if normalized in {
        "cholera vibrio",
        "cholera vibrios",
        "microfilariae",
    }:
        return "ORGANISM"

    if normalized == "basophilic staining" and predicted_type == "PROCEDURE":
        return "FINDING"

    if normalized == "agarose gel" and predicted_type == "EQUIPMENT":
        return "REAGENT"

    if normalized == "humans":
        return "ORGANISM"

    # These are specimen materials, not anatomical structures. Apply the
    # correction to new and restored mentions alike.
    if normalized in {"stool", "stools", "sputum"}:
        return "SPECIMEN"

    if normalized in {"thick film", "thick films"}:
        return "SPECIMEN"

    if normalized in MICROSCOPE_PART_NAMES:
        return "EQUIPMENT"

    if normalized in {"slide", "slides"} and predicted_type == "SPECIMEN":
        return "EQUIPMENT"

    if normalized in CELL_TERMS:
        return "CELL"

    if re.search(r"\bspp\.?$", normalized, re.IGNORECASE):
        return "ORGANISM"

    if re.search(
        r"\b(?:bacillus|bacilli|parasite|parasites|helminth eggs)$",
        normalized,
        re.IGNORECASE,
    ):
        return "ORGANISM"

    if normalized == "blood" and predicted_type in {"ANATOMICAL_SITE", "REAGENT"}:
        return "SPECIMEN"

    return predicted_type


def keep_entity(mention_text, entity_type, context_text=""):
    normalized = normalize_name(mention_text)

    if len(normalized) < 2 or re.fullmatch(r"[a-z]\d*", normalized):
        return False

    if re.search(r"\bternatively\b|plungerternatively", normalized):
        return False

    if entity_type == "ANATOMICAL_SITE" and normalized in {
        "2 o’clock",
        "opening",
        "openings",
    }:
        return False

    if entity_type == "PROCEDURE" and normalized in GENERIC_PROCEDURE_NAMES:
        return False

    if entity_type == "EQUIPMENT" and normalized in GENERIC_EQUIPMENT_NAMES:
        return False

    if re.search(r"-\s+", mention_text):
        return False

    if "\n" in mention_text and entity_type not in {
        "PROCEDURE",
        "REAGENT",
    }:
        return False

    if normalized in GENERIC_ENTITY_NAMES:
        return False

    if normalized in {"cell", "cells"}:
        return False

    if TABLE_FRAGMENT_PATTERN.search(normalized):
        return False

    if entity_type == "REAGENT":
        if normalized in {
            "hucker",
            "modified hucker",
            "crystal violet",
            "salt",
        }:
            return False

        if (
            normalized != "agarose gel"
            and not REAGENT_CONTEXT_PATTERN.search(context_text)
        ):
            return False

    if entity_type == "EQUIPMENT":
        if normalized in INVALID_EQUIPMENT_NAMES:
            return False

        if INVALID_EQUIPMENT_PATTERN.search(normalized):
            return False

    return True


def clean_evidence(text):
    return re.sub(r"\s+", " ", text).strip()


def read_chunks(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("chunks.csv has no header.")

        required = {
            "chunk_id",
            "page_id",
            "pdf_page",
            "printed_page",
            "start_char",
            "chunk_text",
        }
        missing = required.difference(reader.fieldnames)

        if missing:
            raise ValueError(
                "Missing required columns: " + ", ".join(sorted(missing))
            )

        chunks = list(reader)

    if not chunks:
        raise ValueError("chunks.csv contains no data rows.")

    chunk_ids = [row["chunk_id"] for row in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id values found.")

    for row in chunks:
        if not row["chunk_text"].strip():
            raise ValueError(
                f"Empty chunk_text found in {row['chunk_id']}."
            )

        try:
            int(row["pdf_page"])
            int(row["start_char"])
        except ValueError as error:
            raise ValueError(
                f"Invalid numeric value in {row['chunk_id']}."
            ) from error

    return chunks


def read_pages(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"page_id", "pdf_page", "normalized_text"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError("pages.csv has an invalid header.")
        rows = list(reader)
    pages = {row["page_id"]: row for row in rows}
    if len(pages) != len(rows):
        raise ValueError("Duplicate page_id values found in pages.csv.")
    return pages


def read_excluded_regions(path, pages_by_id, allow_unreviewed=False):
    if not path.is_file():
        raise FileNotFoundError(
            f"Table exclusion mask not found: {path.resolve()}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("excluded_regions.csv has no header.")
        missing = EXCLUDED_REGION_REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "Missing exclusion columns: " + ", ".join(sorted(missing))
            )
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    regions_by_page = {}
    seen_ids = set()
    for row in rows:
        region_id = row["region_id"]
        if region_id in seen_ids:
            raise ValueError(f"Duplicate exclusion region: {region_id}")
        seen_ids.add(region_id)
        if row["region_type"] != "TABLE":
            raise ValueError(f"Unsupported region type in {region_id}.")
        if row["reviewed"].strip().lower() != "true" and not allow_unreviewed:
            raise RuntimeError(
                f"Unreviewed exclusion region: {region_id}. "
                "Run --audit-mask before semantic extraction."
            )
        page = pages_by_id.get(row["page_id"])
        if page is None or page["pdf_page"] != row["pdf_page"]:
            raise ValueError(f"Invalid page reference in {region_id}.")
        start = int(row["start_char"])
        end = int(row["end_char_exclusive"])
        page_text = page["normalized_text"]
        if start < 0 or end <= start or end > len(page_text):
            raise ValueError(f"Invalid offsets in {region_id}.")
        actual_hash = hashlib.sha256(
            page_text[start:end].encode("utf-8")
        ).hexdigest()
        if actual_hash != row["text_sha256"]:
            raise ValueError(f"Text hash mismatch in {region_id}.")
        normalized = {**row, "start": start, "end": end}
        regions_by_page.setdefault(row["page_id"], []).append(normalized)

    for page_regions in regions_by_page.values():
        page_regions.sort(key=lambda item: (item["start"], item["end"]))
        for previous, current in zip(page_regions, page_regions[1:]):
            if current["start"] < previous["end"]:
                raise ValueError(
                    f"Overlapping exclusion regions: {previous['region_id']} "
                    f"and {current['region_id']}"
                )
    return rows, fieldnames, regions_by_page


def approve_excluded_regions(path, rows, fieldnames):
    approved = [{**row, "reviewed": "true"} for row in rows]
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(approved)
    os.replace(temporary_path, path)


def span_overlaps_excluded(page_id, start, end, regions_by_page):
    return any(
        start < region["end"] and end > region["start"]
        for region in regions_by_page.get(page_id, [])
    )


def mask_chunk_text(chunk, regions_by_page):
    text = chunk["chunk_text"]
    page_start = int(chunk["start_char"])
    page_end = page_start + len(text)
    characters = list(text)
    for region in regions_by_page.get(chunk["page_id"], []):
        overlap_start = max(page_start, region["start"])
        overlap_end = min(page_end, region["end"])
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - page_start
        local_end = overlap_end - page_start
        for index in range(local_start, local_end):
            if characters[index] != "\n":
                characters[index] = " "
    return "".join(characters)


def sentence_spans(text):
    spans = []

    for match in SENTENCE_PATTERN.finditer(text):
        raw_text = match.group()

        if not raw_text.strip():
            continue

        leading_whitespace = len(raw_text) - len(raw_text.lstrip())
        trailing_end = len(raw_text.rstrip())

        start = match.start() + leading_whitespace
        end = match.start() + trailing_end

        spans.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

    return spans


def inference_spans(text, max_words=120, overlap_words=20):
    spans = []

    for sentence in sentence_spans(text):
        word_matches = list(re.finditer(r"\S+", sentence["text"]))

        if len(word_matches) <= max_words:
            spans.append(sentence)
            continue

        step = max_words - overlap_words

        for word_start in range(0, len(word_matches), step):
            word_end = min(word_start + max_words, len(word_matches))
            local_start = word_matches[word_start].start()
            local_end = word_matches[word_end - 1].end()
            absolute_start = sentence["start"] + local_start
            absolute_end = sentence["start"] + local_end

            spans.append(
                {
                    "start": absolute_start,
                    "end": absolute_end,
                    "text": text[absolute_start:absolute_end],
                }
            )

            if word_end == len(word_matches):
                break

    return spans


def find_sentence_for_span(sentences, start, end):
    for sentence in sentences:
        if start >= sentence["start"] and end <= sentence["end"]:
            return sentence

    return None


def is_conversion_table_chunk(text):
    normalized_text = re.sub(r"\s+", " ", text)

    return any(
        marker.lower() in normalized_text.lower()
        for marker in CONVERSION_TABLE_MARKERS
    )


def expand_known_entity_span(sentence_text, prediction):
    mention_text = prediction.get("text", "").strip()
    local_start = prediction.get("start")
    local_end = prediction.get("end")

    if normalize_name(mention_text) == "hucker":
        pattern = re.compile(
            r"crystal violet,\s*modified\s+Hucker",
            re.IGNORECASE,
        )
        match = pattern.search(sentence_text)

        if match is not None:
            return (
                match.group(),
                match.start(),
                match.end(),
            )

    return mention_text, local_start, local_end


def extract_known_compound_reagents(text):
    reagents = []

    pattern = re.compile(
        r"crystal violet,\s*modified\s+Hucker",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        reagents.append(
            {
                "text": match.group(),
                "label": "REAGENT",
                "score": 1.0,
                "start": match.start(),
                "end": match.end(),
                "method": "explicit_reagent_rule",
            }
        )

    return reagents


def extract_measurements(text):
    measurements = []
    occupied = set()
    sentences = sentence_spans(text)

    for pattern in (
        DIMENSION_PATTERN,
        PH_PATTERN,
        MEASUREMENT_PATTERN,
    ):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())

            if any(
                match.start() >= used_start
                and match.end() <= used_end
                for used_start, used_end in occupied
            ):
                continue

            sentence = find_sentence_for_span(
                sentences,
                match.start(),
                match.end(),
            )

            if sentence is None:
                continue

            if re.search(
                r"\broom should\s+measure\b",
                sentence["text"],
                re.IGNORECASE,
            ):
                continue

            if not PROCEDURAL_MEASUREMENT_CUE.search(sentence["text"]):
                continue

            if pattern is PH_PATTERN:
                numbers = re.findall(
                    r"\d+(?:\.\d+)?",
                    match.group(),
                )

                if (
                    not numbers
                    or any(float(value) > 14 for value in numbers)
                ):
                    continue

            occupied.add(span)

            measurements.append(
                {
                    "text": match.group().strip(),
                    "label": "MEASUREMENT",
                    "score": 1.0,
                    "start": match.start(),
                    "end": match.end(),
                    "method": "contextual_measurement_rule",
                }
            )

    return measurements


def valid_entity(entity, text):
    start = entity.get("start")
    end = entity.get("end")
    mention_text = entity.get("text", "").strip()

    if not isinstance(start, int) or not isinstance(end, int):
        return False

    if start < 0 or end <= start or end > len(text):
        return False

    if not mention_text:
        return False

    actual_text = text[start:end]

    return normalize_name(actual_text) == normalize_name(mention_text)


def entity_key(entity_type, normalized_name):
    return entity_type, normalized_name


def write_csv_temporary(path, fieldnames, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    return temporary_path


def recover_interrupted_save(paths):
    for path in paths:
        backup_path = path.with_suffix(path.suffix + ".bak")
        temporary_path = path.with_suffix(path.suffix + ".tmp")

        if backup_path.exists():
            os.replace(backup_path, path)

        if temporary_path.exists():
            temporary_path.unlink()


def commit_csv_outputs(specifications):
    temporary_paths = []
    backup_paths = []

    try:
        for path, fieldnames, rows in specifications:
            temporary_paths.append(
                write_csv_temporary(path, fieldnames, rows)
            )

        for path, _, _ in specifications:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if path.exists():
                shutil.copy2(path, backup_path)
                backup_paths.append(backup_path)

        for (path, _, _), temporary_path in zip(
            specifications, temporary_paths
        ):
            os.replace(temporary_path, path)

        for backup_path in backup_paths:
            backup_path.unlink()
    except Exception:
        for path, _, _ in specifications:
            backup_path = path.with_suffix(path.suffix + ".bak")
            if backup_path.exists():
                os.replace(backup_path, path)
        raise


def read_existing_mentions(
    entity_path,
    mention_path,
    chunks_by_id,
    excluded_regions_by_page,
):
    if not entity_path.exists() and not mention_path.exists():
        return [], 0

    if entity_path.exists() != mention_path.exists():
        raise RuntimeError(
            "Existing semantic output is incomplete: entities.csv and "
            "rel_chunk_entity.csv must either both exist or both be absent."
        )

    with entity_path.open("r", encoding="utf-8-sig", newline="") as file:
        entities = {row["entity_id"]: row for row in csv.DictReader(file)}

    restored = []
    excluded_count = 0

    with mention_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            entity = entities.get(row["entity_id"])
            chunk = chunks_by_id.get(row["chunk_id"])

            if entity is None or chunk is None:
                raise RuntimeError(
                    f"Cannot restore existing mention {row.get('mention_id', '')}."
                )

            local_start = int(row["start_char"])
            local_end = int(row["end_char_exclusive"])
            page_offset = int(chunk["start_char"])

            absolute_start = page_offset + local_start
            absolute_end = page_offset + local_end
            if span_overlaps_excluded(
                row["page_id"],
                absolute_start,
                absolute_end,
                excluded_regions_by_page,
            ):
                excluded_count += 1
                continue

            entity_type = corrected_entity_type(
                row["mention_text"],
                entity["entity_type"],
            )

            sentences = sentence_spans(chunk["chunk_text"])
            mention_sentence = find_sentence_for_span(
                sentences,
                local_start,
                local_end,
            )

            context_text = (
                mention_sentence["text"]
                if mention_sentence is not None
                else ""
            )

            if not keep_entity(
                row["mention_text"],
                entity_type,
                context_text,
            ):
                continue

            normalized_name = normalize_entity_name(
                row["mention_text"],
                entity_type,
            )

            restored.append(
                {
                    "chunk_id": row["chunk_id"],
                    "page_id": row["page_id"],
                    "pdf_page": row["pdf_page"],
                    "printed_page": row["printed_page"],
                    "mention_text": row["mention_text"],
                    "start_char": local_start,
                    "end_char_exclusive": local_end,
                    "absolute_start": absolute_start,
                    "absolute_end": absolute_end,
                    "entity_type": entity_type,
                    "normalized_name": normalized_name,
                    "canonical_name": row["mention_text"],
                    "extraction_method": row["extraction_method"],
                    "confidence": float(row["confidence"]),
                }
            )

    return restored, excluded_count


def detect_relation(source, target, sentence):
    if int(source["start_char"]) >= int(target["start_char"]):
        return None

    sentence_text = sentence["text"]
    sentence_start = int(sentence["start"])
    source_end = int(source["end_char_exclusive"]) - sentence_start
    target_start = int(target["start_char"]) - sentence_start
    between_text = sentence_text[source_end:target_start]
    candidates = []

    # A semicolon normally separates independent tests or procedures in this
    # manual. Do not connect entities across those clauses.
    if ";" in between_text:
        return None

    for rule in RELATION_RULES:
        if source["_entity_type"] not in rule["source_types"]:
            continue

        if target["_entity_type"] not in rule["target_types"]:
            continue

        cue_found = rule["pattern"].search(between_text) is not None

        if rule["relation_type"] == "USES_REAGENT" and not cue_found:
            source_prefix = sentence_text[:source_end]
            cue_found = bool(
                re.search(
                    r"\breagents?\s+needed\s+for\s+[^.;:]{0,80}$",
                    source_prefix,
                    re.IGNORECASE,
                )
                and re.search(r"\bis\b", between_text, re.IGNORECASE)
            )

        if not cue_found:
            continue

        if rule["relation_type"] == "USES_EQUIPMENT":
            if re.search(
                r"\bwithout\s+(?:a|an|the)?\s*$",
                between_text,
                re.IGNORECASE,
            ):
                continue

            if re.search(
                r"\bnot\b.{0,80}\b(?:confuse|mistake)\b.{0,80}\bwith\b",
                between_text,
                re.IGNORECASE | re.DOTALL,
            ):
                continue

            # In biopsy instructions the equipment belongs to the tissue-
            # removal action, not to a preceding examination heading.
            if re.search(
                r"\b(?:physician|technician|operator)\s+"
                r"(?:removes?|collects?|cuts?)\b",
                between_text,
                re.IGNORECASE,
            ):
                continue

            # The urine dipstick is presented as an alternative detection
            # method, not as equipment used by the preceding Benedict method.
            if (
                source["_normalized_name"] == "benedict method"
                and re.search(
                    r"\bcan\s+also\s+be\s+detected\s+using\b",
                    between_text,
                    re.IGNORECASE,
                )
            ):
                continue

            # A following examination action is not equipment usage by the
            # preceding smear-preparation heading.
            if (
                source["_normalized_name"] == "preparation of smears"
                and re.search(r"\bexamine\b", between_text, re.IGNORECASE)
            ):
                continue

            # Urine cultures are indicated after dipstick screening; the
            # culture procedure does not itself use the dipstick.
            if (
                source["_normalized_name"] == "urine cultures"
                and re.search(
                    r"\bindicated\b.*\bdetected\b.*\busing\b",
                    between_text,
                    re.IGNORECASE | re.DOTALL,
                )
            ):
                continue

            # Equipment listed after use of the centrifuged supernatant
            # belongs to the broader test, not to the centrifuge action.
            if (
                source["_normalized_name"] == "centrifuge"
                and re.search(
                    r"\buse\s+the\s+supernatant\s+fluid\b.*\bG\b",
                    between_text,
                    re.IGNORECASE | re.DOTALL,
                )
            ):
                continue

        if rule["relation_type"] == "DETECTS":
            # Collection prepares a specimen for a later diagnostic test; it
            # does not itself detect the organism or disease.
            if source["_normalized_name"] in {
                "specimen collection",
            }:
                continue

            source_prefix = sentence_text[:source_end]
            if re.search(
                r"\bdesigned\s+for\s+[^.;:]{0,80}$",
                source_prefix,
                re.IGNORECASE,
            ):
                continue

            # A condition that makes detection difficult is not itself the
            # diagnostic procedure performing the detection.
            if re.search(
                r"\bdifficult\s+to\s+detect\b",
                between_text,
                re.IGNORECASE,
            ):
                continue

            if re.search(
                r"\bnot\s+sufficient\s+to\s+identify\b",
                between_text,
                re.IGNORECASE,
            ):
                continue

            if re.search(
                r"\b(?:do|does|did)\s+not\s+detect\b",
                between_text,
                re.IGNORECASE,
            ):
                continue

            # In "diseases caused by bacteria", the detection target is the
            # disease, not the organism appearing after "caused by".
            if re.search(
                r"\bcaused\s+by\s*$",
                between_text,
                re.IGNORECASE,
            ):
                continue

        if rule["relation_type"] == "CAUSES":
            # Coordinated lists followed by "respectively" require explicit
            # positional alignment. Reject the cross-product instead of
            # emitting incorrect organism-disease pairs.
            if re.search(r"\brespectively\b", sentence_text, re.IGNORECASE):
                continue

        if rule["relation_type"] == "FOUND_IN":
            if re.search(
                r"\b(?:sparse\s+or\s+)?absent\s+in\b",
                between_text,
                re.IGNORECASE,
            ):
                continue

        if rule["relation_type"] == "HAS_MEASUREMENT":
            # Measurements embedded in a Materials and reagents inventory
            # describe listed equipment or containers, not the procedure
            # named in the section heading.
            if re.search(
                r"\bmaterials\s+and\s+reagents\b",
                sentence_text,
                re.IGNORECASE,
            ):
                continue

        relation_tuple = (
            source["_entity_type"],
            rule["relation_type"],
            target["_entity_type"],
        )

        if relation_tuple not in ALLOWED_RELATIONS:
            continue

        candidates.append(
            (
                rule["relation_type"],
                rule["confidence"],
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def main():
    args = parse_arguments()

    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("--threshold must be greater than 0 and at most 1.")

    all_chunks = read_chunks(args.chunks)
    index_start = next(
        (
            index
            for index, chunk in enumerate(all_chunks)
            if INDEX_HEADING_PATTERN.search(chunk["chunk_text"])
        ),
        len(all_chunks),
    )
    index_chunk_ids = {
        chunk["chunk_id"] for chunk in all_chunks[index_start:]
    }
    pages_by_id = read_pages(args.pages)
    mask_rows, mask_fieldnames, excluded_regions_by_page = (
        read_excluded_regions(
            args.excluded_regions,
            pages_by_id,
            allow_unreviewed=args.audit_mask,
        )
    )

    if args.start < 0:
        raise ValueError("--start must be zero or greater.")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.inference_batch_size < 1:
        raise ValueError("--inference-batch-size must be at least 1.")
    if not args.audit_mask and args.start >= len(all_chunks):
        raise ValueError(
            f"--start must be smaller than the {len(all_chunks)} input chunks."
        )

    stop = min(args.start + args.limit, len(all_chunks))
    chunks = (
        []
        if args.audit_mask
        else [
            chunk
            for chunk in all_chunks[args.start:stop]
            if chunk["chunk_id"] not in index_chunk_ids
        ]
    )
    all_chunks_by_id = {chunk["chunk_id"]: chunk for chunk in all_chunks}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "entities": args.output_dir / "entities.csv",
        "mentions": args.output_dir / "rel_chunk_entity.csv",
        "relations": args.output_dir / "rel_entity_entity.csv",
    }

    if not args.audit_mask and args.start == 0 and not args.overwrite:
        existing = [path for path in output_paths.values() if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Output file already exists: {names}. Use --overwrite "
                "only when intentionally restarting from chunk 0."
            )

    if args.audit_mask and args.overwrite:
        raise ValueError("--audit-mask cannot be combined with --overwrite.")

    if not args.audit_mask and args.start > 0 and args.overwrite:
        raise ValueError("--overwrite may only be used together with --start 0.")

    model = None
    labels = list(ENTITY_LABELS.keys())
    if args.audit_mask:
        print(f"[INFO] Validated table regions: {len(mask_rows)}")
        print("[INFO] Audit mode: GLiNER will not be loaded")
    else:
        from gliner import GLiNER

        print(f"[INFO] Loading model: {MODEL_NAME}")
        print("[INFO] Device: CPU")
        model = GLiNER.from_pretrained(MODEL_NAME)
        model.to("cpu")
        model.eval()

    # Restore completed batches before adding this batch. Existing relations are
    # rebuilt from the merged mentions, so entity IDs always stay consistent.
    if args.overwrite:
        raw_mentions = []
        excluded_restored_mentions = 0
    else:
        raw_mentions, excluded_restored_mentions = read_existing_mentions(
            output_paths["entities"],
            output_paths["mentions"],
            all_chunks_by_id,
            excluded_regions_by_page,
        )
    restored_before_index_cleanup = len(raw_mentions)
    raw_mentions = [
        mention
        for mention in raw_mentions
        if mention["chunk_id"] not in index_chunk_ids
    ]
    excluded_index_mentions = (
        restored_before_index_cleanup - len(raw_mentions)
    )
    seen_mentions = {
        (
            mention["page_id"],
            mention["absolute_start"],
            mention["absolute_end"],
            mention["entity_type"],
            mention["normalized_name"],
        )
        for mention in raw_mentions
    }

    recover_interrupted_save(output_paths.values())
    rejected_entities = 0

    if args.audit_mask:
        print("[INFO] Selected chunk indexes: none (cleanup only)")
    else:
        print(
            f"[INFO] Selected chunk indexes: {args.start}-{stop - 1} "
            f"({len(chunks)} chunks)"
        )
    print(f"[INFO] Restored mentions: {len(raw_mentions)}")
    print(
        "[INFO] Removed restored index mentions: "
        f"{excluded_index_mentions}"
    )
    print(
        "[INFO] Removed restored mentions overlapping tables: "
        f"{excluded_restored_mentions}"
    )

    for index, chunk in enumerate(chunks, start=1):
        source_text = chunk["chunk_text"]
        text = mask_chunk_text(chunk, excluded_regions_by_page)
        page_offset = int(chunk["start_char"])

        if len(text.split()) <= 10:
            if index == 1 or index % 10 == 0 or index == len(chunks):
                print(
                    f"[INFO] Processed chunks: {index}/{len(chunks)}",
                    flush=True,
                )
            continue

        if int(chunk["pdf_page"]) < MIN_SEMANTIC_PDF_PAGE:
            if index == 1 or index % 10 == 0 or index == len(chunks):
                print(
                    f"[INFO] Processed chunks: {index}/{len(chunks)}",
                    flush=True,
                )
            continue

        if is_conversion_table_chunk(text):
            if index == 1 or index % 10 == 0 or index == len(chunks):
                print(
                    f"[INFO] Processed chunks: {index}/{len(chunks)}",
                    flush=True,
                )
            continue

        extracted = []
        sentences = inference_spans(text)

        # Every sentence is processed in full. Small inference batches reduce
        # repeated model overhead without dropping or truncating document text.
        for batch_start in range(0, len(sentences), args.inference_batch_size):
            sentence_batch = sentences[
                batch_start:batch_start + args.inference_batch_size
            ]
            batch_predictions = model.batch_predict_entities(
                [sentence["text"] for sentence in sentence_batch],
                labels,
                threshold=args.threshold,
            )

            if len(batch_predictions) != len(sentence_batch):
                raise RuntimeError("GLiNER returned an incomplete sentence batch.")

            for sentence, predictions in zip(sentence_batch, batch_predictions):
                for prediction in predictions:
                    label = prediction.get("label", "").strip().lower()
                    mapped_type = ENTITY_LABELS.get(label)

                    if mapped_type is None:
                        rejected_entities += 1
                        continue

                    mapped_type = corrected_entity_type(
                        prediction.get("text", ""),
                        mapped_type,
                    )

                    mention_text, local_start, local_end = expand_known_entity_span(
                        sentence["text"],
                        prediction,
                    )

                    if not keep_entity(
                        mention_text,
                        mapped_type,
                        sentence["text"],
                    ):
                        rejected_entities += 1
                        continue

                    if not isinstance(local_start, int) or not isinstance(local_end, int):
                        rejected_entities += 1
                        continue

                    candidate = {
                        "text": mention_text,
                        "label": mapped_type,
                        "score": float(prediction.get("score", 0.0)),
                        "start": sentence["start"] + local_start,
                        "end": sentence["start"] + local_end,
                        "method": f"gliner:{MODEL_NAME}:sentence_batch",
                    }

                    if not valid_entity(candidate, text):
                        rejected_entities += 1
                        continue

                    extracted.append(candidate)

        extracted.extend(extract_measurements(text))
        extracted.extend(extract_known_compound_reagents(text))

        for item in extracted:
            mention_text = source_text[
                item["start"]:item["end"]
            ].strip()
            item["label"] = corrected_entity_type(
                mention_text,
                item["label"],
            )

            mention_sentence = find_sentence_for_span(
                sentence_spans(text),
                item["start"],
                item["end"],
            )

            context_text = (
                mention_sentence["text"]
                if mention_sentence is not None
                else ""
            )

            if not keep_entity(
                mention_text,
                item["label"],
                context_text,
            ):
                rejected_entities += 1
                continue

            normalized = normalize_entity_name(
                mention_text,
                item["label"],
            )

            if not normalized:
                rejected_entities += 1
                continue

            absolute_start = page_offset + item["start"]
            absolute_end = page_offset + item["end"]

            if span_overlaps_excluded(
                chunk["page_id"],
                absolute_start,
                absolute_end,
                excluded_regions_by_page,
            ):
                rejected_entities += 1
                continue

            # This removes duplicated mentions caused by chunk overlap.
            deduplication_key = (
                chunk["page_id"],
                absolute_start,
                absolute_end,
                item["label"],
                normalized,
            )

            if deduplication_key in seen_mentions:
                continue

            seen_mentions.add(deduplication_key)

            raw_mentions.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "page_id": chunk["page_id"],
                    "pdf_page": chunk["pdf_page"],
                    "printed_page": chunk["printed_page"],
                    "mention_text": mention_text,
                    "start_char": item["start"],
                    "end_char_exclusive": item["end"],
                    "absolute_start": absolute_start,
                    "absolute_end": absolute_end,
                    "entity_type": item["label"],
                    "normalized_name": normalized,
                    "canonical_name": mention_text,
                    "extraction_method": item["method"],
                    "confidence": round(float(item["score"]), 4),
                }
            )

        if index == 1 or index % 10 == 0 or index == len(chunks):
            print(
                f"[INFO] Processed chunks: {index}/{len(chunks)}",
                flush=True,
            )

    # Use the most frequent surface form as the canonical name.
    surface_forms = {}

    for mention in raw_mentions:
        key = entity_key(
            mention["entity_type"],
            mention["normalized_name"],
        )
        surface_forms.setdefault(key, Counter())
        surface_forms[key][mention["canonical_name"]] += 1

    sorted_keys = sorted(
        surface_forms,
        key=lambda item: (item[0], item[1]),
    )

    entity_id_by_key = {}
    entity_rows = []

    for number, key in enumerate(sorted_keys, start=1):
        entity_type, normalized_name = key
        entity_id = f"E_{number:06d}"
        canonical_name = surface_forms[key].most_common(1)[0][0]

        entity_id_by_key[key] = entity_id
        entity_rows.append(
            {
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "normalized_name": normalized_name,
                "entity_type": entity_type,
            }
        )

    raw_mentions.sort(
        key=lambda row: (
            int(row["pdf_page"]),
            row["absolute_start"],
            row["absolute_end"],
            row["entity_type"],
        )
    )

    mention_rows = []

    for number, mention in enumerate(raw_mentions, start=1):
        key = entity_key(
            mention["entity_type"],
            mention["normalized_name"],
        )

        mention_rows.append(
            {
                "mention_id": f"M_{number:07d}",
                "chunk_id": mention["chunk_id"],
                "entity_id": entity_id_by_key[key],
                "page_id": mention["page_id"],
                "pdf_page": mention["pdf_page"],
                "printed_page": mention["printed_page"],
                "mention_text": mention["mention_text"],
                "start_char": mention["start_char"],
                "end_char_exclusive": mention["end_char_exclusive"],
                "extraction_method": mention["extraction_method"],
                "confidence": mention["confidence"],
                "_entity_type": mention["entity_type"],
                "_normalized_name": mention["normalized_name"],
            }
        )

    mentions_by_chunk = {}

    for mention in mention_rows:
        mentions_by_chunk.setdefault(mention["chunk_id"], []).append(
            mention
        )

    chunk_by_id = all_chunks_by_id
    raw_relations = []
    seen_relations = set()
    rejected_relations = 0

    for chunk_id, chunk_mentions in mentions_by_chunk.items():
        chunk = chunk_by_id[chunk_id]
        text = mask_chunk_text(chunk, excluded_regions_by_page)
        sentences = sentence_spans(text)

        mentions_with_sentences = []

        for mention in chunk_mentions:
            sentence = find_sentence_for_span(
                sentences,
                int(mention["start_char"]),
                int(mention["end_char_exclusive"]),
            )

            if sentence is not None:
                mentions_with_sentences.append((mention, sentence))

        for source, source_sentence in mentions_with_sentences:
            for target, target_sentence in mentions_with_sentences:
                if source["mention_id"] == target["mention_id"]:
                    continue

                if source_sentence["start"] != target_sentence["start"]:
                    continue

                if source["entity_id"] == target["entity_id"]:
                    continue

                result = detect_relation(
                    source,
                    target,
                    source_sentence,
                )

                if result is None:
                    continue

                relation_type, confidence = result

                relation_key = (
                    source["entity_id"],
                    relation_type,
                    target["entity_id"],
                    clean_evidence(source_sentence["text"]),
                )

                if relation_key in seen_relations:
                    continue

                seen_relations.add(relation_key)

                allowed_key = (
                    source["_entity_type"],
                    relation_type,
                    target["_entity_type"],
                )

                if allowed_key not in ALLOWED_RELATIONS:
                    rejected_relations += 1
                    continue

                raw_relations.append(
                    {
                        "source_entity_id": source["entity_id"],
                        "relation_type": relation_type,
                        "target_entity_id": target["entity_id"],
                        "source_chunk_id": chunk_id,
                        "source_page_id": chunk["page_id"],
                        "pdf_page": chunk["pdf_page"],
                        "printed_page": chunk["printed_page"],
                        "evidence_text": clean_evidence(source_sentence["text"]),
                        "extraction_method": "explicit_sentence_rule",
                        "confidence": confidence,
                    }
                )

        # Measurements are connected only to a meaningful procedural or
        # facility anchor in the same evidence sentence.
        measurement_anchors = [
            item for item in mentions_with_sentences
            if item[0]["_entity_type"] == "PROCEDURE"
        ]
        measurement_mentions = [
            item for item in mentions_with_sentences
            if item[0]["_entity_type"] == "MEASUREMENT"
        ]

        for anchor, anchor_sentence in measurement_anchors:
            for measurement, measurement_sentence in measurement_mentions:
                if anchor_sentence["start"] != measurement_sentence["start"]:
                    continue

                # Values in a Materials and reagents inventory describe a
                # listed item (for example, a tube capacity), not the
                # procedure heading that precedes the inventory.
                if re.search(
                    r"\bmaterials\s+and\s+reagents\b",
                    anchor_sentence["text"],
                    re.IGNORECASE,
                ):
                    continue

                # The procedural anchor must introduce the value. This avoids
                # attaching a specimen's thickness to a later word such as
                # "fixation" in a parenthetical explanation.
                if int(anchor["start_char"]) >= int(measurement["start_char"]):
                    continue

                anchor_end = int(anchor["end_char_exclusive"])
                measurement_start = int(measurement["start_char"])
                sentence_start = int(anchor_sentence["start"])
                between_text = anchor_sentence["text"][
                    anchor_end - sentence_start:
                    measurement_start - sentence_start
                ]

                if ";" in between_text:
                    continue

                relation_key = (
                    anchor["entity_id"],
                    "HAS_MEASUREMENT",
                    measurement["entity_id"],
                    clean_evidence(anchor_sentence["text"]),
                )

                if relation_key in seen_relations:
                    continue

                seen_relations.add(relation_key)

                raw_relations.append(
                    {
                        "source_entity_id": anchor["entity_id"],
                        "relation_type": "HAS_MEASUREMENT",
                        "target_entity_id": measurement["entity_id"],
                        "source_chunk_id": chunk_id,
                        "source_page_id": chunk["page_id"],
                        "pdf_page": chunk["pdf_page"],
                        "printed_page": chunk["printed_page"],
                        "evidence_text": clean_evidence(anchor_sentence["text"]),
                        "extraction_method": "explicit_sentence_rule",
                        "confidence": 0.88,
                    }
                )

    raw_relations.sort(
        key=lambda row: (
            int(row["pdf_page"]),
            row["source_chunk_id"],
            row["source_entity_id"],
            row["relation_type"],
            row["target_entity_id"],
        )
    )

    relation_rows = []

    for number, relation in enumerate(raw_relations, start=1):
        relation_rows.append(
            {
                "relation_id": f"R_{number:07d}",
                **relation,
            }
        )

    entity_ids = {row["entity_id"] for row in entity_rows}
    chunk_ids = set(all_chunks_by_id)

    for mention in mention_rows:
        if mention["entity_id"] not in entity_ids:
            raise RuntimeError(
                f"Unknown entity in mention {mention['mention_id']}."
            )

        if mention["chunk_id"] not in chunk_ids:
            raise RuntimeError(
                f"Unknown chunk in mention {mention['mention_id']}."
            )

    for relation in relation_rows:
        if relation["source_entity_id"] not in entity_ids:
            raise RuntimeError(
                f"Unknown source entity in {relation['relation_id']}."
            )

        if relation["target_entity_id"] not in entity_ids:
            raise RuntimeError(
                f"Unknown target entity in {relation['relation_id']}."
            )

        if relation["source_chunk_id"] not in chunk_ids:
            raise RuntimeError(
                f"Unknown source chunk in {relation['relation_id']}."
            )

        if not relation["evidence_text"]:
            raise RuntimeError(
                f"Missing evidence in {relation['relation_id']}."
            )

    commit_csv_outputs(
        [
            (output_paths["entities"], ENTITY_FIELDS, entity_rows),
            (output_paths["mentions"], MENTION_FIELDS, mention_rows),
            (output_paths["relations"], RELATION_FIELDS, relation_rows),
        ]
    )

    if args.audit_mask:
        approve_excluded_regions(
            args.excluded_regions,
            mask_rows,
            mask_fieldnames,
        )

    entity_type_counts = Counter(
        row["entity_type"] for row in entity_rows
    )
    relation_type_counts = Counter(
        row["relation_type"] for row in relation_rows
    )

    print()
    if args.audit_mask:
        print("[OK] Table-mask audit and semantic cleanup completed")
        print("[OK] Chunks processed in this run: 0")
        print(f"[OK] Approved table regions: {len(mask_rows)}")
        print(
            "[OK] Removed table-derived mentions: "
            f"{excluded_restored_mentions}"
        )
    else:
        print("[OK] Semantic graph extraction completed")
        print(f"[OK] Chunks processed in this run: {len(chunks)}")
        print(f"[OK] Completed chunk range: {args.start}-{stop - 1}")
    print(f"[OK] Entities: {len(entity_rows)}")
    print(f"[OK] Mentions: {len(mention_rows)}")
    print(f"[OK] Relations: {len(relation_rows)}")
    print(f"[OK] Rejected entity predictions: {rejected_entities}")
    print(f"[OK] Rejected relations: {rejected_relations}")
    print("[OK] Entity counts by type:")

    for entity_type in sorted(entity_type_counts):
        print(
            f"     {entity_type}: "
            f"{entity_type_counts[entity_type]}"
        )

    print("[OK] Relation counts by type:")

    if relation_type_counts:
        for relation_type in sorted(relation_type_counts):
            print(
                f"     {relation_type}: "
                f"{relation_type_counts[relation_type]}"
            )
    else:
        print("     No evidence-based relations were extracted.")

    print(f"[OK] Wrote: {output_paths['entities'].resolve()}")
    print(f"[OK] Wrote: {output_paths['mentions'].resolve()}")
    print(f"[OK] Wrote: {output_paths['relations'].resolve()}")


if __name__ == "__main__":
    main()

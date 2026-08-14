import argparse
import csv
import difflib
import hashlib
import os
import re
import unicodedata
from pathlib import Path

import pymupdf


EXPECTED_PAGE_COUNT = 398
EXPECTED_PDF_SHA256 = (
    "005ff66f9f3445a3bd5c1e4d17a92b7b54b35c909feafb1fc7ae77cd45bcb8a2"
)

OUTPUT_FIELDS = [
    "region_id",
    "page_id",
    "pdf_page",
    "start_char",
    "end_char_exclusive",
    "region_type",
    "detection_method",
    "reviewed",
    "x0",
    "y0",
    "x1",
    "y1",
    "row_count",
    "col_count",
    "token_count",
    "text_sha256",
    "text_preview",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Detect table regions in the audited WHO PDF and map them to "
            "character offsets in pages.csv without modifying source data."
        )
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw/who.pdf"),
    )
    parser.add_argument(
        "--pages",
        type=Path,
        default=Path("data/graph_v2/pages.csv"),
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/graph_v2/chunks.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/graph_v2/excluded_regions.csv"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing excluded_regions.csv after full validation.",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines).strip()


def canonical_token(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return value.casefold()


def read_csv_rows(path, required_fields):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise RuntimeError(f"Missing CSV header: {path.resolve()}")
        missing = set(required_fields).difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"Missing columns in {path.name}: {', '.join(sorted(missing))}"
            )
        return list(reader)


def token_spans(text):
    return [
        {
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(),
            "key": canonical_token(match.group()),
        }
        for match in re.finditer(r"\S+", text)
    ]


def find_subsequence(page_tokens, target_keys):
    if not target_keys or len(target_keys) > len(page_tokens):
        return []
    first = target_keys[0]
    matches = []
    limit = len(page_tokens) - len(target_keys) + 1
    for start in range(limit):
        if page_tokens[start]["key"] != first:
            continue
        if all(
            page_tokens[start + offset]["key"] == key
            for offset, key in enumerate(target_keys)
        ):
            matches.append((start, start + len(target_keys)))
    return matches


def rect_union(first, second):
    return pymupdf.Rect(
        min(first.x0, second.x0),
        min(first.y0, second.y0),
        max(first.x1, second.x1),
        max(first.y1, second.y1),
    )


def overlap_ratio(first, second):
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    intersection_area = intersection.get_area()
    smaller_area = min(first.get_area(), second.get_area())
    if smaller_area <= 0:
        return 0.0
    return intersection_area / smaller_area


def detect_table_candidates(page):
    candidates = []
    for strategy in ("lines_strict", "text"):
        finder = page.find_tables(strategy=strategy)
        for table in finder.tables:
            if table.row_count < 2 or table.col_count < 2:
                continue
            bbox = pymupdf.Rect(table.bbox)
            if table.header is not None and table.header.external:
                bbox = rect_union(bbox, pymupdf.Rect(table.header.bbox))
            candidates.append(
                {
                    "bbox": bbox,
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                    "methods": {strategy},
                }
            )

    merged = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["bbox"].y0,
            item["bbox"].x0,
            item["bbox"].y1,
            item["bbox"].x1,
        ),
    ):
        matching = None
        for existing in merged:
            if overlap_ratio(candidate["bbox"], existing["bbox"]) >= 0.80:
                matching = existing
                break
        if matching is None:
            merged.append(candidate)
        else:
            matching["bbox"] = rect_union(
                matching["bbox"], candidate["bbox"]
            )
            matching["row_count"] = max(
                matching["row_count"], candidate["row_count"]
            )
            matching["col_count"] = max(
                matching["col_count"], candidate["col_count"]
            )
            matching["methods"].update(candidate["methods"])
    return merged


def words_inside(page_words, bbox):
    selected = []
    for word in page_words:
        word_rect = pymupdf.Rect(word[:4])
        center = pymupdf.Point(
            (word_rect.x0 + word_rect.x1) / 2,
            (word_rect.y0 + word_rect.y1) / 2,
        )
        if bbox.contains(center):
            selected.append(word)
    return selected


def map_candidate_to_offsets(page_text, page_words, candidate):
    selected_indexes = []
    for index, word in enumerate(page_words):
        word_rect = pymupdf.Rect(word[:4])
        center = pymupdf.Point(
            (word_rect.x0 + word_rect.x1) / 2,
            (word_rect.y0 + word_rect.y1) / 2,
        )
        if candidate["bbox"].contains(center):
            selected_indexes.append(index)

    if len(selected_indexes) < 4:
        raise RuntimeError("Detected table contains fewer than four text tokens.")

    page_token_rows = token_spans(page_text)
    word_keys = [canonical_token(word[4]) for word in page_words]
    page_keys = [token["key"] for token in page_token_rows]

    matcher = difflib.SequenceMatcher(
        None,
        word_keys,
        page_keys,
        autojunk=False,
    )
    word_to_page = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            word_to_page[block.a + offset] = block.b + offset

    mapped_indexes = [
        word_to_page[index]
        for index in selected_indexes
        if index in word_to_page
    ]
    coverage = len(mapped_indexes) / len(selected_indexes)

    if coverage < 0.80:
        raise RuntimeError(
            "Table words could not be aligned reliably with pages.csv "
            f"(selected={len(selected_indexes)}, mapped={len(mapped_indexes)}, "
            f"coverage={coverage:.3f})."
        )

    start_token = min(mapped_indexes)
    end_token = max(mapped_indexes)
    start_char = page_token_rows[start_token]["start"]
    end_char = page_token_rows[end_token]["end"]
    region_text = page_text[start_char:end_char]
    return start_char, end_char, region_text, len(selected_indexes)


def merge_character_regions(regions):
    merged = []
    for region in sorted(
        regions,
        key=lambda item: (item["start_char"], item["end_char_exclusive"]),
    ):
        if not merged or region["start_char"] > merged[-1]["end_char_exclusive"]:
            merged.append(region)
            continue
        previous = merged[-1]
        previous["end_char_exclusive"] = max(
            previous["end_char_exclusive"], region["end_char_exclusive"]
        )
        previous["bbox"] = rect_union(previous["bbox"], region["bbox"])
        previous["row_count"] = max(previous["row_count"], region["row_count"])
        previous["col_count"] = max(previous["col_count"], region["col_count"])
        previous["methods"].update(region["methods"])
        previous["token_count"] += region["token_count"]
    return merged


def validate_chunks(chunks, pages_by_id):
    seen = set()
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen:
            raise RuntimeError(f"Duplicate chunk_id: {chunk_id}")
        seen.add(chunk_id)
        page = pages_by_id.get(chunk["page_id"])
        if page is None:
            raise RuntimeError(f"Unknown page_id in {chunk_id}")
        start = int(chunk["start_char"])
        end = int(chunk["end_char_exclusive"])
        expected = page["normalized_text"][start:end]
        if expected != chunk["chunk_text"]:
            raise RuntimeError(f"Chunk cannot be reconstructed: {chunk_id}")


def write_atomic(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_arguments()
    for path in (args.pdf, args.pages, args.chunks):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output.resolve()}. "
            "Use --overwrite only after reviewing the existing file."
        )

    pdf_hash = file_sha256(args.pdf)
    if pdf_hash != EXPECTED_PDF_SHA256:
        raise RuntimeError(
            "WHO PDF hash mismatch.\n"
            f"Expected: {EXPECTED_PDF_SHA256}\nActual:   {pdf_hash}"
        )

    pages = read_csv_rows(
        args.pages,
        {"page_id", "pdf_page", "normalized_text", "normalized_text_sha256"},
    )
    chunks = read_csv_rows(
        args.chunks,
        {
            "chunk_id",
            "page_id",
            "pdf_page",
            "start_char",
            "end_char_exclusive",
            "chunk_text",
        },
    )
    if len(pages) != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} pages, found {len(pages)}."
        )
    pages_by_id = {row["page_id"]: row for row in pages}
    if len(pages_by_id) != len(pages):
        raise RuntimeError("Duplicate page_id in pages.csv.")
    validate_chunks(chunks, pages_by_id)

    document = pymupdf.open(args.pdf)
    if document.page_count != EXPECTED_PAGE_COUNT:
        raise RuntimeError("Unexpected PDF page count.")

    all_regions = []
    pages_with_tables = set()
    for pdf_index, page in enumerate(document):
        pdf_page = pdf_index + 1
        page_id = f"P_{pdf_page:04d}"

        # Front matter / contents pages are outside the semantic corpus and
        # contain dense layouts that PyMuPDF can mistake for tables.
        if pdf_page < 13:
            continue
        page_row = pages_by_id.get(page_id)
        if page_row is None:
            raise RuntimeError(f"Missing {page_id} in pages.csv.")
        extracted_normalized = normalize_text(page.get_text("text", sort=True))
        if extracted_normalized != page_row["normalized_text"]:
            raise RuntimeError(
                f"PDF extraction no longer matches pages.csv on PDF page {pdf_page}."
            )
        stored_hash = hashlib.sha256(
            page_row["normalized_text"].encode("utf-8")
        ).hexdigest()
        if stored_hash != page_row["normalized_text_sha256"]:
            raise RuntimeError(f"Stored page hash mismatch on PDF page {pdf_page}.")

        page_words = page.get_text("words", sort=True)
        page_regions = []
        for candidate in detect_table_candidates(page):
            try:
                start, end, region_text, count = map_candidate_to_offsets(
                    page_row["normalized_text"], page_words, candidate
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"PDF page {pdf_page}: {error}"
                ) from error
            page_regions.append(
                {
                    "page_id": page_id,
                    "pdf_page": pdf_page,
                    "start_char": start,
                    "end_char_exclusive": end,
                    "bbox": candidate["bbox"],
                    "row_count": candidate["row_count"],
                    "col_count": candidate["col_count"],
                    "methods": set(candidate["methods"]),
                    "token_count": count,
                }
            )

        page_regions = merge_character_regions(page_regions)
        if page_regions:
            pages_with_tables.add(pdf_page)
        for region in page_regions:
            region_text = page_row["normalized_text"][
                region["start_char"]:region["end_char_exclusive"]
            ]
            bbox = region["bbox"]
            all_regions.append(
                {
                    "page_id": page_id,
                    "pdf_page": pdf_page,
                    "start_char": region["start_char"],
                    "end_char_exclusive": region["end_char_exclusive"],
                    "region_type": "TABLE",
                    "detection_method": "+".join(sorted(region["methods"])),
                    "reviewed": "false",
                    "x0": round(bbox.x0, 3),
                    "y0": round(bbox.y0, 3),
                    "x1": round(bbox.x1, 3),
                    "y1": round(bbox.y1, 3),
                    "row_count": region["row_count"],
                    "col_count": region["col_count"],
                    "token_count": region["token_count"],
                    "text_sha256": hashlib.sha256(
                        region_text.encode("utf-8")
                    ).hexdigest(),
                    "text_preview": re.sub(r"\s+", " ", region_text)[:160],
                }
            )

        if pdf_page == 1 or pdf_page % 25 == 0 or pdf_page == document.page_count:
            print(
                f"[INFO] Audited PDF pages: {pdf_page}/{document.page_count}",
                flush=True,
            )

    document.close()
    all_regions.sort(
        key=lambda row: (
            int(row["pdf_page"]),
            int(row["start_char"]),
            int(row["end_char_exclusive"]),
        )
    )
    for number, row in enumerate(all_regions, start=1):
        row["region_id"] = f"XR_{number:06d}"

    write_atomic(args.output, all_regions)
    print()
    print("[OK] Table exclusion mask candidate file created")
    print(f"[OK] PDF hash validated: {pdf_hash}")
    print(f"[OK] PDF pages validated: {len(pages)}")
    print(f"[OK] Chunks reconstructed: {len(chunks)}")
    print(f"[OK] Detected table regions: {len(all_regions)}")
    print(f"[OK] Pages with table regions: {len(pages_with_tables)}")
    print("[OK] All regions marked reviewed=false")
    print(f"[OK] Wrote: {args.output.resolve()}")
    print("[NEXT] Audit excluded_regions.csv before applying any mask")


if __name__ == "__main__":
    main()
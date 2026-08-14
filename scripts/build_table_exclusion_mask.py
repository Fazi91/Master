import argparse
import collections
import csv
import hashlib
import os
import re
import unicodedata
from pathlib import Path

import pymupdf

try:
    import pymupdf.layout  # noqa: F401 - activates Page.get_layout()
except ImportError as error:
    raise RuntimeError(
        "Missing table-layout dependency. Install it with: "
        "python -m pip install pymupdf-layout==1.26.6"
    ) from error


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


def box_text(page, bbox):
    return re.sub(
        r"\s+",
        " ",
        page.get_text("text", clip=bbox, sort=True),
    ).strip()


def horizontal_overlap(first, second):
    overlap = min(first.x1, second.x1) - max(first.x0, second.x0)
    return max(0.0, overlap)


def trim_picture_to_table_words(page, picture_bbox):
    """Stop a table-like picture before a following figure or text block."""
    words = []
    for word in page.get_text("words", clip=picture_bbox, sort=True):
        word_rect = pymupdf.Rect(word[:4])
        center = pymupdf.Point(
            (word_rect.x0 + word_rect.x1) / 2,
            (word_rect.y0 + word_rect.y1) / 2,
        )
        if picture_bbox.contains(center):
            words.append(word)
    if len(words) < 4:
        return picture_bbox

    line_ranges = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        if not line_ranges or word[1] - line_ranges[-1][0] > 2.0:
            line_ranges.append([word[1], word[3]])
        else:
            line_ranges[-1][1] = max(line_ranges[-1][1], word[3])
    if len(line_ranges) < 3:
        return picture_bbox

    gaps = [
        line_ranges[index + 1][0] - line_ranges[index][1]
        for index in range(len(line_ranges) - 1)
    ]
    for index, gap in enumerate(gaps):
        if index >= 1 and gap >= 30.0:
            return pymupdf.Rect(
                picture_bbox.x0,
                picture_bbox.y0,
                picture_bbox.x1,
                line_ranges[index][1] + 1.0,
            )
    return picture_bbox


def detect_table_candidates(page):
    page.get_layout()
    layout_boxes = [
        {
            "bbox": pymupdf.Rect(item[:4]),
            "class": item[4],
        }
        for item in page.layout_information
    ]
    for item in layout_boxes:
        item["text"] = box_text(page, item["bbox"])

    candidates = []

    for item in layout_boxes:
        if item["class"] != "table":
            continue
        bbox = item["bbox"]
        has_table_label = bool(
            re.search(r"\btable\b", item["text"], re.IGNORECASE)
        )
        labels_to_merge = []
        for label in layout_boxes:
            label_words = label["text"].split()
            is_table_label = (
                label["class"] == "caption"
                or len(label_words) <= 12
            ) and bool(
                re.search(r"\btable\b", label["text"], re.IGNORECASE)
            )
            if not is_table_label:
                continue
            label_bbox = label["bbox"]
            close_above = (
                0 <= item["bbox"].y0 - label_bbox.y1 <= 30
                and horizontal_overlap(item["bbox"], label_bbox) > 0
            )
            close_side = (
                0 <= item["bbox"].x0 - label_bbox.x1 <= 30
                and max(item["bbox"].y0, label_bbox.y0)
                < min(item["bbox"].y1, label_bbox.y1)
            )
            if close_above or close_side:
                has_table_label = True
                labels_to_merge.append(label_bbox)
        if not has_table_label:
            continue
        for label_bbox in labels_to_merge:
            bbox = rect_union(bbox, label_bbox)
        candidates.append(
            {
                "bbox": bbox,
                "row_count": 0,
                "col_count": 0,
                "methods": {"pymupdf_layout"},
            }
        )

    # Some complex tables are classified as a picture.  Accept that fallback
    # only when an explicit numbered Table caption immediately precedes it.
    captions = [
        item
        for item in layout_boxes
        if item["class"] == "caption"
        and re.match(r"(?i)^Table\s*\d", item["text"])
    ]
    for caption in captions:
        if any(
            0 <= candidate["bbox"].y0 - caption["bbox"].y1 <= 35
            for candidate in candidates
        ):
            continue
        pictures = [
            item
            for item in layout_boxes
            if item["class"] == "picture"
            and 0 <= item["bbox"].y0 - caption["bbox"].y1 <= 35
            and horizontal_overlap(item["bbox"], caption["bbox"]) > 0
        ]
        if not pictures:
            continue
        picture = min(pictures, key=lambda item: item["bbox"].y0)
        table_bbox = trim_picture_to_table_words(page, picture["bbox"])
        candidates.append(
            {
                "bbox": rect_union(caption["bbox"], table_bbox),
                "row_count": 0,
                "col_count": 0,
                "methods": {"pymupdf_layout_caption_fallback"},
            }
        )

    # A small number of tables near figures are emitted as a picture or as a
    # numbered section-header followed by one text block.  The explicit table
    # number keeps this fallback from masking ordinary images or prose.
    for label in layout_boxes:
        if not re.match(r"(?i)^Table\s*\d", label["text"]):
            continue
        if label["class"] == "picture":
            fallback_bbox = label["bbox"]
        elif label["class"] == "section-header":
            following = [
                item
                for item in layout_boxes
                if item["class"] == "text"
                and 0 <= item["bbox"].y0 - label["bbox"].y1 <= 20
                and horizontal_overlap(item["bbox"], label["bbox"]) > 0
            ]
            if not following:
                continue
            fallback_bbox = rect_union(
                label["bbox"],
                min(following, key=lambda item: item["bbox"].y0)["bbox"],
            )
        else:
            continue
        if any(
            candidate["bbox"].intersects(fallback_bbox)
            for candidate in candidates
        ):
            continue
        candidates.append(
            {
                "bbox": fallback_bbox,
                "row_count": 0,
                "col_count": 0,
                "methods": {"pymupdf_layout_numbered_fallback"},
            }
        )

    return candidates


def best_token_window(page_keys, target_keys):
    target_count = collections.Counter(target_keys)
    target_length = len(target_keys)
    minimum_length = max(4, int(target_length * 0.75))
    maximum_length = min(
        len(page_keys),
        int(target_length * 1.15) + 8,
    )
    best = None
    for start in range(len(page_keys)):
        window_count = collections.Counter()
        overlap = 0
        stop_limit = min(len(page_keys), start + maximum_length)
        for end in range(start, stop_limit):
            key = page_keys[end]
            if window_count[key] < target_count.get(key, 0):
                overlap += 1
            window_count[key] += 1
            window_length = end - start + 1
            if window_length < minimum_length:
                continue
            recall = overlap / target_length
            precision = overlap / window_length
            f1 = (
                2 * recall * precision / (recall + precision)
                if recall + precision
                else 0.0
            )
            score = (f1, recall, -abs(window_length - target_length))
            if best is None or score > best[0]:
                best = (score, start, end + 1, recall, precision)
    return best


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

    target_keys = [word_keys[index] for index in selected_indexes]
    best = best_token_window(page_keys, target_keys)
    if best is None:
        raise RuntimeError("No character-offset window was found for table.")
    _, start_token, end_token_exclusive, recall, precision = best
    if recall < 0.70 or precision < 0.70:
        raise RuntimeError(
            "Table words could not be aligned reliably with pages.csv "
            f"(selected={len(selected_indexes)}, recall={recall:.3f}, "
            f"precision={precision:.3f})."
        )

    start_char = page_token_rows[start_token]["start"]
    end_char = page_token_rows[end_token_exclusive - 1]["end"]
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

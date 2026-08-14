import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf


EXPECTED_PAGE_COUNT = 398
EXPECTED_PDF_SHA256 = (
    "005ff66f9f3445a3bd5c1e4d17a92b7b54b35c909feafb1fc7ae77cd45bcb8a2"
)
MAIN_BODY_START_PDF_PAGE = 13
MAIN_BODY_END_PDF_PAGE = 397

ROMAN_RE = re.compile(
    r"^(?=[ivxlcdm]+$)[ivxlcdm]+$",
    flags=re.IGNORECASE,
)

ARABIC_RE = re.compile(r"^\d{1,4}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    return "\n".join(lines).strip()


def non_empty_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def page_number_candidates(text: str) -> list[int]:
    """
    Search for a standalone Arabic page number near the header
    or footer of the extracted PDF page.

    Numbers in the middle of the page are ignored because they may
    belong to tables, procedures, measurements or figure captions.
    """
    lines = non_empty_lines(text)

    if not lines:
        return []

    boundary_lines = lines[:5] + lines[-5:]
    candidates = []

    for line in boundary_lines:
        cleaned_line = line.strip()

        if ARABIC_RE.fullmatch(cleaned_line):
            number = int(cleaned_line)

            if number not in candidates:
                candidates.append(number)

    return candidates


def roman_page_candidate(text: str) -> str:
    lines = non_empty_lines(text)

    for line in lines[:3]:
        if ROMAN_RE.fullmatch(line):
            return line.lower()

    for line in lines[-2:]:
        if ROMAN_RE.fullmatch(line):
            return line.lower()

    return ""


def detect_main_page_offset(page_texts: list[str]) -> int:
    """
    Determine the dominant difference between physical PDF page
    and printed Arabic page.

    For this WHO PDF the expected dominant offset is 12:
    PDF page 187 -> printed page 175.
    """
    offsets = []

    for pdf_page, text in enumerate(page_texts, start=1):
        for printed_candidate in page_number_candidates(text):
            offset = pdf_page - printed_candidate

            if 0 <= offset <= 50:
                offsets.append(offset)

    if not offsets:
        raise RuntimeError(
            "Could not determine the PDF-to-printed-page offset."
        )

    offset_counts = Counter(offsets)
    offset, count = offset_counts.most_common(1)[0]

    if count < 10:
        raise RuntimeError(
            "Printed-page offset is not supported by enough pages. "
            f"Detected offset={offset}, supporting_pages={count}"
        )

    return offset


def detect_printed_page(
    pdf_page: int,
    text: str,
    main_offset: int,
) -> tuple[str, str]:
    """
    Determine the printed page without assigning inferred page
    numbers outside the verified main body.

    PDF pages 13 to 397 correspond to printed pages 1 to 385.
    PDF page 398 is the unnumbered back cover.
    """
    arabic_candidates = page_number_candidates(text)
    roman_candidate = roman_page_candidate(text)

    if MAIN_BODY_START_PDF_PAGE <= pdf_page <= MAIN_BODY_END_PDF_PAGE:
        expected_printed_page = pdf_page - main_offset

        if expected_printed_page in arabic_candidates:
            return (
                str(expected_printed_page),
                "detected_arabic_boundary",
            )

        return (
            str(expected_printed_page),
            "inferred_from_verified_offset",
        )

    if pdf_page < MAIN_BODY_START_PDF_PAGE and roman_candidate:
        return (
            roman_candidate,
            "detected_roman_boundary",
        )

    if pdf_page > MAIN_BODY_END_PDF_PAGE:
        return "", "unnumbered_back_matter"

    return "", "not_printed_or_not_detected"


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_pages(
    rows: list[dict],
    pdf_sha256: str,
    main_offset: int,
) -> None:
    if len(rows) != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} pages, "
            f"but generated {len(rows)} rows."
        )

    expected_pdf_pages = list(
        range(1, EXPECTED_PAGE_COUNT + 1)
    )

    actual_pdf_pages = [
        int(row["pdf_page"])
        for row in rows
    ]

    if actual_pdf_pages != expected_pdf_pages:
        raise RuntimeError(
            "PDF page sequence is incomplete or out of order."
        )

    page_ids = [row["page_id"] for row in rows]

    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Duplicate page_id detected.")

    if pdf_sha256 != EXPECTED_PDF_SHA256:
        raise RuntimeError(
            "The PDF SHA-256 does not match the audited WHO PDF."
        )

    if main_offset != 12:
        raise RuntimeError(
            f"Expected offset 12, but found {main_offset}."
        )

    # Validate the continuous main-body mapping.
    main_body_rows = [
        row
        for row in rows
        if (
            MAIN_BODY_START_PDF_PAGE
            <= int(row["pdf_page"])
            <= MAIN_BODY_END_PDF_PAGE
        )
    ]

    for row in main_body_rows:
        pdf_page = int(row["pdf_page"])
        expected_printed_page = pdf_page - main_offset

        if row["printed_page"] != str(expected_printed_page):
            raise RuntimeError(
                "Incorrect continuous page mapping: "
                f"PDF page {pdf_page} should map to "
                f"printed page {expected_printed_page}, "
                f"but found {row['printed_page']}."
            )

        allowed_methods = {
            "detected_arabic_boundary",
            "inferred_from_verified_offset",
        }

        if row["printed_page_method"] not in allowed_methods:
            raise RuntimeError(
                "Invalid printed-page method for "
                f"PDF page {pdf_page}: "
                f"{row['printed_page_method']}"
            )

    last_main_body_page = rows[MAIN_BODY_END_PDF_PAGE - 1]

    if last_main_body_page["printed_page"] != "385":
        raise RuntimeError(
            "PDF page 397 must map to printed page 385."
        )

    back_cover = rows[397]

    if back_cover["pdf_page"] != 398:
        raise RuntimeError("PDF page 398 is missing.")

    if back_cover["printed_page"] != "":
        raise RuntimeError(
            "PDF page 398 is the back cover and must not "
            "have a printed page number."
        )

    if back_cover["printed_page_method"] != "unnumbered_back_matter":
        raise RuntimeError(
            "PDF page 398 must be classified as "
            "unnumbered_back_matter."
        )

    page_187 = rows[186]

    if page_187["printed_page"] != "175":
        raise RuntimeError(
            "PDF page 187 must map to printed page 175."
        )

    normalized_test_text = (
        page_187["normalized_text"]
        .replace("ﬁ", "fi")
        .lower()
    )

    required_phrase = "staining blood films with giemsa stain"

    if required_phrase not in normalized_test_text:
        raise RuntimeError(
            "The Giemsa reference phrase was not found "
            "on PDF page 187."
        )

    detected_count = sum(
        row["printed_page_method"]
        == "detected_arabic_boundary"
        for row in rows
    )

    inferred_count = sum(
        row["printed_page_method"]
        == "inferred_from_verified_offset"
        for row in rows
    )

    roman_count = sum(
        row["printed_page_method"]
        == "detected_roman_boundary"
        for row in rows
    )

    unnumbered_count = sum(
        row["printed_page_method"]
        == "not_printed_or_not_detected"
        for row in rows
    )

    if detected_count < 10:
        raise RuntimeError(
            "Too few directly detected Arabic page anchors: "
            f"{detected_count}"
        )

    print(f"[OK] Direct Arabic anchors: {detected_count}")
    print(f"[OK] Offset-inferred pages: {inferred_count}")
    print(f"[OK] Detected Roman pages: {roman_count}")
    print(f"[OK] Unnumbered front-matter pages: {unnumbered_count}")
    print(
        "[OK] Continuous main-body mapping: "
        "PDF 13 -> printed 1 through "
        "PDF 397 -> printed 385"
    )
    print("[OK] PDF page 398 classified as unnumbered back cover")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the original WHO PDF.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory for page-aware CSV output.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    output_dir = Path(args.outdir).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pdf_sha256 = file_sha256(pdf_path)

    if pdf_sha256 != EXPECTED_PDF_SHA256:
        raise RuntimeError(
            "Wrong or incomplete WHO PDF.\n"
            f"Expected SHA-256: {EXPECTED_PDF_SHA256}\n"
            f"Actual SHA-256:   {pdf_sha256}"
        )

    document = pymupdf.open(pdf_path)

    if document.page_count != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} PDF pages, "
            f"but found {document.page_count}."
        )

    page_texts = []

    for page in document:
        raw_text = page.get_text(
            "text",
            sort=True,
        )
        page_texts.append(raw_text)

    main_offset = detect_main_page_offset(page_texts)

    if main_offset != 12:
        raise RuntimeError(
            "Unexpected printed-page offset. "
            f"Expected 12, detected {main_offset}."
        )

    rows = []

    for pdf_page, raw_text in enumerate(
        page_texts,
        start=1,
    ):
        normalized_text = normalize_text(raw_text)

        printed_page, printed_page_method = (
            detect_printed_page(
                pdf_page=pdf_page,
                text=raw_text,
                main_offset=main_offset,
            )
        )

        raw_text_sha256 = hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest()

        normalized_text_sha256 = hashlib.sha256(
            normalized_text.encode("utf-8")
        ).hexdigest()

        rows.append(
            {
                "page_id": f"P_{pdf_page:04d}",
                "doc_id": "DOC_WHO_2003",
                "pdf_page": pdf_page,
                "printed_page": printed_page,
                "printed_page_method": printed_page_method,
                "char_count_raw": len(raw_text),
                "char_count_normalized": len(normalized_text),
                "raw_text_sha256": raw_text_sha256,
                "normalized_text_sha256": (
                    normalized_text_sha256
                ),
                "raw_text": raw_text,
                "normalized_text": normalized_text,
            }
        )

    document.close()

    validate_pages(
        rows=rows,
        pdf_sha256=pdf_sha256,
        main_offset=main_offset,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_rows = [
        {
            "doc_id": "DOC_WHO_2003",
            "file_name": pdf_path.name,
            "pdf_sha256": pdf_sha256,
            "pdf_page_count": EXPECTED_PAGE_COUNT,
            "main_printed_page_offset": main_offset,
            "extraction_library": "PyMuPDF",
            "extraction_mode": "text_sort_true",
        }
    ]

    write_csv(
        output_dir / "document.csv",
        [
            "doc_id",
            "file_name",
            "pdf_sha256",
            "pdf_page_count",
            "main_printed_page_offset",
            "extraction_library",
            "extraction_mode",
        ],
        document_rows,
    )

    write_csv(
        output_dir / "pages.csv",
        [
            "page_id",
            "doc_id",
            "pdf_page",
            "printed_page",
            "printed_page_method",
            "char_count_raw",
            "char_count_normalized",
            "raw_text_sha256",
            "normalized_text_sha256",
            "raw_text",
            "normalized_text",
        ],
        rows,
    )

    print("[OK] WHO PDF validated")
    print(f"[OK] PDF pages: {len(rows)}")
    print(f"[OK] PDF SHA-256: {pdf_sha256}")
    print(f"[OK] Main printed-page offset: {main_offset}")
    print("[OK] PDF page 187 -> printed page 175")
    print("[OK] Giemsa reference text verified")
    print(f"[OK] Wrote: {output_dir / 'document.csv'}")
    print(f"[OK] Wrote: {output_dir / 'pages.csv'}")


if __name__ == "__main__":
    main()
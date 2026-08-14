import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pymupdf


EXPECTED_PAGE_COUNT = 398
TARGET_PDF_PAGE = 242


def configure_csv_field_limit() -> None:
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            break
        except OverflowError:
            maximum_limit //= 10


def read_pages_csv(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)

    required_fields = {
        "page_id",
        "pdf_page",
        "printed_page",
        "normalized_text",
    }

    missing_fields = required_fields - fieldnames

    if missing_fields:
        raise RuntimeError(
            "pages.csv is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    return rows


def validate_inputs(
    document: pymupdf.Document,
    pages: list[dict],
) -> None:
    if document.page_count != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} PDF pages, "
            f"but found {document.page_count}."
        )

    if len(pages) != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} rows in pages.csv, "
            f"but found {len(pages)}."
        )

    actual_pdf_pages = [
        int(row["pdf_page"])
        for row in pages
    ]

    expected_pdf_pages = list(
        range(1, EXPECTED_PAGE_COUNT + 1)
    )

    if actual_pdf_pages != expected_pdf_pages:
        raise RuntimeError(
            "pages.csv is not ordered as PDF pages 1 through 398."
        )


def digest_to_hex(digest: object) -> str:
    if isinstance(digest, bytes):
        return digest.hex()

    return str(digest)


def audit_pdf(
    document: pymupdf.Document,
    pages: list[dict],
) -> None:
    total_occurrences = 0
    pages_with_images = 0
    unique_digests = set()

    pages_with_no_text = []
    pages_with_no_text_but_images = []
    pages_with_no_text_or_images = []

    occurrence_counts = Counter()
    target_page_records = []
    target_drawing_count = 0

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        pdf_page = page_index + 1
        page_row = pages[page_index]

        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        page_area = page_width * page_height

        image_info = page.get_image_info(
            hashes=True,
            xrefs=True,
        )

        drawing_count = len(
            page.get_drawings()
        )

        if image_info:
            pages_with_images += 1

        total_occurrences += len(image_info)

        if not page_row["normalized_text"].strip():
            pages_with_no_text.append(pdf_page)

            if image_info:
                pages_with_no_text_but_images.append(
                    pdf_page
                )
            else:
                pages_with_no_text_or_images.append(
                    pdf_page
                )

        for image_index, info in enumerate(
            image_info,
            start=1,
        ):
            bbox = pymupdf.Rect(info["bbox"])

            width_points = float(bbox.width)
            height_points = float(bbox.height)

            bbox_area = max(
                0.0,
                width_points * height_points,
            )

            page_coverage = (
                bbox_area / page_area
                if page_area > 0
                else 0.0
            )

            pixel_width = int(
                info.get("width", 0)
            )
            pixel_height = int(
                info.get("height", 0)
            )

            digest_hex = digest_to_hex(
                info.get("digest", "")
            )

            if digest_hex:
                unique_digests.add(digest_hex)

            if pixel_width < 50 or pixel_height < 50:
                size_group = "under_50_px"
            elif pixel_width < 100 or pixel_height < 100:
                size_group = "50_to_99_px"
            elif pixel_width < 200 or pixel_height < 200:
                size_group = "100_to_199_px"
            else:
                size_group = "at_least_200_px"

            occurrence_counts[size_group] += 1

            if page_coverage < 0.005:
                coverage_group = "under_0.5_percent"
            elif page_coverage < 0.02:
                coverage_group = "0.5_to_2_percent"
            elif page_coverage < 0.10:
                coverage_group = "2_to_10_percent"
            else:
                coverage_group = "at_least_10_percent"

            occurrence_counts[coverage_group] += 1

            if pdf_page == TARGET_PDF_PAGE:
                target_page_records.append(
                    {
                        "occurrence": image_index,
                        "xref": int(
                            info.get("xref", 0)
                        ),
                        "pixel_width": pixel_width,
                        "pixel_height": pixel_height,
                        "bbox": (
                            round(bbox.x0, 2),
                            round(bbox.y0, 2),
                            round(bbox.x1, 2),
                            round(bbox.y1, 2),
                        ),
                        "page_coverage_percent": round(
                            page_coverage * 100,
                            3,
                        ),
                        "digest": digest_hex,
                    }
                )

        if pdf_page == TARGET_PDF_PAGE:
            target_drawing_count = drawing_count

    print("[OK] Validated PDF and pages.csv")
    print(
        f"[AUDIT] PDF pages: {document.page_count}"
    )
    print(
        "[AUDIT] Total image occurrences: "
        f"{total_occurrences}"
    )
    print(
        "[AUDIT] Pages containing image occurrences: "
        f"{pages_with_images}"
    )
    print(
        "[AUDIT] Unique image digests: "
        f"{len(unique_digests)}"
    )

    print("\n=== PIXEL-SIZE GROUPS ===")
    print(
        "Under 50 px on at least one side:",
        occurrence_counts["under_50_px"],
    )
    print(
        "50-99 px on at least one side:",
        occurrence_counts["50_to_99_px"],
    )
    print(
        "100-199 px on at least one side:",
        occurrence_counts["100_to_199_px"],
    )
    print(
        "At least 200 px on both sides:",
        occurrence_counts["at_least_200_px"],
    )

    print("\n=== PAGE-COVERAGE GROUPS ===")
    print(
        "Under 0.5%:",
        occurrence_counts["under_0.5_percent"],
    )
    print(
        "0.5% to under 2%:",
        occurrence_counts["0.5_to_2_percent"],
    )
    print(
        "2% to under 10%:",
        occurrence_counts["2_to_10_percent"],
    )
    print(
        "At least 10%:",
        occurrence_counts["at_least_10_percent"],
    )

    print("\n=== PAGES WITHOUT EXTRACTED TEXT ===")
    print("All:", pages_with_no_text)
    print(
        "With raster image occurrences:",
        pages_with_no_text_but_images,
    )
    print(
        "Without raster image occurrences:",
        pages_with_no_text_or_images,
    )

    print(
        f"\n=== PDF PAGE {TARGET_PDF_PAGE} ==="
    )
    print(
        "Printed page:",
        pages[TARGET_PDF_PAGE - 1]["printed_page"],
    )
    print(
        "Normalized-text characters:",
        len(
            pages[TARGET_PDF_PAGE - 1][
                "normalized_text"
            ]
        ),
    )
    print(
        "Raster image occurrences:",
        len(target_page_records),
    )
    print(
        "Vector drawing objects:",
        target_drawing_count,
    )

    if target_page_records:
        for record in target_page_records:
            print(record)
    else:
        print(
            "No raster image occurrence detected "
            "on PDF page 242."
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the source PDF.",
    )

    parser.add_argument(
        "--pages",
        required=True,
        help="Path to the validated pages.csv file.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    pages_path = Path(args.pages).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if not pages_path.is_file():
        raise FileNotFoundError(
            f"pages.csv not found: {pages_path}"
        )

    configure_csv_field_limit()
    pages = read_pages_csv(pages_path)

    with pymupdf.open(pdf_path) as document:
        validate_inputs(
            document=document,
            pages=pages,
        )

        audit_pdf(
            document=document,
            pages=pages,
        )


if __name__ == "__main__":
    main()
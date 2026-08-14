import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf


EXPECTED_PAGE_COUNT = 398
TOP_REPEATED_DIGESTS = 30


PIXEL_GROUPS = [
    "under_50_px",
    "50_to_99_px",
    "100_to_199_px",
    "at_least_200_px",
]


COVERAGE_GROUPS = [
    "under_0.5_percent",
    "0.5_to_2_percent",
    "2_to_10_percent",
    "at_least_10_percent",
]


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


def classify_pixel_size(
    pixel_width: int,
    pixel_height: int,
) -> str:
    if pixel_width < 50 or pixel_height < 50:
        return "under_50_px"

    if pixel_width < 100 or pixel_height < 100:
        return "50_to_99_px"

    if pixel_width < 200 or pixel_height < 200:
        return "100_to_199_px"

    return "at_least_200_px"


def classify_page_coverage(
    page_coverage: float,
) -> str:
    if page_coverage < 0.005:
        return "under_0.5_percent"

    if page_coverage < 0.02:
        return "0.5_to_2_percent"

    if page_coverage < 0.10:
        return "2_to_10_percent"

    return "at_least_10_percent"


def build_occurrence_records(
    document: pymupdf.Document,
    pages: list[dict],
) -> list[dict]:
    records = []

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_row = pages[page_index]

        pdf_page = page_index + 1
        page_width = float(page.rect.width)
        page_height = float(page.rect.height)
        page_area = page_width * page_height

        image_info = page.get_image_info(
            hashes=True,
            xrefs=True,
        )

        for occurrence_index, info in enumerate(
            image_info,
            start=1,
        ):
            bbox = pymupdf.Rect(info["bbox"])

            bbox_width = max(
                0.0,
                float(bbox.width),
            )

            bbox_height = max(
                0.0,
                float(bbox.height),
            )

            bbox_area = bbox_width * bbox_height

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

            digest = digest_to_hex(
                info.get("digest", "")
            )

            xref = int(
                info.get("xref", 0)
            )

            pixel_group = classify_pixel_size(
                pixel_width=pixel_width,
                pixel_height=pixel_height,
            )

            coverage_group = classify_page_coverage(
                page_coverage=page_coverage,
            )

            records.append(
                {
                    "pdf_page": pdf_page,
                    "printed_page": page_row[
                        "printed_page"
                    ],
                    "occurrence_index": occurrence_index,
                    "xref": xref,
                    "digest": digest,
                    "pixel_width": pixel_width,
                    "pixel_height": pixel_height,
                    "bbox_x0": float(bbox.x0),
                    "bbox_y0": float(bbox.y0),
                    "bbox_x1": float(bbox.x1),
                    "bbox_y1": float(bbox.y1),
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                    "page_coverage": page_coverage,
                    "pixel_group": pixel_group,
                    "coverage_group": coverage_group,
                }
            )

    return records


def print_cross_table(
    records: list[dict],
) -> None:
    cross_counts = Counter(
        (
            record["pixel_group"],
            record["coverage_group"],
        )
        for record in records
    )

    print("\n=== PIXEL SIZE × PAGE COVERAGE ===")

    header = ["Pixel group"] + COVERAGE_GROUPS + ["Total"]

    print(" | ".join(header))
    print("-" * 125)

    for pixel_group in PIXEL_GROUPS:
        row_values = []

        for coverage_group in COVERAGE_GROUPS:
            row_values.append(
                cross_counts[
                    (
                        pixel_group,
                        coverage_group,
                    )
                ]
            )

        print(
            " | ".join(
                [pixel_group]
                + [
                    str(value)
                    for value in row_values
                ]
                + [str(sum(row_values))]
            )
        )

    column_totals = []

    for coverage_group in COVERAGE_GROUPS:
        column_total = sum(
            cross_counts[
                (
                    pixel_group,
                    coverage_group,
                )
            ]
            for pixel_group in PIXEL_GROUPS
        )

        column_totals.append(column_total)

    print("-" * 125)
    print(
        " | ".join(
            ["Total"]
            + [
                str(value)
                for value in column_totals
            ]
            + [str(sum(column_totals))]
        )
    )


def print_digest_reuse_distribution(
    records: list[dict],
) -> None:
    digest_counts = Counter(
        record["digest"]
        for record in records
        if record["digest"]
    )

    reuse_distribution = Counter(
        digest_counts.values()
    )

    unique_once = sum(
        1
        for count in digest_counts.values()
        if count == 1
    )

    repeated_digests = sum(
        1
        for count in digest_counts.values()
        if count > 1
    )

    repeated_occurrences = sum(
        count
        for count in digest_counts.values()
        if count > 1
    )

    duplicate_occurrences_beyond_first = sum(
        count - 1
        for count in digest_counts.values()
        if count > 1
    )

    print("\n=== DIGEST REUSE SUMMARY ===")
    print(
        "Unique digests:",
        len(digest_counts),
    )
    print(
        "Digests occurring once:",
        unique_once,
    )
    print(
        "Digests occurring more than once:",
        repeated_digests,
    )
    print(
        "Occurrences belonging to repeated digests:",
        repeated_occurrences,
    )
    print(
        "Duplicate occurrences beyond first use:",
        duplicate_occurrences_beyond_first,
    )

    print("\n=== DIGEST OCCURRENCE DISTRIBUTION ===")

    for occurrence_count in sorted(
        reuse_distribution
    ):
        number_of_digests = reuse_distribution[
            occurrence_count
        ]

        print(
            f"Used {occurrence_count} time(s): "
            f"{number_of_digests} digest(s)"
        )


def print_top_repeated_digests(
    records: list[dict],
) -> None:
    records_by_digest = defaultdict(list)

    for record in records:
        if record["digest"]:
            records_by_digest[
                record["digest"]
            ].append(record)

    ordered_digests = sorted(
        records_by_digest.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )

    print(
        f"\n=== TOP {TOP_REPEATED_DIGESTS} "
        "REPEATED DIGESTS ==="
    )

    printed_count = 0

    for digest, digest_records in ordered_digests:
        if len(digest_records) <= 1:
            continue

        pages = sorted(
            {
                record["pdf_page"]
                for record in digest_records
            }
        )

        xrefs = sorted(
            {
                record["xref"]
                for record in digest_records
            }
        )

        pixel_dimensions = sorted(
            {
                (
                    record["pixel_width"],
                    record["pixel_height"],
                )
                for record in digest_records
            }
        )

        coverage_values = [
            record["page_coverage"] * 100
            for record in digest_records
        ]

        page_preview = pages[:15]

        page_suffix = (
            ""
            if len(pages) <= 15
            else f" ... (+{len(pages) - 15} more)"
        )

        print(
            {
                "digest": digest,
                "occurrences": len(digest_records),
                "distinct_pages": len(pages),
                "pages": (
                    f"{page_preview}{page_suffix}"
                ),
                "xrefs": xrefs,
                "pixel_dimensions": pixel_dimensions,
                "min_coverage_percent": round(
                    min(coverage_values),
                    4,
                ),
                "max_coverage_percent": round(
                    max(coverage_values),
                    4,
                ),
            }
        )

        printed_count += 1

        if printed_count == TOP_REPEATED_DIGESTS:
            break

    if printed_count == 0:
        print("No repeated digest found.")


def print_xref_summary(
    records: list[dict],
) -> None:
    zero_xref_records = [
        record
        for record in records
        if record["xref"] == 0
    ]

    nonzero_xref_records = [
        record
        for record in records
        if record["xref"] != 0
    ]

    unique_nonzero_xrefs = {
        record["xref"]
        for record in nonzero_xref_records
    }

    print("\n=== XREF SUMMARY ===")
    print(
        "Occurrences with xref = 0:",
        len(zero_xref_records),
    )
    print(
        "Occurrences with non-zero xref:",
        len(nonzero_xref_records),
    )
    print(
        "Unique non-zero xrefs:",
        len(unique_nonzero_xrefs),
    )

    if zero_xref_records:
        zero_xref_pages = sorted(
            {
                record["pdf_page"]
                for record in zero_xref_records
            }
        )

        print(
            "Pages containing xref = 0:",
            zero_xref_pages,
        )


def print_dimension_consistency(
    records: list[dict],
) -> None:
    dimensions_by_digest = defaultdict(set)
    xrefs_by_digest = defaultdict(set)

    for record in records:
        digest = record["digest"]

        if not digest:
            continue

        dimensions_by_digest[digest].add(
            (
                record["pixel_width"],
                record["pixel_height"],
            )
        )

        xrefs_by_digest[digest].add(
            record["xref"]
        )

    inconsistent_dimensions = {
        digest: dimensions
        for digest, dimensions
        in dimensions_by_digest.items()
        if len(dimensions) > 1
    }

    multiple_xrefs = {
        digest: xrefs
        for digest, xrefs
        in xrefs_by_digest.items()
        if len(xrefs) > 1
    }

    print("\n=== DIGEST CONSISTENCY ===")
    print(
        "Digests with multiple pixel dimensions:",
        len(inconsistent_dimensions),
    )
    print(
        "Digests associated with multiple xrefs:",
        len(multiple_xrefs),
    )

    if inconsistent_dimensions:
        print(
            "\nDigests with inconsistent dimensions:"
        )

        for digest, dimensions in sorted(
            inconsistent_dimensions.items()
        ):
            print(
                {
                    "digest": digest,
                    "dimensions": sorted(dimensions),
                }
            )


def print_anomaly_groups(
    records: list[dict],
) -> None:
    small_pixels_large_display = [
        record
        for record in records
        if (
            record["pixel_group"] == "under_50_px"
            and record["page_coverage"] >= 0.02
        )
    ]

    large_pixels_tiny_display = [
        record
        for record in records
        if (
            record["pixel_group"]
            == "at_least_200_px"
            and record["page_coverage"] < 0.005
        )
    ]

    zero_dimensions = [
        record
        for record in records
        if (
            record["pixel_width"] <= 0
            or record["pixel_height"] <= 0
        )
    ]

    empty_digests = [
        record
        for record in records
        if not record["digest"]
    ]

    print("\n=== STRUCTURAL ANOMALY GROUPS ===")
    print(
        "Under 50 px but displayed on at least 2% "
        "of page:",
        len(small_pixels_large_display),
    )
    print(
        "At least 200 px but displayed under 0.5% "
        "of page:",
        len(large_pixels_tiny_display),
    )
    print(
        "Occurrences with zero pixel dimension:",
        len(zero_dimensions),
    )
    print(
        "Occurrences without digest:",
        len(empty_digests),
    )

    if small_pixels_large_display:
        print(
            "\nSmall-pixel / large-display records:"
        )

        for record in small_pixels_large_display:
            print(
                {
                    "pdf_page": record["pdf_page"],
                    "occurrence_index": record[
                        "occurrence_index"
                    ],
                    "xref": record["xref"],
                    "pixel_dimensions": (
                        record["pixel_width"],
                        record["pixel_height"],
                    ),
                    "coverage_percent": round(
                        record["page_coverage"] * 100,
                        4,
                    ),
                    "digest": record["digest"],
                }
            )

    if large_pixels_tiny_display:
        print(
            "\nFirst 30 large-pixel / tiny-display "
            "records:"
        )

        for record in large_pixels_tiny_display[:30]:
            print(
                {
                    "pdf_page": record["pdf_page"],
                    "occurrence_index": record[
                        "occurrence_index"
                    ],
                    "xref": record["xref"],
                    "pixel_dimensions": (
                        record["pixel_width"],
                        record["pixel_height"],
                    ),
                    "coverage_percent": round(
                        record["page_coverage"] * 100,
                        4,
                    ),
                    "digest": record["digest"],
                }
            )


def audit_raster_candidates(
    document: pymupdf.Document,
    pages: list[dict],
) -> None:
    records = build_occurrence_records(
        document=document,
        pages=pages,
    )

    pages_with_occurrences = {
        record["pdf_page"]
        for record in records
    }

    unique_digests = {
        record["digest"]
        for record in records
        if record["digest"]
    }

    print("[OK] Validated PDF and pages.csv")
    print(
        "[AUDIT] Total raster occurrences:",
        len(records),
    )
    print(
        "[AUDIT] Pages with raster occurrences:",
        len(pages_with_occurrences),
    )
    print(
        "[AUDIT] Unique non-empty digests:",
        len(unique_digests),
    )

    print_cross_table(records)
    print_digest_reuse_distribution(records)
    print_top_repeated_digests(records)
    print_xref_summary(records)
    print_dimension_consistency(records)
    print_anomaly_groups(records)

    print("\n[OK] Audit completed")
    print(
        "[OK] No image file or CSV file was created"
    )
    print(
        "[OK] No raster candidate was removed"
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

        audit_raster_candidates(
            document=document,
            pages=pages,
        )


if __name__ == "__main__":
    main()
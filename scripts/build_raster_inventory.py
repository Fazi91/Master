import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import pymupdf


EXPECTED_PAGE_COUNT = 398


IMAGE_FIELDS = [
    "image_id",
    "digest",
    "pixel_width",
    "pixel_height",
    "file_path",
    "file_format",
    "occurrence_count",
    "first_pdf_page",
    "classification_status",
]


RELATION_FIELDS = [
    "relation_id",
    "page_id",
    "image_id",
    "relation_type",
    "pdf_page",
    "printed_page",
    "occurrence_index_on_page",
    "xref",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "page_coverage",
]


def configure_csv_field_limit() -> None:
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            return
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

    expected_pdf_pages = list(
        range(1, EXPECTED_PAGE_COUNT + 1)
    )

    actual_pdf_pages = [
        int(row["pdf_page"])
        for row in pages
    ]

    if actual_pdf_pages != expected_pdf_pages:
        raise RuntimeError(
            "pages.csv must contain PDF pages 1 through 398 "
            "in the correct order."
        )

    page_ids = [
        row["page_id"].strip()
        for row in pages
    ]

    if any(not page_id for page_id in page_ids):
        raise RuntimeError(
            "pages.csv contains an empty page_id."
        )

    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError(
            "pages.csv contains duplicate page_id values."
        )


def digest_to_hex(digest: object) -> str:
    if isinstance(digest, bytes):
        return digest.hex()

    if isinstance(digest, memoryview):
        return digest.tobytes().hex()

    return str(digest)


def normalize_extension(extension: str) -> str:
    extension = extension.lower().strip().lstrip(".")

    supported_extensions = {
        "png",
        "jpg",
        "jpeg",
        "jp2",
        "jpx",
        "jbig2",
        "tif",
        "tiff",
        "bmp",
        "pam",
        "pbm",
        "pgm",
        "ppm",
    }

    if extension not in supported_extensions:
        return "png"

    if extension == "jpeg":
        return "jpg"

    if extension == "jpx":
        return "jp2"

    if extension == "tiff":
        return "tif"

    return extension


def build_asset_key(
    digest: str,
    pixel_width: int,
    pixel_height: int,
) -> tuple[str, int, int]:
    return (
        digest,
        pixel_width,
        pixel_height,
    )


def pixmap_digest(pixmap: pymupdf.Pixmap) -> str:
    return digest_to_hex(pixmap.digest)


def extract_payload_from_xref(
    document: pymupdf.Document,
    xref: int,
    expected_key: tuple[str, int, int],
) -> tuple[bytes, str] | None:
    if xref <= 0:
        return None

    try:
        extracted = document.extract_image(xref)
    except Exception:
        return None

    if not extracted:
        return None

    image_bytes = extracted.get("image")

    if not image_bytes:
        return None

    try:
        pixmap = pymupdf.Pixmap(image_bytes)
    except Exception:
        return None

    expected_width = expected_key[1]
    expected_height = expected_key[2]
    actual_width = int(pixmap.width)
    actual_height = int(pixmap.height)

    # The digest may change after decoding an image with a mask
    # or transparency. Dimensions are stable enough to verify
    # that the extracted payload belongs to this occurrence.
    if (
        actual_width != expected_width
        or actual_height != expected_height
    ):
        return None

    extension = normalize_extension(
        str(extracted.get("ext", "png"))
    )

    return image_bytes, extension


def collect_page_block_payloads(
    page: pymupdf.Page,
) -> dict[tuple[str, int, int], tuple[bytes, str]]:
    payloads = {}

    page_dictionary = page.get_text("dict")

    for block in page_dictionary.get("blocks", []):
        if block.get("type") != 1:
            continue

        image_bytes = block.get("image")

        if not image_bytes:
            continue

        try:
            pixmap = pymupdf.Pixmap(image_bytes)
        except Exception:
            continue

        asset_key = build_asset_key(
            digest=pixmap_digest(pixmap),
            pixel_width=int(pixmap.width),
            pixel_height=int(pixmap.height),
        )

        extension = normalize_extension(
            str(block.get("ext", "png"))
        )

        if asset_key not in payloads:
            payloads[asset_key] = (
                image_bytes,
                extension,
            )

    return payloads


def collect_inventory(
    document: pymupdf.Document,
    pages: list[dict],
) -> tuple[
    list[dict],
    dict[tuple[str, int, int], tuple[bytes, str]],
]:
    occurrences = []
    payloads = {}

    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_row = pages[page_index]

        pdf_page = page_index + 1
        page_area = float(page.rect.width * page.rect.height)

        block_payloads = collect_page_block_payloads(page)

        for key, payload in block_payloads.items():
            if key not in payloads:
                payloads[key] = payload

        image_info = page.get_image_info(
            hashes=True,
            xrefs=True,
        )

        for occurrence_index, info in enumerate(
            image_info,
            start=1,
        ):
            digest = digest_to_hex(
                info.get("digest", "")
            )

            pixel_width = int(
                info.get("width", 0)
            )

            pixel_height = int(
                info.get("height", 0)
            )

            xref = int(
                info.get("xref", 0)
            )

            if not digest:
                raise RuntimeError(
                    f"Missing digest on PDF page {pdf_page}, "
                    f"occurrence {occurrence_index}."
                )

            if pixel_width <= 0 or pixel_height <= 0:
                raise RuntimeError(
                    f"Invalid pixel dimensions on PDF page "
                    f"{pdf_page}, occurrence {occurrence_index}."
                )

            asset_key = build_asset_key(
                digest=digest,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
            )

            if asset_key not in payloads:
                extracted_payload = extract_payload_from_xref(
                    document=document,
                    xref=xref,
                    expected_key=asset_key,
                )

                if extracted_payload is not None:
                    payloads[asset_key] = extracted_payload

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

            occurrences.append(
                {
                    "asset_key": asset_key,
                    "page_id": page_row["page_id"],
                    "pdf_page": pdf_page,
                    "printed_page": page_row[
                        "printed_page"
                    ],
                    "occurrence_index_on_page":
                        occurrence_index,
                    "xref": xref,
                    "bbox_x0": float(bbox.x0),
                    "bbox_y0": float(bbox.y0),
                    "bbox_x1": float(bbox.x1),
                    "bbox_y1": float(bbox.y1),
                    "page_coverage": page_coverage,
                }
            )

    return occurrences, payloads


def validate_collected_inventory(
    occurrences: list[dict],
    payloads: dict[
        tuple[str, int, int],
        tuple[bytes, str],
    ],
) -> list[tuple[str, int, int]]:
    if len(occurrences) != 1131:
        raise RuntimeError(
            "Expected 1131 raster occurrences, "
            f"but collected {len(occurrences)}."
        )

    asset_keys = sorted(
        {
            record["asset_key"]
            for record in occurrences
        }
    )

    missing_payloads = [
        asset_key
        for asset_key in asset_keys
        if asset_key not in payloads
    ]

    if missing_payloads:
        preview = missing_payloads[:10]

        raise RuntimeError(
            f"Exact image data could not be extracted for "
            f"{len(missing_payloads)} asset(s). "
            f"First missing keys: {preview}"
        )

    empty_payloads = [
        asset_key
        for asset_key in asset_keys
        if not payloads[asset_key][0]
    ]

    if empty_payloads:
        raise RuntimeError(
            f"{len(empty_payloads)} asset payload(s) are empty."
        )

    return asset_keys


def prepare_output_paths(
    images_directory: Path,
    images_csv_path: Path,
    relations_csv_path: Path,
) -> tuple[Path, Path, Path]:
    if images_directory.exists():
        raise FileExistsError(
            f"Output directory already exists: "
            f"{images_directory}\n"
            "Remove it only if it was created by an earlier "
            "failed run."
        )

    if images_csv_path.exists():
        raise FileExistsError(
            f"Output file already exists: {images_csv_path}"
        )

    if relations_csv_path.exists():
        raise FileExistsError(
            f"Output file already exists: "
            f"{relations_csv_path}"
        )

    images_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relations_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_images_directory = (
        images_directory.parent
        / f"{images_directory.name}_building"
    )

    temporary_images_csv = images_csv_path.with_suffix(
        ".building.csv"
    )

    temporary_relations_csv = (
        relations_csv_path.with_suffix(".building.csv")
    )

    temporary_paths = [
        temporary_images_directory,
        temporary_images_csv,
        temporary_relations_csv,
    ]

    for temporary_path in temporary_paths:
        if temporary_path.exists():
            raise FileExistsError(
                f"Temporary output already exists: "
                f"{temporary_path}\n"
                "This may be left from an earlier failed run."
            )

    return (
        temporary_images_directory,
        temporary_images_csv,
        temporary_relations_csv,
    )


def write_outputs(
    occurrences: list[dict],
    payloads: dict[
        tuple[str, int, int],
        tuple[bytes, str],
    ],
    asset_keys: list[tuple[str, int, int]],
    images_directory: Path,
    images_csv_path: Path,
    relations_csv_path: Path,
) -> tuple[int, int]:
    (
        temporary_images_directory,
        temporary_images_csv,
        temporary_relations_csv,
    ) = prepare_output_paths(
        images_directory=images_directory,
        images_csv_path=images_csv_path,
        relations_csv_path=relations_csv_path,
    )

    temporary_images_directory.mkdir(
        parents=False,
        exist_ok=False,
    )

    asset_id_by_key = {
        asset_key: f"img_{index:06d}"
        for index, asset_key in enumerate(
            asset_keys,
            start=1,
        )
    }

    occurrence_counts = Counter(
        record["asset_key"]
        for record in occurrences
    )

    first_pages = {}

    for record in occurrences:
        asset_key = record["asset_key"]
        pdf_page = record["pdf_page"]

        if asset_key not in first_pages:
            first_pages[asset_key] = pdf_page
        else:
            first_pages[asset_key] = min(
                first_pages[asset_key],
                pdf_page,
            )

    image_rows = []
    relation_rows = []

    try:
        for asset_key in asset_keys:
            digest, pixel_width, pixel_height = asset_key
            image_id = asset_id_by_key[asset_key]
            image_bytes, extension = payloads[asset_key]

            filename = f"{image_id}.{extension}"

            temporary_file_path = (
                temporary_images_directory / filename
            )

            temporary_file_path.write_bytes(image_bytes)

            if temporary_file_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Created an empty image file: {filename}"
                )

            image_rows.append(
                {
                    "image_id": image_id,
                    "digest": digest,
                    "pixel_width": pixel_width,
                    "pixel_height": pixel_height,
                    "file_path": (
                        f"data/processed/images/{filename}"
                    ),
                    "file_format": extension,
                    "occurrence_count":
                        occurrence_counts[asset_key],
                    "first_pdf_page":
                        first_pages[asset_key],
                    "classification_status": "pending",
                }
            )

        for relation_index, record in enumerate(
            occurrences,
            start=1,
        ):
            relation_rows.append(
                {
                    "relation_id":
                        f"rel_page_image_{relation_index:06d}",
                    "page_id": record["page_id"],
                    "image_id": asset_id_by_key[
                        record["asset_key"]
                    ],
                    "relation_type": "CONTAINS_IMAGE",
                    "pdf_page": record["pdf_page"],
                    "printed_page":
                        record["printed_page"],
                    "occurrence_index_on_page":
                        record[
                            "occurrence_index_on_page"
                        ],
                    "xref": record["xref"],
                    "bbox_x0": round(
                        record["bbox_x0"],
                        6,
                    ),
                    "bbox_y0": round(
                        record["bbox_y0"],
                        6,
                    ),
                    "bbox_x1": round(
                        record["bbox_x1"],
                        6,
                    ),
                    "bbox_y1": round(
                        record["bbox_y1"],
                        6,
                    ),
                    "page_coverage": round(
                        record["page_coverage"],
                        10,
                    ),
                }
            )

        with temporary_images_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=IMAGE_FIELDS,
            )

            writer.writeheader()
            writer.writerows(image_rows)

        with temporary_relations_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=RELATION_FIELDS,
            )

            writer.writeheader()
            writer.writerows(relation_rows)

        extracted_file_count = len(
            list(temporary_images_directory.iterdir())
        )

        if extracted_file_count != len(image_rows):
            raise RuntimeError(
                f"Expected {len(image_rows)} image files, "
                f"but created {extracted_file_count}."
            )

        if len(relation_rows) != 1131:
            raise RuntimeError(
                f"Expected 1131 relation rows, "
                f"but created {len(relation_rows)}."
            )

        temporary_images_directory.rename(
            images_directory
        )

        temporary_images_csv.replace(
            images_csv_path
        )

        temporary_relations_csv.replace(
            relations_csv_path
        )

    except Exception:
        if temporary_images_directory.exists():
            shutil.rmtree(temporary_images_directory)

        if temporary_images_csv.exists():
            temporary_images_csv.unlink()

        if temporary_relations_csv.exists():
            temporary_relations_csv.unlink()

        raise

    return len(image_rows), len(relation_rows)


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
        help="Path to the validated pages.csv.",
    )

    parser.add_argument(
        "--images-dir",
        default="data/processed/images",
        help="Directory for extracted raster assets.",
    )

    parser.add_argument(
        "--images-csv",
        default="data/graph_v2/images.csv",
        help="Output path for images.csv.",
    )

    parser.add_argument(
        "--relations-csv",
        default="data/graph_v2/rel_page_image.csv",
        help="Output path for rel_page_image.csv.",
    )

    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    pages_path = Path(args.pages).resolve()
    images_directory = Path(
        args.images_dir
    ).resolve()
    images_csv_path = Path(
        args.images_csv
    ).resolve()
    relations_csv_path = Path(
        args.relations_csv
    ).resolve()

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

        print("[1/4] Collecting raster occurrences...")

        occurrences, payloads = collect_inventory(
            document=document,
            pages=pages,
        )

        print(
            "[OK] Raster occurrences:",
            len(occurrences),
        )

        print("[2/4] Validating exact image payloads...")

        asset_keys = validate_collected_inventory(
            occurrences=occurrences,
            payloads=payloads,
        )

        digest_count = len(
            {
                asset_key[0]
                for asset_key in asset_keys
            }
        )

        print(
            "[OK] Unique digests:",
            digest_count,
        )

        print(
            "[OK] Unique digest-dimension assets:",
            len(asset_keys),
        )

        print("[3/4] Writing final outputs...")

        image_count, relation_count = write_outputs(
            occurrences=occurrences,
            payloads=payloads,
            asset_keys=asset_keys,
            images_directory=images_directory,
            images_csv_path=images_csv_path,
            relations_csv_path=relations_csv_path,
        )

    print("[4/4] Final validation completed")
    print("[OK] Image files:", image_count)
    print("[OK] images.csv rows:", image_count)
    print("[OK] rel_page_image.csv rows:", relation_count)
    print("[OK] No raster occurrence was removed")
    print("[OK] classification_status = pending")


if __name__ == "__main__":
    main()
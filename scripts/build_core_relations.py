import argparse
import csv
import sys
from pathlib import Path


EXPECTED_DOCUMENT_ID = "DOC_WHO_2003"
EXPECTED_PAGE_COUNT = 398
EXPECTED_CHUNK_COUNT = 767


def configure_csv_field_limit() -> None:
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            break
        except OverflowError:
            maximum_limit //= 10


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    return fieldnames, rows


def require_fields(
    path: Path,
    fieldnames: list[str],
    required_fields: set[str],
) -> None:
    missing_fields = required_fields - set(fieldnames)

    if missing_fields:
        raise RuntimeError(
            f"{path.name} is missing required fields: "
            f"{sorted(missing_fields)}"
        )


def validate_inputs(
    documents: list[dict],
    pages: list[dict],
    chunks: list[dict],
) -> None:
    if len(documents) != 1:
        raise RuntimeError(
            "document.csv must contain exactly one document."
        )

    if documents[0]["doc_id"] != EXPECTED_DOCUMENT_ID:
        raise RuntimeError(
            "Unexpected document ID in document.csv: "
            f"{documents[0]['doc_id']}"
        )

    if len(pages) != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} pages, "
            f"but found {len(pages)}."
        )

    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CHUNK_COUNT} chunks, "
            f"but found {len(chunks)}."
        )

    page_ids = [row["page_id"] for row in pages]
    chunk_ids = [row["chunk_id"] for row in chunks]

    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Duplicate page_id detected.")

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("Duplicate chunk_id detected.")

    expected_pdf_pages = list(
        range(1, EXPECTED_PAGE_COUNT + 1)
    )

    actual_pdf_pages = [
        int(row["pdf_page"])
        for row in pages
    ]

    if actual_pdf_pages != expected_pdf_pages:
        raise RuntimeError(
            "pages.csv is not ordered as PDF pages 1 through 398."
        )

    page_lookup = {
        row["page_id"]: row
        for row in pages
    }

    for page in pages:
        if page["doc_id"] != EXPECTED_DOCUMENT_ID:
            raise RuntimeError(
                "Unexpected doc_id on "
                f"PDF page {page['pdf_page']}."
            )

    for chunk in chunks:
        page_id = chunk["page_id"]

        if chunk["doc_id"] != EXPECTED_DOCUMENT_ID:
            raise RuntimeError(
                f"Unexpected doc_id in {chunk['chunk_id']}."
            )

        if page_id not in page_lookup:
            raise RuntimeError(
                f"Unknown page_id in {chunk['chunk_id']}: "
                f"{page_id}"
            )

        page = page_lookup[page_id]

        if chunk["pdf_page"] != page["pdf_page"]:
            raise RuntimeError(
                "PDF-page mismatch between chunk and page: "
                f"{chunk['chunk_id']}"
            )

        if chunk["printed_page"] != page["printed_page"]:
            raise RuntimeError(
                "Printed-page mismatch between chunk and page: "
                f"{chunk['chunk_id']}"
            )


def build_document_page_relations(
    pages: list[dict],
) -> list[dict]:
    relations = []

    for page in pages:
        relations.append(
            {
                "relation_id": (
                    f"R_DOC_PAGE_{int(page['pdf_page']):04d}"
                ),
                "start_id": EXPECTED_DOCUMENT_ID,
                "end_id": page["page_id"],
                "relation_type": "HAS_PAGE",
                "pdf_page": page["pdf_page"],
            }
        )

    return relations


def build_page_chunk_relations(
    chunks: list[dict],
) -> list[dict]:
    relations = []

    for chunk in chunks:
        relations.append(
            {
                "relation_id": (
                    f"R_PAGE_CHUNK_{chunk['chunk_id']}"
                ),
                "start_id": chunk["page_id"],
                "end_id": chunk["chunk_id"],
                "relation_type": "HAS_CHUNK",
                "pdf_page": chunk["pdf_page"],
                "chunk_index_on_page": (
                    chunk["chunk_index_on_page"]
                ),
            }
        )

    return relations


def validate_relations(
    pages: list[dict],
    chunks: list[dict],
    document_page_relations: list[dict],
    page_chunk_relations: list[dict],
) -> None:
    if len(document_page_relations) != len(pages):
        raise RuntimeError(
            "Document-page relation count does not match "
            "the number of pages."
        )

    if len(page_chunk_relations) != len(chunks):
        raise RuntimeError(
            "Page-chunk relation count does not match "
            "the number of chunks."
        )

    all_relation_ids = [
        relation["relation_id"]
        for relation in (
            document_page_relations
            + page_chunk_relations
        )
    ]

    if len(all_relation_ids) != len(set(all_relation_ids)):
        raise RuntimeError(
            "Duplicate relation_id detected."
        )

    page_ids = {
        page["page_id"]
        for page in pages
    }

    chunk_ids = {
        chunk["chunk_id"]
        for chunk in chunks
    }

    related_page_ids = {
        relation["end_id"]
        for relation in document_page_relations
    }

    related_chunk_ids = {
        relation["end_id"]
        for relation in page_chunk_relations
    }

    if related_page_ids != page_ids:
        raise RuntimeError(
            "Document-page relations do not cover all pages."
        )

    if related_chunk_ids != chunk_ids:
        raise RuntimeError(
            "Page-chunk relations do not cover all chunks."
        )

    for relation in document_page_relations:
        if relation["start_id"] != EXPECTED_DOCUMENT_ID:
            raise RuntimeError(
                "Invalid start_id in document-page relation: "
                f"{relation['relation_id']}"
            )

        if relation["relation_type"] != "HAS_PAGE":
            raise RuntimeError(
                "Invalid document-page relation type."
            )

    for relation in page_chunk_relations:
        if relation["start_id"] not in page_ids:
            raise RuntimeError(
                "Unknown page start node in relation: "
                f"{relation['relation_id']}"
            )

        if relation["end_id"] not in chunk_ids:
            raise RuntimeError(
                "Unknown chunk end node in relation: "
                f"{relation['relation_id']}"
            )

        if relation["relation_type"] != "HAS_CHUNK":
            raise RuntimeError(
                "Invalid page-chunk relation type."
            )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--document",
        required=True,
        help="Path to document.csv.",
    )

    parser.add_argument(
        "--pages",
        required=True,
        help="Path to pages.csv.",
    )

    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to chunks.csv.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for relation CSV files.",
    )

    args = parser.parse_args()

    document_path = Path(args.document).resolve()
    pages_path = Path(args.pages).resolve()
    chunks_path = Path(args.chunks).resolve()
    output_directory = Path(args.outdir).resolve()

    for input_path in (
        document_path,
        pages_path,
        chunks_path,
    ):
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input CSV not found: {input_path}"
            )

    configure_csv_field_limit()

    document_fields, documents = read_csv(document_path)
    page_fields, pages = read_csv(pages_path)
    chunk_fields, chunks = read_csv(chunks_path)

    require_fields(
        document_path,
        document_fields,
        {"doc_id"},
    )

    require_fields(
        pages_path,
        page_fields,
        {
            "page_id",
            "doc_id",
            "pdf_page",
            "printed_page",
        },
    )

    require_fields(
        chunks_path,
        chunk_fields,
        {
            "chunk_id",
            "doc_id",
            "page_id",
            "pdf_page",
            "printed_page",
            "chunk_index_on_page",
        },
    )

    validate_inputs(
        documents=documents,
        pages=pages,
        chunks=chunks,
    )

    document_page_relations = (
        build_document_page_relations(pages)
    )

    page_chunk_relations = (
        build_page_chunk_relations(chunks)
    )

    validate_relations(
        pages=pages,
        chunks=chunks,
        document_page_relations=document_page_relations,
        page_chunk_relations=page_chunk_relations,
    )

    document_page_path = (
        output_directory / "rel_document_page.csv"
    )

    page_chunk_path = (
        output_directory / "rel_page_chunk.csv"
    )

    write_csv(
        path=document_page_path,
        fieldnames=[
            "relation_id",
            "start_id",
            "end_id",
            "relation_type",
            "pdf_page",
        ],
        rows=document_page_relations,
    )

    write_csv(
        path=page_chunk_path,
        fieldnames=[
            "relation_id",
            "start_id",
            "end_id",
            "relation_type",
            "pdf_page",
            "chunk_index_on_page",
        ],
        rows=page_chunk_relations,
    )

    print("[OK] Validated document.csv")
    print("[OK] Validated pages.csv")
    print("[OK] Validated chunks.csv")
    print(
        "[OK] Document-page relations: "
        f"{len(document_page_relations)}"
    )
    print(
        "[OK] Page-chunk relations: "
        f"{len(page_chunk_relations)}"
    )
    print("[OK] Every page is linked to the document")
    print("[OK] Every chunk is linked to exactly one page")
    print(f"[OK] Wrote: {document_page_path}")
    print(f"[OK] Wrote: {page_chunk_path}")


if __name__ == "__main__":
    main()
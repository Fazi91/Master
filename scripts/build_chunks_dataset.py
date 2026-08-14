import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path


MAX_WORDS = 220
OVERLAP_WORDS = 40

EXPECTED_DOCUMENT_ID = "DOC_WHO_2003"
EXPECTED_PAGE_COUNT = 398

WORD_PATTERN = re.compile(r"\S+")


def configure_csv_field_limit() -> None:
    """
    Allow Python to read large page-text fields from pages.csv.
    """
    maximum_limit = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maximum_limit)
            break
        except OverflowError:
            maximum_limit //= 10


def text_sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def read_pages(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    required_fields = {
        "page_id",
        "doc_id",
        "pdf_page",
        "printed_page",
        "normalized_text",
        "normalized_text_sha256",
    }

    available_fields = set(reader.fieldnames or [])

    missing_fields = required_fields - available_fields

    if missing_fields:
        raise RuntimeError(
            "pages.csv is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    return rows


def validate_pages_input(rows: list[dict]) -> None:
    if len(rows) != EXPECTED_PAGE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAGE_COUNT} page rows, "
            f"but found {len(rows)}."
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
            "pages.csv does not contain a complete and ordered "
            "PDF-page sequence."
        )

    for row in rows:
        if row["doc_id"] != EXPECTED_DOCUMENT_ID:
            raise RuntimeError(
                "Unexpected document ID on "
                f"PDF page {row['pdf_page']}: "
                f"{row['doc_id']}"
            )

        normalized_text = row["normalized_text"]
        calculated_hash = text_sha256(normalized_text)

        if calculated_hash != row["normalized_text_sha256"]:
            raise RuntimeError(
                "Normalized-text hash mismatch on "
                f"PDF page {row['pdf_page']}."
            )


def build_page_chunks(page: dict) -> list[dict]:
    text = page["normalized_text"]

    word_matches = list(
        WORD_PATTERN.finditer(text)
    )

    if not word_matches:
        return []

    chunks = []
    chunk_index = 1
    start_word = 0

    while start_word < len(word_matches):
        end_word = min(
            start_word + MAX_WORDS,
            len(word_matches),
        )

        start_char = word_matches[start_word].start()
        end_char = word_matches[end_word - 1].end()

        chunk_text = text[start_char:end_char]

        chunks.append(
            {
                "chunk_id": (
                    f"C_{int(page['pdf_page']):04d}_"
                    f"{chunk_index:03d}"
                ),
                "doc_id": page["doc_id"],
                "page_id": page["page_id"],
                "pdf_page": page["pdf_page"],
                "printed_page": page["printed_page"],
                "chunk_index_on_page": chunk_index,
                "start_word_index": start_word,
                "end_word_index_exclusive": end_word,
                "start_char": start_char,
                "end_char_exclusive": end_char,
                "word_count": end_word - start_word,
                "chunk_text_sha256": text_sha256(chunk_text),
                "chunk_text": chunk_text,
            }
        )

        if end_word == len(word_matches):
            break

        next_start_word = end_word - OVERLAP_WORDS

        if next_start_word <= start_word:
            raise RuntimeError(
                "Chunking did not advance on "
                f"PDF page {page['pdf_page']}."
            )

        start_word = next_start_word
        chunk_index += 1

    return chunks


def build_all_chunks(pages: list[dict]) -> list[dict]:
    chunks = []

    for page in pages:
        chunks.extend(
            build_page_chunks(page)
        )

    return chunks


def validate_chunks(
    pages: list[dict],
    chunks: list[dict],
) -> None:
    if not chunks:
        raise RuntimeError("No chunks were generated.")

    page_lookup = {
        row["page_id"]: row
        for row in pages
    }

    chunk_ids = [
        row["chunk_id"]
        for row in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(
            "Duplicate chunk_id detected."
        )

    previous_page_number = 0
    previous_chunk_index = 0

    for chunk in chunks:
        page_id = chunk["page_id"]

        if page_id not in page_lookup:
            raise RuntimeError(
                f"Unknown page_id in chunk: {page_id}"
            )

        page = page_lookup[page_id]

        if chunk["doc_id"] != page["doc_id"]:
            raise RuntimeError(
                f"Document mismatch in {chunk['chunk_id']}."
            )

        if chunk["pdf_page"] != page["pdf_page"]:
            raise RuntimeError(
                f"PDF-page mismatch in {chunk['chunk_id']}."
            )

        if chunk["printed_page"] != page["printed_page"]:
            raise RuntimeError(
                f"Printed-page mismatch in {chunk['chunk_id']}."
            )

        start_char = int(chunk["start_char"])
        end_char = int(chunk["end_char_exclusive"])

        reconstructed_text = page["normalized_text"][
            start_char:end_char
        ]

        if reconstructed_text != chunk["chunk_text"]:
            raise RuntimeError(
                "Chunk text cannot be reconstructed from "
                f"its page: {chunk['chunk_id']}."
            )

        if (
            text_sha256(chunk["chunk_text"])
            != chunk["chunk_text_sha256"]
        ):
            raise RuntimeError(
                f"Chunk hash mismatch: {chunk['chunk_id']}."
            )

        word_count = int(chunk["word_count"])

        if word_count < 1 or word_count > MAX_WORDS:
            raise RuntimeError(
                "Invalid word count in "
                f"{chunk['chunk_id']}: {word_count}"
            )

        current_page_number = int(chunk["pdf_page"])
        current_chunk_index = int(
            chunk["chunk_index_on_page"]
        )

        if current_page_number == previous_page_number:
            if current_chunk_index != previous_chunk_index + 1:
                raise RuntimeError(
                    "Non-continuous chunk index on "
                    f"PDF page {current_page_number}."
                )
        else:
            if current_page_number < previous_page_number:
                raise RuntimeError(
                    "Chunks are not ordered by PDF page."
                )

            if current_chunk_index != 1:
                raise RuntimeError(
                    "The first chunk of PDF page "
                    f"{current_page_number} must have index 1."
                )

        previous_page_number = current_page_number
        previous_chunk_index = current_chunk_index

    chunks_by_page = {}

    for chunk in chunks:
        chunks_by_page.setdefault(
            chunk["page_id"],
            [],
        ).append(chunk)

    for page_id, page_chunks in chunks_by_page.items():
        for index in range(1, len(page_chunks)):
            previous_chunk = page_chunks[index - 1]
            current_chunk = page_chunks[index]

            previous_end = int(
                previous_chunk["end_word_index_exclusive"]
            )
            current_start = int(
                current_chunk["start_word_index"]
            )

            actual_overlap = previous_end - current_start

            if actual_overlap != OVERLAP_WORDS:
                raise RuntimeError(
                    "Incorrect overlap between "
                    f"{previous_chunk['chunk_id']} and "
                    f"{current_chunk['chunk_id']}: "
                    f"{actual_overlap}"
                )

    page_ids_with_text = {
        page["page_id"]
        for page in pages
        if WORD_PATTERN.search(page["normalized_text"])
    }

    page_ids_with_chunks = {
        chunk["page_id"]
        for chunk in chunks
    }

    if page_ids_with_chunks != page_ids_with_text:
        missing = sorted(
            page_ids_with_text - page_ids_with_chunks
        )
        unexpected = sorted(
            page_ids_with_chunks - page_ids_with_text
        )

        raise RuntimeError(
            "Chunk coverage does not match pages containing text. "
            f"Missing={missing}, unexpected={unexpected}"
        )


def write_chunks(
    output_path: Path,
    rows: list[dict],
) -> None:
    fieldnames = [
        "chunk_id",
        "doc_id",
        "page_id",
        "pdf_page",
        "printed_page",
        "chunk_index_on_page",
        "start_word_index",
        "end_word_index_exclusive",
        "start_char",
        "end_char_exclusive",
        "word_count",
        "chunk_text_sha256",
        "chunk_text",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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
        "--pages",
        required=True,
        help="Path to the validated pages.csv file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated chunks.csv file.",
    )

    args = parser.parse_args()

    pages_path = Path(args.pages).resolve()
    output_path = Path(args.output).resolve()

    if not pages_path.is_file():
        raise FileNotFoundError(
            f"pages.csv not found: {pages_path}"
        )

    configure_csv_field_limit()

    pages = read_pages(pages_path)
    validate_pages_input(pages)

    chunks = build_all_chunks(pages)

    validate_chunks(
        pages=pages,
        chunks=chunks,
    )

    write_chunks(
        output_path=output_path,
        rows=chunks,
    )

    pages_with_chunks = len(
        {
            chunk["page_id"]
            for chunk in chunks
        }
    )

    print("[OK] Validated pages.csv")
    print(f"[OK] Input PDF pages: {len(pages)}")
    print(f"[OK] Pages with chunks: {pages_with_chunks}")
    print(f"[OK] Generated chunks: {len(chunks)}")
    print(f"[OK] Maximum words per chunk: {MAX_WORDS}")
    print(f"[OK] Word overlap: {OVERLAP_WORDS}")
    print("[OK] No chunk crosses a PDF-page boundary")
    print("[OK] Every chunk is reconstructable from pages.csv")
    print(f"[OK] Wrote: {output_path}")


if __name__ == "__main__":
    main()
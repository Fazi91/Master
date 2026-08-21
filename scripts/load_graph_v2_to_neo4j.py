"""Validate and load data/graph_v2 into Neo4j Aura.

Validation is read-only and does not require a Neo4j connection. Loading is
intentionally destructive and requires both --replace and a confirmation
token so the legacy graph cannot be deleted accidentally.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_DIR = ROOT / "data" / "graph_v2"
BATCH_SIZE = 500
CONFIRMATION_TOKEN = "REPLACE_WITH_GRAPH_V2"
IMAGE_MODEL = "openai/clip-vit-base-patch32"

ENTITY_LABELS = {
    "ANATOMICAL_SITE": "AnatomicalSite",
    "CELL": "Cell",
    "DISEASE": "Disease",
    "EQUIPMENT": "Equipment",
    "FINDING": "Finding",
    "MEASUREMENT": "Measurement",
    "ORGANISM": "Organism",
    "PROCEDURE": "Procedure",
    "REAGENT": "Reagent",
    "SPECIMEN": "Specimen",
    "VECTOR": "Vector",
}

ENTITY_RELATION_TYPES = {
    "CAUSES",
    "DETECTS",
    "EXAMINES",
    "FOUND_IN",
    "HAS_FINDING",
    "HAS_MEASUREMENT",
    "TRANSMITTED_BY",
    "USES_EQUIPMENT",
    "USES_REAGENT",
}

FILES = {
    "document": "document.csv",
    "pages": "pages.csv",
    "chunks": "chunks.csv",
    "images": "images.csv",
    "entities": "entities.csv",
    "document_page": "rel_document_page.csv",
    "page_chunk": "rel_page_chunk.csv",
    "page_image": "rel_page_image.csv",
    "chunk_image": "rel_chunk_image.csv",
    "mentions": "rel_chunk_entity.csv",
    "entity_relations": "rel_entity_entity.csv",
}

REQUIRED_FIELDS = {
    "document": {"doc_id", "file_name", "pdf_sha256", "pdf_page_count"},
    "pages": {"page_id", "doc_id", "pdf_page", "printed_page", "normalized_text"},
    "chunks": {"chunk_id", "doc_id", "page_id", "pdf_page", "chunk_text"},
    "images": {"image_id", "file_path", "pixel_width", "pixel_height"},
    "entities": {"entity_id", "canonical_name", "normalized_name", "entity_type"},
    "document_page": {"relation_id", "start_id", "end_id", "relation_type"},
    "page_chunk": {"relation_id", "start_id", "end_id", "relation_type"},
    "page_image": {"relation_id", "page_id", "image_id", "relation_type"},
    "chunk_image": {
        "relation_id", "chunk_id", "image_id", "relation_type", "page_id",
        "pdf_page", "semantic_score", "image_type", "link_method",
    },
    "mentions": {
        "mention_id", "chunk_id", "entity_id", "page_id", "pdf_page",
        "mention_text", "start_char", "end_char_exclusive",
        "extraction_method", "confidence",
    },
    "entity_relations": {
        "relation_id", "source_entity_id", "relation_type",
        "target_entity_id", "source_chunk_id", "source_page_id",
        "pdf_page", "evidence_text", "extraction_method", "confidence",
    },
}


def configure_csv_limit():
    value = sys.maxsize
    while True:
        try:
            csv.field_size_limit(value)
            return
        except OverflowError:
            value //= 10


def parse_args():
    parser = argparse.ArgumentParser(description="Load graph_v2 into Neo4j Aura")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--confirm-replace", default="")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--build-image-links", action="store_true")
    parser.add_argument("--image-model", default=IMAGE_MODEL)
    parser.add_argument("--image-threshold", type=float, default=0.20)
    parser.add_argument("--image-min-confidence", type=float, default=0.55)
    parser.add_argument("--image-max-links", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=16)
    return parser.parse_args()


def build_chunk_image_relations(
    graph_dir, model_name, threshold, minimum_confidence,
    max_links_per_image, batch_size,
):
    """Create same-page semantic Chunk -> Image links with CLIP."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    if not 0 <= threshold <= 1:
        raise ValueError("--image-threshold must be between 0 and 1")
    if max_links_per_image < 1 or batch_size < 1:
        raise ValueError("Image link counts and batch size must be at least 1")

    _, chunks = read_csv(graph_dir / "chunks.csv")
    _, images = read_csv(graph_dir / "images.csv")
    _, occurrences = read_csv(graph_dir / "rel_page_image.csv")
    image_by_id = {row["image_id"]: row for row in images}
    allowed_types = {"microscopy", "clinical_or_laboratory", "diagram_or_chart"}
    candidates = []
    seen = set()
    for occurrence in occurrences:
        image = image_by_id.get(occurrence["image_id"])
        if not image:
            continue
        image_type = (
            image.get("final_type") or image.get("predicted_type") or ""
        ).strip()
        confidence = float(image.get("classification_confidence") or 0)
        width = int(image.get("pixel_width") or 0)
        height = int(image.get("pixel_height") or 0)
        key = (occurrence["page_id"], image["image_id"])
        if (key in seen or image_type not in allowed_types
                or confidence < minimum_confidence or min(width, height) < 96):
            continue
        image_path = ROOT / image["file_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        candidates.append({
            **occurrence, **image, "effective_type": image_type,
            "image_path": image_path,
        })
        seen.add(key)
    if not candidates:
        raise RuntimeError("No meaningful image candidates passed the filters")

    def normalized(features):
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features / (features.norm(dim=-1, keepdim=True) + 1e-12)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading CLIP: {model_name}")
    print(f"[INFO] Device: {device}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device).eval()

    text_features = []
    prompts = [
        "Laboratory manual passage: " + " ".join(row["chunk_text"].split())
        for row in chunks
    ]
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            inputs = processor(
                text=prompts[start:start + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            ).to(device)
            text_features.append(normalized(model.get_text_features(**inputs)).cpu())
    text_features = torch.cat(text_features)

    image_features = []
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            opened = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            try:
                inputs = processor(images=opened, return_tensors="pt").to(device)
                image_features.append(
                    normalized(model.get_image_features(**inputs)).cpu()
                )
            finally:
                for opened_image in opened:
                    opened_image.close()
    image_features = torch.cat(image_features)

    chunks_by_page = defaultdict(list)
    for index, chunk in enumerate(chunks):
        chunks_by_page[chunk["page_id"]].append(index)
    relations = []
    for image_index, image in enumerate(candidates):
        indexes = chunks_by_page.get(image["page_id"], [])
        if not indexes:
            continue
        scores = text_features[indexes] @ image_features[image_index]
        ranked = sorted(
            zip(indexes, scores.tolist()), key=lambda item: item[1], reverse=True
        )
        for chunk_index, score in ranked[:max_links_per_image]:
            if score < threshold:
                continue
            chunk = chunks[chunk_index]
            relations.append({
                "chunk_id": chunk["chunk_id"],
                "image_id": image["image_id"],
                "relation_type": "ILLUSTRATED_BY",
                "page_id": image["page_id"],
                "pdf_page": image["pdf_page"],
                "semantic_score": f"{score:.6f}",
                "image_type": image["effective_type"],
                "link_method": f"clip_same_page:{model_name}",
            })
    relations.sort(
        key=lambda row: (int(row["pdf_page"]), row["chunk_id"], row["image_id"])
    )
    for index, row in enumerate(relations, start=1):
        row["relation_id"] = f"rel_chunk_image_{index:06d}"
    output_fields = [
        "relation_id", "chunk_id", "image_id", "relation_type", "page_id",
        "pdf_page", "semantic_score", "image_type", "link_method",
    ]
    output = graph_dir / "rel_chunk_image.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(relations)
    print(f"[OK] Meaningful image candidates: {len(candidates)}")
    print(f"[OK] ILLUSTRATED_BY relations: {len(relations)}")
    print(f"[OK] Wrote: {output}")


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        return fields, list(reader)


def require_unique(rows, field, name):
    values = [row[field] for row in rows]
    if len(values) != len(set(values)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        raise RuntimeError(f"Duplicate {field} in {name}: {duplicates[:10]}")


def load_and_validate(graph_dir):
    data = {}
    for key, filename in FILES.items():
        path = graph_dir / filename
        fields, rows = read_csv(path)
        missing = REQUIRED_FIELDS[key] - set(fields)
        if missing:
            raise RuntimeError(f"{filename} is missing fields: {sorted(missing)}")
        data[key] = rows

    if len(data["document"]) != 1:
        raise RuntimeError("document.csv must contain exactly one row")

    for key, field in (
        ("document", "doc_id"),
        ("pages", "page_id"),
        ("chunks", "chunk_id"),
        ("images", "image_id"),
        ("entities", "entity_id"),
        ("document_page", "relation_id"),
        ("page_chunk", "relation_id"),
        ("page_image", "relation_id"),
        ("chunk_image", "relation_id"),
        ("mentions", "mention_id"),
        ("entity_relations", "relation_id"),
    ):
        require_unique(data[key], field, FILES[key])

    doc_ids = {row["doc_id"] for row in data["document"]}
    page_ids = {row["page_id"] for row in data["pages"]}
    chunk_ids = {row["chunk_id"] for row in data["chunks"]}
    image_ids = {row["image_id"] for row in data["images"]}
    entity_ids = {row["entity_id"] for row in data["entities"]}

    for row in data["pages"]:
        if row["doc_id"] not in doc_ids:
            raise RuntimeError(f"Unknown document on page {row['page_id']}")
    for row in data["chunks"]:
        if row["doc_id"] not in doc_ids or row["page_id"] not in page_ids:
            raise RuntimeError(f"Invalid chunk parent: {row['chunk_id']}")
    for row in data["document_page"]:
        if row["start_id"] not in doc_ids or row["end_id"] not in page_ids:
            raise RuntimeError(f"Invalid HAS_PAGE relation: {row['relation_id']}")
        if row["relation_type"] != "HAS_PAGE":
            raise RuntimeError(f"Unexpected document-page type: {row['relation_type']}")
    for row in data["page_chunk"]:
        if row["start_id"] not in page_ids or row["end_id"] not in chunk_ids:
            raise RuntimeError(f"Invalid HAS_CHUNK relation: {row['relation_id']}")
        if row["relation_type"] != "HAS_CHUNK":
            raise RuntimeError(f"Unexpected page-chunk type: {row['relation_type']}")
    for row in data["page_image"]:
        if row["page_id"] not in page_ids or row["image_id"] not in image_ids:
            raise RuntimeError(f"Invalid CONTAINS_IMAGE relation: {row['relation_id']}")
        if row["relation_type"] != "CONTAINS_IMAGE":
            raise RuntimeError(f"Unexpected page-image type: {row['relation_type']}")
    for row in data["chunk_image"]:
        if row["chunk_id"] not in chunk_ids or row["image_id"] not in image_ids:
            raise RuntimeError(f"Invalid ILLUSTRATED_BY relation: {row['relation_id']}")
        if row["page_id"] not in page_ids:
            raise RuntimeError(f"Unknown chunk-image page: {row['relation_id']}")
        if row["relation_type"] != "ILLUSTRATED_BY":
            raise RuntimeError(f"Unexpected chunk-image type: {row['relation_type']}")
    for row in data["mentions"]:
        if row["chunk_id"] not in chunk_ids or row["entity_id"] not in entity_ids:
            raise RuntimeError(f"Invalid mention: {row['mention_id']}")
        if row["page_id"] not in page_ids:
            raise RuntimeError(f"Unknown mention page: {row['mention_id']}")
    for row in data["entity_relations"]:
        if row["source_entity_id"] not in entity_ids:
            raise RuntimeError(f"Unknown relation source: {row['relation_id']}")
        if row["target_entity_id"] not in entity_ids:
            raise RuntimeError(f"Unknown relation target: {row['relation_id']}")
        if row["source_chunk_id"] not in chunk_ids:
            raise RuntimeError(f"Unknown relation chunk: {row['relation_id']}")
        if row["source_page_id"] not in page_ids:
            raise RuntimeError(f"Unknown relation page: {row['relation_id']}")
        if row["relation_type"] not in ENTITY_RELATION_TYPES:
            raise RuntimeError(
                f"Unsupported entity relation type: {row['relation_type']}"
            )
    for row in data["entities"]:
        if row["entity_type"] not in ENTITY_LABELS:
            raise RuntimeError(f"Unsupported entity type: {row['entity_type']}")

    return data


def to_int(value):
    return int(value) if str(value).strip() else None


def to_float(value):
    return float(value) if str(value).strip() else None


def to_text(value):
    return str(value).strip() if value is not None and str(value).strip() else None


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def run_batches(session, query, rows, size):
    for batch in batches(rows, size):
        session.run(query, rows=batch).consume()


def print_validation_summary(data):
    print("[OK] graph_v2 validation passed")
    for key in FILES:
        print(f"[OK] {key}: {len(data[key])}")
    print("[OK] Entity types:")
    for name, count in sorted(Counter(r["entity_type"] for r in data["entities"]).items()):
        print(f"     {name}: {count}")
    print("[OK] Entity relation types:")
    for name, count in sorted(Counter(r["relation_type"] for r in data["entity_relations"]).items()):
        print(f"     {name}: {count}")


def create_schema(session):
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT page_id IF NOT EXISTS FOR (n:Page) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT image_id IF NOT EXISTS FOR (n:Image) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX entity_normalized_name IF NOT EXISTS FOR (n:Entity) ON (n.normalized_name)",
        "CREATE INDEX page_pdf_page IF NOT EXISTS FOR (n:Page) ON (n.pdf_page)",
    ]
    for statement in statements:
        session.run(statement).consume()


def load_graph(session, data, batch_size):
    run_batches(session, """
        UNWIND $rows AS row
        CREATE (n:Document {id: row.doc_id})
        SET n.file_name = row.file_name,
            n.pdf_sha256 = row.pdf_sha256,
            n.pdf_page_count = toInteger(row.pdf_page_count),
            n.graph_version = 'v2'
    """, data["document"], batch_size)

    page_rows = [dict(row, pdf_page=to_int(row["pdf_page"]),
                      printed_page=to_text(row.get("printed_page")))
                 for row in data["pages"]]
    run_batches(session, """
        UNWIND $rows AS row
        CREATE (n:Page {id: row.page_id})
        SET n.pdf_page = row.pdf_page, n.printed_page = row.printed_page,
            n.text = row.normalized_text, n.graph_version = 'v2'
    """, page_rows, batch_size)

    chunk_rows = [dict(row, pdf_page=to_int(row["pdf_page"]),
                       printed_page=to_text(row.get("printed_page")),
                       chunk_index_on_page=to_int(row.get("chunk_index_on_page")))
                  for row in data["chunks"]]
    run_batches(session, """
        UNWIND $rows AS row
        CREATE (n:Chunk {id: row.chunk_id})
        SET n.pdf_page = row.pdf_page, n.printed_page = row.printed_page,
            n.chunk_index_on_page = row.chunk_index_on_page,
            n.text = row.chunk_text, n.text_sha256 = row.chunk_text_sha256,
            n.graph_version = 'v2'
    """, chunk_rows, batch_size)

    image_rows = [dict(row, pixel_width=to_int(row.get("pixel_width")),
                       pixel_height=to_int(row.get("pixel_height")),
                       occurrence_count=to_int(row.get("occurrence_count")),
                       first_pdf_page=to_int(row.get("first_pdf_page")),
                       classification_confidence=to_float(row.get("classification_confidence")),
                       classification_margin=to_float(row.get("classification_margin")))
                  for row in data["images"]]
    run_batches(session, """
        UNWIND $rows AS row
        CREATE (n:Image {id: row.image_id})
        SET n.file_path = row.file_path, n.digest = row.digest,
            n.pixel_width = row.pixel_width, n.pixel_height = row.pixel_height,
            n.occurrence_count = row.occurrence_count,
            n.first_pdf_page = row.first_pdf_page,
            n.predicted_type = row.predicted_type,
            n.final_type = row.final_type,
            n.content_relevance = row.content_relevance,
            n.classification_status = row.classification_status,
            n.classification_confidence = row.classification_confidence,
            n.graph_version = 'v2'
    """, image_rows, batch_size)

    grouped_entities = defaultdict(list)
    for row in data["entities"]:
        grouped_entities[row["entity_type"]].append(row)
    for entity_type, rows in grouped_entities.items():
        label = ENTITY_LABELS[entity_type]
        query = f"""
            UNWIND $rows AS row
            CREATE (n:Entity:{label} {{id: row.entity_id}})
            SET n.canonical_name = row.canonical_name,
                n.normalized_name = row.normalized_name,
                n.entity_type = row.entity_type,
                n.graph_version = 'v2'
        """
        run_batches(session, query, rows, batch_size)

    run_batches(session, """
        UNWIND $rows AS row
        MATCH (d:Document {id: row.start_id}), (p:Page {id: row.end_id})
        CREATE (d)-[:HAS_PAGE {id: row.relation_id}]->(p)
    """, data["document_page"], batch_size)

    run_batches(session, """
        UNWIND $rows AS row
        MATCH (p:Page {id: row.start_id}), (c:Chunk {id: row.end_id})
        CREATE (p)-[:HAS_CHUNK {id: row.relation_id}]->(c)
    """, data["page_chunk"], batch_size)

    page_image_rows = [dict(row,
        pdf_page=to_int(row.get("pdf_page")),
        occurrence_index_on_page=to_int(row.get("occurrence_index_on_page")),
        xref=to_int(row.get("xref")),
        bbox_x0=to_float(row.get("bbox_x0")), bbox_y0=to_float(row.get("bbox_y0")),
        bbox_x1=to_float(row.get("bbox_x1")), bbox_y1=to_float(row.get("bbox_y1")),
        page_coverage=to_float(row.get("page_coverage")))
        for row in data["page_image"]]
    run_batches(session, """
        UNWIND $rows AS row
        MATCH (p:Page {id: row.page_id}), (i:Image {id: row.image_id})
        CREATE (p)-[:CONTAINS_IMAGE {
            id: row.relation_id, occurrence_index: row.occurrence_index_on_page,
            xref: row.xref, bbox_x0: row.bbox_x0, bbox_y0: row.bbox_y0,
            bbox_x1: row.bbox_x1, bbox_y1: row.bbox_y1,
            page_coverage: row.page_coverage
        }]->(i)
    """, page_image_rows, batch_size)

    chunk_image_rows = [dict(
        row, pdf_page=to_int(row.get("pdf_page")),
        semantic_score=to_float(row.get("semantic_score")),
    ) for row in data["chunk_image"]]
    run_batches(session, """
        UNWIND $rows AS row
        MATCH (c:Chunk {id: row.chunk_id}), (i:Image {id: row.image_id})
        CREATE (c)-[:ILLUSTRATED_BY {
            id: row.relation_id, page_id: row.page_id,
            pdf_page: row.pdf_page, semantic_score: row.semantic_score,
            image_type: row.image_type, link_method: row.link_method
        }]->(i)
    """, chunk_image_rows, batch_size)

    mention_rows = [dict(row, pdf_page=to_int(row["pdf_page"]),
                         start_char=to_int(row["start_char"]),
                         end_char_exclusive=to_int(row["end_char_exclusive"]),
                         confidence=to_float(row["confidence"]))
                    for row in data["mentions"]]
    run_batches(session, """
        UNWIND $rows AS row
        MATCH (c:Chunk {id: row.chunk_id}), (e:Entity {id: row.entity_id})
        CREATE (c)-[:MENTIONS {
            id: row.mention_id, page_id: row.page_id, pdf_page: row.pdf_page,
            text: row.mention_text, start_char: row.start_char,
            end_char_exclusive: row.end_char_exclusive,
            extraction_method: row.extraction_method,
            confidence: row.confidence
        }]->(e)
    """, mention_rows, batch_size)

    grouped_relations = defaultdict(list)
    for row in data["entity_relations"]:
        converted = dict(row, pdf_page=to_int(row["pdf_page"]),
                         confidence=to_float(row["confidence"]))
        grouped_relations[row["relation_type"]].append(converted)
    for relation_type, rows in grouped_relations.items():
        query = f"""
            UNWIND $rows AS row
            MATCH (s:Entity {{id: row.source_entity_id}}),
                  (t:Entity {{id: row.target_entity_id}})
            CREATE (s)-[:{relation_type} {{
                id: row.relation_id, source_chunk_id: row.source_chunk_id,
                source_page_id: row.source_page_id, pdf_page: row.pdf_page,
                evidence_text: row.evidence_text,
                extraction_method: row.extraction_method,
                confidence: row.confidence
            }}]->(t)
        """
        run_batches(session, query, rows, batch_size)


def verify_loaded_graph(session, data):
    expected_nodes = {
        "Document": len(data["document"]), "Page": len(data["pages"]),
        "Chunk": len(data["chunks"]), "Image": len(data["images"]),
        "Entity": len(data["entities"]),
    }
    for label, expected in expected_nodes.items():
        actual = session.run(
            f"MATCH (n:{label}) RETURN count(n) AS count"
        ).single()["count"]
        if actual != expected:
            raise RuntimeError(f"{label}: expected {expected}, loaded {actual}")

    expected_relationships = (
        len(data["document_page"]) + len(data["page_chunk"]) +
        len(data["page_image"]) + len(data["mentions"]) +
        len(data["chunk_image"]) + len(data["entity_relations"])
    )
    actual_relationships = session.run(
        "MATCH ()-[r]->() RETURN count(r) AS count"
    ).single()["count"]
    if actual_relationships != expected_relationships:
        raise RuntimeError(
            f"Relationships: expected {expected_relationships}, "
            f"loaded {actual_relationships}"
        )
    print("[OK] Neo4j verification passed")
    for label, count in expected_nodes.items():
        print(f"[OK] {label}: {count}")
    print(f"[OK] Relationships: {expected_relationships}")


def main():
    configure_csv_limit()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.build_image_links:
        build_chunk_image_relations(
            args.graph_dir, args.image_model, args.image_threshold,
            args.image_min_confidence, args.image_max_links,
            args.image_batch_size,
        )
        return
    data = load_and_validate(args.graph_dir)
    print_validation_summary(data)

    if args.validate_only:
        return
    if not args.replace:
        raise RuntimeError("Use --validate-only, or explicitly use --replace")
    if args.confirm_replace != CONFIRMATION_TOKEN:
        raise RuntimeError(
            "Replacement not confirmed. Required: "
            f"--confirm-replace {CONFIRMATION_TOKEN}"
        )

    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv(ROOT / ".env")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not uri or not user or not password:
        raise RuntimeError("Missing NEO4J_URI, NEO4J_USER, or NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            print("[INFO] Deleting legacy graph...")
            session.run("MATCH (n) DETACH DELETE n").consume()
            print("[INFO] Creating graph_v2 schema...")
            create_schema(session)
            print("[INFO] Loading graph_v2...")
            load_graph(session, data, args.batch_size)
            verify_loaded_graph(session, data)
    finally:
        driver.close()


if __name__ == "__main__":
    main()


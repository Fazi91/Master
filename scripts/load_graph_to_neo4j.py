"""
Load graph CSV files into Neo4j Aura database from local files (data/graph/*).
"""
import csv
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

ROOT = Path(__file__).parent.parent
CSV_FILES = {
    "nodes_text": ROOT / "data" / "graph" / "nodes_text.csv",
    "nodes_image": ROOT / "data" / "graph" / "nodes_image.csv",
    "rel_text_text": ROOT / "data" / "graph" / "rel_text_text.csv",
    "rel_text_image": ROOT / "data" / "graph" / "rel_text_image.csv",
}

# Add parent directory to path
sys.path.insert(0, str(ROOT))
from scripts.graph_client import GraphClient


def _read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_nodes_and_relationships():
    """Load Text/Image nodes and their relationships into Neo4j."""

    # Initialize connection
    gc = GraphClient()

    # 1. Load Text nodes
    print("Loading Text nodes from", CSV_FILES["nodes_text"])
    text_nodes = _read_csv(CSV_FILES["nodes_text"])

    with gc._driver.session(database=gc.database) as session:
        # Create constraint for uniqueness
        session.run("CREATE CONSTRAINT text_id IF NOT EXISTS FOR (t:Text) REQUIRE t.id IS UNIQUE")

        # Batch insert Text nodes
        batch_size = 500
        for i in tqdm(range(0, len(text_nodes), batch_size), desc="Text nodes"):
            batch = text_nodes[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS node
                MERGE (t:Text {id: node.chunk_id})
                SET t.page = toInteger(node.page),
                    t.tokens = toInteger(node.tokens),
                    t.text = node.text
                """,
                batch=batch,
            )

    # 2. Load Image nodes
    print("\nLoading Image nodes from", CSV_FILES["nodes_image"])
    image_nodes = _read_csv(CSV_FILES["nodes_image"])

    with gc._driver.session(database=gc.database) as session:
        # Create constraint for uniqueness
        session.run("CREATE CONSTRAINT image_id IF NOT EXISTS FOR (i:Image) REQUIRE i.id IS UNIQUE")

        # Batch insert Image nodes
        for i in tqdm(range(0, len(image_nodes), batch_size), desc="Image nodes"):
            batch = image_nodes[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS node
                MERGE (img:Image {id: node.img_id})
                SET img.page = toInteger(node.page),
                    img.path = node.path
                """,
                batch=batch,
            )

    # 3. Load Text-Text relationships
    print("\nLoading Text-Text relationships from", CSV_FILES["rel_text_text"])
    text_rels = _read_csv(CSV_FILES["rel_text_text"])

    with gc._driver.session(database=gc.database) as session:
        for i in tqdm(range(0, len(text_rels), batch_size), desc="Text-Text rels"):
            batch = text_rels[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS rel
                MATCH (src:Text {id: rel.src_chunk_id})
                MATCH (dst:Text {id: rel.dst_chunk_id})
                MERGE (src)-[r:SIMILAR_TO]->(dst)
                SET r.score = toFloat(rel.cosine)
                """,
                batch=batch,
            )

    # 4. Load Text-Image relationships
    print("\nLoading Text-Image relationships from", CSV_FILES["rel_text_image"])
    image_rels = _read_csv(CSV_FILES["rel_text_image"])

    with gc._driver.session(database=gc.database) as session:
        for i in tqdm(range(0, len(image_rels), batch_size), desc="Text-Image rels"):
            batch = image_rels[i:i+batch_size]
            session.run(
                """
                UNWIND $batch AS rel
                MATCH (t:Text {id: rel.chunk_id})
                MATCH (img:Image {id: rel.img_id})
                MERGE (t)-[r:HAS_IMAGE]->(img)
                SET r.score = toFloat(coalesce(rel.score, "0.61"))
                """,
                batch=batch,
            )

    # 5. Print summary
    print("\n=== Graph Summary ===")
    with gc._driver.session(database=gc.database) as session:
        text_count = session.run("MATCH (t:Text) RETURN count(t) AS count").single()["count"]
        image_count = session.run("MATCH (i:Image) RETURN count(i) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

        print(f"Text nodes: {text_count}")
        print(f"Image nodes: {image_count}")
        print(f"Relationships: {rel_count}")

    print("\n✓ Graph loaded successfully!")
    gc.close()


if __name__ == "__main__":
    load_nodes_and_relationships()

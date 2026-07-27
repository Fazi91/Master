"""
Load Text-Image relationships from GitHub CSV.
"""
from dotenv import load_dotenv
load_dotenv()

import csv
from io import StringIO
from pathlib import Path
from tqdm import tqdm
import requests
from scripts.graph_client import GraphClient

GITHUB_BASE = "https://raw.githubusercontent.com/Fazi91/Master/main/data/graph"

def fetch_csv_from_github(url):
    """Fetch CSV from GitHub and return list of dictionaries."""
    print(f"Fetching {url}...")
    response = requests.get(url)
    response.raise_for_status()
    reader = csv.DictReader(StringIO(response.text))
    return list(reader)

# Load and create relationships
gc = GraphClient()
batch_size = 500

print("Loading Text-Image relationships from GitHub...")
image_rels = fetch_csv_from_github(
    f"{GITHUB_BASE}/rel_text_image.csv"
)

print(f"Total relationships to load: {len(image_rels)}")

with gc._driver.session(database=gc.database) as session:
    for i in tqdm(range(0, len(image_rels), batch_size), desc="Text-Image rels"):
        batch = image_rels[i:i+batch_size]
        session.run("""
            UNWIND $batch AS rel
            MATCH (t:Text {id: rel.chunk_id})
            MATCH (img:Image {id: rel.img_id})
            MERGE (t)-[r:HAS_IMAGE]->(img)
            SET r.score = toFloat(rel.score),
                r.method = rel.method
        """, batch=batch)

# Verify
with gc._driver.session(database=gc.database) as session:
    count = session.run('MATCH ()-[r:HAS_IMAGE]->() RETURN count(r) AS n').single()['n']
    print(f"\nText-Image relationships created: {count}")

gc.close()
print("Done!")

"""
Clear all data from Neo4j before fresh load.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.graph_client import GraphClient

gc = GraphClient()

with gc._driver.session(database=gc.database) as session:
    print("Dropping constraints...")
    try:
        session.run("DROP CONSTRAINT text_id IF EXISTS")
    except:
        pass
    try:
        session.run("DROP CONSTRAINT image_id IF EXISTS")
    except:
        pass
    
    print("Deleting all nodes and relationships...")
    session.run("MATCH (n) DETACH DELETE n")
    
    count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    print(f"Nodes remaining: {count}")

gc.close()
print("✓ Graph cleared!")

"""
Build mapping from semantic_X IDs (FAISS) to T_XXXXX IDs (Neo4j).
"""
import json
from pathlib import Path

def build_mapping():
    """Build and save semantic_X -> T_XXXXX mapping."""
    
    chunks_path = Path("outputs/who_chunks_semantic.jsonl")
    mapping_path = Path("outputs/semantic_to_neo4j_mapping.json")
    
    mapping = {}  # semantic_X -> T_XXXXX
    
    # Read semantic chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        idx = 0
        for line in f:
            chunk = json.loads(line)
            semantic_id = chunk.get("id")  # e.g., "semantic_0"
            
            # Map to Neo4j ID: T_00001, T_00002, etc.
            # Since chunks are sequential, just number them
            neo4j_id = f"T_{idx+1:05d}"  # T_00001, T_00002, ...
            
            mapping[semantic_id] = neo4j_id
            idx += 1
    
    # Save mapping
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ Mapping saved to {mapping_path}")
    print(f"  Total mappings: {len(mapping)}")
    print(f"  Sample: semantic_0 -> {mapping.get('semantic_0')}")
    print(f"  Sample: semantic_5 -> {mapping.get('semantic_5')}")

if __name__ == "__main__":
    build_mapping()

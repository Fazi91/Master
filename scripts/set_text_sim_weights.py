from pathlib import Path
import numpy as np
from neo4j import GraphDatabase
import json

ROOT = Path(".")
EMB_NPY  = ROOT / "outputs" / "text" / "embeddings.npy"
META_NPY = ROOT / "outputs" / "text" / "chunk_meta.npy"

def load_neo_cfg(cfg_path="config.neo4j.json"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 1) بارگذاری متا و امبدینگ‌ها
    assert META_NPY.exists(), f"Missing meta: {META_NPY}"
    assert EMB_NPY.exists(),  f"Missing embeddings: {EMB_NPY}"
    meta = np.load(META_NPY, allow_pickle=True)
    E = np.load(EMB_NPY).astype("float32")
    # نرمال تا ضرب داخلی = کوساین
    # اگر قبلاً نرمال شده، دوباره نرمال آسیب نمی‌زند
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    E = E / norms

    # map از chunk_id به ایندکس ردیف
    id2idx = {dict(m)["chunk_id"]: i for i, m in enumerate(meta)}

    # 2) لیست یال‌های موجود TEXT_SIMILAR
    cfg = load_neo_cfg()
    drv = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    with drv.session() as s:
        rels = s.run("""
            MATCH (a:Chunk)-[r:TEXT_SIMILAR]->(b:Chunk)
            RETURN a.chunk_id AS src, b.chunk_id AS dst
        """).data()

    if not rels:
        print("No TEXT_SIMILAR relationships found. Nothing to update.")
        return

    print(f"Found {len(rels)} TEXT_SIMILAR edges. Computing weights...")

    # 3) وزن هر یال = dot(E[src], E[dst])
    rows = []
    missing = 0
    for r in rels:
        i = id2idx.get(r["src"])
        j = id2idx.get(r["dst"])
        if i is None or j is None:
            missing += 1
            continue
        w = float(np.dot(E[i], E[j]))
        rows.append({"src": r["src"], "dst": r["dst"], "w": w})

    print(f"Prepared {len(rows)} rows (skipped {missing} missing ids).")

    # 4) آپدیت در بچ‌های کوچک
    BATCH = 1000
    q = """
    UNWIND $rows AS row
    MATCH (a:Chunk {chunk_id: row.src})-[r:TEXT_SIMILAR]->(b:Chunk {chunk_id: row.dst})
    SET r.weight = row.w
    """
    with drv.session() as s:
        for k in range(0, len(rows), BATCH):
            chunk = rows[k:k+BATCH]
            s.run(q, rows=chunk).consume()
            print(f"Updated {k+len(chunk)}/{len(rows)}")

    print("Done. Weights set on TEXT_SIMILAR.")
    drv.close()

if __name__ == "__main__":
    main()

from pathlib import Path
import csv, sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# مسیرها
ROOT = Path(".")
CSV_PATHS = [
    ROOT / "data" / "graph" / "nodes_text.csv",
    ROOT / "outputs" / "text" / "nodes_text.csv",   # اگه قبلا کپی کرده‌ای
]
OUT_DIR = ROOT / "outputs" / "text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_NPY   = OUT_DIR / "chunk_meta.npy"
EMB_NPY    = OUT_DIR / "embeddings.npy"
INDEX_PATH = OUT_DIR / "minilm_index.faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def find_csv():
    for p in CSV_PATHS:
        if p.exists():
            return p
    print("ERROR: nodes_text.csv پیدا نشد. یکی از مسیرهای زیر باید وجود داشته باشد:")
    for p in CSV_PATHS:
        print(" -", p)
    sys.exit(1)

def read_nodes_csv(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # نام ستون‌ها را مقاوم به اسم‌های مختلف می‌گیریم
        cols = {k.lower(): k for k in reader.fieldnames}
        def pick(d, candidates, default=None):
            for c in candidates:
                if c in cols:
                    return d[cols[c]]
            return default
        for r in reader:
            text = pick(r, ["text","content","chunk","body"], "")
            cid  = pick(r, ["chunk_id","id"], "")
            page = pick(r, ["page","page_num","pageid"], "")
            rows.append({"chunk_id": cid, "page": page, "text": text})
    if not rows:
        print("ERROR: nodes_text.csv خالی است یا ستون‌های لازم را ندارد.")
        sys.exit(1)
    return rows

def build_embeddings(texts):
    model = SentenceTransformer(MODEL_NAME)
    E = model.encode(texts, normalize_embeddings=True).astype("float32")
    return E

def build_index(E: np.ndarray, out_path: Path):
    faiss.normalize_L2(E)   # برای IP که معادل cosine شود
    d = E.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(E)
    faiss.write_index(index, str(out_path))
    return index.ntotal

if __name__ == "__main__":
    csv_path = find_csv()
    print(f"Using CSV: {csv_path}")
    rows = read_nodes_csv(csv_path)
    texts = [r["text"] or "" for r in rows]

    print(f"Encoding {len(texts)} chunks with MiniLM...")
    E = build_embeddings(texts)

    print("Saving embeddings and meta...")
    np.save(EMB_NPY, E)
    np.save(META_NPY, np.array(rows, dtype=object))

    print("Building FAISS index (IP/cosine-like)...")
    n = build_index(E, INDEX_PATH)
    print(f"OK: wrote {INDEX_PATH} with {n} vectors.")
    print(f"OK: wrote {META_NPY} and {EMB_NPY}.")

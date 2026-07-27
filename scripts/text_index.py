# scripts/text_index.py
# Build and query a FAISS index for text chunks.
# Modes:
#   build  -> create FAISS IP index from JSONL chunks and write meta
#   search -> run ad-hoc queries against the index

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def encode_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 128) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embs, list):
        embs = np.array(embs)
    return embs.astype("float32")


def build_index(chunks_jsonl: Path, index_path: Path, meta_path: Path, model_name: str, batch_size: int):
    # 1) load chunks
    rows = load_jsonl(chunks_jsonl)
    texts = [r["text"] for r in rows]

    # 2) encode
    model = SentenceTransformer(model_name, device="cpu")
    embs = encode_texts(texts, model, batch_size=batch_size)  # L2-normalized already

    # 3) build FAISS IP index (cosine)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # 4) save index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    # 5) write meta aligned with vector order
    meta_rows = []
    for i, r in enumerate(rows):
        meta_rows.append({
            "row_id": i,
            "id": r.get("id", f"row_{i}"),
            "method": r.get("method", "unknown"),
            "start_token": r.get("start_token", None),
            "end_token": r.get("end_token", None),
            "n_tokens": r.get("n_tokens", None),
        })
    save_jsonl(meta_rows, meta_path)

    print(f"Built index @ {index_path} | ntotal={index.ntotal} | dim={dim}")
    print(f"Meta written @ {meta_path}")
    print("Done.")


def search_index(index_path: Path, meta_path: Path, model_name: str, query: str, top_k: int, chunks_jsonl: Path):
    # load index + meta + text
    index = faiss.read_index(str(index_path))
    meta = load_jsonl(meta_path)
    chunks = load_jsonl(chunks_jsonl)

    # encode query
    model = SentenceTransformer(model_name, device="cpu")
    q = encode_texts([query], model)  # normalized
    D, I = index.search(q, top_k)
    sims = D[0]
    idxs = I[0]

    print(f"Query: {query}")
    for rank, (doc_id, score) in enumerate(zip(idxs, sims), 1):
        m = meta[int(doc_id)]
        txt = chunks[int(doc_id)]["text"].replace("\n", " ")
        preview = txt[:220] + ("..." if len(txt) > 220 else "")
        print(f"{rank}. score={score:.4f} | row_id={int(doc_id)} | id={m['id']} | method={m['method']} | n_tokens={m['n_tokens']}")
        print(f"   > {preview}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build FAISS IP index from chunks JSONL")
    b.add_argument("--chunks", required=True, help="Path to chunks JSONL")
    b.add_argument("--index", required=True, help="Output FAISS index path")
    b.add_argument("--meta", required=True, help="Output meta JSONL path")
    b.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--batch_size", type=int, default=128)

    s = sub.add_parser("search", help="Query FAISS index")
    s.add_argument("--chunks", required=True, help="Path to chunks JSONL (for preview)")
    s.add_argument("--index", required=True, help="FAISS index path")
    s.add_argument("--meta", required=True, help="Meta JSONL path")
    s.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    s.add_argument("--query", required=True)
    s.add_argument("--top_k", type=int, default=5)

    args = ap.parse_args()
    if args.cmd == "build":
        build_index(Path(args.chunks), Path(args.index), Path(args.meta), args.model, args.batch_size)
    else:
        search_index(Path(args.index), Path(args.meta), args.model, args.query, args.top_k, Path(args.chunks))


if __name__ == "__main__":
    main()

# scripts/compare_chunks.py
# Compare retrieval quality between baseline and semantic chunking using the same encoder.
# Input:
#   --baseline_jsonl outputs\who_chunks_baseline.jsonl
#   --semantic_jsonl outputs\who_chunks_semantic.jsonl
#   --queries_file queries.txt      (one query per line)
# Output:
#   outputs\compare\retrieval_report.md  (human-readable side-by-side top-k)
#   Prints a compact score summary to stdout.

import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_queries(path: Path) -> List[str]:
    qs = [q.strip() for q in path.read_text(encoding="utf-8").splitlines()]
    return [q for q in qs if q]


def build_faiss_ip_index(embs: np.ndarray) -> faiss.Index:
    # normalize for cosine, then use Inner Product
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


def encode_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embs, list):
        embs = np.array(embs)
    return embs.astype("float32")


def keyword_coverage_score(query: str, text: str) -> float:
    # very simple proxy: fraction of unique query tokens present in the chunk text
    import re
    toks_q = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not toks_q:
        return 0.0
    toks_t = set(re.findall(r"\w+", text.lower()))
    hit = sum(1 for t in set(toks_q) if t in toks_t)
    return hit / len(set(toks_q))


def search_topk(
    queries: List[str],
    index: faiss.Index,
    doc_texts: List[str],
    doc_meta: List[Dict],
    model: SentenceTransformer,
    top_k: int
) -> List[List[Tuple[int, float, float]]]:
    # returns per-query list of tuples: (doc_idx, cosine_sim, keyword_score)
    q_embs = encode_texts(queries, model)
    D, I = index.search(q_embs, top_k)
    results = []
    for qi, (scores, idxs) in enumerate(zip(D, I)):
        row = []
        for j, (score, doc_id) in enumerate(zip(scores, idxs)):
            kw = keyword_coverage_score(queries[qi], doc_texts[doc_id])
            row.append((int(doc_id), float(score), float(kw)))
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_jsonl", required=True)
    ap.add_argument("--semantic_jsonl", required=True)
    ap.add_argument("--queries_file", required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    p_base = Path(args.baseline_jsonl)
    p_sem = Path(args.semantic_jsonl)
    p_q = Path(args.queries_file)

    base = load_jsonl(p_base)
    sem = load_jsonl(p_sem)
    queries = read_queries(p_q)

    model = SentenceTransformer(args.model_name, device="cpu")

    base_texts = [r["text"] for r in base]
    sem_texts = [r["text"] for r in sem]

    base_embs = encode_texts(base_texts, model)
    sem_embs  = encode_texts(sem_texts,  model)

    base_index = build_faiss_ip_index(base_embs.copy())
    sem_index  = build_faiss_ip_index(sem_embs.copy())

    base_hits = search_topk(queries, base_index, base_texts, base, model, args.top_k)
    sem_hits  = search_topk(queries, sem_index,  sem_texts,  sem,  model, args.top_k)

    # Score summary: average cosine and keyword coverage@k
    def summarize(hits):
        cos_means, kw_means = [], []
        for row in hits:
            cos_means.append(np.mean([x[1] for x in row]))
            kw_means.append(np.mean([x[2] for x in row]))
        return float(np.mean(cos_means)), float(np.mean(kw_means))

    base_cos, base_kw = summarize(base_hits)
    sem_cos,  sem_kw  = summarize(sem_hits)

    out_dir = Path("outputs/compare")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "retrieval_report.md"

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Retrieval Comparison (Baseline vs Semantic)\n\n")
        f.write(f"- top_k = {args.top_k}\n")
        f.write(f"- model = {args.model_name}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Baseline avg cosine@{args.top_k}: **{base_cos:.4f}** | keyword@{args.top_k}: **{base_kw:.4f}**\n")
        f.write(f"- Semantic avg cosine@{args.top_k}: **{sem_cos:.4f}** | keyword@{args.top_k}: **{sem_kw:.4f}**\n\n")

        for qi, q in enumerate(queries):
            f.write(f"---\n\n### Q{qi+1}: {q}\n\n")
            f.write("**Baseline top-k**\n\n")
            for rank, (doc_id, cos, kw) in enumerate(base_hits[qi], 1):
                txt = base_texts[doc_id][:280].replace("\n", " ")
                f.write(f"{rank}. cos={cos:.4f} | kw={kw:.2f} | id=baseline_{doc_id}\n\n> {txt}...\n\n")
            f.write("\n**Semantic top-k**\n\n")
            for rank, (doc_id, cos, kw) in enumerate(sem_hits[qi], 1):
                txt = sem_texts[doc_id][:280].replace("\n", " ")
                f.write(f"{rank}. cos={cos:.4f} | kw={kw:.2f} | id=semantic_{doc_id}\n\n> {txt}...\n\n")

    print("Report:", str(out_md))
    print(f"Baseline avg cosine@{args.top_k}: {base_cos:.4f} | keyword@{args.top_k}: {base_kw:.4f}")
    print(f"Semantic  avg cosine@{args.top_k}: {sem_cos:.4f} | keyword@{args.top_k}: {sem_kw:.4f}")
    print("Done.")
    

if __name__ == "__main__":
    main()

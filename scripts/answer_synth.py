# scripts/answer_synth.py
# Answer synthesis for RAG (extractive).
# Modes:
#   answer -> retrieve (+optional rerank) and write a concise, cited markdown answer
#
# Inputs:
#   --query "..."                       : user question
#   --chunks outputs\who_chunks_semantic.jsonl
#   --index  outputs\text\bge_index.faiss
#   --meta   outputs\text\bge_meta.jsonl
#   --model  BAAI/bge-small-en-v1.5
#   --top_k  5
#   --rerank (store_true)               : optional cross-encoder rerank
#   --cross_encoder cross-encoder/ms-marco-MiniLM-L-6-v2
#   --attach_images (store_true)        : attach page images if mappable
#   --page_map outputs\images\page_to_images.json
#   --clean_txt outputs\who_clean.txt   : needed to map token offsets -> page index
#   --out_md outputs\answers\answer.md
#
# Output:
#   A compact markdown answer with cited quotes and, if available, image file hints.

import argparse, json, re, os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- IO ----------
def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

# ---------- Embedding helpers ----------
def encode_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embs, list):
        import numpy as np
        embs = np.array(embs)
    return embs.astype("float32")

# ---------- Retrieval ----------
def retrieve(index_path: Path, chunks_path: Path, meta_path: Path, model_name: str, query: str, top_k: int) -> List[Dict]:
    index = faiss.read_index(str(index_path))
    chunks = load_jsonl(chunks_path)
    meta = load_jsonl(meta_path)
    model = SentenceTransformer(model_name, device="cpu")
    q = encode_texts([query], model)
    D, I = index.search(q, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    out = []
    for rank, (i, s) in enumerate(zip(idxs, scores), 1):
        row = {
            "rank": rank,
            "score": float(s),
            "row_id": int(i),
            "id": meta[i].get("id"),
            "method": meta[i].get("method"),
            "n_tokens": meta[i].get("n_tokens"),
            "start_token": meta[i].get("start_token"),
            "end_token": meta[i].get("end_token"),
            "text": chunks[i]["text"]
        }
        out.append(row)
    return out

# ---------- Optional rerank ----------
def rerank(query: str, candidates: List[Dict], cross_encoder_name: str, top_k: int) -> List[Dict]:
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(cross_encoder_name, device="cpu")
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    for r, c in enumerate(ranked, 1):
        c["rank"] = r
    return ranked

# ---------- Simple sentence ranking for extractive summary ----------
_SENT_SPLIT = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|…|؟)\s+')

def split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    if len(parts) <= 1:
        parts = re.split(r'\n{2,}|(?<=\.|\?|!)\n', text)
    return [p.strip() for p in parts if p.strip()]

def keyword_score(query: str, sentence: str) -> float:
    toks_q = set(t for t in re.findall(r"\w+", query.lower()) if len(t) > 2)
    toks_s = set(re.findall(r"\w+", sentence.lower()))
    if not toks_q:
        return 0.0
    hit = sum(1 for t in toks_q if t in toks_s)
    return hit / len(toks_q)

def pick_best_sentences(query: str, texts: List[str], limit: int = 6) -> List[str]:
    scored = []
    for t in texts:
        for s in split_sentences(t):
            if len(s) < 20:  # skip too-short
                continue
            ks = keyword_score(query, s)
            if ks == 0:
                continue
            scored.append((ks, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    uniq, seen = [], set()
    for _, s in scored:
        key = s[:160]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
        if len(uniq) >= limit:
            break
    return uniq

# ---------- Token -> page mapping (optional) ----------
# We map chunk token offsets back to page by tokenizing the full clean text split by <<<PAGE_BREAK>>>
import tiktoken
def token_to_page(start_token: int, clean_text: str, encoding_name: str = "cl100k_base") -> int:
    # returns 1-based page index; if cannot infer, returns -1
    if start_token is None:
        return -1
    parts = clean_text.split("<<<PAGE_BREAK>>>")
    enc = tiktoken.get_encoding(encoding_name)
    acc = 0
    for i, part in enumerate(parts, 1):
        n = len(enc.encode(part))
        if start_token < acc + n:
            return i
        acc += n
    return -1

# ---------- Build markdown answer ----------
def build_markdown_answer(query: str, hits: List[Dict], attach_images: bool, page_map_path: Path, clean_txt_path: Path) -> str:
    lines = []
    lines.append(f"# Answer\n")
    lines.append(f"**Query:** {query}\n")
    # concise bullets from best sentences
    best_sents = pick_best_sentences(query, [h["text"] for h in hits], limit=6)
    if best_sents:
        lines.append("## Summary\n")
        for s in best_sents:
            lines.append(f"- {s}")
        lines.append("")

    # quoted evidence
    lines.append("## Evidence (top hits)\n")
    clean_txt = None
    page_map = None
    if attach_images and clean_txt_path and clean_txt_path.exists():
        clean_txt = clean_txt_path.read_text(encoding="utf-8")
    if attach_images and page_map_path and page_map_path.exists():
        try:
            page_map = json.loads(page_map_path.read_text(encoding="utf-8"))
        except Exception:
            page_map = None

    for h in hits:
        preview = h["text"].strip().replace("\n", " ")
        preview = preview[:480] + ("..." if len(preview) > 480 else "")
        page_info = ""
        images_info = ""
        if clean_txt is not None:
            p = token_to_page(h.get("start_token"), clean_txt)
            if p != -1:
                page_info = f" | page={p}"
                if page_map and str(p) in page_map and page_map[str(p)]:
                    images = page_map[str(p)][:2]  # at most 2
                    images_info = " | images: " + ", ".join(images)
        lines.append(f"- **{h['id']}** | score={h['score']:.4f}{page_info}{images_info}\n  > {preview}")
    lines.append("")
    return "\n".join(lines)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--attach_images", action="store_true")
    ap.add_argument("--page_map", default="outputs/images/page_to_images.json")
    ap.add_argument("--clean_txt", default="outputs/who_clean.txt")
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    hits = retrieve(Path(args.index), Path(args.chunks), Path(args.meta), args.model, args.query, args.top_k)
    if args.rerank:
        hits = rerank(args.query, hits, args.cross_encoder, args.top_k)

    md = build_markdown_answer(
        args.query,
        hits,
        attach_images=args.attach_images,
        page_map_path=Path(args.page_map),
        clean_txt_path=Path(args.clean_txt),
    )
    save_text(Path(args.out_md), md)
    print(f"Wrote: {args.out_md}")

if __name__ == "__main__":
    main()

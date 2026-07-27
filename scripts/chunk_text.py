# scripts/chunk_text.py
# One file, two modes: baseline and semantic chunking.
# Minimal deps: tiktoken, sentence-transformers (for semantic), torch (CPU OK).

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import tiktoken


# ---------- Tokenization helpers ----------

def get_encoder(name: str = "cl100k_base"):
    return tiktoken.get_encoding(name)

def encode(text: str, enc) -> List[int]:
    return enc.encode(text)

def decode(tokens: List[int], enc) -> str:
    return enc.decode(tokens)

def trim_to_tokens(text: str, max_tokens: int, enc) -> str:
    toks = encode(text, enc)
    if len(toks) <= max_tokens:
        return text
    return decode(toks[:max_tokens], enc)


# ---------- Baseline chunking ----------

def chunk_baseline(
    text: str,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 150,
    encoding_name: str = "cl100k_base",
) -> List[Dict]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    enc = get_encoder(encoding_name)
    toks = encode(text, enc)
    n = len(toks)
    out, start, idx = [], 0, 0

    while start < n:
        end = min(start + chunk_tokens, n)
        piece = decode(toks[start:end], enc)
        out.append({
            "id": f"baseline_{idx}",
            "method": "baseline",
            "start_token": start,
            "end_token": end,
            "n_tokens": end - start,
            "text": piece,
        })
        idx += 1
        if end == n:
            break
        start = max(0, end - overlap_tokens)
    return out


# ---------- Semantic chunking ----------

_SENT_SPLIT_RE = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|…|।|؟)\s+'  # split on sentence end punctuation + space
)

def sentence_split(text: str) -> List[str]:
    # conservative split; keeps punctuation
    parts = _SENT_SPLIT_RE.split(text.strip())
    # fallback in case text is one huge paragraph
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r'\n{2,}|(?<=\.|\?|!)\n', text) if s.strip()]
    return [p.strip() for p in parts if p.strip()]

def chunk_semantic(
    text: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_tokens: int = 1000,
    overlap_tokens: int = 150,
    sim_threshold: float = 0.55,
    min_chunk_tokens: int = 400,
    encoding_name: str = "cl100k_base",
) -> List[Dict]:
    """
    Strategy:
      1) Split into sentences.
      2) Embed sentences.
      3) Greedily pack sentences until token budget ~chunk_tokens.
      4) If similarity with previous sentence drops below sim_threshold and current chunk already has >= min_chunk_tokens,
         cut a new chunk to respect topical boundaries.
      5) Add token-level overlap between chunks (by carrying the last `overlap_tokens` tokens).
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens.")

    sentences = sentence_split(text)
    enc = get_encoder(encoding_name)

    # If very short, just return single chunk
    if len(encode(text, enc)) <= chunk_tokens:
        return [{
            "id": "semantic_0",
            "method": "semantic",
            "start_token": 0,
            "end_token": len(encode(text, enc)),
            "n_tokens": len(encode(text, enc)),
            "text": text,
        }]

    # Lazy import to avoid torch load in baseline mode
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer(model_name, device="cpu")
    embs = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)

    out = []
    cur_sent_idx = 0
    chunk_idx = 0

    while cur_sent_idx < len(sentences):
        chunk_sents = []
        chunk_tokens_used = 0
        start_token = None

        # If we have a previous chunk, start with overlap prefix
        overlap_prefix = ""
        if out:
            # take last overlap_tokens from previous chunk, prepend as context
            prev_text = out[-1]["text"]
            prev_toks = encode(prev_text, enc)
            prefix = decode(prev_toks[-overlap_tokens:], enc) if len(prev_toks) > overlap_tokens else prev_text
            overlap_prefix = prefix

        # Account for overlap in budget
        if overlap_prefix:
            tok_ov = encode(overlap_prefix, enc)
            chunk_tokens_used += len(tok_ov)
            chunk_sents.append(overlap_prefix)

        # pack sentences
        while cur_sent_idx < len(sentences):
            sent = sentences[cur_sent_idx]
            sent_toks = encode(sent, enc)
            # similarity cut
            should_cut_by_sim = False
            if (len(chunk_sents) > 0 and not overlap_prefix) or (len(chunk_sents) > 1 and overlap_prefix):
                prev_idx = cur_sent_idx - 1
                if prev_idx >= 0:
                    cos = float(util.cos_sim(embs[prev_idx], embs[cur_sent_idx]).item())
                    if cos < sim_threshold and chunk_tokens_used >= min_chunk_tokens:
                        should_cut_by_sim = True

            if should_cut_by_sim:
                break

            if chunk_tokens_used + len(sent_toks) > chunk_tokens:
                # stop if adding this sentence would exceed budget
                if chunk_tokens_used >= min_chunk_tokens:
                    break
                # if chunk is still tiny, force-add and then trim at the end
                chunk_sents.append(sent)
                chunk_tokens_used += len(sent_toks)
                cur_sent_idx += 1
                break

            chunk_sents.append(sent)
            chunk_tokens_used += len(sent_toks)
            cur_sent_idx += 1

        chunk_text = " ".join(chunk_sents).strip()
        # Hard trim to token budget (safety)
        chunk_text = trim_to_tokens(chunk_text, chunk_tokens, enc)

        if start_token is None:
            # approximate: use previous end to infer start; precise token indices are optional here
            start_token = 0 if not out else out[-1]["end_token"] - overlap_tokens
            if start_token < 0:
                start_token = 0
        end_token = start_token + len(encode(chunk_text, enc))

        out.append({
            "id": f"semantic_{chunk_idx}",
            "method": "semantic",
            "start_token": start_token,
            "end_token": end_token,
            "n_tokens": end_token - start_token,
            "text": chunk_text,
        })
        chunk_idx += 1

    return out


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True, help="Input cleaned text file")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL path")
    ap.add_argument("--method", choices=["baseline", "semantic"], required=True)
    ap.add_argument("--chunk_tokens", type=int, default=1000)
    ap.add_argument("--overlap_tokens", type=int, default=150)
    ap.add_argument("--encoding", default="cl100k_base")
    # semantic opts
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--min_chunk_tokens", type=int, default=400)
    args = ap.parse_args()

    in_path = Path(args.in_txt)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = in_path.read_text(encoding="utf-8")

    if args.method == "baseline":
        chunks = chunk_baseline(
            text,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            encoding_name=args.encoding,
        )
    else:
        chunks = chunk_semantic(
            text,
            model_name=args.model_name,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            sim_threshold=args.sim_threshold,
            min_chunk_tokens=args.min_chunk_tokens,
            encoding_name=args.encoding,
        )

    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

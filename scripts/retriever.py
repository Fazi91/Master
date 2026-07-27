# scripts/retriever.py — app-only, no CLI, no rerank
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.graph_client import GraphClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GRAPH = GraphClient()  # از env یا پیش‌فرض می‌خوانَد

# Load semantic->Neo4j ID mapping
def _load_mapping():
    mapping_path = Path("outputs/semantic_to_neo4j_mapping.json")
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return {}

SEMANTIC_TO_NEO4J = _load_mapping()

# Global BM25 index (lazy-loaded)
_BM25_VECTORIZER = None
_BM25_MATRIX = None
_BM25_CHUNKS = None

def _load_bm25():
    global _BM25_VECTORIZER, _BM25_MATRIX, _BM25_CHUNKS
    if _BM25_VECTORIZER is not None:
        return
    
    chunks = load_jsonl(Path("outputs/who_chunks_semantic.jsonl"))
    texts = [c['text'] for c in chunks]
    
    _BM25_VECTORIZER = TfidfVectorizer(
        max_features=2000,
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2)
    )
    _BM25_MATRIX = _BM25_VECTORIZER.fit_transform(texts)
    _BM25_CHUNKS = chunks

# ---------- IO ----------
def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f]

def encode_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(embs, list):
        embs = np.array(embs)
    return embs.astype("float32")

# ---------- Base FAISS retrieval ----------
def retrieve(index_path: Path, chunks_path: Path, meta_path: Path, model_name: str, query: str, top_k: int) -> List[Dict]:
    index = faiss.read_index(str(index_path))
    chunks = load_jsonl(chunks_path)
    meta = load_jsonl(meta_path)

    model = SentenceTransformer(model_name, device="cpu")
    q = encode_texts([query], model)  # normalized
    D, I = index.search(q, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    out: List[Dict[str, Any]] = []
    for rank, (i, s) in enumerate(zip(idxs, scores), 1):
        out.append({
            "rank": rank,
            "score": float(s),
            "row_id": int(i),
            "id": meta[i].get("id"),
            "method": meta[i].get("method"),
            "n_tokens": meta[i].get("n_tokens"),
            "preview": chunks[i]["text"].replace("\n", " ")[:500]
        })
    return out

# ---------- BM25 retrieval ----------
def retrieve_bm25(query: str, top_k: int) -> List[Dict]:
    """BM25-like keyword search using TF-IDF"""
    _load_bm25()
    
    query_vec = _BM25_VECTORIZER.transform([query])
    scores = (_BM25_MATRIX * query_vec.T).toarray().flatten()
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    out: List[Dict[str, Any]] = []
    for rank, i in enumerate(top_indices, 1):
        if i >= len(_BM25_CHUNKS):
            continue
        chunk = _BM25_CHUNKS[i]
        out.append({
            "rank": rank,
            "score": float(scores[i]),
            "row_id": int(i),
            "id": chunk.get("id"),
            "n_tokens": chunk.get("n_tokens"),
            "preview": chunk["text"].replace("\n", " ")[:500]
        })
    return out

# ---------- Hybrid Semantic + BM25 ----------
def hybrid_search_semantic_bm25(index_path: Path,
                                chunks_path: Path,
                                meta_path: Path,
                                model_name: str,
                                query: str,
                                top_k: int = 10,
                                semantic_weight: float = 0.6,
                                bm25_weight: float = 0.4) -> List[Dict]:
    """Mix semantic and BM25 results"""
    
    # 1) Get both results
    semantic_results = retrieve(index_path, chunks_path, meta_path, model_name, query, top_k)
    bm25_results = retrieve_bm25(query, top_k)
    
    # 2) Normalize scores by method
    def normalize_scores(results: List[Dict]) -> Dict[str, float]:
        if not results:
            return {}
        scores = {r["id"]: float(r["score"]) for r in results}
        vals = list(scores.values())
        if not vals or max(vals) == min(vals):
            return {k: 1.0 for k in scores}
        min_v, max_v = min(vals), max(vals)
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
    
    semantic_scores = normalize_scores(semantic_results)
    bm25_scores = normalize_scores(bm25_results)
    
    # 3) Combine all unique IDs
    all_ids = set(list(semantic_scores.keys()) + list(bm25_scores.keys()))
    
    # 4) Calculate final scores
    final_scores: Dict[str, Dict] = {}
    for cid in all_ids:
        sem_score = semantic_scores.get(cid, 0.0)
        bm25_score = bm25_scores.get(cid, 0.0)
        final = semantic_weight * sem_score + bm25_weight * bm25_score
        
        # Find original data
        result_from_semantic = next((r for r in semantic_results if r["id"] == cid), None)
        result_from_bm25 = next((r for r in bm25_results if r["id"] == cid), None)
        source_result = result_from_semantic or result_from_bm25
        
        final_scores[cid] = {
            "final_score": final,
            "semantic_score": sem_score,
            "bm25_score": bm25_score,
            "row_id": source_result["row_id"] if source_result else None,
            "n_tokens": source_result.get("n_tokens") if source_result else None,
            "preview": source_result.get("preview") if source_result else ""
        }
    
    # 5) Sort and return
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
    
    out = []
    for rank, (cid, data) in enumerate(sorted_results, 1):
        out.append({
            "rank": rank,
            "id": cid,
            "score": data["final_score"],
            "semantic_score": data["semantic_score"],
            "bm25_score": data["bm25_score"],
            "row_id": data["row_id"],
            "n_tokens": data["n_tokens"],
            "preview": data["preview"]
        })
    
    return out

# ---------- Hybrid (FAISS+BM25 + Neo4j) ----------
def hybrid_search_query(index_path: Path,
                        chunks_path: Path,
                        meta_path: Path,
                        model_name: str,
                        query: str,
                        faiss_topk: int = 12,
                        graph_neigh_limit: int = 15,
                        alpha: float = 0.65,
                        beta: float = 0.35,
                        min_unique_texts: int = 8,
                        use_bm25: bool = True,
                        semantic_weight: float = 0.65,
                        bm25_weight: float = 0.35) -> Dict[str, Any]:
    # 1) Get hybrid semantic+BM25 results (or just semantic)
    if use_bm25:
        hybrid_rows = hybrid_search_semantic_bm25(
            index_path, chunks_path, meta_path, model_name, query,
            top_k=faiss_topk,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight
        )
    else:
        hybrid_rows = retrieve(index_path, chunks_path, meta_path, model_name, query, faiss_topk)
    
    hybrid_hits: List[Tuple[str, float]] = [(r["id"], float(r["score"])) for r in hybrid_rows if r.get("id")]
    
    # Map semantic IDs to Neo4j IDs
    seed_ids = [SEMANTIC_TO_NEO4J.get(cid, cid) for cid, _ in hybrid_hits]
    seed_ids = [sid for sid in seed_ids if sid]  # filter empty

    # 2) Graph neighbors
    graph_scores: Dict[str, float] = {}
    for sid in seed_ids:
        for n in GRAPH.neighbors(sid, limit=graph_neigh_limit):
            nid = n["id"]; gscore = float(n.get("score", 0.0))
            if nid not in graph_scores or graph_scores[nid] < gscore:
                graph_scores[nid] = gscore

    # 3) normalize
    def _nz(d: Dict[str, float]) -> Dict[str, float]:
        if not d: return {}
        vals = list(d.values()); lo, hi = min(vals), max(vals)
        if hi == lo: return {k: 1.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    # FAISS scores use semantic IDs, but we need Neo4j IDs for retrieval
    hybrid_scores_semantic = _nz({cid: sc for cid, sc in hybrid_hits})
    hybrid_scores = {SEMANTIC_TO_NEO4J.get(cid, cid): sc for cid, sc in hybrid_scores_semantic.items()}
    graph_scores = _nz(graph_scores)

    # 4) fuse (now both use Neo4j IDs)
    final_scores: Dict[str, float] = {}
    for k in set(list(hybrid_scores.keys()) + list(graph_scores.keys())):
        f = hybrid_scores.get(k, 0.0); g = graph_scores.get(k, 0.0)
        final_scores[k] = alpha * f + beta * g

    # 5) pick text/image ids
    text_ids = [k for k in final_scores if k and k.startswith("T_")]
    image_ids = [k for k in final_scores if k and k.startswith("I_")]
    text_ids.sort(key=lambda x: final_scores[x], reverse=True)
    image_ids.sort(key=lambda x: final_scores[x], reverse=True)

    chosen_texts = text_ids[:max(min_unique_texts, len(seed_ids))]
    chosen_images = image_ids[:6]

    text_docs = GRAPH.fetch_text_chunks(chosen_texts) if chosen_texts else []
    image_docs = GRAPH.fetch_images(chosen_images) if chosen_images else []

    return {
        "seed_hybrid": hybrid_hits,
        "graph_neighbors_count": len(graph_scores),
        "final_scores": final_scores,
        "texts": text_docs,
        "images": image_docs
    }

def build_llm_context(hybrid_result: Dict[str, Any], max_chars: int = 3500) -> str:
    scores = hybrid_result["final_scores"]
    texts = list(hybrid_result["texts"])
    texts.sort(key=lambda r: scores.get(r["id"], 0.0), reverse=True)

    ctx_parts: List[str] = []; used = 0
    for r in texts:
        piece = f"[page {r['page']}] {r['content']}\n"
        if used + len(piece) > max_chars: break
        ctx_parts.append(piece); used += len(piece)

    images = hybrid_result["images"]
    if images:
        refs = " | ".join([f"{x['id']}@p{x['page']}" for x in images])
        ctx_parts.append(f"\n[images] {refs}\n")

    return "".join(ctx_parts)

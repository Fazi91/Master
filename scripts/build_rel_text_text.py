import csv, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(r"C:\Users\farza\llm-hallu-pipeline")
GRAPH = ROOT / "data" / "graph"
NODES = GRAPH / "nodes_text.csv"
OUT = GRAPH / "rel_text_text.csv"

K = 5          # همسایه برای هر چانک
MIN_COS = 0.25 # آستانه شباهت

def norm(s): 
    s = (s or "").replace("\u200c"," ")
    return re.sub(r"\s+"," ", s).strip()

# 1) خواندن چانک‌ها
ids, texts = [], []
with open(NODES, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        ids.append(r["chunk_id"])
        texts.append(norm(r["text"]))
print(f"[read] chunks: {len(ids)}")

# 2) TF-IDF
vec = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
X = vec.fit_transform(texts)
print(f"[tfidf] shape: {X.shape}")

# 3) kNN در فضای کسینوسی
nn = NearestNeighbors(n_neighbors=min(K+1, len(ids)), metric="cosine", algorithm="brute")
nn.fit(X)
dist, idx = nn.kneighbors(X)

# 4) نوشتن یال‌ها
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["src_chunk_id","dst_chunk_id","cosine"])
    edges = 0
    for i, (drow, irow) in enumerate(zip(dist, idx)):
        src = ids[i]
        for d, j in zip(drow[1:], irow[1:]):     # اولی خودش است
            cos = 1.0 - float(d)
            if cos >= MIN_COS:
                w.writerow([src, ids[int(j)], f"{cos:.4f}"]); edges += 1
print(f"[write] edges: {edges} -> {OUT}")

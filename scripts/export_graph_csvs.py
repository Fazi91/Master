import json, csv, os, re
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Users\farza\llm-hallu-pipeline")
OUT = ROOT / "outputs"
GRAPH_DIR = ROOT / "data" / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = OUT / "who_chunks_semantic.jsonl"
PAGE_IMG_PATH = OUT / "images" / "page_to_images.json"

def norm_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u200c", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ---------- 1) خواندن چانک‌ها و شمردن PAGE_BREAK ----------
print("[1/4] Reading chunks and assigning pages...")
chunks = []
page = 1
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        text = obj.get("text", "")
        count = text.count("<<PAGE_BREAK>>")
        chunks.append({
            "chunk_id": f"T_{idx:05d}",
            "page": page,
            "text": norm_text(text.replace("<<PAGE_BREAK>>", "")),
            "tokens": len(text.split())
        })
        page += count

print(f"  chunks: {len(chunks)} | last page: {page}")

# ---------- 2) خواندن مپ تصاویر ----------
print("[2/4] Reading image mappings...")
nodes_image = []
pages_image = set()
if PAGE_IMG_PATH.exists():
    with open(PAGE_IMG_PATH, "r", encoding="utf-8") as f:
        p2i = json.load(f)
    img_counter = 1
    for k, items in p2i.items():
        try:
            pg = int(re.findall(r"\d+", str(k))[0])
        except:
            pg = -1
        for rel_name in items:
            nodes_image.append({
                "img_id": f"I_{img_counter:05d}",
                "page": pg,
                "path": f"outputs/images/{rel_name}"
            })
            pages_image.add(pg)
            img_counter += 1
else:
    print("!! page_to_images.json not found!")

print(f"  images: {len(nodes_image)} | distinct pages: {len(pages_image)}")

# ---------- 3) ساخت رابطه متن-تصویر ----------
print("[3/4] Linking chunks to images...")
page_to_imgids = defaultdict(list)
for img in nodes_image:
    page_to_imgids[img["page"]].append(img["img_id"])

rels_ti = []
for ch in chunks:
    pg = ch["page"]
    if pg in page_to_imgids:
        for img_id in page_to_imgids[pg]:
            rels_ti.append({
                "chunk_id": ch["chunk_id"],
                "img_id": img_id,
                "score": 0.61,
                "method": "page_match"
            })

print(f"  links created: {len(rels_ti)}")

# ---------- 4) نوشتن فایل‌های CSV ----------
print("[4/4] Writing CSVs...")
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):>5} rows -> {path.name}")

write_csv(GRAPH_DIR / "nodes_text.csv",
           ["chunk_id", "page", "tokens", "text"],
           chunks)

write_csv(GRAPH_DIR / "nodes_image.csv",
           ["img_id", "page", "path"],
           nodes_image)

write_csv(GRAPH_DIR / "rel_text_image.csv",
           ["chunk_id", "img_id", "score", "method"],
           rels_ti)

write_csv(GRAPH_DIR / "rel_text_text.csv",
           ["src_chunk_id", "dst_chunk_id", "cosine"],
           [])

print(f"\n[Done] CSVs are in: {GRAPH_DIR}")

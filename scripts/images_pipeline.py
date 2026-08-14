import os, json, argparse
from pathlib import Path
from io import BytesIO
from typing import List

import pymupdf
from PIL import Image

def _lazy_import_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer

def _lazy_import_faiss():
    import faiss
    return faiss

# ---------- Utils ----------
def _valid_png(path: Path, min_bytes: int = 256) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.convert("RGB")
        return path.stat().st_size > min_bytes
    except Exception:
        return False

def _save_via_bytes(data: bytes, out_path: Path) -> bool:
    try:
        im = Image.open(BytesIO(data)).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, format="PNG")
        return _valid_png(out_path)
    except Exception:
        return False

def _save_via_pixmap(doc: fitz.Document, xref: int, out_path: Path) -> bool:
    try:
        pix = fitz.Pixmap(doc, xref)
        need_convert = True
        try:
            if pix.colorspace is not None and pix.colorspace == fitz.csRGB and not pix.alpha and pix.n == 3:
                need_convert = False
        except Exception:
            need_convert = True
        if need_convert:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        return _valid_png(out_path)
    except Exception:
        return False

def _save_via_clip_render(page: fitz.Page, rect: fitz.Rect, dpi: int, out_path: Path) -> bool:
    try:
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        return _valid_png(out_path)
    except Exception:
        return False

def _save_full_page(page: fitz.Page, dpi: int, out_path: Path) -> bool:
    try:
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_path))
        return _valid_png(out_path)
    except Exception:
        return False

# ---------- Commands ----------
def cmd_extract(pdf: str, outdir: str, min_wh: int, max_per_page: int, dpi: int, allow_full_page: bool, manifest_path: str):
    pdf_path = Path(pdf)
    out_dir = Path(outdir)
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    saved = skipped = failed = 0

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for pno, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            count_this_page = 0
            for info in images:
                if count_this_page >= max_per_page:
                    break
                xref = info[0]
                width = info[2] if len(info) > 2 else 0
                height = info[3] if len(info) > 3 else 0

                if min(width, height) and min(width, height) < min_wh:
                    skipped += 1
                    continue

                out_name = f"p{pno:03d}_xref{xref}.png"
                out_path = files_dir / out_name

                ok = False
                # 1) raw bytes
                try:
                    img_dict = doc.extract_image(xref)
                    data = img_dict.get("image", b"")
                    if data:
                        ok = _save_via_bytes(data, out_path)
                except Exception:
                    ok = False
                # 2) pixmap
                if not ok:
                    ok = _save_via_pixmap(doc, xref, out_path)
                # 3) bbox render
                if not ok:
                    try:
                        rect = page.get_image_bbox(xref)
                        ok = _save_via_clip_render(page, rect, dpi, out_path)
                    except Exception:
                        ok = False
                # 4) full page (optional)
                if not ok and allow_full_page:
                    ok = _save_full_page(page, dpi, out_path)

                if ok:
                    rec = {"id": f"p{pno:03d}_xref{xref}", "page": pno, "xref": xref, "file_path": f"files/{out_name}"}
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved += 1
                    count_this_page += 1
                else:
                    failed += 1

    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Images folder: {files_dir}")
    print(f"[STATS] saved={saved} skipped_small={skipped} failed={failed}")
    if failed and not allow_full_page:
        print("[HINT] Re-run with --allow_full_page to eliminate residual fails.")

def cmd_verify(root: str, pattern: str, min_bytes: int, report: str, delete_bad: bool):
    rootp = Path(root)
    files = sorted(rootp.rglob(pattern))
    bad = []
    for p in files:
        if not _valid_png(p, min_bytes=min_bytes):
            bad.append(p)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        for p in bad:
            f.write(str(p) + "\n")
    print(f"total_files: {len(files)}")
    print(f"bad_count: {len(bad)}")
    print(f"report: {report}")
    if delete_bad and bad:
        for p in bad:
            try: p.unlink(missing_ok=True)
            except: pass
        print(f"deleted_bad_files: {len(bad)}")

def cmd_map(manifest: str, out_json: str, out_md: str | None):
    from collections import defaultdict
    page2imgs = defaultdict(list)
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            p = rec.get("page"); fp = rec.get("file_path")
            if p is None or not fp: continue
            page2imgs[int(p)].append(fp)
    out = {int(k): v for k, v in page2imgs.items()}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] page→images:", out_json)
    if out_md:
        lines = []
        for p in sorted(out.keys()):
            lines.append(f"## Page {p}")
            lines += [f"- {fp}" for fp in out[p]]
        Path(out_md).write_text("\n".join(lines), encoding="utf-8")
        print("[OK] md:", out_md)

def _embed_images(model, records: List[dict], base_dir: Path, batch_size: int = 32):
    embs = []
    paths = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        imgs = []
        local_paths = []
        for rec in batch:
            p = (base_dir / rec["file_path"]).resolve()
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            imgs.append(im); local_paths.append(str(p))
        if not imgs: continue
        arr = model.encode(imgs, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        embs.append(arr); paths += local_paths
    import numpy as np
    if not embs:
        return np.zeros((0, 512), dtype="float32"), []
    return np.vstack(embs).astype("float32"), paths

def cmd_index(manifest: str, index_path: str, meta_path: str, basedir: str, model_name: str, batch_size: int):
    records = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[INFO] records: {len(records)}")

    SentenceTransformer = _lazy_import_sentence_transformers()
    faiss = _lazy_import_faiss()

    model = SentenceTransformer(model_name)
    img_embs, used_paths = _embed_images(model, records, Path(basedir), batch_size=batch_size)
    print(f"[INFO] embedded images: {img_embs.shape}")

    dim = img_embs.shape[1] if img_embs.size else 512
    index = faiss.IndexFlatIP(dim)
    if img_embs.size:
        index.add(img_embs)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print("[OK] index:", index_path)

    with open(meta_path, "w", encoding="utf-8") as mf:
        for p in used_paths:
            mf.write(json.dumps({"file_path": p}) + "\n")
    print("[OK] meta:", meta_path)

def cmd_search(index_path: str, meta_path: str, query: str, top_k: int, model_name: str):
    SentenceTransformer = _lazy_import_sentence_transformers()
    faiss = _lazy_import_faiss()

    index = faiss.read_index(index_path)
    paths = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            paths.append(json.loads(line)["file_path"])

    model = SentenceTransformer(model_name)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q, top_k)
    print("[RESULTS]")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
        if 0 <= idx < len(paths):
            print(f"{rank}. score={score:.4f} | {paths[idx]}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(prog="images_pipeline", description="Unified image pipeline for PDF-based RAG.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ext = sub.add_parser("extract", help="Extract images and write manifest.jsonl")
    ap_ext.add_argument("--pdf", required=True)
    ap_ext.add_argument("--outdir", required=True)
    ap_ext.add_argument("--min_wh", type=int, default=32)
    ap_ext.add_argument("--max_per_page", type=int, default=100)
    ap_ext.add_argument("--dpi", type=int, default=300)
    ap_ext.add_argument("--allow_full_page", action="store_true")
    ap_ext.add_argument("--manifest", default=None)

    ap_v = sub.add_parser("verify", help="Verify PNGs under a folder")
    ap_v.add_argument("--root", required=True)
    ap_v.add_argument("--pattern", default="*.png")
    ap_v.add_argument("--min_bytes", type=int, default=256)
    ap_v.add_argument("--report", default="outputs/images/verify_report.txt")
    ap_v.add_argument("--delete_bad", action="store_true")

    ap_map = sub.add_parser("map", help="Build page->images mapping from manifest.jsonl")
    ap_map.add_argument("--manifest", required=True)
    ap_map.add_argument("--out_json", required=True)
    ap_map.add_argument("--out_md", default=None)

    ap_idx = sub.add_parser("index", help="Build CLIP index (faiss) + meta")
    ap_idx.add_argument("--manifest", required=True)
    ap_idx.add_argument("--index", required=True)
    ap_idx.add_argument("--meta", required=True)
    ap_idx.add_argument("--basedir", required=True)
    ap_idx.add_argument("--model", default="sentence-transformers/clip-ViT-B-32")
    ap_idx.add_argument("--batch_size", type=int, default=32)

    ap_s = sub.add_parser("search", help="Search images by text")
    ap_s.add_argument("--index", required=True)
    ap_s.add_argument("--meta", required=True)
    ap_s.add_argument("--query", required=True)
    ap_s.add_argument("--top_k", type=int, default=5)
    ap_s.add_argument("--model", default="sentence-transformers/clip-ViT-B-32")

    args = ap.parse_args()

    if args.cmd == "extract":
        manifest = args.manifest or str(Path(args.outdir) / "manifest.jsonl")
        cmd_extract(args.pdf, args.outdir, args.min_wh, args.max_per_page, args.dpi, args.allow_full_page, manifest)
    elif args.cmd == "verify":
        cmd_verify(args.root, args.pattern, args.min_bytes, args.report, args.delete_bad)
    elif args.cmd == "map":
        cmd_map(args.manifest, args.out_json, args.out_md)
    elif args.cmd == "index":
        cmd_index(args.manifest, args.index, args.meta, args.basedir, args.model, args.batch_size)
    elif args.cmd == "search":
        cmd_search(args.index, args.meta, args.query, args.top_k, args.model)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()

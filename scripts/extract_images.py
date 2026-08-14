import os, json, hashlib, argparse
from pathlib import Path
import pymupdf

def sha1_bytes(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_wh", type=int, default=32)
    ap.add_argument("--max_per_page", type=int, default=100)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.outdir)
    img_dir = out_dir / "files"
    ensure_dir(img_dir)

    doc = pymupdf.open(str(pdf_path))
    manifest_path = out_dir / "manifest.jsonl"

    total_saved = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for pno, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            count_this_page = 0
            for info in images:
                if count_this_page >= args.max_per_page:
                    break
                xref = info[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                except Exception:
                    continue

                # کوچک‌ها را رد کن
                if min(pix.width, pix.height) < args.min_wh:
                    continue

                # اگر آلفا یا CMYK داشت، به RGB تبدیل کن
                try:
                    if pix.alpha or pix.n > 4:
                        pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                        pix = pix_converted
                except Exception:
                    pass

                # فایل خروجی
                file_name = f"p{pno:03d}_x{xref}_sha{sha1_bytes(pix.samples)[:12]}.png"
                rel = f"files/{file_name}"
                abs_path = img_dir / file_name

                try:
                    pix.save(str(abs_path))
                except Exception:
                    continue
                finally:
                    pix = None  # آزادسازی حافظه

                if not abs_path.exists() or abs_path.stat().st_size == 0:
                    continue

                rec = {
                    "id": f"p{pno:03d}_xref{xref}",
                    "page": pno,
                    "xref": xref,
                    "width": abs_path.stat().st_size,  # فقط برای دیباگ، اگر خواستی width/height را هم بنویس
                    "file_path": rel
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_saved += 1
                count_this_page += 1

    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Images folder: {img_dir}")
    print(f"[STATS] total_records={total_saved}")

if __name__ == "__main__":
    main()

import os, re, json, unicodedata, argparse, traceback
from collections import Counter

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.replace("\x00"," "))
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s

def try_extract_with_pypdf(pdf_path, debug=False):
    try:
        if debug: print("[DBG] importing pypdf...")
        from pypdf import PdfReader
        r = PdfReader(pdf_path)
        if debug: print(f"[DBG] pypdf: pages={len(r.pages)}")
        texts = []
        for i, p in enumerate(r.pages, 1):
            try:
                t = p.extract_text() or ""
            except Exception as e:
                if debug: print(f"[WARN] pypdf extract page {i} -> {e}")
                t = ""
            texts.append(t)
        return "pypdf", texts
    except Exception as e:
        if debug:
            print("[ERR] pypdf failed:")
            traceback.print_exc()
        return None, None

def try_extract_with_pypdf2(pdf_path, debug=False):
    try:
        if debug: print("[DBG] importing PyPDF2...")
        import PyPDF2
        r = PyPDF2.PdfReader(pdf_path)
        if debug: print(f"[DBG] PyPDF2: pages={len(r.pages)}")
        texts = []
        for i, p in enumerate(r.pages, 1):
            try:
                t = p.extract_text() or ""
            except Exception as e:
                if debug: print(f"[WARN] PyPDF2 extract page {i} -> {e}")
                t = ""
            texts.append(t)
        return "PyPDF2", texts
    except Exception as e:
        if debug:
            print("[ERR] PyPDF2 failed:")
            traceback.print_exc()
        return None, None

def try_extract_with_pdfminer(pdf_path, debug=False):
    try:
        if debug: print("[DBG] importing pdfminer.six...")
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        texts = []
        count = 0
        for layout in extract_pages(pdf_path):
            page_chunks = []
            for el in layout:
                if isinstance(el, LTTextContainer):
                    page_chunks.append(el.get_text())
            texts.append("".join(page_chunks))
            count += 1
        if debug: print(f"[DBG] pdfminer: pages={count}")
        return "pdfminer", texts
    except Exception as e:
        if debug:
            print("[ERR] pdfminer failed:")
            traceback.print_exc()
        return None, None

def looks_like_page_number(s: str) -> bool:
    s2 = s.strip()
    if re.fullmatch(r"\d{1,4}", s2): return True
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,6}", s2): return True
    if re.fullmatch(r"(PART|Part)\s+[IVXLCDM]+\s+\d{1,4}", s2): return True
    return False

def fix_hyphenation_safe(text: str) -> str:
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith("-") and i+1 < len(lines):
            nxt = lines[i+1]
            if nxt and re.match(r"^[a-zäöüßà-úışğœµα-ωа-яё۰-۹]", nxt, flags=re.IGNORECASE):
                out.append(line[:-1] + nxt.lstrip()); i += 2; continue
        out.append(line); i += 1
    return "\n".join(out)

def smart_unwrap(text: str) -> str:
    lines = text.split("\n")
    out = []
    buf = ""
    def end_of_sentence(s): 
        return re.search(r"[\.!?:;\)\]\}»]$", s or "") is not None
    for ln in lines:
        st = ln.strip()
        if st == "<<<PAGE_BREAK>>>":
            if buf:
                out.append(buf.strip()); buf = ""
            out.append("<<<PAGE_BREAK>>>")
            continue
        if st == "":
            if buf:
                out.append(buf.strip()); buf = ""
            out.append("")
            continue
        if buf and not end_of_sentence(buf):
            buf = buf + " " + st
        else:
            if buf:
                out.append(buf.strip())
            buf = st
    if buf: out.append(buf.strip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to input PDF")
    ap.add_argument("--outdir", default="outputs", help="Where to write outputs")
    ap.add_argument("--header_footer_threshold_ratio", type=float, default=0.3)
    ap.add_argument("--edge_window", type=int, default=8)
    ap.add_argument("--force_backend", choices=["pypdf","pypdf2","pdfminer"], default=None)
    ap.add_argument("--debug", action="store_true", help="Verbose debug logs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    backend = None
    pages = None

    order = [args.force_backend] if args.force_backend else ["pypdf","pypdf2","pdfminer"]
    if args.debug: print("[DBG] Try order:", order)

    for be in order:
        if be == "pypdf":
            backend, pages = try_extract_with_pypdf(args.pdf, debug=args.debug)
        elif be == "pypdf2":
            backend, pages = try_extract_with_pypdf2(args.pdf, debug=args.debug)
        elif be == "pdfminer":
            backend, pages = try_extract_with_pdfminer(args.pdf, debug=args.debug)
        if pages is not None:
            break

    if pages is None:
        raise RuntimeError("Failed to extract text with any backend (pypdf / PyPDF2 / pdfminer). Check --debug output.")

    pages = [normalize_text(t or "") for t in pages]
    num_pages = len(pages)
    if args.debug: print(f"[DBG] Using backend={backend} | pages={num_pages}")

    # detect repeated header/footer
    from collections import Counter
    top_counter, bottom_counter = Counter(), Counter()
    page_lines = []
    for t in pages:
        lines = [ln for ln in t.split("\n")]
        non_empty = [ln.strip() for ln in lines if ln.strip() != ""]
        page_lines.append(lines)
        top_candidates = non_empty[:5]
        bottom_candidates = non_empty[-5:]
        for ln in top_candidates[:3]:
            if len(ln.strip()) >= 3: top_counter[ln.strip()] += 1
        for ln in bottom_candidates[-3:]:
            if len(ln.strip()) >= 3: bottom_counter[ln.strip()] += 1

    threshold = max(3, int(args.header_footer_threshold_ratio * max(1, num_pages)))
    repeated = set([l for l,c in top_counter.items() if c >= threshold] + 
                   [l for l,c in bottom_counter.items() if c >= threshold])

    cleaned_pages = []
    for lines in page_lines:
        new_lines = []
        n = len(lines)
        for idx, ln in enumerate(lines):
            st = (ln or "").strip()
            is_edge = (idx < args.edge_window) or (idx > n - args.edge_window - 1)
            if is_edge and (st in repeated or looks_like_page_number(st)):
                continue
            new_lines.append(ln)
        cleaned_pages.append("\n".join(new_lines))

    text = "\n\n<<<PAGE_BREAK>>>\n\n".join(cleaned_pages)
    text = normalize_text(text)
    text = fix_hyphenation_safe(text)
    text = smart_unwrap(text)

    base = os.path.splitext(os.path.basename(args.pdf))[0]
    out_txt = os.path.join(args.outdir, f"{base}_clean.txt")
    out_json = os.path.join(args.outdir, f"{base}_preview.json")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    meta = {
        "pdf": args.pdf,
        "extraction_backend": backend,
        "num_pages": num_pages,
        "chars": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
        "preview_head": "\n".join(text.split("\n")[:120])
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Backend: {backend} | pages: {num_pages}")
    print(f"[OK] Clean text:", out_txt)
    print(f"[OK] Meta preview:", out_json)

if __name__ == "__main__":
    main()

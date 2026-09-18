import os, re, sys, json
from pypdf import PdfReader

REFDIR = r"E:\paper\refs"
out = {}
for fn in sorted(os.listdir(REFDIR)):
    if not fn.lower().endswith(".pdf"):
        continue
    path = os.path.join(REFDIR, fn)
    rec = {"file": fn, "bytes": os.path.getsize(path)}
    try:
        r = PdfReader(path)
        rec["npages"] = len(r.pages)
        meta = r.metadata or {}
        rec["meta"] = {k: str(v) for k, v in meta.items()}
        txt = []
        for p in r.pages[:2]:
            try:
                txt.append(p.extract_text() or "")
            except Exception as e:
                txt.append("<<extract error: %s>>" % e)
        t = "\n".join(txt)
        t = re.sub(r"[ \t]+", " ", t)
        rec["head"] = t[:2600]
        # also capture arXiv stamp line anywhere in first 2 pages
        m = re.findall(r"arXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\s*\[[^\]]*\]\s*[^\n]*", t)
        rec["arxiv_stamps"] = m[:4]
    except Exception as e:
        rec["error"] = repr(e)
    out[fn] = rec

with open(r"E:\paper\_audit\refs_meta.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

for fn, rec in out.items():
    print("=" * 100)
    print("FILE:", fn, "| pages:", rec.get("npages"), "| bytes:", rec["bytes"])
    if "error" in rec:
        print("  ERROR:", rec["error"])
        continue
    m = rec["meta"]
    print("  META:", {k: v for k, v in m.items() if k not in ("/Producer",)})
    print("  STAMPS:", rec["arxiv_stamps"])
    print("  --- head ---")
    print(rec["head"][:1400])

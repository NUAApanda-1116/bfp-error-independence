"""Regenerate SHA256SUMS.txt over the distribution files (excludes working dirs)."""
import os, hashlib

ROOT = r"E:\paper"
EXCLUDE_DIRS = {"_audit", "_build", "refs"}   # working/verification dirs + third-party PDFs

entries = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = sorted(d for d in dn if d not in EXCLUDE_DIRS)
    for f in sorted(fn):
        if f == "SHA256SUMS.txt":
            continue
        p = os.path.join(dp, f)
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest().upper()
        entries.append((rel, h))

entries.sort(key=lambda t: t[0])
out = os.path.join(ROOT, "SHA256SUMS.txt")
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    for rel, h in entries:
        fh.write(f"{h}  {rel}\n")

print(f"wrote {out}: {len(entries)} entries")
for rel, h in entries:
    print(f"  {h[:12]}…  {rel}")

# sanity: every listed file must exist and match
bad = 0
for rel, h in entries:
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest().upper() != h:
        print("  !! MISMATCH", rel); bad += 1
print("self-check mismatches:", bad)

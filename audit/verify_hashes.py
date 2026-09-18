import hashlib, os, re

root = r"E:\paper"
man = os.path.join(root, "SHA256SUMS.txt")
lines = [l.rstrip("\n") for l in open(man, encoding="utf-8") if l.strip()]
print("manifest entries:", len(lines))
ok = bad = missing = 0
for l in lines:
    m = re.match(r"^([0-9A-Fa-f]{64})\s+(.*)$", l)
    if not m:
        print("  MALFORMED:", l); continue
    want, rel = m.group(1).upper(), m.group(2).strip()
    p = os.path.join(root, rel.replace("/", os.sep))
    if not os.path.exists(p):
        print(f"  MISSING FILE : {rel}")
        missing += 1
        continue
    h = hashlib.sha256(open(p, "rb").read()).hexdigest().upper()
    if h == want:
        ok += 1
    else:
        print(f"  HASH MISMATCH: {rel}\n      manifest {want}\n      actual   {h}")
        bad += 1
print(f"\nOK={ok}  MISMATCH={bad}  MISSING={missing}")

# duplicate hashes in the manifest
seen = {}
for l in lines:
    m = re.match(r"^([0-9A-Fa-f]{64})\s+(.*)$", l)
    if m:
        seen.setdefault(m.group(1).upper(), []).append(m.group(2).strip())
dups = {k: v for k, v in seen.items() if len(v) > 1}
print("\nduplicate hashes across different files in the manifest:", dups)

# files in the tree not covered by the manifest
covered = {m.group(2).strip().replace("/", os.sep) for m in
           (re.match(r"^([0-9A-Fa-f]{64})\s+(.*)$", l) for l in lines) if m}
extra = []
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ("_audit",)]
    for f in fn:
        rel = os.path.relpath(os.path.join(dp, f), root)
        if rel not in covered and not rel.startswith("refs" + os.sep) and rel != "SHA256SUMS.txt":
            extra.append(rel)
print("\nfiles present but NOT in manifest (excluding refs/):")
for e in sorted(extra):
    print("   ", e)

import os, hashlib, sys

REMOTE = r"E:\paper\_audit\repo\bfp-error-independence-main"
LOCAL = r"E:\paper\scripts"

def walk(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out

r = walk(REMOTE)
l = walk(LOCAL)
print("remote files:", len(r), "| local files:", len(l))
onlyR = sorted(set(r) - set(l))
onlyL = sorted(set(l) - set(r))
diff = sorted(k for k in set(r) & set(l) if r[k] != l[k])
print("\nONLY ON GITHUB (%d):" % len(onlyR)); [print("   +", k) for k in onlyR]
print("\nONLY LOCALLY (%d):" % len(onlyL)); [print("   -", k) for k in onlyL]
print("\nCONTENT DIFFERS (%d):" % len(diff)); [print("   ~", k) for k in diff]

# compare the two README files that share a checksum in SHA256SUMS
for name in ["README.md", "GITHUB_README.md"]:
    for root, tag in ((REMOTE, "github"), (LOCAL, "local")):
        p = os.path.join(root, name)
        if os.path.exists(p):
            t = open(p, encoding="utf-8").read()
            print(f"\n[{tag}] {name}: {len(t)} chars, first line: {t.splitlines()[0][:80]}")

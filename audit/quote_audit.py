import re, sys, os
from pypdf import PdfReader

OUT = open(r"E:\paper\_audit\quote_audit.txt", "w", encoding="utf-8")

def dump(fname, patterns, maxpages=None, ctx=260, label=""):
    path = os.path.join(r"E:\paper\refs", fname)
    OUT.write("\n" + "#"*100 + "\n# %s  (%s)\n" % (label or fname, fname) + "#"*100 + "\n")
    try:
        r = PdfReader(path)
    except Exception as e:
        OUT.write("OPEN FAIL %r\n" % e); return
    npg = len(r.pages)
    OUT.write("pages=%d\n" % npg)
    lim = npg if maxpages is None else min(npg, maxpages)
    for i in range(lim):
        try:
            t = r.pages[i].extract_text() or ""
        except Exception as e:
            continue
        flat = re.sub(r"\s+", " ", t)
        for p in patterns:
            for m in re.finditer(p, flat, re.I):
                a = max(0, m.start()-ctx); b = min(len(flat), m.end()+ctx)
                OUT.write("\n--- p%d /%s/ ---\n...%s...\n" % (i+1, p, flat[a:b]))

# 1. song2018 verbatim quote
dump("song2017_AAAI-OA.pdf", [r"zero mean", r"Kalliojarvi", r"Kallioj", r"variance"],
     label="song2018 (AAAI) - zero mean / Kalliojarvi / variance quotes")

# 2. han2025 quote + reference [31]
dump("han2025_arXiv2504.15721.pdf", [r"zero-mean", r"quantisation error", r"quantization error",
     r"Kallioj", r"\[31\]", r"variance"],
     label="han2025 (BBAL) - zero-mean quote and ref [31]")

# 3. wu2026mxfp4 title page author order
r = PdfReader(os.path.join(r"E:\paper\refs", "wu2026mxfp4_arXiv2605.20402.pdf"))
OUT.write("\n"+"#"*100+"\n# wu2026mxfp4 page 1 raw text\n"+"#"*100+"\n")
OUT.write(r.pages[0].extract_text()[:2500])

# 4. cim2026 quote check
dump("cim2026mxfp4_arXiv2605.09825.pdf", [r"structured micro", r"insufficient stochastic",
     r"stochastic rounding and", r"Hadamard"], label="cim2026mxfp4 - quote check")

# 5. zhang2026bfp - shared exponent coupling
dump("zhang2026bfp_arXiv2604.10494.pdf", [r"coupl", r"shared exponent", r"shared-exponent",
     r"fault"], label="zhang2026bfp - shared exponent coupling claim")

# 6. ang2026 formula
dump("ang2026quant_arXiv2603.19559.pdf", [r"k\(k-1\)", r"k − 1", r"companding", r"E\[D", r"designed away", r"design away"],
     label="ang2026quant - MSE formula / companding")

# 7. sao2026 conditionally unbiased + centred/non-centred
dump("sao2026trees_arXiv2607.18758.pdf", [r"conditionally unbiased", r"non-?centred", r"non-?centered",
     r"centred", r"centered"], label="sao2026trees - conditional unbiasedness / centred dichotomy")

# 8. zhao2026 - sign symmetry / amplitude space
dump("zhao2026shrinkage_arXiv2606.20381.pdf", [r"shrinkage bias", r"negative rounding error", r"sign"],
     maxpages=8, label="zhao2026shrinkage")

# 9. hallman2023 - sqrt n
dump("hallman2023_arXiv2203.15928.pdf", [r"sqrt", r"probabilistic bound", r"summation"],
     maxpages=4, label="hallman2023")

# 10. higham2019/2022 first-page publication info already captured
# 11. drineas2024, elarar2026 fine.

# 12. L2: lian2019 quote is paywalled; check har2025 no.
# Also check fan2018's citation of Kalliojarvi and of Song
dump("fan2018_Imperial-authorcopy.pdf", [r"Kallioj", r"Song", r"\[1\]", r"zero mean", r"zero-mean"],
     label="fan2018 - does it cite Kalliojarvi1996 and Song?")

OUT.close()
print("written")

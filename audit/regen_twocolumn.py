"""Regenerate paper-twocolumn.tex from paper.tex (adds the twocolumn class option)."""
import re, os

SRC = r"E:\paper\paper.tex"
DST = r"E:\paper\paper-twocolumn.tex"

HEADER = """%-----------------------------------------------------------------------
%  paper-twocolumn.tex
%  Generated from paper.tex by adding the ``twocolumn'' class option.
%  Identical text, numbers, floats and bibliography; two-column layout.
%  Regenerate after editing paper.tex:
%    python .texbuild/make_twocolumn.py
%-----------------------------------------------------------------------
"""

src = open(SRC, encoding="utf-8").read()
pat = "\\documentclass["
n = src.count(pat)
print("occurrences of %r : %d" % (pat, n))
for i, line in enumerate(src.splitlines(), 1):
    if pat in line:
        print(f"  line {i}: {line[:120]}")
assert n == 1, "documentclass header not unique"
new = HEADER + src
new = new.replace(pat, "\\documentclass[twocolumn,")
open(DST, "w", encoding="utf-8", newline="\n").write(new)
print("wrote", DST, len(new), "chars")

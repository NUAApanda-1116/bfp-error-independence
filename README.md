# The Independence Assumption for Rounding Errors in Block Floating Point

Source, bibliography and reproduction material for:

**The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation**
Yongkang Xiong —
College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics

## Layout

```
paper/          manuscript source and bibliography
  paper.tex                 one-column source (SIAM siamart251216 class)
  paper-twocolumn.tex       generated from paper.tex with the twocolumn option
  references.bib            28 entries, every one carrying a DOI or arXiv ID
  COMPILE.txt               build instructions
  siamart251216.cls         SIAM class
  siamplain.bst             SIAM bibliography style
scripts/        reproduction package (see scripts/README.md, scripts/NOTES.zh.md)
  src/                      experiment scripts v1-v9 and src/exact/ quadrature checks
  outputs/                  archived terminal output that backs every table
  tools/check_paper_numbers.py
audit/          scripts used to verify the bibliography and the numbers
```

## Build

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

The two-column variant is regenerated from the one-column source by adding
`twocolumn` to the `\documentclass` options
(`audit/regen_twocolumn.py` does this and nothing else).

## Verify the numbers in the paper

```bash
python3 scripts/tools/check_paper_numbers.py     # 11/11 groups pass
bash scripts/run_all.sh                          # full re-run into scripts/outputs/rerun/
```

## Verification of the bibliography

Every entry has been checked against the primary source (Crossref, arXiv, or the
published PDF). The checks and their output are in `audit/`:

| script | what it does |
|---|---|
| `extract_refs.py` | dumps title/author metadata and the first two pages of every reference PDF |
| `check_dois.py` | queries Crossref for every DOI in `references.bib` |
| `check_updates.py` | checks every DOI for retraction and correction links |
| `verify_hashes.py` | validates a checksum manifest against the files on disk |
| `verify_numerics.py` | independently recomputes the dead-zone response functions delta(s), gamma(s) |
| `check_peak2.py` | independently evaluates the population curve of the inflation factor |
| `quote_audit.py` | locates the passages cited in the manuscript inside the source PDFs |
| `diff_repo.py` | compares this repository against a local working copy |

## Citation

See `CITATION.cff`. Prefer citing the paper once a DOI or arXiv identifier is
available.

## License

Code: MIT (`LICENSE`). Paper text: CC BY 4.0.

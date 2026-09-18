# bfp-error-independence

Reproduction code for:

**The Independence Assumption for Rounding Errors in Block Floating Point:
Failure of Conditional Symmetry and Its Effect on Summation Error Accumulation**

Yongkang Xiong  
College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics  
nuaapanda@outlook.com

Monte Carlo experiments and high-precision quadrature checks for block floating
point (BFP) rounding-error independence.

## Script versions

| Stage | Work |
|---|---|
| v1 | Pilot study; first estimator (later rejected) |
| v2 | Bias *z*-values and classical variance ratios |
| v3 | Rounding-rule sensitivity; rejected negative inflation factors |
| v4 | Third-round grouped estimator |
| v5 | Inflation factor vs block length |
| v6 | Stochastic-rounding falsification test |
| v7 | Conditional decomposition given the shared exponent |
| v8 | Bounded vs unbounded support |
| `exact/` | High-precision quadrature and logic audit |
| v9 | Large-*n* robustness check |

Scripts are numbered v1–v9 in the order they were developed. Older versions are
kept because they record how the estimator was corrected and which hypotheses
were rejected.

Random seeds in the scripts are **fixed experiment identifiers** that produced
the archived numbers. They are not calendar dates of each revision and they do
not date the scripts. Keep them unchanged if you want to compare a re-run with
the paper tables.

The git history does not reflect the development process: the files were
committed together after the reproduction package was assembled.

## Environment

```bash
pip install -r requirements.txt
```

- Python >= 3.9
- numpy (all experiment scripts)
- matplotlib (only `src/make_figures.py`)

## Verify paper numbers

```bash
python3 tools/check_paper_numbers.py
```

This checks that archived terminal outputs under `outputs/` contain the key
figures reported in the paper tables.

## Full re-run

```bash
bash run_all.sh
```

New results are written to `outputs/rerun/` and do **not** overwrite the
archived outputs that back the paper tables.

## Layout

```
CITATION.cff
LICENSE
README.md
requirements.txt
run_all.sh
outputs/                 archived terminal outputs
  README.txt
  out_v2.txt … out_v9.txt
  rerun/out_v3.txt
src/
  bfp_error_pilot.py … bfp_error_pilot_v9.py
  make_figures.py
  exact/                 high-precision / audit scripts
tools/
  check_paper_numbers.py
```

## Script ↔ paper mapping

| Script | Paper material | Archive |
|---|---|---|
| `src/bfp_error_pilot.py` (v1) | background for estimator history | — |
| `src/bfp_error_pilot_v2.py` | bias *z*, variance ratios, Figs. 1–2 | `outputs/out_v2.txt` |
| `src/bfp_error_pilot_v3.py` | rounding modes, Fig. 5 | `outputs/rerun/out_v3.txt` |
| `src/bfp_error_pilot_v4.py` | third-round estimator | `outputs/out_v4.txt` |
| `src/bfp_error_pilot_v5.py` | inflation vs *n*, Fig. 3 | `outputs/out_v5.txt` |
| `src/bfp_error_pilot_v6.py` | stochastic rounding test, Fig. 4 | `outputs/out_v6.txt` |
| `src/bfp_error_pilot_v7.py` | conditional decomposition; reduced model | `outputs/out_v7.txt` |
| `src/bfp_error_pilot_v8.py` | bounded vs unbounded support | `outputs/out_v8.txt` |
| `src/bfp_error_pilot_v9.py` | large-*n* robustness | `outputs/out_v9.txt` |
| `src/exact/*` | population model / lattice checks | printed at runtime |

See `NOTES.zh.md` in this repository for the Chinese notes, known gaps, and
seed policy.

## Citation

See `CITATION.cff`. Prefer citing the accompanying paper once a DOI/arXiv ID
is available.

## License

Code: MIT (`LICENSE`). Paper text: CC BY 4.0.

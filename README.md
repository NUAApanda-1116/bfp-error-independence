# Reproduction package: rounding-error independence in block floating point

Scripts and archived terminal output behind every numerical result in

> **The Independence Assumption for Rounding Errors in Block Floating Point:
> Failure of Conditional Symmetry and Its Effect on Summation Error
> Accumulation**
> Yongkang Xiong — Nanjing University of Aeronautics and Astronautics

Code is MIT (`LICENSE`); the paper text is CC BY 4.0.

---

## Contents

```
src/                 experiment scripts, in development order v1 → v9
  bfp_error_pilot.py            v1
  bfp_error_pilot_v2.py ... v9.py
  make_figures.py               single-panel figures 1–5
  make_figures_compact.py       the two compound floats used in the paper
  exact/                        high-precision and independent-check scripts
outputs/             archived terminal output — the raw record behind the tables
  out_v2.txt ... out_v9.txt
  rerun/out_v3.txt
tools/
  check_paper_numbers.py        checks the archived output against the paper tables
run_all.sh           re-runs everything into outputs/rerun/
requirements.txt     numpy (matplotlib for the figures only)
CITATION.cff         how to cite
```

Comments in the scripts are in Chinese; this file and `outputs/README.txt` are in
English.

---

## Requirements and quick start

Python ≥ 3.9.

```bash
pip install -r requirements.txt

# 1. does the archived output still contain the numbers printed in the paper?
python3 tools/check_paper_numbers.py        # exit 0 = all groups pass

# 2. the high-precision checks (seconds to a minute)
bash run_all.sh fast

# 3. everything: the full Monte Carlo grid plus the figures
bash run_all.sh                             # minutes to about an hour
```

`run_all.sh` never overwrites `outputs/`: everything it produces goes to
`outputs/rerun/`. Do not edit the files in `outputs/` — they are the original
record of the numbers in the paper.

`tools/check_paper_numbers.py` uses only the standard library, so it runs even
without numpy. It is an *existence* check: it confirms that representative
numbers from each table are present in the corresponding archived output. It does
not replace re-running the experiments.

---

## Which script produces what

| Script (`src/`) | Result in the paper | Archived output | Seed |
|---|---|---|---|
| `bfp_error_pilot.py` (v1) | initial pilot; background to the estimator history in sec 4.2 | — | 20260913 |
| `bfp_error_pilot_v2.py` | `tab:var`; the \|z\| values in sec 5.1; figures 1–2 | `out_v2.txt` | 20260914 |
| `bfp_error_pilot_v3.py` | rounding-rule \|z\| values in sec 5.4 (right panel of `fig:srmodes`); the two rejected negative inflation factors in sec 4.2 | `rerun/out_v3.txt` (see the note below) | 20260915 |
| `bfp_error_pilot_v4.py` | sec 4.2, third-round estimator (grouped mean ± standard error) | `out_v4.txt` | 20260916 |
| `bfp_error_pilot_v5.py` | `tab:infl` (both mantissa widths) and the local log-slopes of sec 5.3 | `out_v5.txt` | 20260917 |
| `bfp_error_pilot_v6.py` | `tab:srfull` (round-to-nearest vs stochastic rounding) | `out_v6.txt` | 20260918 |
| `bfp_error_pilot_v7.py` | `tab:cond`, `tab:model` (measured column), sec 5.6 | `out_v7.txt` | 20260920 |
| `bfp_error_pilot_v8.py` | `tab:bounded` and the $\mathbb P(\xi=\text{mode})$ column | `out_v8.txt` | 20260921 / 20260922 |
| `bfp_error_pilot_v9.py` | `tab:largen` (two independent seeds) | `out_v9.txt` | 20260930 / 20260923 |

### High-precision and independent checks (`src/exact/`)

These print to the terminal and have no archived output.

| Script | What it computes |
|---|---|
| `model_pop.py` | the population curve of $\mathcal I_n$ for the reduced model (peak $5.9\times10^4$ near $n\approx10^5$; $3.61\times10^4$ at $n=10^6$, $9.2\times10^3$ at $n=10^7$). Uses differences of survival functions and a two-pass variance to avoid cancellation in $1-\mathrm{CDF}$ |
| `pareto_asymptotic.py` | the same curve in 60-decimal arithmetic, taking $n$ up to $10^{30}$ |
| `lattice_check2.py` | high-precision $\delta(s),\gamma(s)$ by Gauss–Legendre quadrature on the unit cell, and a numerical test of the lattice identity |
| `check_lattice_identity.py` | earlier version of the above, kept to record the derivation |
| `audit_logic.py` | fourth, independent check (tests A and C) |
| `verify_numerics.py` | independent recomputation of the dead-zone law $\delta(s)$ and of the v5 estimator at small $n$ |

### Figures

`make_figures.py` draws figures 1–5 as single-panel PDFs into `src/figures/`.
`make_figures_compact.py` draws the two compound floats actually used in the
paper, `fig_biasvar` (panels 1+2) and `fig_srmodes` (panels 4+5), from the same
arrays; `fig3_inflation` is used as produced by `make_figures.py`.

Both scripts are **redrawn from hard-coded arrays — they do not re-run an
experiment.** To change a plotted value you must edit the array in
`make_figures.py`. That is deliberate: the figures record the numbers in the
archived output, and the arrays in the script match the published tables.

---

1. **The original output for `tab:modes`/sec 5.4 was not preserved.**
   `outputs/rerun/out_v3.txt` is a re-run with the original seed `20260915`; it
   reproduces the published digits exactly, including the two rejected negative
   inflation factors ($-17.487$ for Student-$t_3$ at $n=512$ and $-6.797$ for the
   half-normal at $n=256$) quoted in sec 4.2. This is the one provenance gap in
   the package, and it is disclosed rather than hidden.

2. **The tables come from different batches of runs.** For example the lognormal
   at $L_m=4$, $n=4096$ is $176.996$ in `out_v5.txt` and $175.7679$ in
   `out_v7.txt`: the estimators and the grouping differ, and the paper says so in
   sec 5.5. **Do not compare digits across tables beyond a few parts in a
   thousand.**

3. **The large-$n$ measurements of sec 5.8 have no archived output.** The
   Pareto$(4)$, $L_m=2$ values at $n=2^{18}$ and above come from a separate
   run whose terminal output was not kept; what the package does provide for
   those block lengths is the parameter-free *prediction* of the reduced model,
   computed exactly by `src/exact/model_pop.py`, which agrees with the published
   values ($3.61\times10^4$ at $n=10^6$, $9.2\times10^3$ at $n=10^7$).

4. **Scripts are archived as they were used.** Their numerical content has not
   been edited. Scripts whose estimates were later rejected (v1, v3) are kept on
   purpose, because they document the corrections described in sec 4.2.



## Verification performed on this package

- `out_v9.txt` was independently re-run on a second machine (Python 3.11.9,
  NumPy 2.3.5) with the original seeds; every value matched digit for digit, and
  only the wall-clock time differed (203.1 s against the recorded 100.9 s).


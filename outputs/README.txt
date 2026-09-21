Archived terminal output. These files are the raw record behind the numbers in
the paper and are reproduced here unedited. Do not overwrite them; re-runs are
written to rerun/.

  out_v2.txt   bfp_error_pilot_v2.py   ->  tab:var, and the |z| values in sec 5.1
  out_v4.txt   bfp_error_pilot_v4.py   ->  sec 4.2, third-round estimator
  out_v5.txt   bfp_error_pilot_v5.py   ->  tab:infl (both mantissa widths)
  out_v6.txt   bfp_error_pilot_v6.py   ->  tab:srfull (RNE vs SR)
  out_v7.txt   bfp_error_pilot_v7.py   ->  tab:cond, tab:model, sec 5.6
  out_v8.txt   bfp_error_pilot_v8.py   ->  tab:bounded (last line: run time)
  out_v9.txt   bfp_error_pilot_v9.py   ->  tab:largen

out_v3.txt is missing: the original terminal output for tab:modes (rounding-rule
sensitivity, sec 5.4) and for the two rejected negative inflation factors in
sec 4.2 of the paper was not saved. The substitute

  rerun/out_v3.txt

is a re-run with the original seed 20260915, and it reproduces the published
digits exactly. Re-run command:

  python3 src/bfp_error_pilot_v3.py > outputs/rerun/out_v3.txt

Also not archived: the large-n (n >= 2^18) measurements of sec 5.8, which came
from a separate run. For those block lengths the package provides the exact
model prediction instead — see src/exact/model_pop.py.

Verify that these files still contain the numbers printed in the paper with

  python3 tools/check_paper_numbers.py

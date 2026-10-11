# Train-only calibration experiment

This experiment tests whether the full-moment bit-allocation decision remains
useful when the source law is estimated from a separate sample. It uses the
fixed 16 x 16 high-contrast diffusion graph and hotspot source envelope from
the preceding experiment. The source law is
`b_i = s_i exp(Z_i)`, with independent `Z_i ~ N(0,1)`.

For each training size `T = 32, 128, 512, 2048`, five independently labeled
training streams are generated. The fit uses the coordinate-wise means of
`log(b_i)` and one pooled residual standard deviation. The fitted scales and
pooled standard deviation determine two train-only maps: a source-scale map
and the full Green-energy greedy map. A uniform map and the oracle maps from
the known law are fixed comparisons. Every map is frozen before validation.

The single validation stream has 30,000 independent vectors and a distinct
stream label containing `20261011`; it is shared across every candidate for
paired comparisons. The candidate list includes the three oracle maps and
all 40 learned maps. Duplicate maps are evaluated once and mapped back to all
candidate records, while every fit and learned map remains in the CSV output.
No validation score is used to choose or remove a candidate.

Run from this directory:

```powershell
python calibration_experiment.py
python calibration_experiment.py --from-saved
```

The first command saves the training fit records, train-only predictions,
all validation energies, summaries, per-training-stream and validation input
digests, and a numerical
source snapshot. The second rebuilds the generated LaTeX section without
sampling or altering retained data. `calibration_prespecification.json`
records stream labels, model, maps, and dependencies before source vectors
are drawn. `calibration_application.tex` is the compact manuscript section.

The validation prediction intervals contain all 6 unique maps (43 candidate records, with repeated maps retained). The fitted
energy map recovers the oracle map in 17 of 20 fits. Its median regret is zero
at all four training sizes because the remaining fits produce maps with the
same energy as the oracle map; all three nonidentical maps are retained. The
source-scale median regret decreases from 45.07% at `T=32` to 25.46% at
`T=2048`. The train-only energy predictions have median validation errors of
-5.43% and -1.01% at those sizes, respectively.

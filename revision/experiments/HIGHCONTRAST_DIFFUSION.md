# Fixed high-contrast diffusion experiment

The additional scientific case uses the 16 x 16 interior nodal grid,
`h=1/17`, and the same independent heterogeneous hotspot lognormal sources
as the primary two-dimensional experiment. The coefficient is 0.01 inside
`(x-.7)^2+(y-.3)^2 <= .05` and 1 elsewhere. Internal and Dirichlet-boundary
face conductivities are harmonic averages. This nodal diffusion graph has
the same weighted-edge energy as its symmetric positive definite matrix.
The low-conductivity disk is spatially separated from the source hotspot.

The coefficient, source, contrast, mesh, and three codec families were
fixed without a parameter sweep. `highcontrast_diffusion_analytic_pilot.csv`
retains the pilot for every candidate. All 14 prescribed block/layout/
placement configurations remain in the experiment, including coincident
maps. Actual packed payloads have exactly 1792 bits. The same population
moment functions and codec as the preceding experiment are used unchanged.

`highcontrast_diffusion_prespecification.json` records all precision maps,
source stream labels, dependencies, and the numerical execution snapshot
before any held-out vectors are drawn. The Monte Carlo uses 30,000 paired
independent complete source vectors and 300 whole-vector bootstrap
replicates. Each vector's energy in every codec is saved in
`highcontrast_diffusion_samples.npz`; the source byte digest is also saved,
and all source vectors can be regenerated from the declared labels.

For 100 vector pairs per codec, both the original and the compressed
source systems are also solved with actual Jacobi-preconditioned conjugate
gradients, at relative tolerance 1e-11. The report includes iterations,
true residuals, energy differences, and a residual energy-norm certificate.
The certificate uses the rigorous operator comparison
`A >= .01 A_Poisson`, rather than a numerically estimated smallest
eigenvalue. Independent auditing reconstructs the coefficient and graph
with exact rational comparisons, checks weighted first-difference energy,
and implements separate manual Jacobi-PCG and Cholesky reference solves.

Reproduce from this directory:

```powershell
python highcontrast_diffusion.py --pilot
python highcontrast_diffusion.py
python ../verify_highcontrast_diffusion.py
```

Rebuild the manuscript section and three-panel precision-map figure without
sampling or repeating solver checks with `python highcontrast_diffusion.py --from-saved`.
This verifies that every retained numerical file and the
historical execution-source snapshot remain unchanged. The current artifact
generator and figure hashes are recorded separately from the executed source.

The generated short manuscript section is `../highcontrast_diffusion.tex`.
Its six-row table is a contiguous-layout slice of all retained results.
The separate paired-comparison and ranking CSVs include every configuration.

Primary exact RMS predictions lie in 13 of 14 nominal 95% intervals, with
maximum RMS discrepancy 0.509%. The full-energy length-32 contiguous map
reduces measured RMS by 19.968% compared with the source-scale map, with
paired interval [19.565%, 20.371%], and by 23.461% compared with uniform
extra-bit placement, with interval [22.984%, 23.875%]. These maps reallocate
one extra-bit block from a boundary row group toward the low-conductivity
disk. The length-128 contiguous improvement over the source-scale map is
11.253%, with interval [10.339%, 12.167%]. All weaker interleaved comparisons
and the single interval miss are retained.

The largest actual PCG energy relative error is 4.65e-11; the maximum
relative energy-norm certificate is 8.42e-9. These are separate checks of
the solver realization and the input-compression population prediction.

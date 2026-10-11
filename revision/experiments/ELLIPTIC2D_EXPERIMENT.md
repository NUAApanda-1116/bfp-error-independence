# Heterogeneous 2D elliptic experiment

The experiment quantizes independent sources `b_i = s_i exp(Z_i)`, with
`Z_i ~ N(0,1)`, for the five-point Dirichlet Poisson operator on the unit
square. The two deterministic spatial envelopes are a smooth sinusoidal
profile and a localized Gaussian hotspot. The meshes have 16 x 16 and
32 x 32 interior points.

`elliptic2d_application.py` integrates all nearest-rounding cells for every
coordinate law and exponent stratum. It propagates the full nonidentical
within-block covariance and the coherent bias through the discrete Green
matrix. DST solves are used for the complete Monte Carlo experiment, and
100 paired original/quantized right-hand sides per codec are also solved by
actual conjugate gradients at relative residual tolerance 1e-11. The energy
of the difference of these two CG solutions agrees with the DST prediction;
its residual supplies a separate energy-norm certificate.

Every codec constructs the actual packed payload of exactly seven bits per
cell. Its fixed map, block layout, and extra-bit locations are recorded in
`elliptic2d_prespecification.json`; coefficient widths and exponent widths
are unchanged from the preceding 1D codec. Block lengths are 8, 32, and 128,
with contiguous and interleaved layouts. At the 32 x 32 mesh, additional
prescribed source-scale and full-energy greedy placements are included when
extra coefficient bits are available. Some policies produce identical maps;
all prescribed policies and their results remain in the package.

The initial experiment used 20,000 whole source vectors per profile on the
smaller mesh and 10,000 on the larger one. The larger-mesh hotspot group had
shared positive deviations across its paired codecs. An additional stream
of 50,000 independent source vectors was then declared for **both** larger
mesh profiles. This decision and its reason are saved before drawing that
stream in `elliptic2d_replication_prespecification.json`. The initial streams
remain unchanged, and no seed was selected or discarded. Primary results
pool all 60,000 larger-mesh vectors. First-only, additional-only, and pooled
results are all saved in `elliptic2d_all_stream_results.csv`.

From this directory, reproduce the first experiment and the declared
replication in order:

```powershell
python elliptic2d_application.py
python elliptic2d_replication.py
python build_elliptic2d_artifacts.py
python elliptic2d_allocation_diagnostic.py
```

The additional stream includes a SHA256 digest of all generated binary64
input vectors. Inputs are reproducible from the saved master seed, stream
labels, and numerical source. Every vector's energy in every declared codec
is retained in the NPZ files. The execution source and dependency hashes are
recorded in the manifests. Running only `build_elliptic2d_artifacts.py`
rebuilds the table, figure, and summary without drawing any source vectors.

Primary results cover 34 of 40 exact RMS predictions with nominal pointwise
95% bootstrap intervals; the maximum RMS discrepancy is 0.827%. The initial
results, including all 14 misses among 40 intervals, are retained. These are
paired codec intervals rather than independent coverage trials. Bootstrap
sampling units are complete source vectors, with 300 replicates.

At the 32 x 32 mesh, the full-energy placement lowers the smooth-profile
length-32 contiguous codec's measured RMS by 22.372%, with paired 95% interval
[22.171%, 22.605%], relative to evenly spaced extra bits. For the hotspot,
full-energy placement improves on the source-scale placement by 0.861%,
with interval [0.814%, 0.906%]. The uniform-cell model chooses the length-32
full-energy codec for the hotspot, whereas the full and exact-marginal models
choose length 8; the former choice has 4.116% higher measured RMS, with paired
interval [3.710%, 4.480%].

The analytical-only post-hoc allocation diagnostic compares full versus
diagonal-covariance greedy placement. They choose the same maps except for
one symmetric tie with identical population energy. Consequently the gain
over the source-scale heuristic is a full-model placement improvement, not
evidence that covariance alone changes the greedy map. Covariance's effect
on the predicted energy and layout remains directly measurable in the
primary experiment.

Independent verification is implemented separately in
`../verify_nonidentical_lognormal_mp.py` and
`../verify_poisson2d_nonidentical.py`. These check analytical block moments against
60/90-digit arithmetic and heterogeneous two-dimensional energy propagation
against exact rational enumeration, respectively.

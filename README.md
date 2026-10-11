# Exact Error Covariances for Block Floating Point Reductions

Reproduction source and retained numerical evidence for the SISC manuscript
**Exact Error Covariances for Block Floating Point Reductions** by Yongkang
Xiong (11 October 2026).

The paper derives finite-block error moments for maximum-scaled block
floating-point quantization and propagates the resulting covariance formulas to
weighted reductions and discrete elliptic computations. The repository keeps
the current manuscript source, experiment drivers, machine-readable results,
verification programs, figures, and retained `.npz` evidence used by the
saved-data checks.

## Reproduce

Requirements are Python 3.14 (or a compatible recent Python), the pinned
packages in `requirements.txt`, TeX Live with `latexmk`, and Poppler's
`pdfinfo` on `PATH`.

```powershell
python -m pip install -r requirements.txt
./reproduce.ps1 -SkipExperiments
```

The saved-data path rebuilds the generated tables and figures, runs the
independent audits, and compiles the 19-page manuscript. To regenerate the
Monte Carlo evidence from the fixed streams, run `./reproduce.ps1` instead.
The detailed experiment order and version notes are in
`revision/README.md`.

The GitHub tree omits the large TIFF rendering together with internal, historical, build, and QA material. The complete SISC supplementary snapshot, including the full source
and retained evidence, is supplied separately in the SISC submission package.

## Layout

- `paper.tex`, `revision/`: current manuscript and its generated sections.
- `revision/experiments/`: experiment drivers, results, manifests, figures,
  and saved numerical evidence.
- `siamart_251216.zip`: the original SIAM template distribution.
- `CITATION.cff`: citation metadata.

The model is an unsaturated uniform integer-mantissa block-floating-point grid
and isolates input quantization from subsequent arithmetic rounding.

## License

Code is MIT. The manuscript and accompanying author-created text retain the
rights stated in the manuscript and SISC submission. The SIAM macro
 distribution retains its own license terms.



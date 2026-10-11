# Reproduction package

Paper: **Exact Error Covariances for Block Floating Point Reductions**.
Snapshot date: 2026-10-11.

This public snapshot contains the current manuscript sources, figures, experiment
drivers, retained numerical results, verification scripts, and pinned research
dependencies. Historical drafts, internal review notes, QA screenshots, and
LaTeX build intermediates are outside this snapshot.

## Main sources

- `../paper.tex` and `../references.bib`: manuscript and bibliography.
- `introduction.tex`, `theory_section.tex`, `computation.tex`, `results.tex`,
  `discussion.tex`: maintained manuscript sections.
- `elliptic2d_application.tex`, `latent_application.tex`,
  `highcontrast_diffusion.tex`, and `calibration_application.tex`: current
  scientific-computing applications.
- `experiments/`: experiment drivers, figures, machine-readable results,
  retained streams, and manifests.

## Reproduction

From the snapshot root, install the pinned research dependencies:

```powershell
python -m pip install -r revision/requirements.txt
./reproduce.ps1 -SkipExperiments
```

The saved-data path rebuilds the generated artifacts, runs the independent
exact, high-precision, quadrature, diffusion, and calibration checks, and
compiles the 19-page manuscript. To regenerate the numerical evidence from the
fixed streams, run `./reproduce.ps1`. The scripts use fixed seeds and retain
all reported interval misses.

The supplied SIAM class and bibliography style are in the snapshot root.
`siamart_251216.zip` is the original template distribution; the class and style
files used by the manuscript are also present directly.

## Verification and packaging

`check_document.py` checks the page cap and PDF layout. `package_revision.py`
regenerates `PACKAGE_MANIFEST.json` and a hashed `SISC_revision_package.zip`.
The package script can use the bundled template distribution when run from an
extracted snapshot.

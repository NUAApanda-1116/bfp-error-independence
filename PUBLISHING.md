# Publishing this GitHub-ready export

This directory is an independent clean export of the 11 October 2026 SISC
revision. It has two local commits and the branch `sisc-revision-2026-10-11`
points to the same final commit as `main`.

The configured remote is:

```text
https://github.com/NUAApanda-1116/bfp-error-independence.git
```

The remote `main` branch contains the earlier repository history. To preserve
that history, publish this export as a new branch after reviewing it:

```powershell
git push origin HEAD:sisc-revision-2026-10-11
```

Merging this export into the existing `main` or replacing `main` requires a
separate repository decision. No network push was performed while preparing
this directory.

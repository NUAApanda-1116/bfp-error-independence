param([switch]$SkipExperiments, [string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $SkipExperiments) {
        & $Python 'revision/experiments/run_revision.py'
        if ($LASTEXITCODE -ne 0) { throw 'Ordinary experiment failed.' }
        & $Python 'revision/experiments/stratified_pareto.py'
        if ($LASTEXITCODE -ne 0) { throw 'Stratified experiment failed.' }
        & $Python 'revision/experiments/elliptic2d_application.py'
        if ($LASTEXITCODE -ne 0) { throw 'Heterogeneous elliptic experiment failed.' }
        & $Python 'revision/experiments/elliptic2d_replication.py'
        if ($LASTEXITCODE -ne 0) { throw 'Independent replication failed.' }
        & $Python 'revision/experiments/latent_elliptic_application.py'
        if ($LASTEXITCODE -ne 0) { throw 'Correlated elliptic experiment failed.' }
        & $Python 'revision/experiments/highcontrast_diffusion.py'
        if ($LASTEXITCODE -ne 0) { throw 'High-contrast diffusion experiment failed.' }
        & $Python 'revision/experiments/calibration_experiment.py'
        if ($LASTEXITCODE -ne 0) { throw 'Calibration experiment failed.' }
        & $Python 'revision/experiments/latent_correlation_sweep.py'
        if ($LASTEXITCODE -ne 0) { throw 'Latent sensitivity experiment failed.' }
    }
    & $Python 'revision/experiments/build_elliptic2d_artifacts.py'
    if ($LASTEXITCODE -ne 0) { throw 'Elliptic artifact generation failed.' }
    & $Python 'revision/experiments/latent_elliptic_application.py' '--from-saved'
    if ($LASTEXITCODE -ne 0) { throw 'Saved correlated artifacts failed.' }
    & $Python 'revision/experiments/highcontrast_diffusion.py' '--from-saved'
    if ($LASTEXITCODE -ne 0) { throw 'Saved diffusion artifacts failed.' }
    & $Python 'revision/experiments/calibration_experiment.py' '--from-saved'
    if ($LASTEXITCODE -ne 0) { throw 'Saved calibration artifacts failed.' }
    & $Python 'revision/experiments/latent_correlation_sweep.py' '--from-saved'
    if ($LASTEXITCODE -ne 0) { throw 'Saved latent sensitivity artifacts failed.' }
    & $Python 'revision/experiments/make_artifacts.py'
    if ($LASTEXITCODE -ne 0) { throw 'Artifact generation failed.' }
    & $Python 'revision/verify_exact_enumeration.py'
    if ($LASTEXITCODE -ne 0) { throw 'Rational audit failed.' }
    & $Python 'revision/verify_nonidentical.py'
    if ($LASTEXITCODE -ne 0) { throw 'Nonidentical-input rational audit failed.' }
    & $Python 'revision/verify_poisson2d_nonidentical.py'
    if ($LASTEXITCODE -ne 0) { throw 'Heterogeneous 2D rational audit failed.' }
    & $Python 'revision/verify_zero_state.py'
    if ($LASTEXITCODE -ne 0) { throw 'Zero-state implementation audit failed.' }
    & $Python 'revision/verify_pareto_mp.py'
    if ($LASTEXITCODE -ne 0) { throw 'High-precision audit failed.' }
    & $Python 'revision/verify_lognormal_mp.py'
    if ($LASTEXITCODE -ne 0) { throw 'Lognormal high-precision audit failed.' }
    & $Python 'revision/verify_nonidentical_lognormal_mp.py'
    if ($LASTEXITCODE -ne 0) { throw 'Heterogeneous high-precision audit failed.' }
    & $Python 'revision/verify_latent_quadrature.py'
    if ($LASTEXITCODE -ne 0) { throw 'Latent quadrature audit failed.' }
    & $Python 'revision/verify_highcontrast_diffusion.py'
    if ($LASTEXITCODE -ne 0) { throw 'Diffusion operator and PCG audit failed.' }
    & $Python 'revision/verify_calibration.py'
    if ($LASTEXITCODE -ne 0) { throw 'Calibration audit failed.' }
    & $Python 'revision/extract_floats.py'
    if ($LASTEXITCODE -ne 0) { throw 'Float extraction failed.' }
    & latexmk '-pdf' '-interaction=nonstopmode' '-halt-on-error' 'paper.tex'
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed.' }
    $PdfMetadata = & pdfinfo 'paper.pdf'
    if ($LASTEXITCODE -ne 0) { throw 'PDF inspection failed.' }
    $PageMatch = [regex]::Match(($PdfMetadata -join "`n"), 'Pages:\s+(\d+)')
    if (-not $PageMatch.Success -or [int]$PageMatch.Groups[1].Value -gt 20) {
        throw 'PDF page count is missing or exceeds 20.'
    }
    Write-Output "Verified PDF: $($PageMatch.Groups[1].Value) pages."
}
finally {
    Pop-Location
}

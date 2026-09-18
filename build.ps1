#-----------------------------------------------------------------------
#  build.ps1 -- compile the manuscript with MiKTeX, without touching the
#  working PDFs in the repository root.
#
#  Usage:   powershell -ExecutionPolicy Bypass -File build.ps1
#           powershell -ExecutionPolicy Bypass -File build.ps1 -Install
#
#  -Install also copies the rebuilt PDFs over paper.pdf / paper-twocolumn.pdf.
#-----------------------------------------------------------------------
param(
    [switch]$Install,
    [string]$BuildDir = "$PSScriptRoot\_build"
)

$ErrorActionPreference = 'Continue'   # MiKTeX writes an update notice to stderr;
                                      # treat it as noise and check $LASTEXITCODE instead

# --- locate MiKTeX (user-scope install) -------------------------------------
$candidates = @(
    "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64",
    "C:\Program Files\MiKTeX\miktex\bin\x64"
)
$mik = $candidates | Where-Object { Test-Path (Join-Path $_ 'pdflatex.exe') } | Select-Object -First 1
if (-not $mik) {
    $cmd = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($cmd) { $mik = Split-Path $cmd.Source } else {
        throw "pdflatex not found. Install MiKTeX:  winget install --id MiKTeX.MiKTeX --exact"
    }
}
$env:Path = $mik + ';' + $env:Path
$env:MIKTEX_AUTOINSTALL = '1'          # fetch missing packages without prompting

Write-Host "MiKTeX bin : $mik"
Write-Host "build dir  : $BuildDir"

# --- stage a clean copy of the sources --------------------------------------
if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
New-Item -ItemType Directory -Path $BuildDir | Out-Null
foreach ($pat in '*.tex', '*.bib', '*.cls', '*.bst') {
    Copy-Item (Join-Path $PSScriptRoot $pat) $BuildDir -Force
}
Copy-Item (Join-Path $PSScriptRoot 'figures') $BuildDir -Recurse -Force

Push-Location $BuildDir
try {
    foreach ($job in 'paper', 'paper-twocolumn') {
        Write-Host ("`n=== $job ===")
        & pdflatex -interaction=nonstopmode -halt-on-error "$job.tex" 1> "$job.pass1.log" 2> $null
        if ($LASTEXITCODE -ne 0) { throw "pdflatex (pass 1) failed for $job -- see $job.pass1.log" }
        & bibtex "$job" 1> "$job.bibtex.log" 2> $null
        & pdflatex -interaction=nonstopmode "$job.tex" 1> "$job.pass2.log" 2> $null
        & pdflatex -interaction=nonstopmode "$job.tex" 1> "$job.pass3.log" 2> $null
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed for $job -- see $job.pass3.log" }

        # report the things that actually matter
        $log   = Get-Content "$job.pass3.log" -Raw
        $undef = ([regex]::Matches($log, 'undefined', 'IgnoreCase')).Count
        $over  = ([regex]::Matches($log, 'Overfull')).Count
        $m     = [regex]::Match($log, 'Output written on .*?\((\d+) pages')
        Write-Host ("  pages={0}  undefined={1}  overfull={2}" -f $m.Groups[1].Value, $undef, $over)
        if ($undef -gt 0) { Write-Warning "$job has undefined references or citations" }
    }
}
finally {
    Pop-Location
}

if ($Install) {
    Copy-Item (Join-Path $BuildDir 'paper.pdf')           (Join-Path $PSScriptRoot 'paper.pdf') -Force
    Copy-Item (Join-Path $BuildDir 'paper-twocolumn.pdf') (Join-Path $PSScriptRoot 'paper-twocolumn.pdf') -Force
    Write-Host "`ninstalled rebuilt PDFs into $PSScriptRoot"
}

$sums = Join-Path $PSScriptRoot 'SHA256SUMS.txt'
Write-Host "`nNOTE: SHA256SUMS.txt is now stale if the sources changed; regenerate it before distributing."
Write-Host "Done."

"""Create a portable, hashed snapshot of the revised manuscript and evidence."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
from pypdf import PdfReader

root = Path(__file__).resolve().parent.parent
required = ['paper.tex', 'paper.pdf', 'paper.bbl', 'references.bib',
            'siamart251216.cls', 'siamplain.bst', 'reproduce.ps1']
files = [root / name for name in required]
stale_logs = {
    'artifact_refresh_console.txt', 'build_current_console.txt',
    'build_final_console.txt', 'reproduction_calibration_console.txt',
    'reproduction_console.txt', 'reproduction_current_console.txt',
    'reproduction_final_console.txt'}
files += [p for p in (root / 'revision').rglob('*') if p.is_file()
          and '__pycache__' not in p.parts
          and not any(part.startswith('before_') for part in p.parts)
          and p.name != 'PACKAGE_MANIFEST.json'
          and p.name not in stale_logs]
manifest = dict(paper_title='Exact Error Covariances for Block Floating Point Reductions',
                revision_date='2026-10-11', pdf_pages=len(PdfReader(root/'paper.pdf').pages),
                source_files=[p.relative_to(root).as_posix() for p in files],
                sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(files)})
manifest_path = root / 'revision' / 'PACKAGE_MANIFEST.json'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
archive = root / 'SISC_revision_package.zip'
with ZipFile(archive, 'w', compression=ZIP_DEFLATED, compresslevel=6) as zipf:
    for path in files + [manifest_path]:
        zipf.write(path, 'SISC_submission/' + path.relative_to(root).as_posix())
    # Include the originally supplied complete template distribution alongside
    # the class/style files, rather than distributing the class by itself.
    template_distribution = root.parent / 'siamart_251216.zip'
    if not template_distribution.exists():
        template_distribution = root / 'siamart_251216.zip'
    if template_distribution.exists():
        zipf.write(template_distribution, 'SISC_submission/siam_template_distribution.zip')
with ZipFile(archive) as zipf:
    assert zipf.testzip() is None
    for path, expected in manifest['sha256'].items():
        assert hashlib.sha256(zipf.read('SISC_submission/' + path)).hexdigest() == expected
print(f'Created {archive.name}: {archive.stat().st_size / (1024 * 1024):.1f} MiB, '
      f'{len(files)} files, hashes verified.')


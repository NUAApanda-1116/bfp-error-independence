"""Document preflight and contact sheets for visual inspection of every page."""
from pathlib import Path
import json
import re
from PIL import Image
from pypdf import PdfReader
import pdfplumber

base = Path(__file__).resolve().parent
root = base.parent
pages = sorted((base / 'qa').glob('page-*.png'))
if pages:
    w, h = Image.open(pages[0]).size
    for start in range(0, len(pages), 4):
        canvas = Image.new('RGB', (2 * w, 2 * h), (230, 230, 230))
        for j, page in enumerate(pages[start:start + 4]):
            canvas.paste(Image.open(page).convert('RGB'), ((j % 2) * w, (j // 2) * h))
        canvas.save(base / 'qa' / f'contact_{start // 4 + 1}.png')

tex = (root / 'paper.tex').read_text(encoding='utf-8')
abstract = tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
running = re.search(r'\\headers\{(.*?)\}', tex).group(1)
reader = PdfReader(root / 'paper.pdf')
assert not pages or len(pages) == len(reader.pages), 'Rendered pages do not match the current PDF.'
log = (root / 'paper.log').read_text(encoding='utf-8', errors='replace')
summary = dict(pages=len(reader.pages), abstract_words=len(abstract.split()),
               running_head=running, running_head_characters=len(running),
               bibliography_entries=(root / 'paper.bbl').read_text().count(r'\bibitem'),
               overfull_boxes=log.count('Overfull'),
               undefined_references='undefined' in log.lower(),
               page_text_checks=[])
with pdfplumber.open(root / 'paper.pdf') as pdf:
    for j, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ''
        chars = page.chars
        outside = [c for c in chars if c['x0'] < -0.5 or c['x1'] > page.width + .5
                   or c['top'] < -.5 or c['bottom'] > page.height + .5]
        summary['page_text_checks'].append(dict(page=j, characters=len(text),
                                                outside_page_characters=len(outside),
                                                replacement_characters=text.count('\ufffd')))
assert summary['pages'] <= 20
assert summary['abstract_words'] <= 250
assert summary['running_head_characters'] <= 50
assert summary['overfull_boxes'] == 0
assert not summary['undefined_references']
assert all(p['outside_page_characters'] == 0 for p in summary['page_text_checks'])
(base / 'qa' / 'preflight.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'page_text_checks'}, ensure_ascii=False))

"""Place data-generated floats near their first discussion in the manuscript."""
from pathlib import Path
import re

base = Path(__file__).resolve().parent
for kind, stem in [('table', 'tables'), ('figure', 'figures')]:
    text = (base / 'experiments' / f'{stem}.tex').read_text(encoding='utf-8')
    blocks = re.findall(r'\\begin\{' + kind + r'\}.*?\\end\{' + kind + r'\}', text, flags=re.S)
    assert len(blocks) == 3
    for name, block in zip(['validation', 'weighted', 'exponential'], blocks):
        (base / f'{kind}_{name}.tex').write_text('% Generated from experiments/' + stem + '.tex\n'
                                                 + block + '\n', encoding='utf-8')
print('Extracted six data-generated floats.')

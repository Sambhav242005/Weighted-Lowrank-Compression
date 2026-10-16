"""Read-only evidence checks and a new export validation record."""
import hashlib
import json
from pathlib import Path

import pdfplumber

root = Path(__file__).resolve().parents[2]
revision = Path(__file__).resolve().parent
paper = root / 'docs/paper-future-scope-20261015'
pdf = paper / 'ieee-conference-gpu-execution-20261016.pdf'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
evidence = json.loads((revision / 'evidence-extraction.json').read_text())
for item in evidence['frozen_inputs'].values():
    assert sha(root / item['path']) == item['sha256'], item['path']
assert sha(revision / 'ieee-conference-before-gpu.tex') == evidence['before_sha256']
outside = []
with pdfplumber.open(pdf) as doc:
    pages = [p.extract_text() or '' for p in doc.pages]
    for i, p in enumerate(doc.pages, 1):
        for c in p.chars:
            if c['x0'] < -0.5 or c['x1'] > p.width + 0.5 or c['top'] < -0.5 or c['bottom'] > p.height + 0.5:
                outside.append({'page': i, 'text': c['text']})
text = '\n'.join(pages)
for expected in ['47.2231', '58.3059', '1866.640', '1921.132', '1940.790', '1939.413', '292,281']:
    assert expected in text, expected
assert not outside, outside
record = {
    'source': str(paper / 'ieee-conference.tex'),
    'source_sha256': sha(paper / 'ieee-conference.tex'),
    'pdf': str(pdf), 'pdf_sha256': sha(pdf), 'page_count': len(pages),
    'compiler': 'Tectonic 0.17.0 x86_64-pc-windows-msvc',
    'frozen_evidence_hashes_match': True,
    'pre_edit_source_preserved': True,
    'all_table_values_found': True, 'text_outside_page_bounds': outside,
    'gpu_table_pages': [i for i, p in enumerate(pages, 1) if '1939.413' in p],
    'new_experiments': False,
    'visual_review': 'Pending rendered-page inspection',
}
(revision / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))

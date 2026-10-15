"""Create a fresh math-readable edition without modifying earlier artifacts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'docs/paper-ieee-preparation-20261015'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE/'manuscript.pdf').exists():
        raise FileExistsError('Preserve the previous math-readable build')
    protected={str(p.relative_to(ROOT)):sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    spec=importlib.util.spec_from_file_location('math_readable_renderer',HERE/'build_paper.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE=PREVIOUS
    module.OUT=HERE
    module.main()
    tex=(HERE/'manuscript.tex').read_text(encoding='utf-8')
    tex='\n'.join(line.rstrip() for line in tex.splitlines()).rstrip()+'\n'
    (HERE/'manuscript.tex').write_text(tex,encoding='utf-8',newline='\n')
    abstract=(HERE/'abstract.md').read_text(encoding='utf-8').strip()
    markdown=(HERE/'manuscript.md').read_text(encoding='utf-8')
    keywords=re.search(r'^Keywords: (.*)$',markdown,re.M)[1]
    helper=module.import_helper('math_edition_latex',ROOT/'docs/export_paper_latex_20261014.py')
    body=tex[tex.index(r'\section{Introduction}'):]
    body=body.replace(r'\section{Acknowledgment}',r'\section*{Acknowledgment}')
    ieee=[r'\documentclass[conference]{IEEEtran}',
        r'\usepackage[T1]{fontenc}',r'\usepackage{amsmath,amssymb,array,booktabs,graphicx}',
        r'\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}',
        r'\setlength{\emergencystretch}{2em}',r'\sloppy',
        '% Provisional conference format; actual venue and metadata remain unresolved.',
        '% Mathematics uses stacked fractions and separate numbered displays.',
        r'\begin{document}',r'\title{'+helper.escape(module.TITLE)+'}',
        r'\author{\IEEEauthorblockN{Sambhav Surana}}',r'\maketitle',
        r'\begin{abstract}'+helper.inline(abstract)+r'\end{abstract}',
        r'\begin{IEEEkeywords}'+helper.escape(keywords)+r'\end{IEEEkeywords}',body]
    ieee_path=HERE/'ieee-conference.tex'
    ieee_path.write_text('\n'.join(ieee).rstrip()+'\n',encoding='utf-8',newline='\n')
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    manifest=json.loads((HERE/'format-manifest.json').read_text(encoding='utf-8'))
    manifest['artifacts']['manuscript.tex']={'sha256':sha(HERE/'manuscript.tex'),
        'bytes':(HERE/'manuscript.tex').stat().st_size}
    (HERE/'format-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    audit={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Readable fractions, floor brackets, displayed equations and inline baselines',
        'source_edition':str(PREVIOUS.relative_to(ROOT)),
        'original_source_hashes':protected,'originals_unchanged':True,
        'manuscript_text_unchanged':sha(HERE/'manuscript.md')==sha(PREVIOUS/'manuscript.md'),
        'abstract_unchanged':sha(HERE/'abstract.md')==sha(PREVIOUS/'abstract.md'),
        'bibliography_unchanged':sha(HERE/'paper_protocol_v2.bib')==sha(PREVIOUS/'paper_protocol_v2.bib'),
        'report_unchanged':sha(HERE/'report.json')==sha(PREVIOUS/'report.json'),
        'math_changes':module.MATH_PRESENTATION,
        'research_experiments_run':False,
        'review_pdf':'Independent ReportLab review rendering; not exported from the IEEE source',
        'ieee_conference_source_sha256':sha(ieee_path)}
    (HERE/'math-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(HERE),'equations':manifest['numbered_pdf_equations'],
        'text_unchanged':audit['manuscript_text_unchanged'],'experiments':False},indent=2))


if __name__=='__main__':
    main()

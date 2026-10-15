"""Render the readable revision and prepare a provisional IEEE conference source.

The conference is unidentified. This produces a review package, not a certified
venue submission or a source-matched IEEE camera-ready PDF. No experiment runs.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import uuid

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREVIOUS=ROOT/'docs/paper-reference-style-20261015'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE/'manuscript.pdf').exists():raise FileExistsError('Preserve the previous review build')
    protected={str(p.relative_to(ROOT)):sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    inputs=ROOT/'docs/proofs'/('ieee-preparation-inputs-'+uuid.uuid4().hex[:8])
    inputs.mkdir(parents=True,exist_ok=False)
    (inputs/'figures').mkdir()
    shutil.copyfile(HERE/'manuscript.source.md',inputs/'manuscript.md')
    shutil.copyfile(PREVIOUS/'report.json',inputs/'report.json')
    shutil.copyfile(PREVIOUS/'figures/provenance.json',inputs/'figures/provenance.json')
    bibliography=(PREVIOUS/'paper_protocol_v2.bib').read_text(encoding='utf-8')
    bibliography+='\n@misc{openaicodex2026,\n  author = {{OpenAI}},\n  title = {Codex},\n  year = {2026},\n  url = {https://openai.com/codex/},\n  note = {Software identified for manuscript-preparation disclosure; accessed 15 October 2026}\n}\n'
    (inputs/'paper_protocol_v2.bib').write_text(bibliography,encoding='utf-8')
    path=PREVIOUS/'build_paper.py'
    # The unchanged renderer records this basename in its output manifest.
    shutil.copyfile(path,HERE/'build_paper.py')
    spec=importlib.util.spec_from_file_location('readable_paper_renderer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.SOURCE=inputs
    module.OUT=HERE
    module.main()
    # Typography-only cleanup; preserve every mathematical token.
    review_tex=(HERE/'manuscript.tex').read_text(encoding='utf-8')
    review_tex='\n'.join(line.rstrip() for line in review_tex.splitlines())+'\n'
    (HERE/'manuscript.tex').write_text(review_tex,encoding='utf-8',newline='\n')
    render_manifest=json.loads((HERE/'format-manifest.json').read_text(encoding='utf-8'))
    render_manifest['artifacts']['manuscript.tex']={'sha256':sha(HERE/'manuscript.tex'),
        'bytes':(HERE/'manuscript.tex').stat().st_size}
    render_manifest['tex_line_end_whitespace_normalized']=True
    (HERE/'format-manifest.json').write_text(json.dumps(render_manifest,indent=2)+'\n',encoding='utf-8')
    abstract=(HERE/'abstract.md').read_text(encoding='utf-8').strip()
    markdown=(HERE/'manuscript.md').read_text(encoding='utf-8')
    keywords=re.search(r'^Keywords: (.*)$',markdown,re.M)[1]
    helper=module.import_helper('preparation_latex',ROOT/'docs/export_paper_latex_20261014.py')
    body=review_tex[review_tex.index(r'\section{Introduction}'):]
    body=body.replace(r'\section{Acknowledgment}',r'\section*{Acknowledgment}')
    ieee=[r'\documentclass[conference]{IEEEtran}',
        r'\usepackage[T1]{fontenc}',r'\usepackage{amsmath,amssymb,array,booktabs,graphicx}',
        r'\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}',
        r'\setlength{\emergencystretch}{2em}',r'\sloppy',
        '% Provisional conference format. Confirm the actual event, paper size, page cap,',
        '% anonymity rules, author affiliation/contact data and permitted appendix policy.',
        r'\begin{document}',r'\title{'+helper.escape(module.TITLE)+'}',
        r'\author{\IEEEauthorblockN{Sambhav Surana}}',r'\maketitle',
        r'\begin{abstract}'+helper.inline(abstract)+r'\end{abstract}',
        r'\begin{IEEEkeywords}'+helper.escape(keywords)+r'\end{IEEEkeywords}',body]
    ieee_path=HERE/'ieee-conference.tex'
    ieee_path.write_text('\n'.join(ieee).rstrip()+'\n',encoding='utf-8',newline='\n')
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'Plain-language abstract, Humanizer prose edits and IEEE preparation',
        'humanizer':{'version':'3.1.0','source':'https://github.com/blader/humanizer/blob/main/SKILL.md',
            'applied_as':'Editorial guidance; no detector-evasion or acceptance claim'},
        'abstract_words':len(abstract.split()),'abstract_single_paragraph':True,
        'original_source_hashes':protected,'originals_unchanged':True,
        'conference_identity':None,'deadline':'User reports submission before 17 October 2026; not independently verified',
        'reported_late_fee':'INR 1,000; event and fee stage unknown; not independently verified',
        'template_status':'Generic IEEEtran conference source, pending actual conference rules',
        'review_pdf':'Independent ReportLab rendering, not an IEEE template submission PDF',
        'ieee_conference_source_sha256':sha(ieee_path),
        'additional_reference':'OpenAI Codex acknowledgment; original 22 research entries unchanged',
        'research_experiments_run':False,'submitted':False,'paid':False}
    (HERE/'preparation-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'abstract_words':len(abstract.split()),'output':str(HERE),'venue':'unidentified','experiments':False},indent=2))


if __name__=='__main__':main()

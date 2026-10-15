"""Build the appendix-free edition with the preserved mathematical layout."""
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
PREVIOUS=ROOT/'docs/paper-math-readable-20261015'

TINYBERT='''
@inproceedings{jiao2020tinybert,
  author = {Jiao, Xiaoqi and Yin, Yichun and Shang, Lifeng and Jiang, Xin and Chen, Xiao and Li, Linlin and Wang, Fang and Liu, Qun},
  title = {TinyBERT: Distilling BERT for Natural Language Understanding},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year = {2020},
  pages = {4163--4174},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2020.findings-emnlp.372},
  url = {https://aclanthology.org/2020.findings-emnlp.372/}
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE/'manuscript.pdf').exists():
        raise FileExistsError('Preserve the existing rendered edition')
    protected={str(p.relative_to(ROOT)):sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    inputs=ROOT/'docs/proofs'/('future-scope-inputs-'+uuid.uuid4().hex[:8])
    (inputs/'figures').mkdir(parents=True,exist_ok=False)
    shutil.copyfile(HERE/'manuscript.source.md',inputs/'manuscript.md')
    shutil.copyfile(PREVIOUS/'report.json',inputs/'report.json')
    shutil.copyfile(PREVIOUS/'figures/provenance.json',inputs/'figures/provenance.json')
    bib=(PREVIOUS/'paper_protocol_v2.bib').read_text(encoding='utf-8')+TINYBERT
    (inputs/'paper_protocol_v2.bib').write_text(bib,encoding='utf-8',newline='\n')
    spec=importlib.util.spec_from_file_location('future_scope_renderer',HERE/'build_paper.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCE=inputs
    module.OUT=HERE
    module.main()
    tex=(HERE/'manuscript.tex').read_text(encoding='utf-8')
    tex='\n'.join(line.rstrip() for line in tex.splitlines()).rstrip()+'\n'
    (HERE/'manuscript.tex').write_text(tex,encoding='utf-8',newline='\n')
    shutil.copyfile(PREVIOUS/'abstract.md',HERE/'abstract.md')
    abstract=(HERE/'abstract.md').read_text(encoding='utf-8').strip()
    markdown=(HERE/'manuscript.md').read_text(encoding='utf-8')
    keywords=re.search(r'^Keywords: (.*)$',markdown,re.M)[1]
    helper=module.import_helper('future_scope_latex',ROOT/'docs/export_paper_latex_20261014.py')
    body=tex[tex.index(r'\section{Introduction}'):]
    body=body.replace(r'\section{Acknowledgment}',r'\section*{Acknowledgment}')
    ieee=[r'\documentclass[conference]{IEEEtran}',
        r'\usepackage[T1]{fontenc}',r'\usepackage{amsmath,amssymb,array,booktabs,graphicx}',
        r'\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}',
        r'\setlength{\emergencystretch}{2em}',r'\sloppy',
        '% Provisional conference format; actual venue and metadata remain unresolved.',
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
        'purpose':'Remove both paper appendices and add Future Scope including a small-network hypothesis',
        'source_edition':str(PREVIOUS.relative_to(ROOT)),
        'previous_edition_hashes':protected,'previous_edition_unchanged':True,
        'abstract_unchanged':sha(HERE/'abstract.md')==sha(PREVIOUS/'abstract.md'),
        'report_unchanged':sha(HERE/'report.json')==sha(PREVIOUS/'report.json'),
        'figure_provenance_unchanged':sha(HERE/'figures/provenance.json')==sha(PREVIOUS/'figures/provenance.json'),
        'new_reference':{'key':'jiao2020tinybert','url':'https://aclanthology.org/2020.findings-emnlp.372/',
            'scope':'Related transformer distillation precedent; no evidence for our untested projection replacement'},
        'network_implemented':False,'research_experiments_run':False,
        'review_pdf':'Independent ReportLab review rendering; not exported from IEEE source',
        'ieee_conference_source_sha256':sha(ieee_path)}
    (HERE/'edition-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(HERE),'equations':manifest['numbered_pdf_equations'],
        'tables':manifest['table_count_preserved'],'experiments':False},indent=2))


if __name__=='__main__':
    main()

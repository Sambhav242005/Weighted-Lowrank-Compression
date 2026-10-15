"""Render the appendix-free manuscript with readable mathematical typography.

Reads frozen editorial/evidence files only. No research code is imported.
PDF is built with ReportLab; standalone TeX is separately compile-checked.
"""
from datetime import datetime, timezone
from html import escape
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'docs/paper-math-readable-20261015'
REFERENCE = Path('C:/Users/NPC/Documents/weighted_lowrank_compression_solo_author.pdf')
TITLE = 'Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost'
AUTHOR = 'Sambhav Surana'
REPO = 'https://github.com/Sambhav242005/Weighted-Lowrank-Compression'
SITE = 'https://sambhav242005.github.io/Weighted-Lowrank-Compression/'

# Presentation changes only: identical operations, operands and bounds.
MATH_PRESENTATION = {
    r'r<oi/(o+i)': r'r<\frac{oi}{o+i}',
    r'r=\lfloor(17/24)oi/(o+i)\rfloor':
        r'r=\left\lfloor\frac{17}{24}\,\frac{oi}{o+i}\right\rfloor',
    r'\lfloor\min(o,i)/d\rfloor':
        r'\left\lfloor\frac{\min(o,i)}{d}\right\rfloor',
    r'D_\ell=\sum_{t\in\mathcal{T}}\|\widehat h_{\ell,t}-h_{\ell,t}\|_2^2/\max(\sum_{t\in\mathcal{T}}\|h_{\ell,t}\|_2^2,10^{-30})':
        r'D_\ell=\frac{\sum_{t\in\mathcal{T}}\|\widehat h_{\ell,t}-h_{\ell,t}\|_2^2}{\max\left(\sum_{t\in\mathcal{T}}\|h_{\ell,t}\|_2^2,\,10^{-30}\right)}',
    r'D=L^{-1}\sum_{\ell=1}^{L}D_\ell':
        r'D=\frac{1}{L}\sum_{\ell=1}^{L}D_\ell',
}


def presentation_form(expression):
    return MATH_PRESENTATION.get(expression, expression).replace(
        r'\lambda_\max', r'\lambda_{\max}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blocks(markdown):
    lines = markdown.splitlines()
    if lines[0] == '---':
        lines = lines[lines.index('---', 1)+1:]
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.startswith('# ') or line.startswith('Editorial revision:'):
            index += 1
            continue
        if line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].startswith('|'):
                row = [c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',c) for c in row):
                    rows.append(row)
                index += 1
            yield 'table', rows
            continue
        if line.startswith('!['):
            yield 'figure', line
        elif line.startswith('##'):
            heading = re.match(r'(#+)\s+(.*)', line)
            yield ('section' if len(heading[1]) == 2 else 'subsection'), heading[2]
        else:
            yield 'paragraph', line
        index += 1


def main():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import mathtext
    from matplotlib.font_manager import FontProperties
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
        NextPageTemplate, FrameBreak, Spacer, Table, TableStyle, Image, KeepTogether)
    outputs = ['manuscript.pdf','manuscript.tex','manuscript.md','format-manifest.json','math','figures','paper_protocol_v2.bib']
    if any((OUT/name).exists() for name in outputs):
        raise FileExistsError('Refusing to overwrite a previous formatted edition')
    protected = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    markdown = (SOURCE/'manuscript.md').read_text(encoding='utf-8')
    shutil.copyfile(SOURCE/'manuscript.md', OUT/'manuscript.md')
    shutil.copyfile(SOURCE/'paper_protocol_v2.bib', OUT/'paper_protocol_v2.bib')
    (OUT/'math').mkdir()
    (OUT/'figures').mkdir()
    shutil.copyfile(SOURCE/'report.json',OUT/'report.json')
    shutil.copyfile(SOURCE/'figures/provenance.json',OUT/'figures/provenance.json')
    provenance=json.loads((SOURCE/'figures/provenance.json').read_text(encoding='utf-8'))
    drift_input=next(i for i in provenance['inputs'] if i['run_id'].startswith('baselines-'))
    from matplotlib import pyplot as plt
    plot_labels={'svd':'SVD','asvd':'ASVD component','whitening':'Whitening','anchored':'Anchor','legacy_ridge':'Extra ridge'}
    palette=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
    fig,ax=plt.subplots(figsize=(3.25,2.55),layout='constrained')
    for color,series in zip(palette,drift_input['figure_data']):
        x=list(range(len(series['mean'])))
        for seed in series['per_seed']:
            ax.plot(x,seed['values'],color=color,linestyle=':',linewidth=.5,alpha=.5)
        ax.plot(x,series['mean'],color=color,linewidth=1,label=plot_labels[series['specification']['method']])
        ax.fill_between(x,[m-s for m,s in zip(series['mean'],series['sample_std'])],
            [m+s for m,s in zip(series['mean'],series['sample_std'])],color=color,alpha=.12)
    ax.set_xlabel('Block index',fontsize=8)
    ax.set_ylabel('Normalized residual error',fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.2,linewidth=.4)
    ax.legend(fontsize=6,frameon=False,loc='upper left')
    fig.savefig(OUT/'figures/gemma-drift.png',dpi=360)
    plt.close(fig)
    helper = import_helper('reference_latex',ROOT/'docs/export_paper_latex_20261014.py')
    bibliography = helper.bibliography_entries((OUT/'paper_protocol_v2.bib').read_text(encoding='utf-8'))
    numbers = {key:i for i,(key,_) in enumerate(bibliography,1)}
    assert set(re.findall(r'\[@([^\]]+)\]',markdown)) == set(numbers)
    for name, file in [('PaperTimes','times.ttf'),('PaperTimes-Bold','timesbd.ttf'),
                       ('PaperTimes-Italic','timesi.ttf'),('PaperTimes-BoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    pdfmetrics.registerFontFamily('PaperTimes',normal='PaperTimes',bold='PaperTimes-Bold',
                                 italic='PaperTimes-Italic',boldItalic='PaperTimes-BoldItalic')
    matplotlib.rcParams.update({'mathtext.fontset':'stix','font.family':'serif'})
    width,height = A4
    margin,gap = 54,18
    usable = width-2*margin
    column = (usable-gap)/2
    styles = {
        'body':ParagraphStyle('Body',fontName='PaperTimes',fontSize=10,leading=13,autoLeading='max',
            alignment=TA_JUSTIFY,spaceAfter=5,splitLongWords=True),
        'title':ParagraphStyle('Title',fontName='PaperTimes-Bold',fontSize=17,leading=19,
            alignment=TA_CENTER,spaceAfter=13),
        'author':ParagraphStyle('Author',fontName='PaperTimes-Bold',fontSize=10,leading=12,
            alignment=TA_CENTER,spaceAfter=2),
        'links':ParagraphStyle('Links',fontName='Courier',fontSize=8,leading=10,
            alignment=TA_CENTER,spaceAfter=10,textColor=colors.blue),
        'abstract_heading':ParagraphStyle('AbstractHeading',fontName='PaperTimes-Bold',fontSize=10,
            leading=12,alignment=TA_CENTER,spaceBefore=8,spaceAfter=9),
        'abstract':ParagraphStyle('Abstract',fontName='PaperTimes-Italic',fontSize=10,leading=12,
            alignment=TA_JUSTIFY,spaceAfter=8),
        'section':ParagraphStyle('Section',fontName='PaperTimes-Bold',fontSize=10,leading=12,
            alignment=TA_CENTER,spaceBefore=14,spaceAfter=8,keepWithNext=True),
        'subsection':ParagraphStyle('Subsection',fontName='PaperTimes-BoldItalic',fontSize=10,
            leading=12,spaceBefore=10,spaceAfter=6,keepWithNext=True),
        'caption':ParagraphStyle('Caption',fontName='PaperTimes',fontSize=8,leading=9.5,
            alignment=TA_JUSTIFY,spaceAfter=5),
        'cell':ParagraphStyle('Cell',fontName='PaperTimes',fontSize=7,leading=8.4,spaceAfter=0),
        'reference':ParagraphStyle('Reference',fontName='PaperTimes',fontSize=8,leading=9.6,
            leftIndent=14,firstLineIndent=-14,alignment=TA_JUSTIFY,spaceAfter=5,splitLongWords=True),
    }
    math_cache = {}
    math_records = []
    math_depths = {}
    display_records = []
    math_parser = mathtext.MathTextParser('path')
    equation_number = 0
    def is_display(expr, measured_width):
        return expr in MATH_PRESENTATION or measured_width>column*.72 or expr.startswith(('J_\\beta(',r'\|Y_b-AX\|',r'A_\beta=',r'D_\ell='))
    def display_form(expr):
        return presentation_form(expr)
    def math_asset(expr,size=10.5):
        key=(expr,size)
        if key in math_cache:return math_cache[key]
        # Mathtext and TeX use equivalent formatting for the supported notation.
        normalized=presentation_form(expr)
        path=OUT/'math'/f'{hashlib.sha256((str(size)+normalized).encode()).hexdigest()[:16]}.png'
        mathtext.math_to_image('$'+normalized+'$',path,prop=FontProperties(size=size),dpi=360,format='png',color='black')
        with PILImage.open(path) as im:
            w,h=im.size
        dims=(path,w*72/360,h*72/360)
        math_cache[key]=dims
        depth=float(math_parser.parse('$'+normalized+'$',dpi=72,prop=FontProperties(size=size)).depth)
        math_depths[key]=depth
        math_records.append({'expression':expr,'presentation':normalized,
            'file':str(path.relative_to(OUT)),'size':size,'baseline_depth_pt':depth})
        return dims
    def inline(text,size=10):
        tokens=[]
        def hold(value):
            tokens.append(value)
            return 'PAPERTOKEN'+str(len(tokens)-1)+'END'
        text=re.sub(r'\$([^$]+)\$',lambda m: hold(math_markup(m[1],size)),text)
        text=re.sub(r'\[@([^\]]+)\]',lambda m: hold(f'<link href="#ref-{m[1]}" color="blue">[{numbers[m[1]]}]</link>'),text)
        text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m: hold(f'<link href="{escape(m[2],quote=True)}" color="blue">{escape(m[1])}</link>'),text)
        text=re.sub(r'`([^`]+)`',lambda m: hold('<font name="Courier" size="7.2">'+escape(m[1])+'</font>'),text)
        text=escape(text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        for i,value in enumerate(tokens):text=text.replace('PAPERTOKEN'+str(i)+'END',value)
        return text
    def math_markup(expr,size):
        size+=.5
        path,w,h=math_asset(expr,size)
        scale=min(1,(column-6)/w)
        depth=math_depths[(expr,size)]*scale
        return f'<img src="{path.as_posix()}" width="{w*scale:.3f}" height="{h*scale:.3f}" valign="{-depth:.3f}"/>'
    def paragraph(text, style=None):
        style=style or styles['body']
        # Long equations are displayed rather than forcing them through a column.
        parts=re.split(r'(\$[^$]+\$)',text)
        pending=''
        result=[]
        for index,part in enumerate(parts):
            if part.startswith('$'):
                expr=part[1:-1]
                path,w,h=math_asset(expr)
                if is_display(expr,w):
                    if pending.strip():result.append(Paragraph(inline(pending),style));pending=''
                    punctuation=''
                    if index+1<len(parts):
                        match=re.match(r'^([,.;:])\s*',parts[index+1])
                        if match:
                            punctuation=match[1]
                            parts[index+1]=parts[index+1][match.end():]
                    result.extend(display_equation(expr,punctuation))
                    continue
            pending+=part
        if pending.strip():result.append(Paragraph(inline(pending),style))
        return result
    def display_equation(expr,punctuation=''):
        nonlocal equation_number
        equation_number+=1
        # Fractions keep the exact numerator/denominator readable in narrow columns.
        original=expr
        expr=display_form(expr)+punctuation
        size=11.5
        path,w,h=math_asset(expr,size)
        if w>column-23:
            size=10.5
            path,w,h=math_asset(expr,size)
        if w>column-23:
            raise ValueError('Equation needs a line break rather than shrinking: '+expr)
        display_records.append({'number':equation_number,'source':original,'presentation':expr,
            'font_size_pt':size,'width_pt':w,'height_pt':h,'scale':1})
        im=Image(str(path),width=w,height=h)
        t=Table([[im,Paragraph(f'({equation_number})',styles['caption'])]],colWidths=[column-20,20])
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),
            ('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),0)]))
        return [Spacer(1,6),t,Spacer(1,9)]
    parsed=list(blocks(markdown))
    abstract=next(value for i,(kind,value) in enumerate(parsed) if kind=='paragraph' and parsed[i-1]==('section','Abstract'))
    keywords=next(value for kind,value in parsed if kind=='paragraph' and value.startswith('Keywords:'))
    header=[Paragraph(TITLE,styles['title']),Paragraph(AUTHOR,styles['author']),
        Paragraph(f'<link href="{REPO}" color="blue">{REPO}</link><br/><link href="{SITE}" color="blue">{SITE}</link>',styles['links']),
        Paragraph('Abstract',styles['abstract_heading']),Paragraph(inline(abstract),styles['abstract']),
        Paragraph('<b>Keywords:</b>'+escape(keywords[len('Keywords:'):]),styles['body']),Spacer(1,8)]
    header_height=sum(item.wrap(usable,10000)[1]+item.getSpaceBefore()+item.getSpaceAfter() for item in header)+7
    first_body_top=height-margin-header_height
    def frame(x,y,w,h,id):return Frame(x,y,w,h,id=id,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    def footer(canvas,doc):
        canvas.setFont('PaperTimes',10)
        canvas.drawCentredString(width/2,28,str(doc.page))
        if doc.page==1:
            canvas.setStrokeColor(colors.HexColor('#777777'));canvas.setLineWidth(.45)
            canvas.line(margin,first_body_top+4,width-margin,first_body_top+4)
    document=BaseDocTemplate(str(OUT/'manuscript.pdf'),pagesize=A4,title=TITLE,author=AUTHOR,
        leftMargin=margin,rightMargin=margin,topMargin=margin,bottomMargin=margin)
    document.addPageTemplates([
        PageTemplate('First',[frame(margin,first_body_top+7,usable,header_height-7,'Title'),
            frame(margin,margin,column,first_body_top-margin,'FirstLeft'),
            frame(margin+column+gap,margin,column,first_body_top-margin,'FirstRight')],onPage=footer),
        PageTemplate('Body',[frame(margin,margin,column,height-2*margin,'Left'),
            frame(margin+column+gap,margin,column,height-2*margin,'Right')],onPage=footer)])
    story=header+[NextPageTemplate('Body'),FrameBreak()]
    table_count=0
    for kind,value in parsed:
        if value=='Abstract' or value==abstract or value==keywords:continue
        if kind in ['section','subsection']:
            story.append(Paragraph(escape(value),styles[kind]))
        elif kind=='table':
            table_count+=1
            count=len(value[0])
            widths=[column/count]*count
            if table_count==1:widths=[column*.30,column*.28,column*.28,column*.14]
            if table_count==2:widths=[column*.15,column*.29,column*.33,column*.23]
            if table_count==3:widths=[column*.21,column*.27,column*.27,column*.25]
            if table_count==4:widths=[column*.24,column*.10,column*.22,column*.22,column*.22]
            if table_count==5:widths=[column*.48,column*.26,column*.26]
            cells=[[Paragraph(inline(c,7),styles['cell']) for c in row] for row in value]
            cells[0]=[Paragraph('<b>'+inline(c,7)+'</b>',styles['cell']) for c in value[0]]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),
                ('LINEBELOW',(0,0),(-1,0),.35,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),2),
                ('RIGHTPADDING',(0,0),(-1,-1),2),('TOPPADDING',(0,0),(-1,-1),3),
                ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
            story.extend([t,Spacer(1,7)])
        elif kind=='figure':
            path=OUT/'figures/gemma-drift.png'
            with PILImage.open(path) as image:w,h=image.size
            caption=re.match(r'!\[([^\]]+)\]',value)[1]
            story.append(KeepTogether([Image(str(path),width=column,height=h*column/w),
                Paragraph(inline(caption,8),styles['caption']),Spacer(1,6)]))
        else:
            story.extend(paragraph(value))
    story.append(Paragraph('References',styles['section']))
    for key,fields in bibliography:
        authors=fields.get('author','').split(' and ')
        author_text=', '.join(authors[:3])+(' et al.' if len(authors)>3 else '')
        venue=fields.get('journal') or fields.get('booktitle') or ('arXiv:'+fields.get('eprint','') if fields.get('eprint') else '')
        text=f'<a name="ref-{key}"/>[{numbers[key]}] '+escape(author_text)+'. '+escape(fields.get('title',''))+'. '+escape(venue)+', '+escape(fields.get('year',''))+'.'
        if fields.get('url'):text+=' <link href="'+escape(fields['url'],quote=True)+'" color="blue">Primary source</link>.'
        story.append(Paragraph(text,styles['reference']))
    document.build(story)
    # Standalone source uses the same two-column visual system and full-width abstract.
    original_tex_inline=helper.inline
    def tex_inline(text):
        text=re.sub(r'\$([^$]+)\$',lambda m:'$'+presentation_form(m[1])+'$',text)
        return original_tex_inline(text)
    helper.inline=tex_inline
    tex=[r'\documentclass[10pt,a4paper,twocolumn]{article}',r'\usepackage[T1]{fontenc}',
        r'\usepackage{mathptmx,amsmath,amssymb,array,booktabs,graphicx}',
        r'\usepackage[margin=0.75in]{geometry}',r'\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}',
        r'\setlength{\columnsep}{0.25in}',r'\setlength{\emergencystretch}{2em}',r'\sloppy',
        r'\makeatletter',r'\renewcommand\section{\@startsection{section}{1}{0pt}{2.2ex plus 1ex minus .2ex}{1.1ex plus .2ex}{\normalfont\normalsize\bfseries\centering}}',
        r'\renewcommand\subsection{\@startsection{subsection}{2}{0pt}{1.8ex plus .8ex minus .2ex}{.8ex plus .2ex}{\normalfont\normalsize\bfseries\itshape}}',r'\makeatother',
        r'\begin{document}',r'\twocolumn[{\centering',r'{\Large\bfseries '+helper.escape(TITLE)+r'\par}',
        r'\vspace{10pt}{\bfseries '+AUTHOR+r'\par}',r'\small\url{'+REPO+r'}\par\url{'+SITE+r'}\par',
        r'\vspace{13pt}\normalsize\textbf{Abstract}\par\vspace{7pt}',r'\begin{minipage}{\textwidth}\itshape '+helper.inline(abstract)+r'\end{minipage}\par',
        r'\vspace{8pt}\noindent\textbf{Keywords:} '+helper.escape(keywords[len('Keywords:'):])+r'\par',r'\vspace{9pt}\hrule\vspace{12pt}}]']
    table_index=0
    for kind,value in parsed:
        if value in ['Abstract',abstract,keywords]:continue
        if kind in ['section','subsection']:
            title=re.sub(r'^\d+(?:\.\d+)*\.?\s+','',value)
            tex.append('\\'+kind+'{'+helper.inline(title)+'}')
        elif kind=='table':
            table_index+=1
            columns=len(value[0])
            fraction=.90/columns
            tex += [r'{\scriptsize\setlength{\tabcolsep}{2pt}',r'\noindent\begin{tabular}{@{}'+('p{'+f'{fraction:.4f}'+r'\columnwidth}')*columns+r'@{}}\toprule']
            for i,row in enumerate(value):
                tex.append(' & '.join(helper.inline(c) for c in row)+r' \\')
                if i==0:tex.append(r'\midrule')
            tex += [r'\bottomrule\end{tabular}}\par\medskip']
        elif kind=='figure':
            # The standalone compiler has no additional-file support. Preserve
            # exact measured coordinates as a native picture instead of an external image.
            provenance=json.loads((SOURCE/'figures/provenance.json').read_text(encoding='utf-8'))
            item=next(v for v in provenance['inputs'] if v['run_id'].startswith('baselines-'))
            labels={'svd':'SVD','asvd':'ASVD component','whitening':'Whitening','anchored':'Anchor','legacy_ridge':'Extra ridge'}
            series=[(labels[s['specification']['method']],list(enumerate(s['mean']))) for s in item['figure_data']]
            native=helper.picture(series,'Block index','Normalized residual squared error')
            tex += [r'\begin{figure}[t]\centering\resizebox{\columnwidth}{!}{\begin{minipage}{350pt}',native,
                    r'\end{minipage}}\caption{Gemma residual drift from the same saved coordinates. Individual-seed variation is retained in the companion PDF figure and original evidence.}\end{figure}']
        else:
            parts=re.split(r'(\$[^$]+\$)',value)
            pending=''
            for index,part in enumerate(parts):
                if part.startswith('$'):
                    expr=part[1:-1]
                    _,w,_=math_asset(expr)
                    if is_display(expr,w):
                        if pending.strip():tex.append(helper.inline(pending.rstrip()));pending=''
                        punctuation=''
                        if index+1<len(parts):
                            match=re.match(r'^([,.;:])\s*',parts[index+1])
                            if match:
                                punctuation=match[1]
                                parts[index+1]=parts[index+1][match.end():]
                        tex.append(r'\begin{equation}'+display_form(expr)+punctuation+r'\end{equation}')
                        continue
                pending+=part
            if pending.strip():tex.append(helper.inline(pending)+r'\par\smallskip')
    tex.extend([r'\begin{thebibliography}{99}'])
    for key,fields in bibliography:
        authors=fields.get('author','').split(' and ')
        author_text=', '.join(authors[:3])+(' et al.' if len(authors)>3 else '')
        venue=fields.get('journal') or fields.get('booktitle') or ('arXiv:'+fields.get('eprint','') if fields.get('eprint') else '')
        text=author_text+'. '+fields.get('title','')+'. '+venue+', '+fields.get('year','')+'.'
        tex.append(r'\bibitem{'+key+'} '+helper.escape(text)+(r' \href{'+fields['url']+r'}{Primary source}.' if fields.get('url') else ''))
    tex += [r'\end{thebibliography}',r'\end{document}']
    (OUT/'manuscript.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    assert all(sha(ROOT/path)==digest for path,digest in protected.items())
    manifest={'generated_utc':datetime.now(timezone.utc).isoformat(),'kind':'future_scope_without_appendices',
        'reference_path':str(REFERENCE),'reference_sha256':sha(REFERENCE),
        'author_and_project_links':'Copied from user-supplied reference; repository URL matches Git origin.',
        'layout':{'page':'A4','columns':2,'body_font':'Times New Roman','body_size_pt':10,'leading_pt':13,
                  'margin_pt':54,'column_gap_pt':18,'first_page_header_pt':header_height},
        'source_manuscript_sha256':sha(SOURCE/'manuscript.md'),
        'scientific_text_unchanged':sha(OUT/'manuscript.md')==sha(SOURCE/'manuscript.md'),
        'table_count_preserved':table_count,'citation_keys_preserved':sorted(numbers),
        'research_experiments_run':False,'protected_source_hashes':protected,
        'paper_appendices':False,'evidence_intervals_retained_in':'report.json',
        'math_assets':math_records,'numbered_pdf_equations':equation_number,
        'math_presentation_changes':MATH_PRESENTATION,'display_equations':display_records,
        'math_typography':{'inline_size_pt':10.5,'display_preferred_size_pt':11.5,
            'display_minimum_size_pt':10.5,'display_shrink_allowed':False,
            'inline_alignment':'Actual math baseline depth; dynamic line height'},
        'pdf_build':'ReportLab with 360-dpi STIX math images; independent of LaTeX',
        'tex_build':'Standalone LaTeX; native measured-coordinate plot avoids external assets',
        'figure_difference':'PDF retains original individual-seed plot; TeX embeds saved mean curves with explicit caption qualification.',
        'artifacts':{name:{'sha256':sha(OUT/name),'bytes':(OUT/name).stat().st_size}
                     for name in ['manuscript.md','manuscript.pdf','manuscript.tex','paper_protocol_v2.bib','build_paper.py']}}
    (OUT/'format-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(OUT),'equations':equation_number,'tables':table_count,'science_unchanged':True},indent=2))


if __name__=='__main__':main()

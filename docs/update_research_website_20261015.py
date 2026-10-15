"""Sync the existing Pages site to the current paper; never run experiments."""
from decimal import Decimal
from html import escape
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/paper-future-scope-20261015'
SITE=ROOT/'results/research-dashboard-20260815'
REPO='https://github.com/Sambhav242005/Weighted-Lowrank-Compression'
PAPER_URL=REPO+'/blob/main/docs/paper-future-scope-20261015/ieee-conference.tex'
REVIEW_URL=REPO+'/blob/main/docs/paper-future-scope-20261015/manuscript.pdf'
EVIDENCE_URL=REPO+'/blob/main/results/paper-v2/revised-report-20261014-151432-07c08d61/report.md'
PUBLICATION_URL=REPO+'/blob/main/docs/paper-ieee-preparation-20261015/IEEE_PUBLICATION_RESEARCH.md'
TITLE='Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost'

EXTRA_CSS='''
    /* current-evidence */
    a{color:var(--link,#0366d6)}
    .skip-link{position:fixed;left:12px;top:-100px;z-index:200;background:white;padding:10px 16px}
    .skip-link:focus{top:12px}
    .page-nav{display:flex;gap:10px 20px;flex-wrap:wrap;justify-content:center;margin-top:20px;font-size:16px}
    .page-nav a{padding:4px 0;min-height:24px}
    .status-note{border-left:3px solid #00628f;background:#f5f9fb;padding:14px 18px;margin:18px 0}
    .chart-viewport{overflow-x:auto;margin:16px 0 8px;border:1px solid var(--rule,#e2e8f0);background:#fff}
    .chart-viewport svg{min-width:820px;overflow:hidden}
    .chart-viewport:focus-visible{outline:2px solid #00628f;outline-offset:3px}
    .chart figure,figure.chart{margin:24px 0}
    figcaption{font-size:16px;line-height:1.6;margin:12px 0}
    caption{text-align:left;font-weight:600;padding:8px 0;caption-side:top}
    th[scope=row]{text-transform:none;letter-spacing:normal;font:inherit;color:var(--ink);background:none}
    summary{min-height:28px;padding:4px 0}
    .table-wrap{max-width:100%;overflow-x:auto}
    .table-wrap table{min-width:540px}
    .wrap ul,.wrap ol{margin-left:0;padding-left:24px;margin-bottom:14px}
    .small{font-size:15px;color:var(--muted)}
    .equation{overflow-x:auto;padding:16px;background:#f5f5f5;font-size:18px}
    .viz-controls{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
    .viz-controls[hidden],.viz-steps[hidden]{display:none}
    button{min-height:32px;padding:6px 12px;border:1px solid #777;background:#fff;color:#1a1a1a;font:inherit;cursor:pointer}
    .viz-step{border:0;cursor:pointer;font-family:inherit;min-height:28px}
    .viz-fallback{padding:24px;color:white;font-size:16px;max-width:650px}
    @media(max-width:600px){.title-block h1{font-size:29px}.wrap,.wrap-wide{width:calc(100% - 32px)}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_table(tex,name):
    text=re.search(r'\\pgfplotstableread\{([^{}]*)\}\\'+name+r'\b',tex,re.S)[1]
    lines=[line.split() for line in text.strip().splitlines()]
    return [dict(zip(lines[0],row)) for row in lines[1:]]


def html_table(headers,rows,caption):
    head=''.join('<th scope="col">'+escape(h)+'</th>' for h in headers)
    body=''
    for row in rows:
        body+='<tr><th scope="row">'+escape(str(row[0]))+'</th>'
        body+=''.join('<td class="num">'+escape(str(c))+'</td>' for c in row[1:])+'</tr>'
    return '<div class="table-wrap"><table><caption>'+escape(caption)+'</caption><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def svg_text(x,y,text,anchor='start',size=17,color='#1a1a1a',weight='400'):
    return f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(str(text))}</text>'


def line(x1,y1,x2,y2,color='#d9dfe3',dash=False):
    extra=' stroke-dasharray="5 5"' if dash else ''
    return f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="1.2"{extra}/>'


def point(x,y,raw,error=None,error_scale=1,color='#00628f'):
    result=''
    if error is not None:
        width=float(error)*error_scale
        result+=line(x-width,y,x+width,y,color)
        result+=line(x-width,y-5,x-width,y+5,color)+line(x+width,y-5,x+width,y+5,color)
    return result+f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.5" fill="{color}"><title>{escape(str(raw))}</title></circle>'


def horizontal_bar(x0,x,y,raw,error=None,error_scale=1,color='#00628f'):
    rect=f'<rect x="{x0:.3f}" y="{y-9:.3f}" width="{max(0,x-x0):.3f}" height="18" fill="{color}" fill-opacity="0.75"><title>{escape(str(raw))}</title></rect>'
    return rect+point(x,y,raw,error,error_scale,color='#1a1a1a')


def svg_start(key,title,description,height):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="img" aria-labelledby="{key}-title" aria-describedby="{key}-desc" font-family="Georgia,serif"><title id="{key}-title">{escape(title)}</title><desc id="{key}-desc">{escape(description)}</desc>'


def axis(x,y,width,minimum,maximum,ticks,top,label):
    scale=lambda value:x+(float(value)-minimum)/(maximum-minimum)*width
    result=line(x,y,x+width,y,'#6a737d')
    for tick in ticks:
        pos=scale(tick)
        result+=line(pos,top,pos,y)+svg_text(pos,y+25,f'{tick:g}','middle',16)
    result+=svg_text(x+width/2,y+55,label,'middle',18)
    return result,scale


def gemma_svg(rows):
    description='Horizontal bars compare five recorded Gemma 3-1B methods. Ours (anchored) is orange. All bars start at zero. Whitening and ours have close means; seed variation remains important.'
    result=svg_start('gemma','Gemma 3-1B horizontal comparison',description,455)
    for metric,sd,x,lo,hi,ticks,title,label in [
        ('nll','nllsd',230,0,6.5,[0,2,4,6],'Prediction loss','NLL — lower is better'),
        ('drift','driftsd',640,0,.1,[0,.025,.05,.075,.1],'Internal representation drift','Residual drift — lower is better')]:
        grid,scale=axis(x,365,270,lo,hi,ticks,75,label)
        result+=grid+svg_text(x+135,40,title,'middle',19,weight='600')
        for index,row in enumerate(rows):
            y=100+index*56
            if metric=='nll':result+=svg_text(210,y+6,row['name'].replace('Diagonal ASVD component','Diagonal ASVD'),'end',17)
            color='#b35400' if row.get('ours') else ('#888' if row['seeds']==1 else '#00628f')
            result+=horizontal_bar(scale(0),scale(row[metric]),y,row[metric],row.get(sd),270/(hi-lo),color)
            result+=svg_text(scale(row[metric])+9,y+5,row[metric],size=15)
        if metric=='nll':
            result+=line(scale(3.451563),75,scale(3.451563),365,'#333',True)
    return result+'</svg>'


def composition_svg(rows):
    result=svg_start('composition','Layer-group changes relative to whitening',
        'Ours (anchored) minus whitening, paired means over three calibration seeds. Values left of zero favor ours. Blocks 11–14 worsen on both metrics; all 26 has a small average improvement that does not hold in every group.',760)
    for metric,x,lo,hi,ticks,mult,title,label in [
        ('nll',230,-8,3.8,[-8,-6,-4,-2,0,2],1000,'Prediction-loss change','ΔNLL, in units of 0.001'),
        ('drift',640,-16,9,[-15,-10,-5,0,5],100000,'Residual-drift change','Δ drift, in units of 0.00001')]:
        grid,scale=axis(x,675,270,lo,hi,ticks,70,label)
        result+=grid+svg_text(x+135,38,title,'middle',19,weight='600')
        result+=line(scale(0),70,scale(0),675,'#555',True)
        for index,row in enumerate(rows):
            y=95+index*58
            value=Decimal(row[metric])*mult
            if metric=='nll':result+=svg_text(210,y+6,row['name']+' ('+row['jointly']+'/3)','end',17)
            color='#b33' if value>0 else '#00628f'
            result+=point(scale(value),y,row[metric],color=color)
            result+=svg_text(scale(value),y+22,f'{value:+.4f}','middle',15,color)
    return result+'</svg>'


def hybrid_svg(models,controls):
    result=svg_start('hybrid','Hybrid-model comparisons',
        'Horizontal bars show dense reference, plain SVD, whitening and orange Ours (anchored) for each tested model. Repeated fits include sample seed standard deviations. Dense and SVD controls are recorded once.',750)
    specifications=[('Qwen','Qwen3.5-0.8B',483,0,4.3,[0,1,2,3,4],0,.09,[0,.03,.06,.09]),
        ('LFM','LFM2.5-1.2B',725,0,5.5,[0,1,2,3,4,5],0,.32,[0,.1,.2,.3])]
    for row_index,(name,display,rank,nlo,nhi,nticks,dlo,dhi,dticks) in enumerate(specifications):
        top=40+row_index*350
        for metric,x,lo,hi,ticks,title in [('nll',180,nlo,nhi,nticks,'NLL'),('drift',660,dlo,dhi,dticks,'Residual drift')]:
            grid,scale=axis(x,top+265,230,lo,hi,ticks,top+28,title+' — lower is better')
            result+=grid+svg_text(x+115,top,display+', rank '+str(rank),'middle',19,weight='600')
            for index,data in enumerate(controls[name]+models[name]):
                y=top+40+index*55
                method=['Dense reference','Plain SVD','Whitening','Ours (anchored)'][index]
                result+=svg_text(x-12,y+5,method,'end',17)
                color='#b35400' if index==3 else ('#888' if index<2 else '#00628f')
                sd=data.get('nllsd' if metric=='nll' else 'driftsd')
                result+=horizontal_bar(scale(0),scale(data[metric]),y,data[metric],sd,230/(hi-lo),color)
                result+=svg_text(scale(data[metric])+8,y+5,f'{float(data[metric]):.6f}',size=15)
    return result+'</svg>'


def figure(key,title,svg,caption,headers,rows):
    return f'''<figure class="chart" id="{key}"><h3>{escape(title)}</h3>
    <div class="chart-viewport" tabindex="0" role="region" aria-label="{escape(title)}; horizontally scrollable">{svg}</div>
    <figcaption>{caption}</figcaption>
    <details><summary>Exact chart values</summary>{html_table(headers,rows,title+' — original reported values')}</details>
    </figure>'''


def section(id,title,content,wide=False):
    return f'<section id="{id}"><div class="'+('wrap-wide' if wide else 'wrap')+f'"><h2>{title}</h2>{content}</div></section>\n'


def main():
    protected={str(p.relative_to(ROOT)):sha(p) for p in [PAPER/'ieee-conference.tex',PAPER/'manuscript.md',PAPER/'manuscript.pdf',PAPER/'report.json',PAPER/'figures/provenance.json']}
    tex=(PAPER/'ieee-conference.tex').read_text(encoding='utf-8')
    abstract=(PAPER/'abstract.md').read_text(encoding='utf-8').strip()
    repeated=block_table(tex,'GemmaRepeatedResults')
    names=['Diagonal ASVD component','Covariance whitening','Ours (anchored)','Extra ridge']
    gemma=[{'name':'Plain SVD','nll':re.search(r'\((5\.\d+),5\)',tex)[1],
        'drift':re.search(r'\((0\.074\d+),5\)',tex)[1],'seeds':1}]
    gemma+=[dict(row,name=name,seeds=3) for row,name in zip(repeated,names)]
    for row in gemma:row['ours']=row['name']=='Ours (anchored)'
    groups=['0','12','25','0–1','12–13','24–25','0–3','11–14','22–25','All 26']
    composition=[dict(row,name=name) for row,name in zip(block_table(tex,'CompositionResults'),groups)]
    hybrid={'Qwen':block_table(tex,'QwenMeans'),'LFM':block_table(tex,'LfmMeans')}
    hybrid_controls={'Qwen':block_table(tex,'QwenControls'),'LFM':block_table(tex,'LfmControls')}
    model_provenance=json.loads((PAPER/'model-chart-data.json').read_text(encoding='utf-8'))
    assert len(gemma)==5 and len(composition)==10 and all(len(rows)==2 for rows in hybrid.values())
    tables=re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',tex,re.S)
    assert len(tables)==2
    normal=[]
    for table in tables:
        body=table.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
        normal.append([[c.strip() for c in row.strip().removesuffix(r'\\').split('&')]
            for row in body.splitlines() if '&' in row])
    data={'updated':'2026-10-15','source_paper':str((PAPER/'ieee-conference.tex').relative_to(ROOT)),
        'source_paper_sha256':sha(PAPER/'ieee-conference.tex'),'evidence_status':'exploratory',
        'gemma':gemma,'composition':composition,'hybrid':hybrid,'hybrid_controls':hybrid_controls,
        'measured_method_label':'Ours (anchored)',
        'chart_orientation':'Horizontal bars for Gemma and hybrid model comparisons',
        'hybrid_model_provenance':model_provenance['models'],
        'bf16_conditions':normal[0],'latency_ratios':normal[1],
        'whole_checkpoint_bytes':{'dense':1999883711,'compressed':1982003507,'saved':17880204},
        'network_proposal_status':'not implemented or evaluated','research_experiments_run':False}
    (SITE/'research-data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    gemma_rows=[[r['name'],r['nll']+(' ± '+r['nllsd'] if 'nllsd' in r else ' (single fit)'),
        r['drift']+(' ± '+r['driftsd'] if 'driftsd' in r else ' (single fit)'),r['seeds']] for r in gemma]
    composition_rows=[[r['name'],r['nll'],r['drift'],r['jointly']+'/3'] for r in composition]
    hybrid_rows=[[name+' / '+method,row['nll']+(' ± '+row['nllsd'] if 'nllsd' in row else ' (single control)'),
        row['drift']+(' ± '+row['driftsd'] if 'driftsd' in row else ' (single control)'),
        '3' if 'nllsd' in row else '1'] for name in hybrid
        for method,row in zip(['Dense reference','Plain SVD','Whitening','Ours (anchored)'],hybrid_controls[name]+hybrid[name])]
    gemma_chart=figure('gemma-results','Gemma 3-1B: horizontal prediction-loss and drift bars',gemma_svg(gemma),
        'Horizontal bars start at zero. Orange identifies Ours (anchored), the measured β = 0.03 fit already reported in the paper. The same rank-384 factor budget is used for all compressed methods. Whiskers are sample standard deviations across three calibration seeds; the single SVD fit has no error bar. The dashed line marks the dense teacher’s NLL, 3.451563. Extra ridge uses β = 0.1 and ρ = 0.01. The historical FP16 numerical-readiness qualification still applies.',
        ['Method','NLL','Residual drift','Seeds'],gemma_rows)
    composition_chart=figure('composition-results','Layer groups: Ours (anchored) minus whitening',composition_svg(composition),
        'Negative differences favor the anchor. Axis units are 0.001 for NLL and 0.00001 for drift; annotations use those units and are rounded to four decimal places. Parentheses show seeds with both lower NLL and lower drift, out of three. They do not indicate statistical significance. Blocks 11–14 worsen on both metrics in all three seeds.',
        ['Blocks','Mean NLL difference','Mean drift difference','Jointly better'],composition_rows)
    hybrid_chart=figure('hybrid-results','Other models: Qwen3.5-0.8B and LFM2.5-1.2B',hybrid_svg(hybrid,hybrid_controls),
        'Horizontal bars start at zero; orange identifies Ours (anchored). Whitening and Ours show three-seed means with sample standard-deviation whiskers. Dense reference and plain SVD are single recorded controls with no error bars. The dense model is an uncompressed reference. The controls and uncertainty values come from the same saved test records, not new experiments. Qwen: rank 483, 8/8 primary-test records and 10/18 validation records. LFM: rank 725, 8/8 test and 18/18 validation records. These differences do not isolate architecture effects.',
        ['Model / method','NLL','Residual drift','Seeds'],hybrid_rows)
    old=(SITE/'index.html').read_text(encoding='utf-8')
    reader=(SITE/'paper-reader.html').read_text(encoding='utf-8')
    def head(document,title,description):
        text=document.split('<body>',1)[0]
        text=re.sub(r'<title>.*?</title>','<title>'+escape(title)+'</title>',text,flags=re.S)
        text=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="'+escape(description,quote=True)+'">',text)
        if '/* current-evidence */' in text:
            text=re.sub(r'\s*/\* current-evidence \*/.*?</style>',lambda m:EXTRA_CSS+'\n  </style>',text,flags=re.S)
        else:text=text.replace('</style>',EXTRA_CSS+'\n  </style>',1)
        return text+'<body>\n<a class="skip-link" href="#content">Skip to main content</a>\n'
    script=re.findall(r'<script>(.*?)</script>',old,re.S)[-1]
    if 'var userPaused' not in script:
        script=script.replace("  var scene = new THREE.Scene();",'''  if (typeof THREE === 'undefined') {
    label.textContent = 'The optional 3D illustration is unavailable. All results and charts below remain readable.';
    document.getElementById('viz-controls').hidden = true;
    return;
  }
  var scene = new THREE.Scene();''')
        script=script.replace('  var renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });','''  var renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
  catch (error) {
    label.textContent = 'WebGL is unavailable. All measured results and charts below remain readable.';
    document.getElementById('viz-controls').hidden = true;
    return;
  }''')
        script=script.replace('Step 3 - Plain SVD: rigid flat plane','Step 3 - Weight-space approximation')
        script=script.replace('Step 4 - Weighted fit bends to traffic','Step 4 - Activation-weighted geometry')
        script=script.replace('Step 5 - Same budget, different result','Step 5 - Global behavior still needs testing')
        start=script.index('  var paused = false;')
        end=script.index('  var lastCycle',start)
        script=script[:start]+'''  var userPaused = prefersReducedMotion;
  var hoverPaused = false;
  var pauseButton = document.getElementById('toggle-animation');
  function syncPauseButton() {
    pauseButton.textContent = userPaused ? 'Play animation' : 'Pause animation';
    pauseButton.setAttribute('aria-pressed', String(!userPaused));
  }
  pauseButton.addEventListener('click', function() { userPaused = !userPaused; syncPauseButton(); });
  stepsEl.forEach(function(step, index) {
    step.addEventListener('click', function() { userPaused = true; applyState(index); syncPauseButton(); });
  });
  document.querySelectorAll('[data-camera]').forEach(function(button) {
    button.addEventListener('click', function() {
      var action = button.dataset.camera;
      if (action === 'left') camTheta -= 0.25;
      if (action === 'right') camTheta += 0.25;
      if (action === 'in') camRadius = Math.max(8, camRadius - 2);
      if (action === 'out') camRadius = Math.min(50, camRadius + 2);
      updateCamera();
    });
  });
  syncPauseButton();
  container.addEventListener('mouseenter', function() { hoverPaused = true; });
  container.addEventListener('mouseleave', function() { hoverPaused = false; });

'''+script[end:]
        script=script.replace('if (!prefersReducedMotion && !isDragging)', 'if (!prefersReducedMotion && !userPaused && !hoverPaused && !isDragging)')
        script=script.replace('if (!paused && !isDragging)', 'if (!userPaused && !hoverPaused && !isDragging)')
    if 'renderer.domElement.setAttribute' not in script:
        script=script.replace('  container.appendChild(renderer.domElement);','''  container.appendChild(renderer.domElement);
  renderer.domElement.setAttribute('aria-hidden', 'true');
  document.querySelector('.viz-steps').hidden = false;
  document.getElementById('viz-controls').hidden = false;''')
        script=script.replace("for (var k = 0; k < stepsEl.length; k++) stepsEl[k].classList.toggle('active', k === s);",'''for (var k = 0; k < stepsEl.length; k++) {
      stepsEl[k].classList.toggle('active', k === s);
      stepsEl[k].setAttribute('aria-pressed', String(k === s));
    }''')
    index=head(old,TITLE,'Current exploratory results on low-rank compression: prediction quality, residual drift, storage, runtime and future research.')
    index+=f'''<header class="title-block"><div class="wrap">
    <h1>{TITLE}</h1><p class="author">Sambhav Surana · Research project</p>
    <p class="meta">Updated <time datetime="2026-10-15">15 October 2026</time> · Exploratory findings</p>
    <p><a href="{PAPER_URL}">Current IEEE paper source</a> · <a href="paper-reader.html">Plain-language reader guide</a> · <a href="{REPO}">Repository</a></p>
    <nav class="page-nav" aria-label="On this page"><a href="#abstract">Abstract</a><a href="#results">Results</a><a href="#practical">Storage &amp; timing</a><a href="#limitations">Limitations</a><a href="#future-scope">Future Scope</a><a href="#paper">Paper &amp; evidence</a></nav>
    </div></header><main id="content" tabindex="-1">'''
    index+=section('abstract','Abstract','<p>'+escape(abstract)+'</p>')
    index+=section('introduction','1. What the study asks','''<p>Can a pretrained transformer use fewer stored weights while retaining the original model’s behavior? We replace selected attention-output matrices with two smaller factors and measure both next-token prediction and the internal signals passed between blocks.</p>
    <div class="status-note"><p><strong>Current finding.</strong> Activation-aware fitting improves on plain SVD in the tested Gemma setting. Adding the weight-space anchor gives no consistent improvement in both prediction quality and residual fidelity across Gemma, Qwen and LFM. A better likelihood score alone cannot establish preservation of the original computation.</p></div>''')
    index+=section('method','2. What is compared','''<p>A dense projection computes <code>Wx + b</code>. Its compact replacement computes <code>U(Vx) + b</code>, applying the original bias once. At a fixed factor budget, covariance whitening weights input directions using calibration activations. The anchored version adds a penalty for changing the original weights.</p>
    <div class="equation">J<sub>β</sub>(A) = ‖(W − A)X‖²<sub>F</sub> + βn‖W − A‖²<sub>F</sub>, with A = UV</div>
    <p>The matched whitening component uses β = ρ = 0; the selected anchor uses β = 0.03 and ρ = 0. The charts label that measured anchored fit <strong>Ours (anchored)</strong> in orange. Gemma’s primary rank is 384, Qwen’s is 483, and LFM’s is 725. Other layers and hybrid mixers remain dense.</p>
    <details><summary>Protocol and interpretation</summary><div class="detail-body"><p>The historical quality cohorts use calibration seeds 11, 23 and 37, paired across methods. WikiText-2 supplies calibration, validation and test splits; the test set has already influenced the project, so these are regression results. The later BF16 practical cohort uses only seeds 11 and 23 and supplies no new drift measurements.</p><p>Exact-arithmetic formulas require their stated rank and spectral assumptions. Failed finite-precision controls prevent attributing all measured changes solely to the mathematical objective. The current paper documents measured-output fitting, extra ridge and spectral cutoffs separately.</p></div></details>''')
    index+=section('viz','3. Concept illustration','''<p>This optional animation uses synthetic points and visual metaphors for different fitting geometries. It is not a measured layer, a literal factorization surface, or evidence that compression preserves computation.</p>
    <div class="viz-wrap"><div class="viz-canvas-wrap"><div id="concept-canvas-container" style="height:360px"><div id="concept-label">Optional concept illustration</div></div></div>
    <div class="viz-steps" hidden><button type="button" class="viz-step active">1. Full matrix</button><button type="button" class="viz-step">2. Calibration directions</button><button type="button" class="viz-step">3. Weight geometry</button><button type="button" class="viz-step">4. Activation geometry</button><button type="button" class="viz-step">5. Test global behavior</button></div>
    <div class="viz-controls" id="viz-controls" hidden><button type="button" id="toggle-animation" aria-pressed="true">Pause animation</button><button type="button" data-camera="left">Rotate left</button><button type="button" data-camera="right">Rotate right</button><button type="button" data-camera="in">Zoom in</button><button type="button" data-camera="out">Zoom out</button></div></div>
    <p class="small">Drag to rotate or use the buttons. Animation starts paused when reduced motion is requested. The measured charts below work without JavaScript or WebGL.</p>''',True)
    index+=section('results','4. Results','''<p>The charts use the same saved numbers as the current IEEE source. On a narrow screen, scroll a chart horizontally or open its exact-value table. NLL is negative log-likelihood; lower NLL means better prediction on the scored text. Residual drift measures internal-state difference from the original model.</p>'''+gemma_chart+'''<p>The anchor’s mean Gemma NLL is 3.565921 versus 3.572814 for whitening. Its average drift is slightly lower, but drift worsens in two of three paired seeds. The average does not establish a reproducible joint improvement.</p>'''+composition_chart+hybrid_chart+'''<p>LFM shows why the metrics must be assessed separately: the anchor improves likelihood in every seed while mean residual drift rises to about 3.86 times the whitening mean. Lower NLL does not establish preservation, denoising, or a particular causal mechanism.</p>''',True)
    index+=section('practical','5. Storage and timing','''<p>The separate BF16 cohort removes 29.17% of weights in the targeted matrices. Complete checkpoint size falls from 1,999,883,711 to 1,982,003,507 bytes: 17,880,204 bytes, or 0.894%. Most model weights remain untouched. Anchor and whitening checkpoints have identical sizes.</p>'''+html_table(['BF16 condition','Seed','Raw NLL','Scaled NLL','Whole GiB'],normal[0],'Separate BF16 practical conditions')+'''<p>Temperature is selected on validation and then frozen; the five saved selections are approximately 1.090508. GiB means 2³⁰ bytes. These are descriptive full-test aggregates without paired block intervals.</p>'''+html_table(['Workload','Seed 11 median ratio','Seed 23 median ratio'],normal[1],'Median latency ratio: anchor / matched whitening')+'''<p>Ratios below one favor the anchor. Seed 11 is slower for the 128-token conditions, while seed 23 favors it in the measured workloads. Timing uses a resident RTX 4070 Ti, batch size one, 10 warmups and 30 synchronized samples. Cached decode excludes the initial prompt pass. These observations establish no consistent anchoring speed advantage or general acceleration over the dense teacher.</p>''')
    index+=section('limitations','6. Limitations and evidence boundaries','''<ul><li>The strict numerical-readiness checks failed. The results describe recorded implementations; they cannot isolate ideal mathematical objectives from numerical error.</li><li>Qwen has all eight primary-test records but only 10 of 18 validation-frontier records. Missing measurements remain missing.</li><li>WikiText-2 has already influenced the project. No fresh-domain confirmation, downstream task accuracy, or free-running generation equivalence has been established.</li><li>Three calibration seeds do not represent independently pretrained models. The later BF16 practical cohort has only two seeds and no new drift measurements.</li><li>Only the selected attention-output matrices are studied. Results do not establish preservation for MLPs, hybrid mixers, larger models, or other rank allocations.</li><li>Model file size, CUDA allocator peaks and warmed-up latency are distinct measurements. Storage savings alone do not establish practical speed or memory savings.</li></ul>''')
    index+=section('future-scope','7. Future Scope','''<h3>A small neural network for compression</h3><p>A proposed next direction is a small bottleneck neural network that learns a compact mapping for selected dense projections from the original model’s activations. A second option is a small correction network alongside lower-rank factors.</p><p>Every retained factor, network weight and bias must count toward a matched storage budget. Any local improvement must be checked for held-out prediction quality and residual drift after replacements are composed across layers, alongside complete file size, peak memory and warmed-up latency. Neither network has been implemented or tested here.</p><h3>Numerical controls and broader evaluation</h3><p>Resolve failed controls and complete missing Qwen measurements before controlled comparisons of group-output matching. Fresh evaluation data, downstream tasks and generation tests need frozen splits, budgets and decision criteria. Fused factor execution and runtime-informed rank allocation remain further hypotheses.</p><p><a href="https://aclanthology.org/2020.findings-emnlp.372/">TinyBERT</a> is a related teacher–student distillation precedent; it does not establish the proposed projection replacement’s effectiveness. The paper’s full Future Scope is Section 8, including the small-network proposal in Section 8.1.</p>''')
    index+=section('paper','8. Paper and reproducible evidence',f'''<ul><li><a href="{PAPER_URL}">Current IEEE paper source</a> — larger mathematical displays and vector charts for dense comparisons; simpler BF16 and timing values remain tables.</li><li><a href="{REVIEW_URL}">Earlier review PDF</a> — the same bounded study and Future Scope, with the earlier table layout. The PDF has not been regenerated for the in-place chart edit.</li><li><a href="{EVIDENCE_URL}">Frozen evidence report</a> — source runs, complete recorded comparisons and unresolved gaps.</li><li><a href="research-data.json">Chart and table data (JSON)</a> — original reported values and a fingerprint of the paper source.</li><li><a href="{PUBLICATION_URL}">IEEE preparation notes</a> — generic conference formatting; the actual venue remains unidentified.</li></ul><p>Both appendices were removed from the current paper. Their experiment records and uncertainty intervals remain in the repository. The existing paper editor stays the place to compile the latest IEEE source.</p>''')
    index+=f'</main><footer><div class="wrap"><p>Sambhav Surana · Updated 15 October 2026 · <a href="{REPO}">Source code and evidence</a> · <a href="paper-reader.html">Reader guide</a></p></div></footer><script>{script}</script></body></html>\n'
    index='\n'.join(line.rstrip() for line in index.splitlines())+'\n'
    (SITE/'index.html').write_text(index,encoding='utf-8',newline='\n')
    guide=head(reader,'Reader guide | '+TITLE,'A plain-language guide to the current exploratory compression study, its mixed results, limitations and future scope.')
    guide+='''<header class="page-header"><div class="wrap"><p><a href="index.html">← Research overview</a></p><p class="eyebrow">Reader guide · Updated 15 October 2026</p><h1>How to understand this research</h1><p class="lead">A guide to the current study: what gets smaller, what changes inside the model, and what still needs testing.</p></div></header><main id="content" tabindex="-1">'''
    guide+=section('main-idea','The main idea','''<div class="callout"><p>A compressed model can predict text better while its internal signals become less like the original model’s. We measure prediction, internal drift, storage and speed separately.</p></div><p>A language model processes text through large grids of numbers called weight matrices. Replacing selected grids with two smaller ones saves stored weights. Even if each replacement fits the original layer well on sample text, errors can change the inputs received by later layers.</p><p>Activation-aware fitting pays attention to the directions used by observed inputs. The study also tests a penalty, called an anchor, for changing the original weights. The anchor adds no extra branch during inference.</p>''')
    guide+=section('method','How the comparison works','''<div class="steps"><div class="step"><h3>Observe the original model</h3><p>Use calibration text to record the signals entering selected attention-output layers. These observations guide which directions the compact representation should fit.</p></div><div class="step"><h3>Use the same storage budget</h3><p>Compare covariance whitening with the anchored fit using the same target shapes, ranks and factor precision. Plain SVD is a separate weight-space control.</p></div><div class="step"><h3>Evaluate the complete model</h3><p>Check predictions and the internal signals passed between blocks, then separately measure file size, memory and warmed-up runtime. A strong single-layer fit alone cannot establish global preservation.</p></div></div>''')
    guide+=section('findings','What the current results say','''<div class="card"><h3>Gemma: a small average difference</h3><p>The anchor has slightly lower mean loss and drift than whitening, but drift worsens in two of three seeds. A full-stack average also hides layer groups where both metrics get worse.</p></div><div class="card"><h3>Qwen: both metrics worsen</h3><p>At the measured primary rank, the anchor has higher prediction loss and higher drift in every calibration seed. Eight of 18 validation-frontier measurements are still missing.</p></div><div class="card"><h3>LFM: better predictions, larger internal change</h3><p>The anchor reduces prediction loss while average residual drift rises to 3.86 times whitening’s. A better prediction score does not establish that the original computation has been preserved.</p></div><div class="card"><h3>Storage and speed: a separate question</h3><p>The selected matrices lose 29.17% of their weights, but the complete model file shrinks by 0.894%. The measured timing advantage varies by seed and workload.</p></div><p><a href="index.html#results">See the measured charts and exact values</a> or <a href="index.html#practical">read the storage and timing tables</a>.</p>''')
    guide+=section('glossary','Terms used in the paper','''<dl class="glossary"><div class="gloss-item"><dt>Low-rank factors</dt><dd>Two smaller matrices whose product approximates a larger matrix. Their combined storage must be counted.</dd></div><div class="gloss-item"><dt>Calibration data</dt><dd>Text used to observe activations and fit the compressed factors. Validation and test splits have separate roles.</dd></div><div class="gloss-item"><dt>Negative log-likelihood (NLL)</dt><dd>Average loss on scored next-token predictions. Lower is better on the same model’s tokenizer and evaluation setup.</dd></div><div class="gloss-item"><dt>Residual drift</dt><dd>A normalized measure of difference between the original and compressed model’s block-output states on the same observed text. It is not a proof of functional equivalence.</dd></div><div class="gloss-item"><dt>Whitening and anchor</dt><dd>Whitening uses activation covariance to guide the fit. The anchor adds a weight-deviation penalty. Its value must be assessed against the matched whitening baseline.</dd></div><div class="gloss-item"><dt>Seed variation</dt><dd>Variation from different calibration draws. It does not mean the pretrained model itself was independently trained several times.</dd></div><div class="gloss-item"><dt>BF16</dt><dd>A 16-bit floating-point format used in the separate practical storage and timing cohort. That cohort is not pooled with the historical FP16 drift study.</dd></div></dl>''')
    guide+=section('limits','Why the conclusions are exploratory','''<ul><li>Some strict numerical accuracy checks failed.</li><li>Qwen’s validation coverage is incomplete.</li><li>The text dataset has already influenced the project, so the results are regression evidence.</li><li>No downstream task or generated-text equivalence has been established.</li><li>The BF16 practical cohort has two seeds and no new drift measurements.</li></ul><p>The paper therefore does not establish a general method for preserving a whole model’s computation, a consistent speedup, or a causal denoising effect.</p>''')
    guide+=section('future-scope','Future Scope: could a small network help?','''<p>A future experiment could train a small bottleneck network to replace a selected dense projection, or use a small network to correct a low-rank replacement’s error. These are proposals; neither was trained or evaluated in this study.</p><p>The new network’s own weights, biases, memory use and processing time must be counted. It would need to improve held-out behavior across composed layers at a matched total budget. The first priorities remain resolving numerical controls and completing missing evaluation.</p><p>Find this proposal in the paper’s <strong>Section 8.1</strong>. Section 8 also discusses broader evaluation and practical execution costs.</p>''')
    guide+=section('read-paper','How to read the current paper',f'''<p><a href="{PAPER_URL}">Open the current IEEE paper source</a>. Start with the abstract and introduction, then Section 5’s results. Read Section 7’s limitations before interpreting gains, and Section 8 for future research. Both appendices have been removed; the supporting records remain in the repository.</p><ul><li><a href="{REVIEW_URL}">Earlier review PDF</a> — readable paper with the earlier table layout.</li><li><a href="{EVIDENCE_URL}">Frozen evidence report</a> — full recorded results and provenance.</li><li><a href="index.html#results">Current website charts</a> — the same saved values as the latest IEEE source.</li><li><a href="{REPO}">Source repository</a> — research code, protocols and audit records.</li></ul>''')
    guide+='</main><footer><div class="wrap"><p><a href="index.html">← Research overview</a> · Updated 15 October 2026</p></div></footer></body></html>\n'
    (SITE/'paper-reader.html').write_text(guide,encoding='utf-8',newline='\n')
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({'website':str(SITE),'paper_and_evidence_unchanged':True,
        'gemma_methods':len(gemma),'composition_groups':len(composition),'new_experiments':False},indent=2))


if __name__=='__main__':
    main()

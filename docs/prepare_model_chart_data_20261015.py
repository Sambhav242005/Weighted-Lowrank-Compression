"""Summarize already-recorded hybrid test rows for figure presentation only."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/paper-future-scope-20261015'
MODELS={
    'Qwen':('Qwen3.5-0.8B',483,'architecture-replication-20261014-093845-97378b5e'),
    'LFM':('LFM2.5-1.2B-Instruct',725,'architecture-replication-20261014-103759-8d32ab84'),
}


def main():
    output={'purpose':'Presentation of existing test observations; no experiment or training',
        'measured_method_label':'Ours (anchored)','method_definition':'Activation-weighted anchored fit, beta=0.03, rho=0',
        'future_small_network_measured':False,'models':{}}
    for family,(name,rank,run) in MODELS.items():
        path=ROOT/'results/paper-v2'/run/'metrics.json'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        records=[r for r in json.loads(path.read_text(encoding='utf-8')) if r['split']=='test']
        assert len(records)==8
        rows=[]
        for source_label,label in [('teacher','Dense reference'),('svd','Plain SVD'),
                                   ('baseline','Whitening'),('solver','Ours (anchored)')]:
            selected=[r for r in records if r['label']==source_label]
            expected=1 if source_label in ['teacher','svd'] else 3
            assert len(selected)==expected and len({r['seed'] for r in selected})==expected
            if expected==3:assert {r['seed'] for r in selected}=={11,23,37}
            if source_label!='teacher':assert all(r['rank']==rank for r in selected)
            row={'source_label':source_label,'label':label,'seeds':[r['seed'] for r in selected],
                'seed_count':expected,'rank':None if source_label=='teacher' else rank,
                'source_values':[{key:r['metrics'][key] for key in ['nll','primary_drift']} for r in selected]}
            for field,key,precision in [('nll','nll',6),('primary_drift','drift',9)]:
                values=[r['metrics'][field] for r in selected]
                row[key+'_mean']=statistics.mean(values)
                row[key+'_display']=f'{row[key+"_mean"]:.{precision}f}'
                row[key+'_sd']=statistics.stdev(values) if expected==3 else None
                row[key+'_sd_display']=f'{row[key+"_sd"]:.{precision}f}' if expected==3 else None
            rows.append(row)
        output['models'][family]={'name':name,'rank':rank,'source_run':run,
            'source_metrics':str(path.relative_to(ROOT)),'source_sha256':digest,'rows':rows}
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (PAPER/'model-chart-data.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({name:[{k:r[k] for k in ['label','nll_display','drift_display','nll_sd_display','drift_sd_display']} for r in model['rows']]
        for name,model in output['models'].items()},indent=2))


if __name__=='__main__':main()

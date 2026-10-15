"""Audit terminal Qwen partial cohort with standard-library CPU file reads only."""
import collections
import datetime
import json
from pathlib import Path
import uuid
from audit_gemma_evidence_20261014 import ROOT, RESULTS, digest, hashes


def main():
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))
    old = RESULTS/'architecture-replication-20261014-082807-5c73f669'
    new = RESULTS/'architecture-replication-20261014-093845-97378b5e'
    original, rows = read(old/'metrics.json'), read(new/'metrics.json')
    contract, status = read(new/'contract.json'), read(new/'status.json')
    check(len(original)==10 and rows[:10]==original, 'Retained ten-row prefix exact equality')
    check(len(rows)==18 and status['state']=='incomplete' and status['budget_limited'] is True, 'Terminal partial cohort')
    config = read(new/'config.json')
    sidecar = config['interrupted_runs'][str(old.resolve())]
    for filename, expected in sidecar['source_artifacts'].items():
        check(digest(old/filename)==expected, 'Immutable interrupted source '+filename)
    for path, expected in contract['implementation_hashes'].items():
        check(digest(path)==expected, 'Frozen implementation '+path)
    ready = read(ROOT/'paper-assets/architecture-code-ready-20261014.json')
    for path, expected in ready['artifacts'].items():
        check(digest(ROOT/path)==expected, 'Architecture ready hash '+path)
    keys = lambda r: (r['split'],r['rank_divisor'],r['label'],r['seed'])
    present = [keys(r) for r in rows]
    check(len(present)==len(set(present)), 'No duplicate scopes')
    expected = [keys(r) for r in contract['comparison_plan']]
    missing = [k for k in expected if k not in present]
    wanted_missing = [('validation',3,l,37) for l in ['baseline','solver']] + [('validation',4,l,s) for s in [11,23,37] for l in ['baseline','solver']]
    check(set(missing)==set(wanted_missing), 'Exact eight omitted validation scopes')
    for i,row in enumerate(rows):
        prefix = f'row {i}'
        targets = 297192 if row['split']=='test' else 261410
        check(row['metrics']['targets']==targets, prefix+' full target count')
        check(sum(b['targets'] for b in row['metrics']['nll_blocks'])==targets, prefix+' complete NLL blocks')
        check(row['teacher_reference_targets']==targets and row['teacher_reference_split']==row['split'], prefix+' teacher split')
        check(len(row['metrics']['residual_by_block'])==24, prefix+' all decoder block drift')
        rank = None if row['label']=='teacher' else (483 if row['split']=='test' else 1024//row['rank_divisor'])
        check(row['rank']==rank, prefix+' rank')
        recipes = row['target_recipes']
        check(len(recipes)==6, prefix+' six full-attention outputs')
        for recipe, block in zip(recipes,[3,7,11,15,19,23]):
            check(recipe['name']==f'model.layers.{block}.self_attn.o_proj' and recipe['block_index']==block and recipe['shape']==[1024,2048]
                  and recipe['rank']==rank and recipe['factor_parameters']==(rank*3072 if rank is not None else None) and recipe['precision']=='bfloat16',prefix+' target recipe')
        if row['label']!='teacher':
            check(digest(row['checkpoint'])==row['metadata']['sha256'],prefix+' checkpoint SHA')
            check(row['metadata']['factor_parameters']==6*rank*3072 and row['metadata']['targets']==recipes,prefix+' checkpoint budget')
        if row['label'] in ['baseline','solver']:
            spec = {'method':'whitening','beta':0.0,'ridge_fraction':0.0} if row['label']=='baseline' else {'method':'anchored','beta':0.03,'ridge_fraction':0.0}
            check(row['specification']==spec,prefix+' frozen method')
    check(read(new/'validation-teacher.json')['metrics']['targets']==261410 if 'metrics' in read(new/'validation-teacher.json') else read(new/'validation-teacher.json')['targets']==261410,'Validation teacher targets')
    charged = sidecar['charged_seconds'] + status['elapsed_seconds']
    check(abs(charged-7200.17339276207)<0.00001,'Cumulative charge')
    check(contract['prior_stage_e_seconds']==sidecar['charged_seconds'],'Prior charge contract')
    output = RESULTS/('qwen-evidence-audit-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    payload = {'state':'passed' if not errors else 'errors','source_run':old.name,'terminal_run':new.name,'source_prefix_rows':10,'terminal_records':len(rows),
               'split_counts':dict(collections.Counter(r['split'] for r in rows)),'missing_scopes':missing,'prior_charged_seconds':sidecar['charged_seconds'],
               'continuation_seconds':status['elapsed_seconds'],'cumulative_seconds':charged,'cooperative_overshoot_seconds':charged-7200,
               'limitations':['Qwen validation cohort incomplete; no restart or budget borrowing.','Retained old running status is immutable reboot evidence; sidecar records interruption.',
                              'Reused rows are not an independent replication.','CPU file audit does not deserialize tensors or re-evaluate predictions.'],
               'files_hashed':hashes,'errors':errors,'script_sha256':digest(__file__)}
    (output/'audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    (output/'report.md').write_text('# Qwen immutable evidence audit\n\nState: '+payload['state']+'\n\n18 records: eight native test and ten validation. Ten source rows preserved exactly.\n\n'+
        f'Cumulative charged seconds: {charged:.9f}; cooperative cap overshoot {charged-7200:.9f} seconds.\n\nMissing scopes: '+str(missing)+'\n\n'+
        '\n'.join('- '+x for x in payload['limitations'])+'\n\nErrors: '+str(errors)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'state':payload['state'],'errors':errors},indent=2))


if __name__=='__main__':
    main()

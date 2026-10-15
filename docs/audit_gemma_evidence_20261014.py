"""CPU-only immutable artifact audit; no model deserialization or source mutation."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/paper-v2'
NAMES = ['baselines-20261013-175255-6a00d4e8', 'composition-20261014-054423-dbd821ff']
errors = []
hashes = {}


def digest(path):
    path = Path(path).resolve()
    if str(path) not in hashes:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        hashes[str(path)] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
    return hashes[str(path)]['sha256']


def check(condition, message):
    if not condition:
        errors.append(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    cohorts = {}
    all_rows = {}
    for name, count in zip(NAMES, [13, 60]):
        directory = RESULTS / name
        rows = load(directory / 'metrics.json')
        all_rows[name] = rows
        check(load(directory / 'status.json')['state'] == 'completed', name + ': terminal state')
        check(len(rows) == count, name + ': row count')
        teacher = load(directory / 'teacher.json')
        check(teacher['targets'] == 292281, name + ': teacher target count')
        check(sum(b['targets'] for b in teacher['nll_blocks']) == 292281, name + ': teacher block coverage')
        for index, row in enumerate(rows):
            prefix = f'{name} row {index}'
            try:
                actual = digest(row['checkpoint'])
                check(actual == row.get('checkpoint_sha256', row['metadata'].get('sha256')), prefix + ': checkpoint SHA')
                check(row['metrics']['targets'] == 292281, prefix + ': target count')
                check(sum(b['targets'] for b in row['metrics']['nll_blocks']) == 292281, prefix + ': block coverage')
                check(len(row['metrics']['residual_by_block']) == 26, prefix + ': drift block count')
                group = row.get('group', list(range(26)))
                targets = row['metadata']['targets']
                check(len(targets) == len(group), prefix + ': recipe coverage')
                for target, block in zip(targets, group):
                    check(target['name'] == f'model.layers.{block}.self_attn.o_proj', prefix + ': target name')
                    check(target['shape'] == [1152, 1024] and target['rank'] == 384 and target['factor_parameters'] == 835584,
                          prefix + ': target shape/rank/budget')
                check(row['metadata']['factor_parameters'] == len(group) * 835584, prefix + ': total budget')
                check(row['metadata']['precision'] == 'float16', prefix + ': precision')
                if row.get('reused_from_run'):
                    check(row.get('independent_replication') is False, prefix + ': reuse independence flag')
                    source = RESULTS / row['reused_from_run'] / 'metrics.json'
                    check(digest(source) == row['source_metrics_sha256'], prefix + ': source metric SHA')
                    source_rows = load(source)
                    if 'source_record_id' in row:
                        matches = [r for r in source_rows if r.get('id') == row['source_record_id']]
                    else:
                        matches = [r for r in source_rows if r.get('label') == row['label'] and r['seed'] == row['seed'] and r.get('group') == group]
                    expected_metrics = row['metrics'] if 'source_record_id' not in row else {k:v for k,v in row['metrics'].items() if k not in ['group_end_drift', 'downstream_drift']}
                    check(len(matches) == 1 and matches[0]['metrics'] == expected_metrics,
                          prefix + ': source metric equality')
            except Exception as exc:
                errors.append(prefix + ': ' + repr(exc))
        cohorts[name] = {'rows': len(rows), 'teacher_targets': teacher['targets'], 'metrics_sha256': digest(directory/'metrics.json'),
                         'methods': dict(collections.Counter(r['specification']['method'] for r in rows)),
                         'origins': dict(collections.Counter(r.get('evaluation_origin', 'original Stage C measurement') for r in rows))}
    c = all_rows[NAMES[0]]
    check(collections.Counter((r['specification']['method'],r['seed']) for r in c) == collections.Counter(
        [('svd',11)] + [(m,s) for m in ['asvd','whitening','anchored','legacy_ridge'] for s in [11,23,37]]), 'C: method/seed grid')
    d = all_rows[NAMES[1]]
    groups = [(0,), (12,), (25,), (0,1), (12,13), (24,25), (0,1,2,3), (11,12,13,14), (22,23,24,25), tuple(range(26))]
    check(collections.Counter((tuple(r['group']),r['label'],r['seed']) for r in d) == collections.Counter(
        (g,l,s) for g in groups for l in ['baseline','solver'] for s in [11,23,37]), 'D: group/method/seed grid')
    for row in d:
        expected = {'method':'whitening','beta':0.0,'ridge_fraction':0.0} if row['label']=='baseline' else {'method':'anchored','beta':0.03,'ridge_fraction':0.0}
        check(row['specification'] == expected, 'D: frozen recipe')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    output = RESULTS / f'evidence-audit-{stamp}-{uuid.uuid4().hex[:8]}'
    output.mkdir(exist_ok=False)
    payload = {'state':'passed' if not errors else 'errors', 'scope':'immutable Gemma C/D artifact evidence; CPU only; no tensor deserialization',
               'cohorts':cohorts,'unique_files_hashed':hashes,'errors':errors,'script_sha256':digest(__file__)}
    (output/'audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    (output/'report.md').write_text('# Gemma C/D immutable evidence audit\n\nState: '+payload['state']+'\n\n'+
        '\n'.join(f'- {n}: {v["rows"]} records; {v["origins"]}' for n,v in cohorts.items())+
        f'\n\nUnique files streamed once: {len(hashes)}. No GPU work or checkpoint deserialization.\n\n'+
        ('Errors:\n'+ '\n'.join('- '+e for e in errors) if errors else 'All declared artifact checks passed.')+'\n',encoding='utf-8')
    print(json.dumps({'output':str(output),'state':payload['state'],'errors':errors},indent=2))


if __name__ == '__main__':
    main()

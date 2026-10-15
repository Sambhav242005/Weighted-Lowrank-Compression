"""Assemble this editorial revision from frozen evidence; never run experiments.

Run from the repository root. Existing outputs cause a refusal. The narrative is
authored in manuscript.source.md; tables and uncertainty are copied/derived from
saved records, not recomputed model evaluations.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'results/paper-v2/revised-report-20261014-151432-07c08d61'
IDS = {
    'gemma': 'baselines-20261013-175255-6a00d4e8',
    'composition': 'composition-20261014-054423-dbd821ff',
    'qwen': 'architecture-replication-20261014-093845-97378b5e',
    'lfm': 'architecture-replication-20261014-103759-8d32ab84',
    'practical': 'practical-bf16-20261014-135107-82168ad5',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table(headers, rows):
    if any(len(row) != len(headers) for row in rows):
        raise ValueError('table width mismatch')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def group(run, method, primary=False):
    found = [g for g in run['method_groups']
             if g['specification']['method'] == method
             and (not primary or (g['scope'].get('split') == 'test'
                                  and g['scope'].get('rank_divisor') is None))]
    if len(found) != 1:
        raise ValueError(f'nonunique group: {run["run_id"]}/{method}')
    return found[0]


def main():
    outputs = ['manuscript.md', 'manuscript.tex', 'manuscript.pdf', 'report.json',
               'revision-manifest.json', 'figures', 'paper_protocol_v2.bib']
    if any((OUT / name).exists() for name in outputs):
        raise FileExistsError('Refusing to overwrite a previous revision build')
    frozen = json.loads((BASE / 'report.json').read_text(encoding='utf-8'))
    source_paths = list(BASE.glob('*')) + list((ROOT / 'src').glob('*.py'))
    protected = {str(p.relative_to(ROOT)): sha(p) for p in source_paths if p.is_file()}
    runs = {r['run_id']: r for r in frozen['runs']}
    selected = {key: runs[value] for key, value in IDS.items()}
    substitutions = {}
    labels = {'svd': 'Plain SVD', 'asvd': 'Diagonal ASVD component',
              'whitening': 'Covariance whitening', 'anchored': 'Anchor (beta=0.03)',
              'legacy_ridge': 'Extra ridge (beta=0.1, rho=0.01)'}
    def metric(g, name):
        value = g['metrics'][name]
        if value['sample_std'] is None:
            return f"{value['mean']:.6f} (-)"
        return f"{value['mean']:.6f} ± {value['sample_std']:.6f}"
    substitutions['GEMMA_TABLE'] = table(['Method', 'NLL', 'Residual drift', 'Seeds'],
        [[labels[m], metric(group(selected['gemma'], m), 'nll'),
          metric(group(selected['gemma'], m), 'primary_drift'),
          len(group(selected['gemma'], m)['seeds'])]
         for m in ['svd', 'asvd', 'whitening', 'anchored', 'legacy_ridge']])
    composition_rows = []
    for comp in selected['composition']['comparisons']:
        records = comp['per_seed']
        assert {row['seed'] for row in records} == {11, 23, 37}
        assert all(row['targets_match'] and row['actual_factor_parameters_match'] for row in records)
        blocks = comp['scope']['group']
        label = ('All 26' if len(blocks) == 26 else str(blocks[0]) if len(blocks) == 1
                 else f'{blocks[0]}-{blocks[-1]}')
        composition_rows.append([label,
            f"{statistics.mean(row['nll_difference'] for row in records):+.7f}",
            f"{statistics.mean(row['drift_difference'] for row in records):+.7g}",
            f"{sum(row['nll_difference'] < 0 and row['drift_difference'] < 0 for row in records)}/3"])
    substitutions['COMPOSITION_TABLE'] = table(['Blocks', 'Mean NLL difference', 'Mean drift difference', 'Jointly better'], composition_rows)
    hybrid_rows = []
    for key, label in [('qwen', 'Qwen, rank 483'), ('lfm', 'LFM, rank 725')]:
        base, anchor = [group(selected[key], m, primary=True) for m in ['whitening', 'anchored']]
        hybrid_rows.append([label,
            f"{base['metrics']['nll']['mean']:.6f} → {anchor['metrics']['nll']['mean']:.6f}",
            f"{base['metrics']['primary_drift']['mean']:.6f} → {anchor['metrics']['primary_drift']['mean']:.6f}",
            '8/8 test; 10/18 validation' if key == 'qwen' else '8/8 test; 18/18 validation'])
    substitutions['HYBRID_TABLE'] = table(['Model / primary rank', 'NLL: whitening → anchor', 'Drift: whitening → anchor', 'Record coverage'], hybrid_rows)
    practical_path = ROOT / 'results/paper-v2' / IDS['practical'] / 'metrics.json'
    raw = json.loads(practical_path.read_text(encoding='utf-8'))
    dense = next(row for row in raw if row['label'] == 'teacher')
    compact = [row for row in raw if row['label'] != 'teacher']
    compact_sizes = {row['whole_checkpoint']['bytes'] for row in compact}
    assert len(raw) == 5 and len(compact_sizes) == 1
    dense_bytes, factor_bytes = dense['whole_checkpoint']['bytes'], compact_sizes.pop()
    assert round(100 * (dense_bytes-factor_bytes)/dense_bytes, 3) == 0.894
    substitutions.update(DENSE_BYTES=f'{dense_bytes:,}', FACTOR_BYTES=f'{factor_bytes:,}',
                         SAVED_BYTES=f'{dense_bytes-factor_bytes:,}')
    practical_rows = []
    for row in raw:
        name = 'Dense teacher' if row['label'] == 'teacher' else ('Whitening' if row['label'] == 'baseline' else 'Anchor')
        practical_rows.append([name, '-' if row['seed'] is None else row['seed'],
            f"{row['calibration']['raw_test']['nll']:.6f}",
            f"{row['calibration']['scaled_test']['nll']:.6f}",
            f"{row['whole_checkpoint']['bytes']/2**30:.6f}"])
    substitutions['PRACTICAL_TABLE'] = table(['BF16 condition', 'Seed', 'Raw NLL', 'Scaled NLL', 'Whole GiB'], practical_rows)
    ratios = selected['practical']['same_bit_cost_comparisons']['comparisons']
    timing_rows = []
    for mode, length in [('prefill', 128), ('decode', 128), ('prefill', 512), ('decode', 512)]:
        values = []
        for seed in [11, 23]:
            match = [r for r in ratios if r['scope']['seed'] == seed
                     and r['scope']['mode'] == mode and r['scope']['prefill_tokens'] == length]
            assert len(match) == 1
            values.append(f"{match[0]['ratios']['median_latency']:.4f}")
        label = f'Prefill {length}' if mode == 'prefill' else f'Decode 128 after prompt {length}'
        timing_rows.append([label] + values)
    substitutions['TIMING_TABLE'] = table(['Workload', 'Seed 11 median ratio', 'Seed 23 median ratio'], timing_rows)
    evidence_rows = [[('LFM' if key == 'lfm' else key.capitalize()), '`' + value + '`', selected[key]['state']]
                     for key, value in IDS.items()]
    substitutions['EVIDENCE_TABLE'] = table(['Evidence', 'Source run', 'Saved state'], evidence_rows)
    text = (OUT / 'manuscript.source.md').read_text(encoding='utf-8')
    for name, value in substitutions.items():
        marker = '{{' + name + '}}'
        assert text.count(marker) == 1, marker
        text = text.replace(marker, value)
    assert not re.search(r'\{\{[A-Z_]+\}\}', text)
    (OUT / 'manuscript.md').write_text(text, encoding='utf-8', newline='\n')
    shutil.copyfile(BASE / 'paper_protocol_v2.bib', OUT / 'paper_protocol_v2.bib')
    figures = OUT / 'figures'
    figures.mkdir()
    original_figure = BASE / 'figures/baselines-20261013-175255-6a00d4e8-drift.png'
    shutil.copyfile(original_figure, figures / 'gemma-drift.png')
    provenance = json.loads((BASE / 'figures/provenance.json').read_text(encoding='utf-8'))
    # Keep the two complete Gemma figures' native-coordinate data for standalone TeX.
    inputs = [i for i in provenance['inputs'] if i.get('run_id') in [IDS['gemma'], IDS['practical']]]
    (figures / 'provenance.json').write_text(json.dumps({'inputs': inputs,
        'source_provenance': str((BASE / 'figures/provenance.json').relative_to(ROOT)),
        'source_sha256': sha(BASE / 'figures/provenance.json'),
        'note': 'Existing measured coordinates retained; no plot or measurement recomputed.'}, indent=2)+'\n', encoding='utf-8')
    interval_runs = []
    for key, label in [('gemma', 'Gemma full stack'), ('composition', 'Gemma blocks 11-14'),
                       ('qwen', 'Qwen primary test'), ('lfm', 'LFM primary test')]:
        candidates = [c for c in selected[key]['comparisons']
                      if c['candidate']['method'] == 'anchored'
                      and (key != 'composition' or c['scope'].get('group') == [11, 12, 13, 14])
                      and (key not in ['qwen', 'lfm'] or c['scope'].get('split') == 'test')]
        assert len(candidates) == 1
        c = deepcopy(candidates[0])
        c['scope'] = {'measurement': label, 'source_run': IDS[key]}
        interval_runs.append({'run_id': label, 'source_run_id': IDS[key], 'comparisons': [c]})
    generated = datetime.now(timezone.utc).isoformat()
    derivative = {'generated_utc': generated, 'promotion_disabled': True,
        'hypothesis_status': 'inconclusive', 'mode': 'editorial revision of frozen evidence',
        'source_report': str((BASE / 'report.json').relative_to(ROOT)),
        'source_report_sha256': sha(BASE / 'report.json'),
        'runs': interval_runs,
        'note': 'Selected existing intervals only. Complete comparisons, failures and provenance remain in source_report.'}
    (OUT / 'report.json').write_text(json.dumps(derivative, indent=2)+'\n', encoding='utf-8')
    bib = (OUT / 'paper_protocol_v2.bib').read_text(encoding='utf-8')
    cited = set(re.findall(r'\[@([^\]]+)\]', text))
    entries = set(re.findall(r'@\w+\{([^,]+),', bib))
    assert cited == entries, (sorted(cited-entries), sorted(entries-cited))
    latex = load_module('revision_latex', ROOT / 'docs/export_paper_latex_20261014.py')
    pdf = load_module('revision_pdf', ROOT / 'docs/export_paper_pdf_20261014.py')
    # Keep the original exporter frozen; improve citation display for this draft.
    reference_numbers = {key: i for i, (key, _) in enumerate(latex.bibliography_entries(bib), 1)}
    original_markup = pdf.text_markup
    def numbered_markup(value):
        return original_markup(re.sub(r'\[@([^\]]+)\]',
            lambda m: '[' + str(reference_numbers[m[1]]) + ']', str(value)))
    pdf.text_markup = numbered_markup
    latex.export_report(OUT)
    tex_path = OUT / 'manuscript.tex'
    tex = tex_path.read_text(encoding='utf-8')
    # Markdown includes human section numbers; TeX supplies its own numbering.
    tex = re.sub(r'(\\(?:sub)?section\{)\d+(?:\.\d+)*\. ?(?=[A-Z])', r'\1', tex)
    tex = re.sub(r'(\\subsection\{)\d+\.\d+ ', r'\1', tex)
    tex_path.write_text(tex, encoding='utf-8', newline='\n')
    pdf.export_pdf(OUT)
    assert all(sha(ROOT / path) == digest for path, digest in protected.items()), 'protected artifact changed'
    manifest = {'generated_utc': generated, 'kind': 'editorial_revision',
        'research_experiments_run': False, 'source_report_sha256': sha(BASE / 'report.json'),
        'original_manuscript_sha256': sha(BASE / 'manuscript.md'),
        'source_runs': IDS, 'substitutions': substitutions,
        'table_extraction': {'1': 'Gemma method_groups metrics and seeds',
            '2': 'Completed composition comparisons per_seed arithmetic means and strict joint signs',
            '3': 'Unique hybrid primary-test method_groups',
            '4': 'Raw BF16 metrics.json calibration and whole_checkpoint.bytes',
            '5': 'BF16 same_bit_cost_comparisons.comparisons median_latency'},
        'citations': {'cited_keys': sorted(cited), 'bibliography_keys': sorted(entries), 'all_keys_resolve': True},
        'protected_originals_unchanged': True, 'protected_hashes': protected,
        'artifacts': {name: {'bytes': (OUT / name).stat().st_size, 'sha256': sha(OUT / name)}
                      for name in ['manuscript.source.md', 'manuscript.md', 'manuscript.tex', 'manuscript.pdf',
                                   'report.json', 'paper_protocol_v2.bib', 'build_revision.py']},
        'review_workflow': 'Manual structural revision with independent reviewer, evidence and citation roles. Installed skill patch/acronym scripts were unavailable; no automated pipeline authorization or submission certification is claimed.',
        'tex_compilation': 'pending separate compiler diagnostic',
        'pdf_rendering': 'Independent ReportLab PDF; not compiled from manuscript.tex'}
    (OUT / 'revision-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'directory': str(OUT), 'citations': len(cited),
        'manuscript_words': len(text.split()), 'originals_unchanged': True}, indent=2))


if __name__ == '__main__':
    main()

"""Serialized handoff for the user's revised precision/architecture study.

No command starts before the verified composition-boundary controller stops
the old dispatcher. Old source is archived before the accounting-only edit.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ready(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get('artifacts'):
        raise ValueError('code readiness manifest is empty')
    for filename, expected in manifest['artifacts'].items():
        if digest(filename) != expected:
            raise ValueError('verified code changed: '+filename)


def main():
    output = Path('results/paper-v2/revised-orchestration-20261014-1')
    output.mkdir(parents=True, exist_ok=False)
    python = str(Path('.venv-cuda/Scripts/python.exe').resolve())
    handoff = Path('results/paper-v2/model-change-handoff-20261014-1/handoff.json')
    ledger = {'state': 'waiting for composition boundary', 'stages': [],
              'excluded_models': ['openai-community/gpt2'], 'max_models': 4,
              'incomplete_cohorts': []}
    def save():
        (output/'status.json').write_text(json.dumps(ledger, indent=2))
    def execute(label, args):
        entry = {'stage': label, 'command': subprocess.list2cmdline(['rtk', python, '-u', *args]),
                 'started_epoch': time.time()}
        ledger['state'] = 'running '+label
        ledger['stages'].append(entry)
        save()
        with (output/(label+'-stdout.log')).open('w', encoding='utf-8') as stdout, (output/(label+'-stderr.log')).open('w', encoding='utf-8') as stderr:
            completed = subprocess.run(['rtk', python, '-u', *args], stdout=stdout, stderr=stderr)
        entry.update(exit_code=completed.returncode, ended_epoch=time.time())
        save()
        return completed.returncode
    def newly_budget_limited(prefix, started):
        states = []
        for directory in Path('results/paper-v2').glob(prefix+'*'):
            status = directory/'status.json'
            if status.exists() and status.stat().st_mtime >= started:
                states.append(json.loads(status.read_text()))
        return bool(states) and all(state.get('budget_limited') is True for state in states)
    def retain_budget_limit(stage):
        ledger['incomplete_cohorts'].append(stage)
        ledger['decision'] = 'Retain budget-limited evidence; run independent authorized stages without borrowing budget'
        save()
    try:
        save()
        while not handoff.exists():
            time.sleep(2)
        event = json.loads(handoff.read_text())
        if event['composition_state']['state'] != 'completed':
            raise RuntimeError('composition was incomplete; retain evidence rather than advance')
        try:
            process = psutil.Process(event['pid'])
            if process.is_running() and 'src.paper_benchmark' in process.cmdline():
                raise RuntimeError('old dispatcher remains active')
            if 'docs.resume_after_second_reboot_20261014' in process.cmdline():
                ancestors = {parent.pid for parent in psutil.Process().parents()}
                completed = Path(event['composition'])/'status.json'
                if process.pid not in ancestors or json.loads(completed.read_text()).get('state') != 'completed':
                    raise RuntimeError('cannot verify completed recovery parent for CUDA-context release')
                # The parent called this dispatcher after saving its finished
                # composition. Close its retained CUDA allocator/context before
                # launching an independent model process.
                process.terminate()
                process.wait(timeout=20)
                ledger['released_completed_recovery_parent'] = event['pid']
                save()
                recovery = Path(__file__).resolve().parent
                prior = recovery/'status.json'
                if (recovery/'composition-config.json').exists() and prior.exists():
                    (recovery/'recovery-parent-status-at-handoff.json').write_bytes(prior.read_bytes())
                    prior.write_text(json.dumps({'state': 'completed',
                        'scope': 'composition recovery and verified handoff only',
                        'composition': event['composition'], 'continuation_ledger': str(output.resolve())}, indent=2))
        except psutil.NoSuchProcess:
            pass
        ledger['handoff'] = event
        checked_ready('paper-assets/precision-code-ready-20261014.json')
        source = Path('src/paper_study.py')
        original = source.read_bytes()
        text = original.decode('utf-8')
        before = 'prefixes.append("svd-diagnosis-")'
        after = 'prefixes.extend(["svd-diagnosis-", "precision-readiness-"])'
        if before not in text or text.count(before) != 1:
            raise RuntimeError('accounting source changed; inspect before applying boundary edit')
        (output/'paper_study-before-accounting.py').write_bytes(original)
        source.write_bytes(text.replace(before, after, 1).encode('utf-8'))
        ledger['accounting_edit'] = {'file': str(source), 'before_sha256': hashlib.sha256(original).hexdigest(),
                                    'after_sha256': digest(source), 'purpose': 'charge BF16/FP8 controls to original readiness cap'}
        save()
        if execute('precision', ['-m', 'src.paper_precision', '--config', 'configs/paper_exploratory_resume_v2.json', '--offload', 'gpu']):
            error = (output/'precision-stderr.log').read_text(encoding='utf-8')
            if 'OutOfMemoryError' not in error and 'out of memory' not in error.lower():
                # The runner retains its traceback in the run, not necessarily stderr.
                runs = sorted(Path('results/paper-v2').glob('precision-readiness-*'))
                error += (runs[-1]/'stderr.log').read_text(encoding='utf-8') if runs else ''
            if 'OutOfMemoryError' in error or 'out of memory' in error.lower():
                ledger['changed_condition'] = 'retry once with explicit active-weight CPU offload; original failure remains charged'
                save()
                if execute('precision-cpu-offload', ['-m', 'src.paper_precision', '--config', 'configs/paper_exploratory_resume_v2.json', '--offload', 'cpu']):
                    raise RuntimeError('precision CPU-offload continuation incomplete')
            elif newly_budget_limited('precision-readiness-', ledger['stages'][-1]['started_epoch']):
                retain_budget_limit('precision controls')
            else:
                raise RuntimeError('precision controls incomplete; no automatic repeated restart')
        ledger['state'] = 'waiting for verified architecture panel code'
        save()
        ready = Path('paper-assets/architecture-code-ready-20261014.json')
        while not ready.exists():
            time.sleep(2)
        checked_ready(ready)
        config_path = Path('configs/paper_architecture_panel_v3.json')
        config = json.loads(config_path.read_text())
        selected = [row['model'] for row in config['models']] + [config['existing_gemma_reference']['model']]
        if len(set(selected)) != 3 or len(selected) > config['max_models'] or any('gpt2' in name.lower() for name in selected):
            raise ValueError('model panel does not match the user selection')
        panel_before = set(Path('results/paper-v2').glob('architecture-panel-*'))
        if execute('architecture-panel', ['-m', 'src.paper_architecture_panel', '--config', str(config_path)]):
            panels = set(Path('results/paper-v2').glob('architecture-panel-*'))-panel_before
            state = json.loads((next(iter(panels))/'status.json').read_text()) if len(panels) == 1 else {}
            if state.get('budget_limited_models'):
                retain_budget_limit('architecture panel: '+', '.join(state['budget_limited_models']))
            else:
                raise RuntimeError('architecture comparison failed unexpectedly; inspect retained evidence')
        if execute('practical', ['-m', 'src.paper_benchmark', 'run', '--config', 'configs/paper_practical_after_panel_v3.json', '--stage', 'local-core']):
            if newly_budget_limited('practical-', ledger['stages'][-1]['started_epoch']):
                retain_budget_limit('Gemma practical measurements')
            else:
                # The original runner calls its cap-limited terminal state incomplete.
                states = [json.loads((p/'status.json').read_text()) for p in Path('results/paper-v2').glob('practical-*')
                          if (p/'status.json').exists() and (p/'status.json').stat().st_mtime >= ledger['stages'][-1]['started_epoch']]
                if states and all(s.get('state') == 'incomplete' and 'cap exceeded' in s.get('error', '') for s in states):
                    retain_budget_limit('Gemma practical measurements')
                else:
                    raise RuntimeError('practical stage incomplete; inspect retained reports')
        ledger['state'] = 'completed with incomplete cohorts' if ledger['incomplete_cohorts'] else 'completed'
        save()
    except Exception as exc:
        ledger.update(state='stopped', error=str(exc))
        save()
        raise
    finally:
        report = Path('docs/report_revised_study_20261014.py')
        if ledger['state'] in {'completed', 'completed with incomplete cohorts', 'stopped'} and report.exists():
            # The existing base interpreter has Matplotlib; experiment modules
            # remain isolated in the CUDA venv. This reporter imports no models.
            reporting_python = str(Path(sys.base_prefix)/'python.exe')
            command = ['rtk', reporting_python, '-u', str(report), '--root', 'results/paper-v2',
                       '--orchestration', str(output)]
            entry = {'command': subprocess.list2cmdline(command), 'started_epoch': time.time()}
            ledger['final_report'] = entry
            save()
            with (output/'final-report-stdout.log').open('w', encoding='utf-8') as stdout, (output/'final-report-stderr.log').open('w', encoding='utf-8') as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr)
            entry.update(exit_code=result.returncode, ended_epoch=time.time())
            save()


if __name__ == '__main__':
    main()

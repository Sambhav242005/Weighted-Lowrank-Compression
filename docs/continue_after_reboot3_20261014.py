"""Continue independent authorized cohorts after the saved review pause.

No completed experiment is repeated and no one-time source patch is reapplied.
This dispatcher imports no model code, so each GPU subprocess releases its CUDA
context before the next cohort starts.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path('results/paper-v2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    output = ROOT/'revised-orchestration-20261014-4'
    output.mkdir(exist_ok=False)
    python = str(Path('.venv-cuda/Scripts/python.exe').resolve())
    ledger = {'state': 'preparing', 'stages': [], 'incomplete_cohorts': [],
        'prior_orchestration': str((ROOT/'revised-orchestration-20261014-3').resolve()),
        'continuation_decision': 'docs/CONTINUATION_20261014.md',
        'interruption_accounting_path': str((ROOT/'interruption-accounting-20261014-reboot3.json').resolve()),
        'excluded_models': ['openai-community/gpt2'], 'max_models': 4,
        'omitted_optional_controls': ['dense_fp8_e4m3', 'dense_fp8_e5m2']}
    def save():
        (output/'status.json').write_text(json.dumps(ledger, indent=2), encoding='utf-8')
    def execute(label, args, prefix):
        before = set(ROOT.glob(prefix+'*'))
        command = ['rtk', python, '-u', *args]
        entry = {'stage': label, 'command': subprocess.list2cmdline(command),
                 'started_epoch': time.time()}
        ledger['stages'].append(entry)
        ledger['state'] = 'running '+label
        save()
        with (output/(label+'-stdout.log')).open('w', encoding='utf-8') as stdout, (output/(label+'-stderr.log')).open('w', encoding='utf-8') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            entry.update(launcher_pid=child.pid, launcher_start_epoch=psutil.Process(child.pid).create_time())
            save()
            code = child.wait()
        added = sorted(set(ROOT.glob(prefix+'*'))-before)
        entry.update(exit_code=code, ended_epoch=time.time(), outputs=[str(p.resolve()) for p in added])
        save()
        return code, added
    try:
        save()
        active = [(p.pid,p.info['cmdline']) for p in psutil.process_iter(['cmdline'])
                  if any(arg in {'src.paper_precision', 'src.paper_architecture_panel', 'src.paper_architecture_recovery', 'src.paper_benchmark'}
                         for arg in p.info['cmdline'] or [])]
        if active:
            raise RuntimeError(f'experiment already active: {active}')
        if read(ROOT/'composition-20261014-054423-dbd821ff/status.json')['state'] != 'completed':
            raise RuntimeError('saved Gemma composition is not complete')
        for filename, expected in read('paper-assets/architecture-code-ready-20261014.json')['artifacts'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise RuntimeError('architecture launch gate changed: '+filename)
        prior = read(ROOT/'precision-readiness-20261014-060840-6db37daf/status.json')
        if prior['state'] != 'cancelled' or len(prior['completed_variants']) != 7:
            raise RuntimeError('unexpected precision review snapshot')
        ledger['preserved_precision_snapshot'] = prior
        save()
        code, panels = execute('architecture-panel', ['-m', 'src.paper_architecture_recovery',
            '--config', 'configs/paper_architecture_recovery_v4.json'], 'architecture-panel-')
        if code:
            status = read(panels[0]/'status.json') if len(panels) == 1 else {}
            if not status.get('budget_limited_models'):
                raise RuntimeError('unexpected architecture failure; inspect retained run before recovery')
            ledger['incomplete_cohorts'].append({'stage': 'architecture panel', 'evidence': status})
            save()
        code, practical = execute('practical', ['-m', 'src.paper_benchmark', 'run',
            '--config', 'configs/paper_practical_after_panel_v3.json', '--stage', 'local-core'], 'practical-')
        if code:
            states = [read(p/'status.json') for p in practical if (p/'status.json').exists()]
            if not states or not all(s.get('state') == 'incomplete' and 'cap exceeded' in s.get('error', '') for s in states):
                raise RuntimeError('unexpected practical failure; inspect retained evidence')
            ledger['incomplete_cohorts'].append({'stage': 'Gemma practical', 'evidence': states})
        ledger['state'] = 'completed with incomplete cohorts' if ledger['incomplete_cohorts'] else 'completed'
        save()
    except Exception as exc:
        ledger.update(state='stopped', error=str(exc))
        save()
        raise
    finally:
        if ledger['state'] in {'completed', 'completed with incomplete cohorts', 'stopped'}:
            command = ['rtk', 'C:/Python314/python.exe', '-u',
                'docs/report_revised_study_20261014.py', '--root', str(ROOT), '--orchestration', str(output)]
            entry = {'command': subprocess.list2cmdline(command), 'started_epoch': time.time()}
            ledger['final_report'] = entry
            save()
            with (output/'final-report-stdout.log').open('w', encoding='utf-8') as stdout, (output/'final-report-stderr.log').open('w', encoding='utf-8') as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr)
            entry.update(exit_code=result.returncode, ended_epoch=time.time())
            save()


if __name__ == '__main__':
    main()

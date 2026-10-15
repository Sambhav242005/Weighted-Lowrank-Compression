"""Serialized handoff for the user's revised precision/architecture study.

No command starts before the verified composition-boundary controller stops
the old dispatcher. Old source is archived before the accounting-only edit.
"""
import hashlib
import json
import subprocess
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
              'excluded_models': ['openai-community/gpt2'], 'max_models': 4}
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
    try:
        save()
        while not handoff.exists():
            time.sleep(2)
        event = json.loads(handoff.read_text())
        try:
            process = psutil.Process(event['pid'])
            if process.is_running() and 'src.paper_benchmark' in process.cmdline():
                raise RuntimeError('old dispatcher remains active')
        except psutil.NoSuchProcess:
            pass
        ledger['handoff'] = event
        if event['composition_state']['state'] != 'completed':
            raise RuntimeError('composition was incomplete; retain evidence rather than advance')
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
        if execute('architecture-panel', ['-m', 'src.paper_architecture_panel', '--config', str(config_path)]):
            raise RuntimeError('architecture comparison incomplete; original practical stage not advanced')
        if execute('practical', ['-m', 'src.paper_benchmark', 'run', '--config', 'configs/paper_practical_after_panel_v3.json', '--stage', 'local-core']):
            raise RuntimeError('practical stage incomplete; inspect retained reports')
        ledger['state'] = 'completed'
        save()
    except Exception as exc:
        ledger.update(state='stopped', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()

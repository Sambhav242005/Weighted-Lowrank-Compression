"""Recover completed records after reboot; never dispatch the excluded GPT-2 stage."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('results/paper-v2')
    evidence = root/'reboot-recovery-20261014-2'
    evidence.mkdir(exist_ok=False)
    old = root/'composition-20261014-032120-486d799f'
    prior_status = json.loads((old/'status.json').read_text())
    if prior_status['state'] != 'running':
        raise RuntimeError('unexpected terminal state; inspect before recovery')
    active = [p.pid for p in psutil.process_iter(['cmdline'])
              if p.pid != psutil.Process().pid and any(
                  term in ' '.join(p.info['cmdline'] or [])
                  for term in ('src.paper_benchmark run', 'src.paper_precision --config',
                               'src.paper_architecture_panel --config'))]
    if active:
        raise RuntimeError(f'experiment process already active: {active}')
    # New boot time is after the old process died, so this charges the whole
    # launch-to-new-boot interval rather than guessing the unlogged tail.
    charge = psutil.boot_time()-1791948078.969369
    event = {'state': 'interrupted', 'reason': 'User-reported PC restart; old process absent',
             'old_pid': 2588, 'new_boot_epoch': psutil.boot_time(),
             'prior_process_start_epoch': 1791948078.969369,
             'elapsed_seconds': charge, 'complete_records': len(json.loads((old/'metrics.json').read_text())),
             'accounting': 'Conservative launch-to-new-boot wall interval, including restart downtime',
             'classification': 'exploratory; earlier first-reboot accounting remains estimated'}
    (evidence/'prior-composition-status.json').write_bytes((old/'status.json').read_bytes())
    save(evidence/'interruption.json', event)
    save(old/'interruption-evidence-second-reboot.json', event)
    save(old/'status.json', event)
    manifest = old/'artifact-integrity.json'
    if manifest.exists():
        raise RuntimeError('refuse to replace an existing integrity manifest')
    save(manifest, {'artifacts': {name: sha(old/name) for name in
          ('metrics.json', 'contract.json', 'config.json', 'token-manifests.json', 'frozen-selection.json', 'teacher.json')}})
    parent = root/'study-20261014-032120-72463b92'
    (evidence/'prior-study-status.json').write_bytes((parent/'status.json').read_bytes())
    save(parent/'status.json', event)
    from src.paper_study import spent_seconds
    config = json.loads(Path('configs/paper_exploratory_composition_resume_v2.json').read_text())
    config.pop('gpt2_model_path', None)
    config['composition_resume_path'] = str(old.resolve())
    remaining = config['stage_gpu_hours']['composition']*3600-spent_seconds(root, 'composition')
    if remaining <= 0:
        raise RuntimeError('composition budget exhausted')
    config['runtime_remaining_seconds'] = {'composition': remaining}
    save(evidence/'composition-config.json', config)
    save(evidence/'status.json', {'state': 'running composition', 'remaining_seconds': remaining})
    before = set(root.glob('composition-*'))
    from src.paper_followups import run_followups
    code = run_followups(config, 'composition')
    added = set(root.glob('composition-*'))-before
    if code or len(added) != 1:
        save(evidence/'status.json', {'state': 'stopped', 'exit_code': code})
        return 1
    completed = added.pop()
    # Reuse the already reviewed serial dispatcher with fresh ledger paths.
    handoff = root/'model-change-handoff-20261014-2'
    handoff.mkdir(exist_ok=False)
    save(handoff/'handoff.json', {'pid': psutil.Process().pid,
         'composition_state': json.loads((completed/'status.json').read_text()),
         'composition': str(completed.resolve()), 'gpt2_removed': True,
         'reason': 'Reboot recovery completed composition; explicit model-panel amendment'})
    driver = Path('docs/run_revised_study_20261014.py').read_text()
    driver = driver.replace('revised-orchestration-20261014-1', 'revised-orchestration-20261014-2')
    driver = driver.replace('model-change-handoff-20261014-1', 'model-change-handoff-20261014-2')
    target = evidence/'revised-driver.py'
    target.write_text(driver, encoding='utf-8')
    save(evidence/'status.json', {'state': 'running revised dispatcher', 'composition': str(completed)})
    return subprocess.call(['rtk', sys.executable, '-u', str(target)])


if __name__ == '__main__':
    raise SystemExit(main())

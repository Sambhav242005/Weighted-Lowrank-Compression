"""Create append-only third-reboot accounting and continuation launch inputs."""
import hashlib
import json
from pathlib import Path

root = Path('results/paper-v2')
old = root/'architecture-replication-20261014-082807-5c73f669'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2)
names = ('contract.json', 'metrics.json', 'readiness.json', 'validation-teacher.json', 'config.json', 'status.json')
entry = {'run_id': old.name, 'run': str(old.resolve()), 'stage': 'replication',
    'model': 'Qwen/Qwen3.5-0.8B', 'state': 'interrupted by reboot',
    'started_epoch': 1791966485.6441517, 'new_boot_epoch': 1791970132.8027894,
    'charged_seconds': 1791970132.8027894-1791966485.6441517,
    'basis': 'Conservative actual Python process launch-to-next-system-boot wall interval; includes reboot gap; original status retained unchanged.',
    'source_artifacts': {name: digest(old/name) for name in names},
    'records_saved': len(json.loads((old/'metrics.json').read_text())),
    'independent_replication': False}
sidecar = root/'interruption-accounting-20261014-reboot3.json'
save(sidecar, {'schema': 'researchcompression-interruption-accounting-v1', 'runs': [entry],
    'prior_orchestration': str((root/'revised-orchestration-20261014-3').resolve()),
    'recovery_orchestration': str((root/'revised-orchestration-20261014-4').resolve()),
    'recovery_sources': {entry['model']: str(old.resolve())},
    'note': 'Preserved source rows are reused verbatim in successor metrics; never sum both copies as independent observations.'})
config = json.loads(Path('configs/paper_architecture_panel_v3.json').read_text())
config.update(interrupted_runs={str(old.resolve()): entry}, recovery_sources={entry['model']: str(old.resolve())},
    interruption_accounting_path=str(sidecar.resolve()))
save('configs/paper_architecture_recovery_v4.json', config)
source = Path('docs/continue_after_review_20261014.py').read_text()
source = source.replace('revised-orchestration-20261014-3', 'revised-orchestration-20261014-4')
source = source.replace("ROOT/'revised-orchestration-20261014-2'", "ROOT/'revised-orchestration-20261014-3'")
source = source.replace("'continuation_decision': 'docs/CONTINUATION_20261014.md',", "'continuation_decision': 'docs/CONTINUATION_20261014.md',\n        'interruption_accounting_path': str((ROOT/'interruption-accounting-20261014-reboot3.json').resolve()),")
source = source.replace("'-m', 'src.paper_architecture_panel'", "'-m', 'src.paper_architecture_recovery'")
source = source.replace('configs/paper_architecture_panel_v3.json', 'configs/paper_architecture_recovery_v4.json')
source = source.replace("'src.paper_precision', 'src.paper_architecture_panel', 'src.paper_benchmark'", "'src.paper_precision', 'src.paper_architecture_panel', 'src.paper_architecture_recovery', 'src.paper_benchmark'")
# Explicit system interpreter avoids package differences in document exporters.
source = source.replace("str(Path(sys.base_prefix)/'python.exe')", "'C:/Python314/python.exe'")
with Path('docs/continue_after_reboot3_20261014.py').open('x', encoding='utf-8') as handle:
    handle.write(source)
print(sidecar)

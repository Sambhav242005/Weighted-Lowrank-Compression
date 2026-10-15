"""Launch only the reviewed additive recovery dispatcher and record its identity."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import psutil

launch = Path('results/paper-v2/continuation-launch-20261014-4')
launch.mkdir(exist_ok=False)
files = ['src/paper_architecture_panel.py', 'src/paper_architecture_recovery.py',
    'configs/paper_architecture_panel_v3.json', 'configs/paper_architecture_recovery_v4.json',
    'docs/continue_after_review_20261014.py', 'docs/continue_after_reboot3_20261014.py',
    'results/paper-v2/revised-orchestration-20261014-3/status.json',
    'results/paper-v2/architecture-panel-20261014-082806-42b906d1/status.json',
    'results/paper-v2/interruption-accounting-20261014-reboot3.json']
hashes = {}
for index, name in enumerate(files):
    content = Path(name).read_bytes()
    (launch/f'{index:02d}-{Path(name).name}').write_bytes(content)
    hashes[name] = hashlib.sha256(content).hexdigest()
command = ['rtk', str(Path('.venv-cuda/Scripts/python.exe').resolve()), '-u', 'docs/continue_after_reboot3_20261014.py']
with (launch/'stdout.log').open('w', encoding='utf-8') as stdout, (launch/'stderr.log').open('w', encoding='utf-8') as stderr:
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    record = {'command': subprocess.list2cmdline(command), 'launcher_pid': process.pid,
        'launcher_start_epoch': psutil.Process(process.pid).create_time(), 'launched_epoch': time.time(),
        'preserved_files': hashes, 'ledger': str(Path('results/paper-v2/revised-orchestration-20261014-4/status.json').resolve())}
    (launch/'launch.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))

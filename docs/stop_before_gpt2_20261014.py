"""Honor the requested model change at the next completed Gemma stage."""
import argparse
import json
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--composition', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=False)
    process = psutil.Process(args.pid)
    command = process.cmdline()
    if abs(process.create_time()-args.start) > .01 or 'src.paper_benchmark' not in command or 'local-core' not in command:
        raise RuntimeError('process identity does not match the authorized study')
    while True:
        if not process.is_running():
            raise RuntimeError('study exited before the stage boundary; inspect retained results')
        try:
            state = json.loads((args.composition/'status.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)
            continue
        if state.get('state') in {'completed', 'failed', 'incomplete', 'cancelled'}:
            break
        time.sleep(1)
    if abs(process.create_time()-args.start) > .01 or process.cmdline() != command:
        raise RuntimeError('process identity changed at boundary')
    process.terminate()
    process.wait(timeout=20)
    event = {'state': 'cancelled', 'reason': 'User replaced GPT-2 with a maximum-four-model architecture study; stop old dispatcher after Gemma composition',
        'composition_state': state, 'pid': args.pid, 'command': command, 'epoch': time.time(),
        'composition': str(args.composition.resolve()), 'gpt2_removed': True}
    partial = []
    for directory in args.study.parent.glob('replication-*'):
        status = directory/'status.json'
        if not status.exists():
            continue
        previous = json.loads(status.read_text())
        if previous.get('state') != 'running':
            continue
        birth = getattr(directory.stat(), 'st_birthtime', directory.stat().st_ctime)
        (args.evidence/(directory.name+'-prior-status.json')).write_text(json.dumps(previous, indent=2))
        previous.update(state='cancelled', reason=event['reason'], elapsed_seconds=max(60, time.time()-birth+10))
        status.write_text(json.dumps(previous, indent=2))
        partial.append(str(directory.resolve()))
    event['partial_replication'] = partial
    parent_status = args.study/'status.json'
    (args.evidence/'prior-study-status.json').write_bytes(parent_status.read_bytes())
    parent_status.write_text(json.dumps(event, indent=2))
    (args.evidence/'handoff.json').write_text(json.dumps(event, indent=2))
    print(json.dumps(event, indent=2), flush=True)


if __name__ == '__main__':
    main()

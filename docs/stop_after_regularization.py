"""Controlled stage-boundary handoff; preserve B and charge partial C time.

This is local process coordination, not a fitting or evaluation implementation.
The process identity and expected command are checked before termination.
"""
import argparse
import json
import time
from pathlib import Path

import psutil


def write_new(path, obj):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected-start", type=float, required=True)
    parser.add_argument("--stage-directory", type=Path, required=True)
    parser.add_argument("--study-directory", type=Path, required=True)
    parser.add_argument("--launch-directory", type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process(args.pid)
    command = process.cmdline()
    if abs(process.create_time() - args.expected_start) > 0.01:
        raise RuntimeError("process identity changed")
    if "src.paper_benchmark" not in command or "local-core" not in command:
        raise RuntimeError("unexpected process command")
    print("Waiting for regularization completion before metric repair", flush=True)
    while True:
        if not process.is_running():
            raise RuntimeError("study exited before boundary handoff")
        status_path = args.stage_directory / "status.json"
        try:
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if status.get("state") == "completed":
            selection_path = args.stage_directory / "selection.json"
            selection = json.loads(selection_path.read_text())
            if selection.get("state") != "frozen":
                raise RuntimeError("completed stage lacks frozen selection")
            break
        if status.get("state") in {"failed", "incomplete", "cancelled"}:
            raise RuntimeError(f"regularization ended unexpectedly: {status}")
        time.sleep(1)
    # Recheck to guard against PID reuse during the wait.
    if abs(process.create_time() - args.expected_start) > 0.01 or process.cmdline() != command:
        raise RuntimeError("process identity changed at boundary")
    process.terminate()
    process.wait(timeout=15)
    event = {"state": "cancelled", "reason": "planned boundary handoff after complete Stage B to repair scale invariance of secondary CKA metric",
             "pid": args.pid, "command": command, "stage_B": str(args.stage_directory.resolve()),
             "selection": str(selection_path.resolve()), "time_utc_epoch": time.time(),
             "resume_stage": "baselines"}
    write_new(args.launch_directory / "boundary-handoff.json", event)
    # The parent may have entered C during the sub-second boundary race. Charge
    # a conservative upper bound and preserve every partial artifact.
    created = []
    for candidate in args.study_directory.parent.glob("baselines-*"):
        config_path = candidate / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if config.get("selection_path") != str(selection_path.resolve()):
            continue
        terminal_path = candidate / "status.json"
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
        if terminal.get("state") in {"completed", "failed", "incomplete"}:
            continue
        birth = getattr(candidate.stat(), "st_birthtime", candidate.stat().st_ctime)
        charged = max(60.0, time.time()-birth+10.0)
        terminal.update(state="cancelled", elapsed_seconds=charged,
            cost_accounting="conservative boundary-handoff charge; no completed C comparison discarded",
            reason=event["reason"])
        terminal_path.write_text(json.dumps(terminal, indent=2)+"\n", encoding="utf-8")
        created.append(str(candidate.resolve()))
    parent_status = args.study_directory / "status.json"
    parent_status.write_text(json.dumps({**event, "partial_C_directories": created}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(event, indent=2), flush=True)


if __name__ == "__main__":
    main()

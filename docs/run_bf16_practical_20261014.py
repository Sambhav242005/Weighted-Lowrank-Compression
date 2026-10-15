"""Serial BF16-only continuation; no legacy experiment dispatcher is invoked."""
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    root = Path("results/paper-v2")
    output = root/"revised-orchestration-20261014-5-bf16"
    output.mkdir(exist_ok=False)
    ledger = {"state": "preparing", "stages": [], "precision": "bfloat16", "max_seeds": 2,
              "seeds": [11, 23], "expected_records": 5,
              "authorization": "Latest human instruction: do remaining tests, BF16 only, maximum two seeds",
              "prior_orchestration": str((root/"revised-orchestration-20261014-4").resolve()),
              "interruption_accounting_path": str((root/"interruption-accounting-20261014-bf16-only.json").resolve()),
              "incomplete_cohorts": [{"stage": "Qwen validation frontier", "saved": 10, "expected": 18,
                                       "reason": "original per-model allocation exhausted; no restart"}],
              "omitted_optional_controls": ["dense_fp8_e4m3", "dense_fp8_e5m2"]}
    def save():
        (output/"status.json").write_text(json.dumps(ledger, indent=2)+"\n")
    save()
    active = [p.pid for p in psutil.process_iter(["cmdline"], ad_value=None)
              if any(a in {"src.paper_benchmark", "src.paper_architecture_recovery", "src.paper_precision", "src.paper_bf16_practical"}
                     for a in (p.info["cmdline"] or []))]
    if active:
        raise RuntimeError(f"another experiment is active: {active}")
    before = set(root.glob("practical-bf16-*"))
    command = ["rtk", str(Path(".venv-cuda/Scripts/python.exe").resolve()), "-u", "-m", "src.paper_bf16_practical",
               "--config", "configs/paper_bf16_practical_two_seeds_20261014.json"]
    entry = {"stage": "BF16 practical", "command": subprocess.list2cmdline(command), "started_epoch": time.time()}
    ledger["stages"].append(entry)
    ledger["state"] = "running BF16 practical"
    save()
    with (output/"practical-stdout.log").open("w") as stdout, (output/"practical-stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        entry.update(launcher_pid=process.pid, launcher_start_epoch=psutil.Process(process.pid).create_time())
        save()
        code = process.wait()
    added = sorted(set(root.glob("practical-bf16-*"))-before)
    entry.update(exit_code=code, ended_epoch=time.time(), outputs=[str(p.resolve()) for p in added])
    ledger["state"] = "completed with incomplete cohorts" if code == 0 else "stopped"
    if code:
        ledger["error"] = "BF16 practical incomplete or failed; inspect retained status, no automatic retry"
    save()
    command = ["rtk", "C:/Python314/python.exe", "-u", "docs/report_revised_study_20261014.py",
               "--root", str(root), "--orchestration", str(output)]
    final = {"command": subprocess.list2cmdline(command), "started_epoch": time.time()}
    ledger["final_report"] = final
    save()
    with (output/"final-report-stdout.log").open("w") as stdout, (output/"final-report-stderr.log").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    final.update(exit_code=result.returncode, ended_epoch=time.time())
    save()


if __name__ == "__main__":
    main()

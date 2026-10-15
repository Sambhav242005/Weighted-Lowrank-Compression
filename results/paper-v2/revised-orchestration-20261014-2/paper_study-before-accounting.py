"""Sequential bounded execution, with cumulative caps and preserved failures."""
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

from .paper_benchmark import create_output, write, environment, command_line


STAGES = ["regularization", "baselines", "composition", "replication", "practical"]


def spent_seconds(root, stage):
    """Use retained terminal elapsed time; reject unknown cost rather than reset it."""
    total = 0.0
    prefixes = [stage+"-"]
    if stage == "readiness":
        prefixes.append("svd-diagnosis-")
    for directory in Path(root).iterdir():
        if not directory.is_dir() or not any(directory.name.startswith(p) for p in prefixes):
            continue
        status_file = directory/"status.json"
        if not status_file.exists():
            raise RuntimeError(f"unresolved prior run: {directory}")
        status = json.loads(status_file.read_text())
        if status["state"] in {"running", "preparing"}:
            raise RuntimeError(f"prior experiment still active: {directory}")
        if status["state"] == "waiting for a human":
            continue
        if status.get("phase") == "initialization" and status.get("observed_process_exit_code") == 1:
            # The retained import failure happened before CUDA model loading.
            continue
        elapsed = status.get("elapsed_seconds")
        if elapsed is None:
            # Older failed readiness artifact: observed log captures duration
            # between first and final checkpoint, conservatively include startup.
            from datetime import datetime
            lines=(directory/"stdout.log").read_text().splitlines() if (directory/"stdout.log").exists() else []
            if not lines:
                raise RuntimeError(f"prior cost unknown: {directory}")
            begin=datetime.fromisoformat(lines[0].split()[0])
            end=datetime.fromisoformat(lines[-1].split()[0])
            elapsed=(end-begin).total_seconds()+60
        if not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError("invalid recorded runtime")
        total += elapsed
    return total


def run_study(config):
    from .paper_comparisons import run_comparisons
    from .paper_followups import run_followups
    from .paper_reporting import report_study
    config=dict(config)
    output=create_output(config,"study")
    ledger={"run_id":output.name,"declared_gpu_hours":config["stage_gpu_hours"],
            "accounting":"conservative elapsed wall time, including fitting and evaluation",
            "stages":[],"readiness_passed":config.get("readiness_passed",False)}
    write(output/"contract.json",{"environment":environment(config),"configuration":config,
                                  "command":command_line(),"ledger":ledger})
    (output/"stderr.log").touch()
    def progress(message):
        print(message,flush=True)
        with (output/"stdout.log").open("a",encoding="utf-8") as handle:
            handle.write(message+"\n")
    try:
        evidence=Path(config["readiness_evidence"])
        if json.loads((evidence/"status.json").read_text())["state"] != "completed":
            raise RuntimeError("diagnosis must complete before downstream exploratory comparisons")
        ledger["readiness_spent_seconds"]=spent_seconds(config["output_root"],"readiness")
        if ledger["readiness_spent_seconds"] > config["stage_gpu_hours"]["readiness"]*3600:
            raise RuntimeError("Stage A allocation exhausted")
        if not config.get("readiness_passed",False):
            progress("Strict readiness gate failed: all downstream results are exploratory; promotion disabled.")
        write(output/"status.json",{"state":"running"})
        start_stage=config.get("start_stage", "regularization")
        if start_stage not in STAGES:
            raise ValueError("unsupported resume stage")
        if start_stage!="regularization":
            selection=Path(config["selection_path"])
            if json.loads((selection.parent/"status.json").read_text())["state"]!="completed":
                raise RuntimeError("resume requires complete preserved Stage B")
            if json.loads(selection.read_text()).get("state")!="frozen":
                raise RuntimeError("resume selection is not frozen")
            ledger["reused_regularization"]={"selection_path":str(selection.resolve()),
                "spent_seconds":spent_seconds(config["output_root"],"regularization")}
        if STAGES.index(start_stage)>STAGES.index("baselines"):
            baseline=Path(config["baseline_run_path"])
            if json.loads((baseline/"status.json").read_text())["state"]!="completed":
                raise RuntimeError("resume requires complete preserved Stage C")
            ledger["reused_baselines"]={"output":str(baseline.resolve()),
                "spent_seconds":spent_seconds(config["output_root"],"baselines")}
        for stage in STAGES[STAGES.index(start_stage):]:
            previous=spent_seconds(config["output_root"],stage)
            remaining=config["stage_gpu_hours"][stage]*3600-previous
            if remaining<=0:
                raise TimeoutError(f"{stage} cumulative allocation exhausted")
            config["runtime_remaining_seconds"]={stage:remaining}
            before=set(Path(config["output_root"]).glob(stage+"-*"))
            progress(f"stage started {stage}; remaining conservative seconds={remaining:.1f}")
            started=time.monotonic()
            code=run_comparisons(config,stage) if stage in {"regularization","baselines"} else run_followups(config,stage)
            after=set(Path(config["output_root"]).glob(stage+"-*"))-before
            if len(after)!=1:
                raise RuntimeError(f"ambiguous stage output for {stage}")
            directory=after.pop()
            entry={"stage":stage,"output":str(directory.resolve()),"exit_code":code,
                   "previous_spent_seconds":previous,"elapsed_seconds":time.monotonic()-started,
                   "remaining_seconds_at_start":remaining}
            ledger["stages"].append(entry)
            write(output/"ledger.json",ledger)
            if entry["elapsed_seconds"] > remaining:
                write(directory/"budget-audit.json",{"stage_complete_within_cumulative_cap":False,
                    "remaining_seconds_at_start":remaining,"observed_elapsed_seconds":entry["elapsed_seconds"],
                    "decision":"retain artifacts as incomplete; do not advance"})
                raise TimeoutError(f"{stage} exceeded cumulative allocation at completion")
            if code!=0:
                raise RuntimeError(f"{stage} stopped; retain partial results in {directory}")
            if stage=="regularization":
                config["selection_path"]=str((directory/"selection.json").resolve())
            elif stage=="baselines":
                config["baseline_run_path"]=str(directory.resolve())
            write(output/"resolved-config.json",config)
            progress(f"stage completed {stage}")
            gc.collect()
            torch.cuda.empty_cache()
        report_study(config,config["study"])
        write(output/"status.json",{"state":"completed","exploratory":not config.get("readiness_passed",False)})
        (output/"report.md").write_text("# Protocol-v2 study\n\nAll declared first-wave stages completed. "
            "Strict readiness status and statistical/promotion limits remain binding; inspect ledger and generated reports.\n",encoding="utf-8")
        return 0
    except Exception as exc:
        (output/"stderr.log").write_text(traceback.format_exc(),encoding="utf-8")
        write(output/"status.json",{"state":"failed","error":str(exc),"exploratory":not config.get("readiness_passed",False)})
        (output/"report.md").write_text(f"# Incomplete study\n\n{exc}\nAll partial artifacts retained.\n",encoding="utf-8")
        progress(f"study stopped: {exc}")
        return 1

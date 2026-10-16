"""Resume only an immutable Gemma270M quality prefix, preserving original research code."""
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from gemma270m_relu_study import (Bottleneck, Budget, chunk_metrics, dump, install,
                                  sha, token_sha, windows, write_report)


def validate_prefix(checkpoint, tokens, methods):
    all_windows = list(windows(tokens))
    completed = checkpoint["completed_windows"]
    if checkpoint["expected_windows"] != len(all_windows) or not 0 < completed < len(all_windows):
        raise ValueError("Checkpoint does not match an unfinished test split")
    expected = sum(end - start - first for start, end, first in all_windows[:completed])
    if set(checkpoint["aggregates"]) != set(methods):
        raise ValueError("Method set changed")
    for a in checkpoint["aggregates"].values():
        if a["count"] != expected or len(a["drift_squared_error"]) != 18 or len(a["drift_teacher_energy"]) != 18:
            raise ValueError("Invalid checkpoint counts or drift arrays")
        values = [a["nll_sum"], a["kl_sum"], *a["drift_squared_error"], *a["drift_teacher_energy"]]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite checkpoint")
    return all_windows


@torch.inference_mode()
def resume_quality(model, dense, variants, tokens, out, budget, checkpoint):
    """Same metrics and iteration order as the original; start after a validated prefix."""
    all_windows = validate_prefix(checkpoint, tokens, ["dense", *variants])
    aggregates = copy.deepcopy(checkpoint["aggregates"])
    completed = checkpoint["completed_windows"]
    with (out / "quality-windows.jsonl").open("x", encoding="utf-8") as log:
        for window_id, (start, end, first) in enumerate(all_windows[completed:], start=completed):
            budget.check()
            ids = tokens[start:end][None].cuda()
            install(model, dense, "cuda")
            teacher = model.model(ids, use_cache=False, output_hidden_states=True)
            th = teacher.last_hidden_state
            residuals = [h.cpu() for h in teacher.hidden_states[1:]]
            for name, mods in [("dense", dense), *variants.items()]:
                budget.check()
                if name == "dense":
                    sh = th
                    student_states = residuals
                else:
                    install(model, mods, "cuda")
                    student = model.model(ids, use_cache=False, output_hidden_states=True)
                    sh = student.last_hidden_state
                    student_states = student.hidden_states[1:]
                a = aggregates[name]
                nll, kl = 0., 0.
                for pos in range(first, end - start, 16):
                    stop = min(pos + 16, end - start)
                    tl = model.lm_head(th[:, pos - 1:stop - 1]).reshape(-1, model.config.vocab_size)
                    sl = tl if name == "dense" else model.lm_head(sh[:, pos - 1:stop - 1]).reshape(-1, model.config.vocab_size)
                    n, k = chunk_metrics(sl, ids[:, pos:stop].reshape(-1), None if name == "dense" else tl)
                    nll += n
                    kl += k
                count = end - start - first
                a["nll_sum"] += nll
                a["kl_sum"] += kl
                a["count"] += count
                for j, (teacher_state, student_state) in enumerate(zip(residuals, student_states)):
                    t = teacher_state[:, first:].float()
                    s = student_state[:, first:].float().cpu()
                    a["drift_squared_error"][j] += float((s - t).square().sum())
                    a["drift_teacher_energy"][j] += float(t.square().sum())
                log.write(json.dumps({"window": window_id, "start": start, "end": end,
                                      "method": name, "nll_sum": nll, "kl_sum": kl, "targets": count}) + "\n")
                log.flush()
                if name != "dense":
                    del student
            if window_id % 10 == 0 or window_id + 1 == len(all_windows):
                print(f"Quality {window_id + 1}/{len(all_windows)} windows, {aggregates['dense']['count']} targets", flush=True)
                # Atomic latest pointer plus immutable numbered checkpoints improve interruption recovery.
                progress = {"completed_windows": window_id + 1, "expected_windows": len(all_windows),
                            "aggregates": aggregates}
                dump(out / f"quality-checkpoint-{window_id + 1:04d}.json", progress)
                temp = out / "quality-progress.tmp"
                dump(temp, progress)
                temp.replace(out / "quality-progress.json")
    result = {}
    for name, a in aggregates.items():
        assert a["count"] == len(tokens) - 1
        result[name] = {"nll": a["nll_sum"] / a["count"],
                        "perplexity": math.exp(a["nll_sum"] / a["count"]),
                        "teacher_to_student_kl": a["kl_sum"] / a["count"], "targets": a["count"],
                        "block_relative_rmse": [math.sqrt(x / max(y, 1e-30)) for x, y in
                                                zip(a["drift_squared_error"], a["drift_teacher_energy"])]}
    return result


def recover(source, out):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.loads((source / "resolved-config.json").read_text())
    checkpoint = json.loads((source / "quality-progress.json").read_text())
    observations = [json.loads(p.read_text()) for p in source.glob("interruption-observation-*.json")]
    observed = min(observations, key=lambda x: x["observed_at_utc"])
    charged = math.ceil((datetime.fromisoformat(observed["observed_at_utc"]) -
                        datetime.fromisoformat(observed["run_directory_created_utc"])).total_seconds())
    remaining = cfg["compute_cap_seconds"] - charged
    if remaining <= 0:
        raise TimeoutError("Original compute allocation exhausted")
    budget = Budget(remaining)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.set_num_interop_threads(1)
    for path, expected in cfg["source_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Original source changed: {path}")
    if torch.cuda.get_device_name() != cfg["gpu"]:
        raise ValueError("GPU changed")
    for package, version in cfg["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Package changed: {package}")
    metrics = json.loads((source / "metrics.json").read_text())
    for name, record in metrics["storage"].items():
        if sha(source / (name + "-full.pt")) != record["full_sha256"]:
            raise ValueError(f"Checkpoint hash changed: {name}")
    cfg["recovery"] = {"source_run": str(source.resolve()), "prefix_windows": checkpoint["completed_windows"],
                       "charged_original_seconds": charged, "remaining_seconds_at_resume": remaining,
                       "accounting": "Original directory creation through first observed interruption, rounded up; includes uncertain tail",
                       "started_utc": datetime.now(timezone.utc).isoformat(),
                       "source_progress_sha256": sha(source / "quality-progress.json"),
                       "source_metrics_sha256": sha(source / "metrics.json"),
                       "source_config_sha256": sha(source / "resolved-config.json"),
                       "recovery_source_sha256": sha(__file__),
                       "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv]),
                       "reused": "Training, timings, storage, parity and quality prefix; no new independent repeats",
                       "discarded_tail": "Any raw windows beyond the last complete drift checkpoint are not counted"}
    dump(out / "resolved-config.json", cfg)
    dump(out / "metrics.json", metrics)
    dump(out / "quality-progress.json", checkpoint)
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    if sha(Path(assets["snapshot"]) / "model.safetensors") != assets["model_sha256"]:
        raise ValueError("Pinned model weights changed")
    tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
    data = load_from_disk(assets["dataset_path"])
    ids = tokenizer("\n\n".join(data["test"]["text"]), add_special_tokens=False,
                    return_attention_mask=False)["input_ids"]
    tokens = torch.tensor(ids, dtype=torch.long)
    if len(tokens) != cfg["splits"]["test"]["tokens"] or token_sha(tokens) != cfg["splits"]["test"]["sha256"]:
        raise ValueError("Test split changed")
    model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
             torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
    model.requires_grad_(False)
    dense = [layer.self_attn.o_proj for layer in model.model.layers]
    variants = {}
    for seed in cfg["seeds"]:
        saved = torch.load(source / f"replacements-seed{seed}.pt", weights_only=True, map_location="cpu")
        for family in ("whitening", "relu"):
            modules = []
            full = torch.load(source / f"{family}_seed{seed}-full.pt", weights_only=True,
                              map_location="cpu", mmap=True)
            for target, state in zip(cfg["targets"], saved[family]):
                for key, tensor in state.items():
                    if not torch.equal(full[target["name"] + "." + key], tensor):
                        raise ValueError("Replacement file differs from verified full checkpoint")
                n_out, n_in = target["shape"]
                nonlinear = family == "relu"
                module = Bottleneck(n_in, n_out, target["relu_width"] if nonlinear else target["linear_rank"], nonlinear)
                module.load_state_dict(state)
                modules.append(module.eval().cuda())
            del full
            variants[f"{family}_seed{seed}"] = modules
    print(f"Verified recovery: {checkpoint['completed_windows']} reused windows, {remaining} seconds remaining", flush=True)
    metrics["quality"] = resume_quality(model, dense, variants, tokens, out, budget, checkpoint)
    metrics["elapsed_compute_seconds"] = charged + time.monotonic() - budget.start
    metrics["recovery_elapsed_seconds"] = time.monotonic() - budget.start
    metrics["artifact_source_run"] = str(source.resolve())
    dump(out / "metrics.json", metrics)
    write_report(out, metrics)
    with (out / "report.md").open("a", encoding="utf-8") as report:
        report.write("\nRecovery reused the interrupted run's fitted weights, CPU/GPU timings and first "
                     f"{checkpoint['completed_windows']} quality windows. Checkpoints remain at `{source.resolve()}`.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "status.json", {"state": "running", "phase": "recovery"})
    with (args.output / "stdout.log").open("x", encoding="utf-8", buffering=1) as stdout, \
         (args.output / "stderr.log").open("x", encoding="utf-8", buffering=1) as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                recover(args.source, args.output)
                dump(args.output / "status.json", {"state": "completed"})
            except BaseException as exc:
                traceback.print_exc()
                dump(args.output / "status.json", {"state": "budget_limited" if isinstance(exc, TimeoutError) else "failed",
                                                    "error": str(exc)})
                raise


if __name__ == "__main__":
    main()

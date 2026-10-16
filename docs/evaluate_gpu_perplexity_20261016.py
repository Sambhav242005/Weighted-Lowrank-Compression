"""GPU-only full native WikiText-2 PPL plus explicit decode-path diagnostic.

No fitting, method selection, new model/data downloads, CPU inference, or changes
to the saved factors. Multi-token production prefill uses cuBLAS in all new paths.
"""
import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu_lowrank_v2 import ReusedGpuLowRank
from gemma270m_relu_study import (Bottleneck, Budget, chunk_metrics, dump, install,
                                  sha, token_sha, windows)


def score_positions(head, hidden, ids, first, chunk=16):
    """Identical shifted-target log-softmax objective to the frozen 270M study."""
    total, count = 0., 0
    for pos in range(first, ids.shape[1], chunk):
        end = min(pos + chunk, ids.shape[1])
        logits = head(hidden[:, pos - 1:end - 1]).reshape(-1, head.out_features)
        nll, _ = chunk_metrics(logits, ids[:, pos:end].reshape(-1))
        total += nll
        count += end - pos
    return total, count


def checkpoint(out, completed, expected, aggregates):
    payload = {"completed_windows": completed, "expected_windows": expected,
               "aggregates": aggregates}
    dump(out / f"checkpoint-{completed:04d}.json", payload)
    temp = out / "progress.tmp"
    dump(temp, payload)
    temp.replace(out / "progress.json")


def verify_coverage(tokens, aggregates):
    for record in aggregates.values():
        if record["targets"] != len(tokens) - 1:
            raise ValueError("Full-test target coverage mismatch")


def run(out):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    budget = Budget(3600)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    original = Path("results/gemma270m-relu/20261016-mvanbe94")
    cfg = json.loads((original / "resolved-config.json").read_text())
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    names = ["dense", "pytorch_lowrank", "staged", "graph_staged", "graph_cublas"]
    contract = {"purpose": "GPU-only test perplexity of existing inference paths; no fitting or selection",
        "compute_cap_seconds": 3600, "model": cfg["model"], "model_revision": cfg["model_revision"],
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1", "dataset_revision": cfg["dataset_revision"],
        "dtype": "float32", "gpu": torch.cuda.get_device_name(), "methods": names,
        "representation": "unchanged whitening factors, all18 attention outputs, rank275",
        "factor_seed": 11, "new_training_seeds": [], "source_run": str(original.resolve()),
        "baseline": "dense and existing GPU PyTorch low-rank, evaluated over identical targets",
        "context": 256, "stride": 128, "logit_chunk": 16, "test_manifest": cfg["splits"]["test"],
        "standard_ppl_path": "multi-token prefill uses existing GPU cuBLAS for new execution modules",
        "decode_diagnostic": {"split": "test", "prefix_tokens": 128, "scored_positions": [128, 256],
                              "teacher_forced": True, "targets": 128, "not_full_test": True},
        "packages": {p: importlib.metadata.version(p) for p in cfg["packages"]},
        "source_hashes": {str(p): sha(p) for p in [Path(__file__), Path("src/gpu_lowrank_v2.py"),
                          Path("src/gpu_lowrank.py"), Path("src/gemma270m_relu_study.py")]},
        "git_revision": subprocess.check_output(["rtk", "git", "rev-parse", "HEAD"], text=True).strip(),
        "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv]),
        "expected_artifacts": ["resolved-config.json", "stdout.log", "stderr.log", "metrics.json",
                               "window-metrics.jsonl", "checkpoint-*.json", "report.md", "audit.json"],
        "limits": "one model/checkpoint and previously seen test set; execution comparison, not fresh generalization"}
    dump(out / "resolved-config.json", contract)
    original_metrics = json.loads((original / "metrics.json").read_text())
    assert sha(original / "whitening_seed11-full.pt") == original_metrics["storage"]["whitening_seed11"]["full_sha256"]
    assert sha(Path(assets["snapshot"]) / "model.safetensors") == assets["model_sha256"]
    saved = torch.load(original / "replacements-seed11.pt", weights_only=True, map_location="cpu")["whitening"]
    full = torch.load(original / "whitening_seed11-full.pt", weights_only=True, map_location="cpu", mmap=True)
    for i, state in enumerate(saved):
        assert all(torch.equal(value, full[f"model.layers.{i}.self_attn.o_proj.{key}"]) for key, value in state.items())
    del full
    tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
    data = load_from_disk(assets["dataset_path"])
    tokens = torch.tensor(tokenizer("\n\n".join(data["test"]["text"]), add_special_tokens=False,
                                    return_attention_mask=False)["input_ids"], dtype=torch.long)
    assert len(tokens) == cfg["splits"]["test"]["tokens"]
    assert token_sha(tokens) == cfg["splits"]["test"]["sha256"]
    model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
    model.requires_grad_(False)
    variants = {"dense": [layer.self_attn.o_proj for layer in model.model.layers],
                "pytorch_lowrank": [], "staged": [], "graph_staged": [], "graph_cublas": []}
    for state in saved:
        layer = Bottleneck(1024, 640, 275).eval().cuda()
        layer.load_state_dict(state)
        layer.requires_grad_(False)
        variants["pytorch_lowrank"].append(layer)
        for mode in ("staged", "graph_staged", "graph_cublas"):
            variants[mode].append(ReusedGpuLowRank(layer.down.weight, layer.up.weight, mode).eval())
    metrics = {"decode_subset": {}, "full_test": {}}
    with torch.inference_mode():
        # This teacher-forced subset exercises decode kernels, unlike standard full-sequence PPL.
        ids = tokens[:256][None].cuda()
        for name, modules in variants.items():
            budget.check()
            install(model, modules, "cuda")
            h = model.model(ids[:, :128], use_cache=True)
            cache = h.past_key_values
            total = 0.
            for pos in range(128, 256):
                budget.check()
                logits = model.lm_head(h.last_hidden_state[:, -1:]).reshape(1, model.config.vocab_size)
                nll, _ = chunk_metrics(logits, ids[:, pos])
                total += nll
                if pos < 255:
                    h = model.model(ids[:, pos:pos + 1], past_key_values=cache, use_cache=True)
                    cache = h.past_key_values
            metrics["decode_subset"][name] = {"targets": 128, "nll_sum": total, "nll": total / 128,
                                               "perplexity": math.exp(total / 128), "scope": "test positions128..255 only"}
            dump(out / "metrics.json", metrics)
            print(f"Decode-only diagnostic complete: {name}, 128 scored targets", flush=True)
        all_windows = list(windows(tokens))
        aggregates = {name: {"nll_sum": 0., "targets": 0} for name in names}
        with (out / "window-metrics.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for index, (start, end, first) in enumerate(all_windows):
                budget.check()
                ids = tokens[start:end][None].cuda()
                for name, modules in variants.items():
                    budget.check()
                    install(model, modules, "cuda")
                    hidden = model.model(ids, use_cache=False).last_hidden_state
                    nll, count = score_positions(model.lm_head, hidden, ids, first)
                    a = aggregates[name]
                    a["nll_sum"] += nll
                    a["targets"] += count
                    log.write(json.dumps({"window": index, "start": start, "end": end,
                                          "first_local_target": first, "method": name,
                                          "targets": count, "nll_sum": nll}) + "\n")
                if index % 10 == 0 or index + 1 == len(all_windows):
                    checkpoint(out, index + 1, len(all_windows), aggregates)
                    print(f"Full test PPL: {index + 1}/{len(all_windows)} windows; "
                          f"{aggregates['dense']['targets']} targets per method", flush=True)
        verify_coverage(tokens, aggregates)
        for name, a in aggregates.items():
            mean = a["nll_sum"] / a["targets"]
            metrics["full_test"][name] = {**a, "nll": mean, "perplexity": math.exp(mean)}
        metrics["elapsed_seconds"] = time.monotonic() - budget.start
        dump(out / "metrics.json", metrics)
    # Host-only artifact audit, no additional inference.
    counts = {name: 0 for name in names}
    nlls = {name: 0. for name in names}
    seen = set()
    with (out / "window-metrics.jsonl").open(encoding="utf-8") as log:
        for line in log:
            row = json.loads(line)
            key = (row["window"], row["method"])
            assert key not in seen
            seen.add(key)
            start, end, first = all_windows[row["window"]]
            assert (row["start"], row["end"], row["first_local_target"]) == (start, end, first)
            assert row["targets"] == end - start - first
            counts[row["method"]] += row["targets"]
            nlls[row["method"]] += row["nll_sum"]
    assert len(seen) == len(names) * len(all_windows)
    assert all(counts[name] == len(tokens) - 1 and nlls[name] == aggregates[name]["nll_sum"] for name in names)
    assert all(sha(path) == value for path, value in contract["source_hashes"].items())
    dump(out / "audit.json", {"state": "passed", "windows_per_method": len(all_windows),
         "targets_per_method": len(tokens) - 1, "source_hashes_match": True,
         "unique_window_method_rows": len(seen), "nll_recomputed_from_rows": True,
         "elapsed_within_cap": metrics["elapsed_seconds"] <= 3600})
    lines = ["# GPU-only WikiText-2 test perplexity", "", "FP32, context/stride256/128, complete native test tail.",
             "Same saved rank275 factors; no ReLU, fitting or precision changes.", "",
             "| Metric | Dense | Old PyTorch low-rank | New staged | New graph staged | New graph cuBLAS |",
             "|---|---:|---:|---:|---:|---:|",
             "| Full test perplexity | " + " | ".join(f"{metrics['full_test'][name]['perplexity']:.8f}" for name in names) + " |",
             "| Full test targets | " + " | ".join(str(metrics['full_test'][name]['targets']) for name in names) + " |",
             "| Decode-subset perplexity (128 targets only) | " + " | ".join(f"{metrics['decode_subset'][name]['perplexity']:.8f}" for name in names) + " |",
             "", "Standard full-sequence PPL exercises the shared cuBLAS prefill path for new modules.",
             "The separate teacher-forced decode subset exercises custom/graph paths; it is not full-test PPL.",
             "These are repeated measurements on previously seen data, not independent generalization evidence."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "status.json", {"state": "running"})
    with (args.output / "stdout.log").open("x", encoding="utf-8", buffering=1) as out, \
         (args.output / "stderr.log").open("x", encoding="utf-8", buffering=1) as err:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                run(args.output)
                dump(args.output / "status.json", {"state": "completed"})
            except BaseException as exc:
                traceback.print_exc()
                dump(args.output / "status.json", {"state": "budget_limited" if isinstance(exc, TimeoutError) else "failed",
                                                    "error": str(exc)})
                raise


if __name__ == "__main__":
    main()

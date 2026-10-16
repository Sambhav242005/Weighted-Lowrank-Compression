"""GPU-only engineering comparison of unchanged Gemma low-rank factors."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu_lowrank import GpuLowRank, fused_lowrank, runtime, SOURCE, source_hash
from gemma270m_relu_study import Bottleneck, Budget, dump, install, sha, token_sha


def measured(fn, budget, repeats=1):
    budget.check()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return {"wall_ms": (time.perf_counter() - begin) * 1000 / repeats,
            "cuda_interval_ms": start.elapsed_time(end) / repeats,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def summarize(samples):
    return {"samples": samples, **{key: {"median": statistics.median(s[key] for s in samples),
                                          "p95": float(np.percentile([s[key] for s in samples], 95))}
                                     for key in ("wall_ms", "cuda_interval_ms")}}


def run(out):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    budget = Budget(900)
    torch.set_num_threads(4)  # Host overhead only; no CPU inference is performed.
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    source = Path("results/gemma270m-relu/20261016-mvanbe94")
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    original = json.loads((source / "resolved-config.json").read_text())
    original_metrics = json.loads((source / "metrics.json").read_text())
    assert sha(source / "whitening_seed11-full.pt") == original_metrics["storage"]["whitening_seed11"]["full_sha256"]
    saved = torch.load(source / "replacements-seed11.pt", weights_only=True, map_location="cpu")["whitening"]
    full = torch.load(source / "whitening_seed11-full.pt", weights_only=True, map_location="cpu", mmap=True)
    for i, state in enumerate(saved):
        assert all(torch.equal(value, full[f"model.layers.{i}.self_attn.o_proj.{key}"]) for key, value in state.items())
    del full
    assert sha(Path(assets["snapshot"]) / "model.safetensors") == assets["model_sha256"]
    rt = runtime(torch.device("cuda", torch.cuda.current_device()))
    (out / "kernel.cu").write_text(SOURCE, encoding="utf-8")
    (out / "kernel.ptx").write_bytes(rt.ptx)
    contract = {"kind": "GPU-only execution optimization; no refitting or activation changes",
        "model": original["model"], "model_revision": original["model_revision"],
        "dataset_revision": original["dataset_revision"], "source_run": str(source.resolve()),
        "factor_seed": 11, "factor_rank": 275, "targets": original["targets"], "dtype": "float32",
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
        "packages": {p: importlib.metadata.version(p) for p in original["packages"]},
        "source_hashes": {__file__: sha(__file__), "src/gpu_lowrank.py": sha("src/gpu_lowrank.py")},
        "kernel_sha256": source_hash(), "nvrtc_options": [x.decode() for x in rt.options],
        "compute_cap_seconds": 900, "git_revision": subprocess.check_output(["rtk", "git", "rev-parse", "HEAD"], text=True).strip(),
        "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv]),
        "micro_grid": {"rows": [1, 4, 16, 128, 256], "tiles": [64, 128, 640],
                       "warmups": 10, "samples": 20, "calls_per_sample": 10},
        "model_timing": {"warmups": 5, "samples": 20, "order": "rotated by sample",
                         "prefill": [128, 256], "decode": "128-token prompt plus 32 cached steps, no EOS stopping"},
        "quality": "Eight original validation windows; last-position logits and full hidden-state numerical parity; no full-test PPL rerun",
        "memory_scope": "Retained reference and candidate modules; not isolated deployment memory",
        "expected_artifacts": ["stdout.log", "stderr.log", "resolved-config.json", "metrics.json", "report.md", "kernel.cu", "kernel.ptx"]}
    dump(out / "resolved-config.json", contract)
    v, u = saved[0]["down.weight"].cuda(), saved[0]["up.weight"].cuda()
    metrics = {"micro": {}, "model": {}, "parity": {}, "factor_parameters_unchanged": True}
    with torch.inference_mode():
        for rows in (1, 4, 16, 128, 256):
            budget.check()
            x = torch.randn(rows, 1024, device="cuda")
            reference = F.linear(F.linear(x, v), u)
            methods = {"pytorch": lambda: F.linear(F.linear(x, v), u)}
            for tile in (64, 128, 640):
                actual = fused_lowrank(x, v, u, tile=tile)
                relative = float((actual - reference).norm() / reference.norm())
                if not relative < 2e-5:
                    raise ValueError(f"Micro parity failed: {rows}, {tile}, {relative}")
                methods[f"fused_tile{tile}"] = lambda tile=tile: fused_lowrank(x, v, u, tile=tile)
            records = {name: [] for name in methods}
            for fn in methods.values():
                for _ in range(10):
                    fn()
            for sample in range(20):
                keys = list(methods)
                order = keys[sample % len(keys):] + keys[:sample % len(keys)]
                for name in order:
                    records[name].append(measured(methods[name], budget, repeats=10))
            metrics["micro"][str(rows)] = {name: summarize(samples) for name, samples in records.items()}
            dump(out / "metrics.json", metrics)
            print(f"Micro complete: rows={rows}", flush=True)
        chosen = min((64, 128, 640), key=lambda t: metrics["micro"]["1"][f"fused_tile{t}"]["wall_ms"]["median"])
        metrics["chosen_tile"] = chosen
        contract["tile_selection"] = {"tile": chosen, "rule": "lowest median 1-row host-wall latency on fixed synthetic micro grid"}
        dump(out / "resolved-config.json", contract)
        tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
        data = load_from_disk(assets["dataset_path"])
        tokens = torch.tensor(tokenizer("\n\n".join(data["validation"]["text"]), add_special_tokens=False,
                                        return_attention_mask=False)["input_ids"], dtype=torch.long)
        assert token_sha(tokens) == original["splits"]["validation"]["sha256"]
        model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
        model.requires_grad_(False)
        dense = [layer.self_attn.o_proj for layer in model.model.layers]
        linear, fused, hybrid = [], [], []
        for state in saved:
            module = Bottleneck(1024, 640, 275, False).eval().cuda()
            module.load_state_dict(state)
            module.requires_grad_(False)
            linear.append(module)
            fused.append(GpuLowRank(module.down.weight, module.up.weight, tile=chosen).eval())
            hybrid.append(GpuLowRank(module.down.weight, module.up.weight, tile=chosen, decode_only=True).eval())
        variants = {"dense": dense, "pytorch_lowrank": linear, "fused_all_rows": fused,
                    "fused_decode_pytorch_prefill": hybrid}
        for pos in original["splits"]["validation"]["positions"]:
            budget.check()
            ids = tokens[pos:pos + 256][None].cuda()
            install(model, linear, "cuda")
            ref = model.model(ids, use_cache=False).last_hidden_state
            ref_logits = model.lm_head(ref[:, -1:])
            for name, mods in [("fused_all_rows", fused), ("fused_decode_pytorch_prefill", hybrid)]:
                install(model, mods, "cuda")
                h = model.model(ids, use_cache=False).last_hidden_state
                logits = model.lm_head(h[:, -1:])
                error = float((logits - ref_logits).norm() / ref_logits.norm())
                record = {"position": pos, "logit_relative_rmse": error,
                          "hidden_relative_rmse": float((h - ref).norm() / ref.norm()),
                          "next_token_agrees": bool(logits.argmax(-1).eq(ref_logits.argmax(-1)).all())}
                if not error < 1e-4:
                    raise ValueError(f"Model parity failed: {name}: {record}")
                metrics["parity"].setdefault(name, []).append(record)
        # Decode parity exercises the custom path used in the hybrid module.
        ids = tokens[:128][None].cuda()
        def decode_collect(mods, count):
            install(model, mods, "cuda")
            h = model.model(ids, use_cache=True)
            cache = h.past_key_values
            logits = []
            for _ in range(count):
                value = model.lm_head(h.last_hidden_state[:, -1:])
                logits.append(value)
                next_id = value.argmax(-1)
                h = model.model(next_id, past_key_values=cache, use_cache=True)
                cache = h.past_key_values
            return logits
        ref = decode_collect(linear, 16)
        for name, mods in [("fused_all_rows", fused), ("fused_decode_pytorch_prefill", hybrid)]:
            actual = decode_collect(mods, 16)
            errors = [float((a - b).norm() / b.norm()) for a, b in zip(actual, ref)]
            agrees = [bool(a.argmax(-1).eq(b.argmax(-1)).all()) for a, b in zip(actual, ref)]
            if max(errors) >= 1e-4 or not all(agrees):
                raise ValueError(f"Decode parity failed: {name}")
            metrics["parity"][name + "_decode"] = {"relative_logit_errors": errors, "token_agreement": agrees}
        del ref, actual
        print(f"All model parity checks passed; chosen output tile={chosen}", flush=True)
        dump(out / "metrics.json", metrics)
        for workload in ("prefill128", "prefill256", "prompt128_decode32"):
            def invoke():
                length = 256 if workload == "prefill256" else 128
                inputs = tokens[:length][None].cuda()
                h = model.model(inputs, use_cache=True)
                cache = h.past_key_values
                value = model.lm_head(h.last_hidden_state[:, -1:])
                if workload == "prompt128_decode32":
                    for _ in range(32):
                        h = model.model(value.argmax(-1), past_key_values=cache, use_cache=True)
                        cache = h.past_key_values
                        value = model.lm_head(h.last_hidden_state[:, -1:])
            # Prompt copy is inside all methods' timed wrapper; identical host input and transfer policy.
            records = {name: [] for name in variants}
            for name, mods in variants.items():
                install(model, mods, "cuda")
                for _ in range(5):
                    budget.check()
                    invoke()
            for sample in range(20):
                keys = list(variants)
                order = keys[sample % len(keys):] + keys[:sample % len(keys)]
                for name in order:
                    install(model, variants[name], "cuda")
                    records[name].append(measured(invoke, budget))
                if sample % 5 == 0:
                    print(f"Model {workload}: sample {sample + 1}/20", flush=True)
            metrics["model"][workload] = {name: summarize(samples) for name, samples in records.items()}
            dump(out / "metrics.json", metrics)
        metrics["elapsed_seconds"] = time.monotonic() - budget.start
        dump(out / "metrics.json", metrics)
        lines = ["# GPU-only U(Vx) execution benchmark", "", "No ReLU, refitting, CPU inference or factor changes.",
                 "The custom kernel uses shared-memory latent storage. Hybrid uses it for one-row decode and GPU PyTorch for prefill.",
                 "This is a numerical execution comparison, not a new compression quality result.", "",
                 "| Workload | Method | Wall median ms | Wall p95 ms | CUDA interval median ms |",
                 "|---|---|---:|---:|---:|"]
        for workload, results in metrics["model"].items():
            for name, result in results.items():
                lines.append(f"| {workload} | {name} | {result['wall_ms']['median']:.3f} | "
                             f"{result['wall_ms']['p95']:.3f} | {result['cuda_interval_ms']['median']:.3f} |")
        lines += ["", "Times include Python/driver wrappers, allocations, and identical prompt transfer. CUDA intervals can include host launch gaps.",
                  "Compile/setup excluded from warmed timings; full run elapsed time includes setup. Methods interleave to reduce order bias.",
                  "Pure fusion can recompute Vx per output tile; this trades extra arithmetic for parallelism and one launch.",
                  "Validation checks cover eight windows and 16 generated positions; complete test PPL was not rerun.",
                  "The saved linear approximation still has its previously measured quality loss versus dense; a kernel does not remove it."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
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

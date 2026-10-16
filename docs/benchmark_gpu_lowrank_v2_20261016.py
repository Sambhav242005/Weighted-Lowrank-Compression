"""Bounded GPU-only reuse/graph ablation; preserves the v1 study and factors."""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu_lowrank import GpuLowRank
from gpu_lowrank_v2 import ReusedGpuLowRank, staged_runtime, SOURCE
from gemma270m_relu_study import Bottleneck, Budget, dump, install, sha, token_sha
from benchmark_gpu_lowrank_20261016 import measured, summarize


def make_variants(saved):
    variants = {name: [] for name in ("pytorch", "v1_fused_decode", "staged", "graph_staged", "graph_cublas")}
    for state in saved:
        linear = Bottleneck(1024, 640, 275).eval().cuda()
        linear.load_state_dict(state)
        linear.requires_grad_(False)
        v, u = linear.down.weight, linear.up.weight
        variants["pytorch"].append(linear)
        variants["v1_fused_decode"].append(GpuLowRank(v, u, tile=64, decode_only=True).eval())
        for mode in ("staged", "graph_staged", "graph_cublas"):
            variants[mode].append(ReusedGpuLowRank(v, u, mode).eval())
    return variants


def run(out):
    from datasets import load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM
    budget = Budget(600)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(11)
    original = Path("results/gemma270m-relu/20261016-mvanbe94")
    cfg = json.loads((original / "resolved-config.json").read_text())
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    metrics_old = json.loads((original / "metrics.json").read_text())
    contract = {"kind": "GPU-only unchanged-factor execution ablation; no training", "compute_cap_seconds": 600,
        "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(), "dtype": "float32",
        "model": cfg["model"], "model_revision": cfg["model_revision"], "dataset_revision": cfg["dataset_revision"],
        "source_factor_seed": 11, "factor_rank": 275, "source_run": str(original.resolve()),
        "targets": cfg["targets"], "validation_split": cfg["splits"]["validation"],
        "baseline": "same two PyTorch linear operations on the same saved factors",
        "changed_variable": "decode execution strategy; GPU cuBLAS prefill fixed for every method",
        "methods": ["pytorch", "v1_fused_decode", "staged", "graph_staged", "graph_cublas"],
        "graph_output_semantics": "input copy, replay, output clone; independent returned tensors; buffers per stream",
        "warmups": 5, "samples_per_condition": 20, "micro_calls_per_sample": 10,
        "measurement_order": "rotate method order each round", "micro_shapes": [1, 128, 256],
        "model_workloads": ["prefill128", "prefill256", "prompt128_decode32"],
        "parity": "eight 256-token validation windows plus16 generated positions; not full-test PPL",
        "source_hashes": {str(p): sha(p) for p in [Path(__file__), Path("src/gpu_lowrank_v2.py"),
                          Path("src/gpu_lowrank.py"), Path("docs/benchmark_gpu_lowrank_20261016.py")]},
        "packages": {p: importlib.metadata.version(p) for p in cfg["packages"]},
        "git_revision": subprocess.check_output(["rtk", "git", "rev-parse", "HEAD"], text=True).strip(),
        "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv]),
        "memory_scope": "retained methods/graphs; not isolated deployment memory"}
    dump(out / "resolved-config.json", contract)
    assert sha(original / "whitening_seed11-full.pt") == metrics_old["storage"]["whitening_seed11"]["full_sha256"]
    assert sha(Path(assets["snapshot"]) / "model.safetensors") == assets["model_sha256"]
    saved = torch.load(original / "replacements-seed11.pt", weights_only=True, map_location="cpu")["whitening"]
    full = torch.load(original / "whitening_seed11-full.pt", weights_only=True, map_location="cpu", mmap=True)
    for i, state in enumerate(saved):
        assert all(torch.equal(value, full[f"model.layers.{i}.self_attn.o_proj.{key}"]) for key, value in state.items())
    del full
    variants = make_variants(saved)
    rt = staged_runtime(torch.device("cuda", torch.cuda.current_device()))
    (out / "kernel.cu").write_text(SOURCE, encoding="utf-8")
    (out / "kernel.ptx").write_bytes(rt.staged_ptx)
    result = {"micro": {}, "model": {}, "parity": {}}
    with torch.inference_mode():
        for rows in (1, 128, 256):
            x = torch.randn(rows, 1024, device="cuda")
            reference = variants["pytorch"][0](x)
            for name, modules in variants.items():
                actual = modules[0](x)
                assert float((actual - reference).norm() / reference.norm()) < 2e-5
                for _ in range(5):
                    modules[0](x)
            raw = {name: [] for name in variants}
            for sample in range(20):
                keys = list(variants)
                for name in keys[sample % 5:] + keys[:sample % 5]:
                    raw[name].append(measured(lambda name=name: variants[name][0](x), budget, repeats=10))
            result["micro"][str(rows)] = {name: summarize(samples) for name, samples in raw.items()}
            dump(out / "metrics.json", result)
            print(f"Reuse/graph micro complete: rows={rows}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
        data = load_from_disk(assets["dataset_path"])
        tokens = torch.tensor(tokenizer("\n\n".join(data["validation"]["text"]), add_special_tokens=False,
                             return_attention_mask=False)["input_ids"], dtype=torch.long)
        assert token_sha(tokens) == cfg["splits"]["validation"]["sha256"]
        model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
                      torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
        model.requires_grad_(False)
        for pos in cfg["splits"]["validation"]["positions"]:
            budget.check()
            ids = tokens[pos:pos + 256][None].cuda()
            install(model, variants["pytorch"], "cuda")
            ref = model.model(ids, use_cache=False).last_hidden_state
            reference_logits = model.lm_head(ref[:, -1:])
            for name in ("staged", "graph_staged", "graph_cublas"):
                install(model, variants[name], "cuda")
                actual = model.model(ids, use_cache=False).last_hidden_state
                logits = model.lm_head(actual[:, -1:])
                error = float((logits - reference_logits).norm() / reference_logits.norm())
                assert error < 1e-4
                result["parity"].setdefault(name, {"prefill": []})["prefill"].append(error)
        prompt = tokens[:128][None].cuda()
        def decode_logits(modules):
            install(model, modules, "cuda")
            h = model.model(prompt, use_cache=True)
            cache = h.past_key_values
            logits = []
            for _ in range(16):
                budget.check()
                value = model.lm_head(h.last_hidden_state[:, -1:])
                logits.append(value)
                h = model.model(value.argmax(-1), past_key_values=cache, use_cache=True)
                cache = h.past_key_values
            return logits
        expected = decode_logits(variants["pytorch"])
        for name in ("staged", "graph_staged", "graph_cublas"):
            actual = decode_logits(variants[name])
            errors = [float((a - b).norm() / b.norm()) for a, b in zip(actual, expected)]
            agrees = [bool(a.argmax(-1).eq(b.argmax(-1)).all()) for a, b in zip(actual, expected)]
            assert max(errors) < 1e-4 and all(agrees)
            result["parity"][name]["decode_logit_errors"] = errors
            result["parity"][name]["decode_token_agreement"] = agrees
        del expected, actual
        dump(out / "metrics.json", result)
        print("All reuse/graph model parity checks passed", flush=True)
        for workload in ("prefill128", "prefill256", "prompt128_decode32"):
            def invoke():
                inputs = tokens[:256 if workload == "prefill256" else 128][None].cuda()
                h = model.model(inputs, use_cache=True)
                cache = h.past_key_values
                value = model.lm_head(h.last_hidden_state[:, -1:])
                if workload == "prompt128_decode32":
                    for _ in range(32):
                        h = model.model(value.argmax(-1), past_key_values=cache, use_cache=True)
                        cache = h.past_key_values
                        value = model.lm_head(h.last_hidden_state[:, -1:])
            raw = {name: [] for name in variants}
            for name, modules in variants.items():
                install(model, modules, "cuda")
                for _ in range(5):
                    budget.check()
                    invoke()
            for sample in range(20):
                keys = list(variants)
                for name in keys[sample % 5:] + keys[:sample % 5]:
                    install(model, variants[name], "cuda")
                    raw[name].append(measured(invoke, budget))
                if sample % 5 == 0:
                    print(f"Reuse/graph {workload}: sample {sample + 1}/20", flush=True)
            result["model"][workload] = {name: summarize(samples) for name, samples in raw.items()}
            dump(out / "metrics.json", result)
        result["elapsed_seconds"] = time.monotonic() - budget.start
        dump(out / "metrics.json", result)
        lines = ["# GPU-only reuse and CUDA Graph execution ablation", "",
                 "Same FP32 factors, no ReLU, refitting or CPU inference. All methods use GPU cuBLAS prefill.",
                 "Staged computes Vx once and U(Vx) once. Graphs capture those stages or existing cuBLAS calls.",
                 "Graph timings include input copy and output clone; no mutable output aliases escape.", "",
                 "| Workload | Method | Wall median ms | Wall p95 ms | CUDA interval median ms |",
                 "|---|---|---:|---:|---:|"]
        for workload, methods in result["model"].items():
            for name, values in methods.items():
                lines.append(f"| {workload} | {name} | {values['wall_ms']['median']:.3f} | "
                             f"{values['wall_ms']['p95']:.3f} | {values['cuda_interval_ms']['median']:.3f} |")
        lines += ["", "Five warmups, twenty rotated-order samples; compilation/capture excluded from warmed timings.",
                  "Prompt transfer, host wrappers, allocations and cache handling included identically. CUDA intervals may include host gaps.",
                  "Buffers retained per stream; factor buffers must not be replaced after graph capture.",
                  "Checked parity on eight validation windows plus16 generated positions; no full-test PPL rerun.",
                  "No quality improvement or broad hardware speedup is implied. Original low-rank quality loss remains."]
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

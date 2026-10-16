"""Isolated, bounded Gemma 3 270M CPU/CUDA ReLU experiment (explicit CLI only)."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import torch
from torch import nn
from torch.nn import functional as F

from paper_math import weighted_factors, factors

MODEL = "google/gemma-3-270m"
REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DATA_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def dump(path, obj):
    # Summary snapshots may be updated within this unique run; prior runs never are.
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_sha(tokens):
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, rank, nonlinear=False):
        super().__init__()
        if not 0 < rank <= min(n_in, n_out):
            raise ValueError("invalid bottleneck width")
        self.down = nn.Linear(n_in, rank, bias=nonlinear)
        self.up = nn.Linear(rank, n_out, bias=nonlinear)
        self.nonlinear = nonlinear

    def forward(self, x):
        x = self.down(x)
        return self.up(F.relu(x) if self.nonlinear else x)


def budget_rank(n_in, n_out, nonlinear=False):
    ceiling = n_in * n_out * 7 // 10
    rank = (ceiling - (n_out if nonlinear else 0)) // (n_in + n_out + int(nonlinear))
    if rank < 1:
        raise ValueError("budget too small")
    return min(rank, n_in, n_out)


def windows(tokens, context=256, stride=128):
    """Every next-token target exactly once, including the tail (HF strided policy)."""
    previous_end = 0
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + context, len(tokens))
        first_target = max(previous_end, start + 1)
        if first_target < end:
            yield start, end, first_target - start
        previous_end = end
        if end == len(tokens):
            break


def chunk_metrics(student, labels, teacher=None):
    s = F.log_softmax(student.float(), dim=-1)
    nll = -s.gather(-1, labels[:, None]).sum()
    kl = s.new_zeros(())
    if teacher is not None:
        t = F.log_softmax(teacher.float(), dim=-1)
        kl = (t.exp() * (t - s)).sum()
    if not torch.isfinite(nll + kl):
        raise FloatingPointError("nonfinite quality metric")
    return float(nll), float(kl)


class Budget:
    def __init__(self, seconds=7200):
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError("Declared active-compute cap reached; preserve partial results")


def prepare(out):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    print("Downloading pinned model", MODEL, REVISION, flush=True)
    snapshot = snapshot_download(MODEL, revision=REVISION, allow_patterns=[
        "config.json", "generation_config.json", "*.safetensors", "tokenizer*", "special_tokens_map.json"
    ])
    print("Downloading pinned WikiText-2", DATA_REVISION, flush=True)
    data_path = Path("paper-assets") / ("wikitext-" + DATA_REVISION)
    if not data_path.exists():
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", revision=DATA_REVISION)
        data.save_to_disk(str(data_path))
    manifest = {"model": MODEL, "revision": REVISION, "snapshot": snapshot,
                "dataset": "Salesforce/wikitext", "dataset_revision": DATA_REVISION,
                "dataset_path": str(data_path.resolve())}
    dump(out / "assets.json", manifest)
    print("Asset preparation complete", flush=True)


def capture(model, tokens, positions, budget):
    captured = [[] for _ in model.model.layers]
    handles = []
    for i, layer in enumerate(model.model.layers):
        def hook(module, args, i=i):
            captured[i].append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for pos in positions:
                budget.check()
                model.model(tokens[pos:pos + 256][None].cuda(), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [torch.cat(x) for x in captured]


def fit_one(weight, x, v, seed, budget):
    n_out, n_in = weight.shape
    rank = budget_rank(n_in, n_out)
    # Stable CPU FP64 solve from the existing study; stored/executed factors FP32.
    left, right = weighted_factors(weight.cpu(), x.T, rank)
    linear = Bottleneck(n_in, n_out, rank)
    with torch.no_grad():
        linear.down.weight.copy_(right)
        linear.up.weight.copy_(left)
    # Method-specific full-rank check on held-out activations.
    full_l, full_r = weighted_factors(weight.cpu(), x.T, min(n_in, n_out))
    expected = v @ weight.cpu().T
    full_error = float(((v @ full_r.T @ full_l.T - expected).square().sum()
                        / expected.square().sum().clamp_min(1e-20)).sqrt())
    if not math.isfinite(full_error) or full_error > 0.01:
        raise ValueError(f"whitening full-rank check failed: {full_error}")
    torch.manual_seed(seed)
    candidate = Bottleneck(n_in, n_out, budget_rank(n_in, n_out, True), True).cuda()
    # Initialize the input subspace from whitening; train all weights/biases.
    with torch.no_grad():
        candidate.down.weight.copy_(right[:candidate.down.out_features].cuda())
        candidate.down.bias.zero_()
        candidate.up.bias.zero_()
    xc, vc, wc = x.cuda(), v.cuda(), weight.cuda()
    with torch.no_grad():
        y, vy = xc @ wc.T, vc @ wc.T
    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
    rng = torch.Generator(device="cuda").manual_seed(seed)
    best_loss = math.inf
    best = None
    history = []
    for step in range(201):
        budget.check()
        if step % 20 == 0:
            with torch.no_grad():
                val = float((candidate(vc) - vy).square().mean() / vy.square().mean().clamp_min(1e-20))
            if not math.isfinite(val):
                raise FloatingPointError("nonfinite validation loss")
            history.append({"step": step, "validation_nmse": val})
            if val < best_loss:
                best_loss = val
                best = {k: t.detach().cpu().clone() for k, t in candidate.state_dict().items()}
        if step == 200:
            break
        indices = torch.randint(len(xc), (256,), generator=rng, device="cuda")
        prediction = candidate(xc[indices])
        loss = (prediction - y[indices]).square().mean() / y[indices].square().mean().clamp_min(1e-20)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    candidate.load_state_dict(best)
    with torch.no_grad():
        linear_nmse = float((linear(v) - expected).square().mean() / expected.square().mean().clamp_min(1e-20))
        svd_l, svd_r = factors(weight.cpu(), rank)
        svd_nmse = float((v @ svd_r.T @ svd_l.T - expected).square().mean() /
                         expected.square().mean().clamp_min(1e-20))
    return linear.eval(), candidate.cpu().eval(), {
        "full_rank_validation_relative_rmse": full_error, "whitening_validation_nmse": linear_nmse,
        "relu_best_validation_nmse": best_loss, "relu_history": history,
        "plain_svd_validation_nmse_diagnostic_only": svd_nmse,
        "linear_parameters": sum(p.numel() for p in linear.parameters()),
        "relu_parameters": sum(p.numel() for p in candidate.parameters()),
        "dense_parameters": weight.numel()}


def install(model, modules, device):
    for layer, module in zip(model.model.layers, modules):
        layer.self_attn.o_proj = module.to(device)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timing(fn, device, budget):
    with torch.inference_mode():
        for _ in range(5):
            budget.check()
            fn()
        sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(20):
            budget.check()
            sync(device)
            start = time.perf_counter()
            fn()
            sync(device)
            samples.append(1000 * (time.perf_counter() - start))
    return {"median_ms": statistics.median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "samples_ms": samples,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "process_peak_rss_bytes": psutil.Process().memory_info().peak_wset if os.name == "nt" else None}


def benchmark(model, tokens, device, budget):
    result = {}
    for length in (128, 256):
        ids = tokens[:length][None].to(device)
        # Compute all hidden states but only final-position logits, as generation prefill does.
        def prefill():
            h = model.model(ids, use_cache=True)
            model.lm_head(h.last_hidden_state[:, -1:])
        result[f"prefill_{length}"] = timing(prefill, device, budget)
    ids = tokens[:128][None].to(device)
    def decode():
        h = model.model(ids, use_cache=True)
        cache = h.past_key_values
        next_id = model.lm_head(h.last_hidden_state[:, -1:]).argmax(-1)
        for _ in range(32):
            h = model.model(next_id, past_key_values=cache, use_cache=True)
            cache = h.past_key_values
            next_id = model.lm_head(h.last_hidden_state[:, -1:]).argmax(-1)
    result["prefill128_plus_decode32"] = timing(decode, device, budget)
    result["decode_definition"] = "128-token prefill plus 32 cached decode steps, greedy, no EOS stopping"
    return result


@torch.inference_mode()
def quality(model, dense, variants, tokens, out, budget):
    aggregates = {name: {"nll_sum": 0., "kl_sum": 0., "count": 0,
                         "drift_squared_error": [0.] * 18, "drift_teacher_energy": [0.] * 18}
                  for name in ["dense", *variants]}
    all_windows = list(windows(tokens))
    with (out / "quality-windows.jsonl").open("x", encoding="utf-8") as log:
        for window_id, (start, end, first) in enumerate(all_windows):
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
                    # Restrict drift to the same nonduplicated target positions as likelihood.
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
                dump(out / "quality-progress.json", {"completed_windows": window_id + 1,
                     "expected_windows": len(all_windows), "aggregates": aggregates})
    result = {}
    for name, a in aggregates.items():
        assert a["count"] == len(tokens) - 1
        result[name] = {"nll": a["nll_sum"] / a["count"],
                        "perplexity": math.exp(a["nll_sum"] / a["count"]),
                        "teacher_to_student_kl": a["kl_sum"] / a["count"], "targets": a["count"],
                        "block_relative_rmse": [math.sqrt(x / max(y, 1e-30)) for x, y in
                                                zip(a["drift_squared_error"], a["drift_teacher_energy"])]}
    return result


def run(out, assets):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    budget = Budget()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not substitute a CPU-only run")
    # Real GEMM check verifies the installed wheel supports this GPU architecture.
    check = torch.randn(32, 32, device="cuda")
    assert torch.isfinite(check @ check).all()
    torch.cuda.synchronize()
    config = {"model": MODEL, "model_revision": REVISION, "dataset_revision": DATA_REVISION,
              "seeds": [11, 23], "dtype": "float32", "gpu": torch.cuda.get_device_name(),
              "gpu_capability": torch.cuda.get_device_capability(), "cuda_arch_list": torch.cuda.get_arch_list(),
              "cuda_runtime": torch.version.cuda, "python": sys.version, "platform": platform.platform(),
              "packages": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "datasets", "huggingface_hub", "numpy", "psutil")},
              "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv]),
              "source_hashes": {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name("paper_math.py"))},
              "cpu_threads": 4, "compute_cap_seconds": 7200, "context": 256, "stride": 128,
              "fit_updates": 200, "fit_batch": 256, "learning_rate": 0.001,
              "parameter_ceiling_fraction": 0.7, "warmups": 5, "measurements": 20,
              "status": "exploratory", "quality_device": "cuda", "cpu_quality": "fixed-window parity only",
              "seed_definition": "ReLU initialization/minibatch seeds; calibration positions shared across seeds",
              "baseline_reuse": "Whitening is deterministic with shared calibration; two seed labels are not independent repeats",
              "budget_match": "Common 70 percent ceiling; exact counts reported, bias overhead makes ReLU slightly larger",
              "residual_drift_definition": "relative RMSE at each hidden-state boundary; final boundary is final-normalized"}
    for args, key in [(["rtk", "git", "rev-parse", "HEAD"], "git_revision"),
                      (["rtk", "git", "diff", "--binary"], "dirty_diff")]:
        config[key] = subprocess.check_output(args, text=True, encoding="utf-8").strip()
    dump(out / "resolved-config.json", config)
    tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
    data = load_from_disk(assets["dataset_path"])
    tokens = {}
    for split in ("train", "validation", "test"):
        encoded = tokenizer("\n\n".join(data[split]["text"]), add_special_tokens=False, return_attention_mask=False)
        tokens[split] = torch.tensor(encoded["input_ids"], dtype=torch.long)
    rng = np.random.default_rng(20261016)
    positions = {"train": sorted(rng.choice(len(tokens["train"]) // 256, 16, replace=False) * 256),
                 "validation": list(np.arange(8) * 256)}
    config["splits"] = {s: {"tokens": len(t), "sha256": token_sha(t),
                               "positions": [int(x) for x in positions.get(s, [])]} for s, t in tokens.items()}
    model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
                torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
    model.requires_grad_(False)
    dense = [layer.self_attn.o_proj for layer in model.model.layers]
    config["targets"] = [{"name": f"model.layers.{i}.self_attn.o_proj", "shape": list(m.weight.shape),
                           "linear_rank": budget_rank(m.in_features, m.out_features),
                           "relu_width": budget_rank(m.in_features, m.out_features, True)} for i, m in enumerate(dense)]
    dump(out / "resolved-config.json", config)
    print("Capturing teacher calibration/validation activations", flush=True)
    train_x = capture(model, tokens["train"], positions["train"], budget)
    val_x = capture(model, tokens["validation"], positions["validation"], budget)
    variants, fitting = {}, {}
    for seed in (11, 23):
        linear, relu, records = [], [], []
        for i, original in enumerate(dense):
            budget.check()
            print(f"Fitting seed={seed} layer={i}", flush=True)
            a, b, record = fit_one(original.weight.detach(), train_x[i], val_x[i], seed, budget)
            linear.append(a)
            relu.append(b)
            records.append(record)
            dump(out / f"fit-seed{seed}.json", records)
        variants[f"whitening_seed{seed}"] = linear
        variants[f"relu_seed{seed}"] = relu
        fitting[str(seed)] = records
        torch.save({"whitening": [m.state_dict() for m in linear], "relu": [m.state_dict() for m in relu]},
                   out / f"replacements-seed{seed}.pt")
    del train_x, val_x
    metrics = {"fitting": fitting, "performance": {}, "storage": {}, "cpu_gpu_parity": {}}
    # Serialize tied-embedding state once for each condition, with no model copy.
    for name, modules in [("dense", dense), *variants.items()]:
        install(model, modules, "cuda")
        path = out / (name + "-full.pt")
        torch.save(model.state_dict(), path)
        payload = io.BytesIO()
        torch.save([m.state_dict() for m in modules], payload)
        metrics["storage"][name] = {"full_checkpoint_bytes": path.stat().st_size,
                                    "target_checkpoint_bytes": payload.tell(), "full_sha256": sha(path),
                                    "target_parameters": sum(p.numel() for m in modules for p in m.parameters())}
    dump(out / "metrics.json", metrics)
    # Device order is frozen; each method uses identical prompt tokens and timing policy.
    for device in ("cuda", "cpu"):
        for name, modules in [("dense", dense), *variants.items()]:
            budget.check()
            print(f"Benchmark device={device} method={name}", flush=True)
            install(model, modules, device)
            model.to(device)
            if device == "cuda":
                torch.cuda.empty_cache()
            metrics["performance"].setdefault(device, {})[name] = benchmark(model, tokens["validation"], device, budget)
            # Compare final hidden states and last-position logits over a fixed 256-token validation window.
            with torch.inference_mode():
                h = model.model(tokens["validation"][:256][None].to(device), use_cache=False).last_hidden_state
                logits = model.lm_head(h[:, -1:]).cpu()
            parity_path = out / (name + "-cuda-parity.pt")
            if device == "cuda":
                torch.save({"hidden": h.cpu(), "logits": logits}, parity_path)
            else:
                ref = torch.load(parity_path, weights_only=True)
                metrics["cpu_gpu_parity"][name] = {
                    "hidden_relative_rmse": float(((h.cpu() - ref["hidden"]).square().sum() /
                                                    ref["hidden"].square().sum()).sqrt()),
                    "last_logit_relative_rmse": float(((logits - ref["logits"]).square().sum() /
                                                        ref["logits"].square().sum()).sqrt()),
                    "last_token_agreement": bool(logits.argmax(-1).eq(ref["logits"].argmax(-1)).all())}
            dump(out / "metrics.json", metrics)
    install(model, dense, "cuda")
    model.cuda()
    metrics["quality"] = quality(model, dense, variants, tokens["test"], out, budget)
    metrics["elapsed_compute_seconds"] = time.monotonic() - budget.start
    dump(out / "metrics.json", metrics)
    write_report(out, metrics)


def write_report(out, metrics):
    lines = ["# Gemma 3 270M exploratory ReLU study", "", "FP32, GTX 1050 and i5-8300H; two seeds.",
             "Standard PyTorch GPU operations; no custom fused CUDA kernel.", "",
             "| Method | Test PPL | Teacher KL | Final drift RMSE | Full bytes |",
             "|---|---:|---:|---:|---:|"]
    for name, q in metrics.get("quality", {}).items():
        lines.append(f"| {name} | {q['perplexity']:.4f} | {q['teacher_to_student_kl']:.6f} | "
                     f"{q['block_relative_rmse'][-1]:.6f} | {metrics['storage'][name]['full_checkpoint_bytes']} |")
    lines += ["", "| Device | Method | Prefill 128 ms | Prefill 256 ms | Prefill128 + decode32 ms |",
              "|---|---|---:|---:|---:|"]
    for device, methods in metrics.get("performance", {}).items():
        for name, p in methods.items():
            lines.append(f"| {device} | {name} | {p['prefill_128']['median_ms']:.2f} | "
                         f"{p['prefill_256']['median_ms']:.2f} | {p['prefill128_plus_decode32']['median_ms']:.2f} |")
    lines += ["", "Prediction quality is evaluated on GPU; CPU uses fixed-window numerical parity checks.",
              "Two seeds and one architecture cannot meet the project's promotion gate. Prior WikiText use limits generalization.",
              "Process peak RSS is cumulative process high-water mark, not an isolated per-method memory delta.",
              "Full checkpoint size uses PyTorch serialization with tied weights retained, not the downloaded BF16 file size."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "status.json", {"state": "running", "mode": args.mode})
    with (args.output / "stdout.log").open("x", encoding="utf-8", buffering=1) as stdout, \
         (args.output / "stderr.log").open("x", encoding="utf-8", buffering=1) as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                if args.mode == "prepare":
                    prepare(args.output)
                else:
                    run(args.output, json.loads(args.assets.read_text(encoding="utf-8")))
                dump(args.output / "status.json", {"state": "completed", "mode": args.mode})
            except BaseException as exc:
                traceback.print_exc()
                state = "budget_limited" if isinstance(exc, TimeoutError) else "failed"
                dump(args.output / "status.json", {"state": state, "error": str(exc)})
                metrics_file = args.output / "metrics.json"
                if metrics_file.exists():
                    write_report(args.output, json.loads(metrics_file.read_text()))
                raise


if __name__ == "__main__":
    main()

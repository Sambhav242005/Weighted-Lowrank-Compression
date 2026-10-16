"""Bounded forward-only diagnostic controls; no optimization or test-split selection."""
import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gemma270m_relu_study import Bottleneck, Budget, capture, dump, install, sha, token_sha


def nmse(actual, expected):
    return float((actual - expected).square().sum() / expected.square().sum().clamp_min(1e-30))


def signed_pair(x, left, right):
    z = x @ right.T
    return F.relu(z) @ left.T - F.relu(-z) @ left.T


def shifted(x, left, right, bias):
    return F.relu(x @ right.T + bias) @ left.T - (bias @ left.T)


def cpu_profile(model, tokens, budget):
    """Exploratory component wall time, not a new comparable latency benchmark."""
    model.cpu()
    ids = tokens[:128][None]
    totals = {"head_seconds": 0., "target_seconds": 0.}
    starts = {}
    handles = []
    def pre(module, args):
        starts[id(module)] = time.perf_counter()
    def after(kind):
        def post(module, args, result):
            totals[kind] += time.perf_counter() - starts[id(module)]
        return post
    with torch.inference_mode():
        def workload():
            h = model.model(ids, use_cache=True)
            cache = h.past_key_values
            next_id = model.lm_head(h.last_hidden_state[:, -1:]).argmax(-1)
            for _ in range(32):
                h = model.model(next_id, past_key_values=cache, use_cache=True)
                cache = h.past_key_values
                next_id = model.lm_head(h.last_hidden_state[:, -1:]).argmax(-1)
        for _ in range(2):
            budget.check()
            workload()
        for module, kind in [(model.lm_head, "head_seconds"), *[
                (layer.self_attn.o_proj, "target_seconds") for layer in model.model.layers]]:
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(after(kind)))
        samples = []
        try:
            for _ in range(3):
                budget.check()
                totals.update(head_seconds=0., target_seconds=0.)
                start = time.perf_counter()
                workload()
                elapsed = time.perf_counter() - start
                samples.append({"total_seconds": elapsed, **totals,
                                "head_fraction": totals["head_seconds"] / elapsed,
                                "target_fraction": totals["target_seconds"] / elapsed})
        finally:
            for handle in handles:
                handle.remove()
    return {"scope": "dense CPU only; 128-token prefill plus 32 cached steps; synchronous hook wall time",
            "warmups": 2, "samples": samples,
            "limitation": "Three instrumented repeats; not a GPU profile or deployment benchmark"}


def run(out):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    original = Path("results/gemma270m-relu/20261016-mvanbe94")
    combined = Path("results/gemma270m-relu/20261016-recovery-mvapk6bo")
    cfg = json.loads((original / "resolved-config.json").read_text())
    saved_metrics = json.loads((combined / "metrics.json").read_text())
    assets = json.loads(Path("paper-assets/gemma270m-study-assets-20261016.json").read_text())
    budget = Budget(600)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    contract = {"purpose": "forward-only diagnosis, no fitting, no test-split probing", "cap_seconds": 600,
                "model": cfg["model"], "model_revision": cfg["model_revision"],
                "dataset_revision": cfg["dataset_revision"], "dtype": "float32",
                "splits": cfg["splits"], "seeds": [11, 23], "target_family": "all18 attention o_proj",
                "baseline": "immutable saved whitening at rank275",
                "controls": "factor-copy, clipping, sign identity, train-derived shift, prefix-input drift, dense CPU hooks",
                "source_sha256": sha(__file__), "original_source_sha256": cfg["source_hashes"],
                "packages": {p: importlib.metadata.version(p) for p in cfg["packages"]},
                "gpu": torch.cuda.get_device_name(), "cpu_threads": 4,
                "git_revision": subprocess.check_output(["rtk", "git", "rev-parse", "HEAD"], text=True).strip(),
                "command": "rtk " + subprocess.list2cmdline([sys.executable, *sys.argv])}
    if contract["gpu"] != cfg["gpu"] or contract["packages"] != cfg["packages"]:
        raise ValueError("Environment changed")
    dump(out / "resolved-config.json", contract)
    tokenizer = AutoTokenizer.from_pretrained(assets["snapshot"], local_files_only=True)
    data = load_from_disk(assets["dataset_path"])
    tokens = {}
    for split in ("train", "validation"):
        tokens[split] = torch.tensor(tokenizer("\n\n".join(data[split]["text"]), add_special_tokens=False,
                                    return_attention_mask=False)["input_ids"], dtype=torch.long)
        assert token_sha(tokens[split]) == cfg["splits"][split]["sha256"]
    model = AutoModelForCausalLM.from_pretrained(assets["snapshot"], local_files_only=True,
                 torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
    model.requires_grad_(False)
    dense = [layer.self_attn.o_proj for layer in model.model.layers]
    train = capture(model, tokens["train"], cfg["splits"]["train"]["positions"], budget)
    validation = capture(model, tokens["validation"], cfg["splits"]["validation"]["positions"], budget)
    saved = {seed: torch.load(original / f"replacements-seed{seed}.pt", weights_only=True,
                             map_location="cpu") for seed in (11, 23)}
    result = {"layers": [], "composition": {}, "fitting_summary": {}}
    with torch.inference_mode():
        for i, layer in enumerate(dense):
            budget.check()
            x, cal = validation[i].cuda(), train[i].cuda()
            teacher = layer(x)
            assert nmse(teacher, x @ layer.weight.T) < 1e-10
            factors = saved[11]["whitening"][i]
            left, right = factors["up.weight"].cuda(), factors["down.weight"].cuda()
            z, zcal = x @ right.T, cal @ right.T
            baseline = z @ left.T
            torch.manual_seed(11)
            initial = Bottleneck(1024, 640, 275, True).cuda()
            initial.down.weight.copy_(right)
            initial.down.bias.zero_()
            initial.up.bias.zero_()
            record = {"layer": i, "whitening_validation_nmse_replayed": nmse(baseline, teacher),
                      "whitening_validation_nmse_recorded": saved_metrics["fitting"]["11"][i]["whitening_validation_nmse"],
                      "original_initialization_nmse": nmse(initial(x), teacher),
                      "copied_both_factors_with_relu_nmse": nmse(F.relu(z) @ left.T, teacher),
                      "initial_latent_negative_fraction": float((z < 0).float().mean()),
                      "initial_latent_negative_energy_fraction": float(z.clamp_max(0).square().sum() / z.square().sum()),
                      "required_up_factor_norm": float(left.norm()),
                      "random_up_factor_norm": float(initial.up.weight.norm()),
                      "sign_pair_550_units_nmse_vs_linear": nmse(signed_pair(x, left, right), baseline),
                      "sign_pair_550_units_parameters": 2 * left.numel() + 2 * right.numel(),
                      "budget_sign_pair_274_units_nmse": nmse(signed_pair(x, left[:, :137], right[:137]), teacher)}
            shift = -zcal.min(dim=0).values + 1e-4
            record.update(shifted_relu_calibration_nmse_vs_linear=nmse(shifted(cal, left, right, shift), zcal @ left.T),
                          shifted_relu_validation_nmse=nmse(shifted(x, left, right, shift), teacher),
                          shifted_relu_validation_clip_fraction=float(((z + shift) < 0).float().mean()),
                          shifted_relu_parameters=275 * (1024 + 640 + 1) + 640)
            for seed in (11, 23):
                trained = Bottleneck(1024, 640, 275, True).cuda()
                trained.load_state_dict(saved[seed]["relu"][i])
                ztrained = trained.down(x)
                record[f"relu_seed{seed}_replayed_nmse"] = nmse(trained(x), teacher)
                record[f"relu_seed{seed}_recorded_nmse"] = saved_metrics["fitting"][str(seed)][i]["relu_best_validation_nmse"]
                record[f"relu_seed{seed}_active_fraction"] = float((ztrained > 0).float().mean())
                record[f"relu_seed{seed}_dead_units_validation"] = int((ztrained <= 0).all(dim=0).sum())
            result["layers"].append(record)
            dump(out / "metrics.json", result)
            print(f"Diagnostic layer {i}: whitening={record['whitening_validation_nmse_replayed']:.5f}, "
                  f"random-init={record['original_initialization_nmse']:.5f}, "
                  f"copy+ReLU={record['copied_both_factors_with_relu_nmse']:.5f}, "
                  f"shifted={record['shifted_relu_validation_nmse']:.5f}", flush=True)
        # Compare the inputs encountered after upstream replacement against teacher inputs.
        for method, seed in (("whitening", 11), ("relu", 11), ("relu", 23)):
            modules = []
            for state in saved[seed][method]:
                module = Bottleneck(1024, 640, 275, method == "relu").cuda().eval()
                module.load_state_dict(state)
                modules.append(module)
            install(model, modules, "cuda")
            actual = capture(model, tokens["validation"], cfg["splits"]["validation"]["positions"], budget)
            result["composition"][f"{method}_seed{seed}"] = {
                "projection_input_relative_rmse": [math.sqrt(nmse(a, t)) for a, t in zip(actual, validation)]}
            dump(out / "metrics.json", result)
        install(model, dense, "cuda")
    for seed in (11, 23):
        rows = saved_metrics["fitting"][str(seed)]
        result["fitting_summary"][str(seed)] = {
            "best_checkpoint_is_final_step_layers": sum(min(r["relu_history"], key=lambda x: x["validation_nmse"])["step"] == 200 for r in rows),
            "median_last20_loss_ratio": statistics.median(r["relu_history"][-1]["validation_nmse"] /
                                                          r["relu_history"][-2]["validation_nmse"] for r in rows)}
    del train, validation
    result["cpu_profile"] = cpu_profile(model, tokens["validation"], budget)
    result["elapsed_seconds"] = time.monotonic() - budget.start
    dump(out / "metrics.json", result)


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
                dump(args.output / "status.json", {"state": "failed", "error": str(exc)})
                raise


if __name__ == "__main__":
    main()

"""BF16-only practical continuation, limited to two predeclared seeds.

Reuses the frozen objective and existing calibration/timing protocol. It never
calls the FP16/INT8 loader or packed-factor constructors. Fitting and metric
accumulation retain the existing higher-precision numerical implementations;
all evaluated model weights, stored factors and forward activations are BF16.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
import traceback
from pathlib import Path

import torch

from .compression_core import sha256_file
from .paper_architecture_panel import frozen_specs
from .paper_benchmark import command_line, create_output, environment, tensor_digest, write
from .paper_followups import (benchmark_inference, calibration_control,
                              dataset_content_hash, model_probe_digest)
from .paper_precision import PrecisionLinear
from .paper_math import weighted_factors
from .paper_protocol import calibration_starts, token_hash


def validate_config(config):
    seeds = config.get("seeds", [])
    if not seeds or len(seeds) > 2 or len(set(seeds)) != len(seeds):
        raise ValueError("one or two distinct seeds required")
    if seeds != [11, 23] or config.get("precision") != "bfloat16":
        raise ValueError("this continuation is BF16 only, with frozen seeds 11/23")
    if config.get("factor_formats") != ["bfloat16"]:
        raise ValueError("no other factor format is authorized")
    if config.get("rank") != 384 or config.get("context") != 512 or config.get("stride") != 256:
        raise ValueError("rank and evaluation protocol must remain frozen")
    if config.get("model") != "google/gemma-3-1b-it" or Path(config["model_path"]).name != config["model_revision"]:
        raise ValueError("pinned Gemma snapshot required")
    prior = config.get("prior_practical_charged_seconds")
    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(prior) or not 0 <= prior < 14400:
        raise ValueError("known prior practical charge below the original four-hour cap required")
    sidecar = json.loads(Path(config["interruption_accounting_path"]).read_text())
    entries = [e for e in sidecar["runs"] if e.get("stage") == "practical"]
    if len(entries) != 1 or entries[0]["charged_seconds"] != prior:
        raise ValueError("practical charge differs from preservation sidecar")
    entry = entries[0]
    if entry["state"] != "cancelled by user" or abs(entry["ended_epoch"]-entry["started_epoch"]-prior) > 1e-6:
        raise ValueError("invalid user-stop accounting interval")
    for name, digest in entry["source_artifacts"].items():
        if Path(name).name != name or sha256_file(Path(entry["run"])/name) != digest:
            raise ValueError("prior practical evidence changed: "+name)


def assert_bf16(model):
    for name, value in model.named_parameters():
        if value.is_floating_point() and value.dtype != torch.bfloat16:
            raise ValueError("non-BF16 model parameter: "+name)
    for name, layer in model.named_modules():
        if isinstance(layer, PrecisionLinear):
            if layer.storage != "bf16" or layer.compute != "bf16":
                raise ValueError("unauthorized factor format: "+name)
            if layer.left.dtype != torch.bfloat16 or layer.right.dtype != torch.bfloat16:
                raise ValueError("factor storage must be BF16")


def save_whole(model, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    assert_bf16(model)
    copied, state, recipes = {}, {}, {}
    for name, value in model.state_dict().items():
        key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in copied:
            copied[key] = value.detach().cpu()
        state[name] = copied[key]
    for name, layer in model.named_modules():
        if isinstance(layer, PrecisionLinear):
            recipes[name] = {"left_shape": list(layer.left.shape), "right_shape": list(layer.right.shape),
                             "storage": "bf16", "compute": "bf16"}
    torch.save({"state_dict": state, "bf16_recipes": recipes,
                "model_config": model.config.to_dict(), "precision": "bfloat16"}, path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path),
            "bf16_recipes": recipes, "precision": "bfloat16"}


def restore_whole(path, *, device, model_factory=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["precision"] != "bfloat16":
        raise ValueError("BF16 checkpoint required")
    if model_factory is None:
        from transformers import AutoConfig, AutoModelForCausalLM
        saved = dict(payload["model_config"])
        kind = saved.pop("model_type")
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(kind, **saved), dtype=torch.bfloat16)
    else:
        model = model_factory(payload["model_config"])
    for name, recipe in payload["bf16_recipes"].items():
        if recipe["storage"] != "bf16" or recipe["compute"] != "bf16":
            raise ValueError("unauthorized checkpoint factor format")
        state = payload["state_dict"]
        layer = PrecisionLinear(state[name+".left"], state[name+".right"], state.get(name+".bias"))
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, layer)
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    assert_bf16(model)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def capture(teacher, tokens, names, seed, progress):
    captured = {name: [] for name in names}
    outputs = {name: [] for name in names}
    handles = []
    for name in names:
        def hook(_layer, args, name=name):
            value = args[0].detach().reshape(-1, args[0].shape[-1])
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("native finite BF16 calibration activations required")
            captured[name].append(value.cpu())
        handles.append(teacher.get_submodule(name).register_forward_pre_hook(hook))
        def output_hook(_layer, _args, result, name=name):
            value = result.detach().reshape(-1, result.shape[-1])
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("native finite BF16 teacher outputs required")
            outputs[name].append(value.cpu())
        handles.append(teacher.get_submodule(name).register_forward_hook(output_hook))
    try:
        for index, start in enumerate(calibration_starts(len(tokens), seed)):
            progress(f"BF16 factor preparation seed={seed} chunk={index}")
            teacher(input_ids=tokens[start:start+512][None].to("cuda"), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return ({name: torch.cat(values).T.contiguous() for name, values in captured.items()},
            {name: torch.cat(values).T.contiguous() for name, values in outputs.items()})


def fit_native(teacher, recipes, inputs, outputs, spec, directory, progress, rcond):
    """Original Gemma measured-output objective, with native BF16 captures."""
    directory.mkdir(exist_ok=False)
    state = {}
    started = time.monotonic()
    for recipe in recipes:
        name = recipe["name"]
        progress(f"BF16 fitting {spec['method']} {name}")
        module = teacher.get_submodule(name)
        left, right = weighted_factors(module.weight.detach().float(), inputs[name].to("cuda"), 384,
                                      outputs=outputs[name].to("cuda"), bias=module.bias,
                                      beta=spec["beta"], ridge_fraction=spec["ridge_fraction"], rcond=rcond)
        state[name] = {"left": left.to(device="cpu", dtype=torch.bfloat16),
                       "right": right.to(device="cpu", dtype=torch.bfloat16),
                       "bias": None if module.bias is None else module.bias.detach().to(device="cpu", dtype=torch.bfloat16)}
    path = directory/"factors.pt"
    torch.save({"targets": recipes, "specification": spec, "state": state,
                "storage": "bfloat16", "compute": "bfloat16"}, path)
    return path, {"factor_parameters": sum(r["factor_parameters"] for r in recipes),
                  "serialized_bytes": path.stat().st_size, "sha256": sha256_file(path),
                  "precision": "bfloat16", "factor_convention": "unbalanced", "targets": recipes,
                  "fit_seconds": time.monotonic()-started, "teacher_outputs": "native BF16 measured outputs"}


def install(model, path, names):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["storage"] != "bfloat16" or payload["compute"] != "bfloat16" or set(payload["state"]) != set(names):
        raise ValueError("BF16 factor recipe mismatch")
    for name, values in payload["state"].items():
        if any(v.dtype != torch.bfloat16 or not torch.isfinite(v).all() for v in values.values() if v is not None):
            raise ValueError("factor dtype or finite check failed")
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, PrecisionLinear(**values))
    assert_bf16(model)
    return model


@torch.inference_mode()
def run(config):
    validate_config(config)
    import psutil
    active = [p.pid for p in psutil.process_iter(["cmdline"], ad_value=None)
              if any(a in {"src.paper_benchmark", "src.paper_architecture_recovery", "src.paper_architecture_panel", "src.paper_precision"}
                     for a in (p.info["cmdline"] or []))]
    if active:
        raise RuntimeError(f"another experiment is active: {active}")
    started = time.monotonic()
    output = create_output(config, "practical-bf16")
    rows = []
    allowance = 14400-config["prior_practical_charged_seconds"]
    def progress(message):
        elapsed = time.monotonic()-started
        if elapsed > allowance:
            raise TimeoutError("practical cumulative cap exceeded; retain incomplete BF16 cohort")
        with (output/"stdout.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{elapsed:.3f} {message}\n")
        print(f"{elapsed:.3f} {message}", flush=True)
    def persist(row):
        rows.append(row)
        write(output/"metrics.json", rows)
    write(output/"config.json", config)
    write(output/"status.json", {"state": "preparing", "elapsed_seconds": 0, "precision": "bfloat16"})
    (output/"stdout.log").touch()
    (output/"stderr.log").touch()
    try:
        env = environment(config)
        write(output/"preflight.json", env)
        if not env["ready"]:
            raise RuntimeError(str(env["missing"]))
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer
        selected, baseline, anchor = frozen_specs(config)
        write(output/"frozen-selection.json", selected)
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        data = load_from_disk(config["dataset_path"])
        if set(data) != {"train", "validation", "test"}:
            raise ValueError("native splits required")
        tokens = {s: tokenizer("\n\n".join(data[s]["text"]), add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0] for s in data}
        manifests = {s: {"tokens": len(t), "hash": token_hash(t)} for s, t in tokens.items()}
        for seed in config["seeds"]:
            manifests.setdefault("calibration", {})[str(seed)] = [
                {"start": start, "tokens": 512, "hash": token_hash(tokens["train"][start:start+512])}
                for start in calibration_starts(len(tokens["train"]), seed)]
        write(output/"token-manifests.json", manifests)
        torch.manual_seed(11)
        teacher = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
                                                      dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
        assert_bf16(teacher)
        recipes = [{"name": n, "shape": list(m.weight.shape), "rank": 384,
                    "factor_parameters": 384*sum(m.weight.shape), "precision": "bfloat16"}
                   for n, m in teacher.named_modules() if n.endswith("self_attn.o_proj")]
        if len(recipes) != 26 or any(r["shape"] != [1152, 1024] for r in recipes):
            raise ValueError("pinned Gemma target recipe changed")
        names = [r["name"] for r in recipes]
        teacher_hash = tensor_digest(teacher)
        contract = {"run_id": output.name, "output_directory": str(output.resolve()), "command": command_line(),
                    "environment": env, "configuration": config, "splits": manifests, "targets": recipes,
                    "teacher_hash": teacher_hash, "selection_hash": sha256_file(config["selection_path"]),
                    "dataset_content_hash": dataset_content_hash(config["dataset_path"]), "baseline": baseline,
                    "selected_solver": anchor, "remaining_seconds_at_start": allowance,
                    "expected_records": 5, "expected_artifacts": ["stdout.log", "stderr.log", "config.json", "metrics.json", "report.md"],
                    "classification": "exploratory two-seed BF16 amendment; no three-seed promotion",
                    "precision_scope": "BF16 evaluated weights, activations and factors; inherited FP64 fitting and FP32 metric accumulations; no other model-format tests"}
        write(output/"contract.json", contract)
        write(output/"status.json", {"state": "running", "precision": "bfloat16", "seeds": config["seeds"]})
        def measure(model, label, directory, seed=None, spec=None, factor_path=None):
            directory.mkdir(exist_ok=True)
            progress(f"BF16 practical {label} seed={seed}: calibration")
            assert_bf16(model)
            control = calibration_control(model, tokens["validation"], tokens["test"], progress=progress,
                                          freeze_path=directory/"frozen-temperature.json", temperatures=config["temperature_grid"])
            progress(f"BF16 practical {label} seed={seed}: timing")
            timing = benchmark_inference(model, tokens["test"], progress=progress)
            before = model_probe_digest(model, tokens["test"])
            whole = save_whole(model, directory/"whole-model.pt")
            model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            progress(f"BF16 practical {label} seed={seed}: whole-model reload")
            restored = restore_whole(directory/"whole-model.pt", device="cuda")
            after = model_probe_digest(restored, tokens["test"])
            if before != after:
                raise RuntimeError("BF16 whole-model reload changed probe logits")
            del restored
            gc.collect()
            torch.cuda.empty_cache()
            parity = {"exact": True, "before": before, "after": after,
                      "scope": "128-token prompt logits and one cached next-token forward"}
            write(directory/"whole-model-reload-parity.json", parity)
            row = {"label": label, "seed": seed, "specification": spec or {"method": "teacher"},
                   "bits": 16, "precision": "bfloat16", "compute_precision": "bfloat16", "placement": "resident",
                   "calibration": control, "timing": timing, "whole_checkpoint": whole,
                   "whole_model_reload_parity": parity}
            if factor_path:
                row.update(checkpoint=str(factor_path.resolve()), factor_checkpoint_bytes=factor_path.stat().st_size,
                           source_sha256=sha256_file(factor_path))
            persist(row)
        measure(teacher, "teacher", output/"teacher")
        dense_bytes = rows[0]["whole_checkpoint"]["bytes"]
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            teacher.to("cuda")
            inputs, outputs = capture(teacher, tokens["train"], names, seed, progress)
            checkpoints = {}
            for label, spec in (("baseline", baseline), ("solver", anchor)):
                path, metadata = fit_native(teacher, recipes, inputs, outputs, spec, output/f"fit-{label}-seed-{seed}",
                                           progress, config["pseudoinverse_rcond"])
                checkpoints[label] = path, metadata, spec
            del inputs, outputs
            if tensor_digest(teacher) != teacher_hash:
                raise RuntimeError("factor preparation mutated teacher")
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            for label in ("baseline", "solver"):
                path, metadata, spec = checkpoints[label]
                model = install(copy.deepcopy(teacher), path, names).to("cuda").eval().requires_grad_(False)
                measure(model, label, output/f"{label}-seed-{seed}-bf16", seed, spec, path)
                rows[-1]["metadata"] = metadata
                rows[-1]["whole_model_savings_fraction"] = 1-rows[-1]["whole_checkpoint"]["bytes"]/dense_bytes
                write(output/"metrics.json", rows)
                del model
                gc.collect()
        if tensor_digest(teacher) != teacher_hash:
            raise RuntimeError("teacher mutated")
        progress("all five BF16-only practical records completed")
        write(output/"status.json", {"state": "completed", "elapsed_seconds": time.monotonic()-started,
                                    "records": len(rows), "precision": "bfloat16", "seeds": config["seeds"], "exploratory": True})
        (output/"report.md").write_text("# BF16-only practical measurements\n\nFive declared records completed with seeds 11/23. Stored factors, evaluated model weights and forward activations use BF16. Original readiness remains failed; two seeds do not establish promotion. Inspect metrics and contract.\n")
        return 0
    except Exception as exc:
        (output/"stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output/"status.json", {"state": "incomplete" if isinstance(exc, TimeoutError) else "failed",
                                    "error": str(exc), "elapsed_seconds": time.monotonic()-started,
                                    "records": len(rows), "precision": "bfloat16", "seeds": config["seeds"]})
        (output/"report.md").write_text(f"# Incomplete BF16-only practical cohort\n\n{exc}\nSaved records: {len(rows)}/5. No automatic restart.\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise SystemExit(run(json.loads(Path(args.config).read_text())))

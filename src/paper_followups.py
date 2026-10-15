"""Reusable, offline Stage D/E/F protocols. Nothing runs on import.

Callers supply the existing comparison runner's fit/evaluation operation. Frozen
selection must come from validation; these helpers never select from test data.
"""
from __future__ import annotations

import math
import copy
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from .compression_core import PackedLowRankLinear, checkpoint_has_dense_weight, sha256_file
from .paper_protocol import windows


def composition_groups(depth):
    if depth < 4:
        raise ValueError("composition protocol requires at least four blocks")
    groups = []
    for size in (1, 2, 4):
        for start in sorted({0, (depth-size)//2, depth-size}):
            groups.append(tuple(range(start, start+size)))
    groups.append(tuple(range(depth)))
    return list(dict.fromkeys(groups))


def frozen_methods(selection):
    if selection.get("selection_split") != "validation":
        raise ValueError("frozen settings must be selected exclusively on validation")
    if not selection.get("source_run_id"):
        raise ValueError("selection requires source-run provenance")
    solver, baseline = selection["solver"], selection["baseline"]
    if baseline["method"] not in {"asvd", "whitening"}:
        raise ValueError("strongest activation-aware comparator must be ASVD or whitening")
    return solver, baseline


def run_composition(selection, depth, execute, *, seeds=(11, 23, 37), check_budget=lambda: None):
    """execute(method_descriptor, seed, rank, group, split) persists each result.

    Parent runner owns unique output/contracts, stage cap and teacher drift hooks.
    A budget failure propagates so incomplete comparisons cannot be promoted.
    """
    solver, baseline = frozen_methods(selection)
    results = []
    for group in composition_groups(depth):
        for method in (baseline, solver):
            for seed in seeds:
                check_budget()
                results.append(execute(method, seed, selection["rank"], group, "test"))
    return results


def run_replication(selection, shapes, execute, *, check_budget=lambda: None):
    """GPT-2 confirmation is rank 256; frontier is validation-only and separate.

    shapes are attention-output [out,in] matrix shapes in block order. Rank
    divisors refer to min(out,in), not factor parameter or whole-model ratios.
    execute receives a rank list for per-target ranks.
    """
    solver, baseline = frozen_methods(selection)
    if not shapes or any(min(s) < 256 for s in shapes):
        raise ValueError("GPT-2 target shape mismatch")
    results = []
    group = tuple(range(len(shapes)))
    for method in ({"method": "teacher"}, {"method": "svd"}, baseline, solver):
        for seed in ((11,) if method["method"] in {"teacher", "svd"} else (11, 23, 37)):
            check_budget()
            results.append(execute(method, seed, [256]*len(shapes), group, "test"))
    for divisor in (2, 3, 4):
        ranks = [min(shape)//divisor for shape in shapes]
        for method in (baseline, solver):
            for seed in (11, 23, 37):
                check_budget()
                results.append(execute(method, seed, ranks, group, "validation"))
    return results


@torch.inference_mode()
def calibration_metrics(model, tokens, *, temperature=1.0, progress=None):
    """Exactly-once test targets; 15 fixed bins and token-weighted ECE."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    device = next(model.parameters()).device
    total = correct = count = 0
    entropy_sum = 0.0
    bin_count = torch.zeros(15, dtype=torch.float64)
    bin_confidence = torch.zeros(15, dtype=torch.float64)
    bin_correct = torch.zeros(15, dtype=torch.float64)
    for index, (start, end, first) in enumerate(windows(len(tokens))):
        if progress:
            progress(f"calibration metrics window={index}")
        ids = tokens[start:end].unsqueeze(0).to(device)
        offset = first-start-1
        scores = model(input_ids=ids, use_cache=False).logits[0, offset:-1].float()/temperature
        labels = ids[0, offset+1:]
        logp = scores.log_softmax(-1)
        losses = F.nll_loss(logp, labels, reduction="none")
        if not torch.isfinite(losses).all() or not torch.isfinite(logp).all():
            raise FloatingPointError("nonfinite calibration metrics")
        confidence, predictions = logp.exp().max(-1)
        hits = predictions.eq(labels)
        bins = (confidence*15).long().clamp(max=14).cpu()
        bin_count += torch.bincount(bins, minlength=15)
        bin_confidence.scatter_add_(0, bins, confidence.double().cpu())
        bin_correct.scatter_add_(0, bins, hits.double().cpu())
        total += losses.double().sum().item()
        entropy_sum += (-(logp.exp()*logp).sum(-1)).double().sum().item()
        correct += hits.sum().item()
        count += len(labels)
    if count != len(tokens)-1 or count <= 0:
        raise ValueError("calibration target count mismatch")
    ece = (bin_correct-bin_confidence).abs().sum().item()/count
    return {"temperature": temperature, "nll": total/count, "ppl": math.exp(total/count),
            "targets": count, "accuracy": correct/count, "entropy": entropy_sum/count,
            "ece_15": ece, "bins": [{"lower": i/15, "upper": (i+1)/15,
                "targets": int(bin_count[i]), "confidence_sum": float(bin_confidence[i]),
                "correct": int(bin_correct[i])} for i in range(15)]}


@torch.inference_mode()
def fit_temperature(model, validation_tokens, *, temperatures=None, progress=None):
    """Fit one scalar from a declared bounded grid without retaining logits.

    This is a grid fit, not an exact continuous optimizer. The grid and boundary
    hit are returned, so a bounded optimum cannot be mistaken for an interior fit.
    """
    grid = list(temperatures if temperatures is not None else [math.exp(-math.log(4)+i*math.log(16)/32) for i in range(33)])
    if not grid or any(not math.isfinite(t) or t <= 0 for t in grid):
        raise ValueError("invalid temperature grid")
    sums, count = [0.0]*len(grid), 0
    device = next(model.parameters()).device
    for index, (start, end, first) in enumerate(windows(len(validation_tokens))):
        if progress:
            progress(f"temperature fit validation window={index}")
        ids = validation_tokens[start:end].unsqueeze(0).to(device)
        offset = first-start-1
        logits = model(input_ids=ids, use_cache=False).logits[0, offset:-1].float()
        labels = ids[0, offset+1:]
        for j, temperature in enumerate(grid):
            loss = F.cross_entropy(logits/temperature, labels, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite temperature loss")
            sums[j] += loss.double().item()
        count += len(labels)
    if count != len(validation_tokens)-1 or count <= 0:
        raise ValueError("temperature validation target count mismatch")
    winner = min(range(len(grid)), key=lambda i: (sums[i], abs(math.log(grid[i]))))
    return {"temperature": grid[winner], "selection_split": "validation", "targets": count,
            "fit": "bounded_grid", "grid": grid, "validation_nll": [s/count for s in sums],
            "boundary_hit": grid[winner] in (min(grid), max(grid))}


def save_reload_factor(layer, path, probe):
    """Persist explicit metadata, then demand exact same-format reload parity."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = {k: v.detach().cpu() for k, v in layer.state_dict().items()}
    if checkpoint_has_dense_weight(state):
        raise ValueError("dense replacement weight found")
    metadata = {"in_features": layer.in_features, "out_features": layer.out_features,
                "rank": layer.rank, "bits": layer.bits}
    torch.save({"metadata": metadata, "state_dict": state}, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    restored = PackedLowRankLinear(**payload["metadata"],
        u=torch.zeros(layer.out_features, layer.rank), v=torch.zeros(layer.rank, layer.in_features),
        bias=state.get("bias"))
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.to(probe.device)
    with torch.inference_mode():
        before, after = layer(probe), restored(probe)
    if not torch.equal(before, after):
        raise RuntimeError("packed checkpoint reload changed outputs")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path),
            "reload_exact": True, "bits": layer.bits, "metadata": metadata}


def calibration_control(model, validation_tokens, test_tokens, *, progress=None, freeze_path=None, temperatures=None):
    """Return raw and frozen-temperature test metrics plus validation provenance."""
    fit = fit_temperature(model, validation_tokens, temperatures=temperatures, progress=progress)
    if freeze_path is not None:
        path = Path(freeze_path)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(fit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    raw = calibration_metrics(model, test_tokens, progress=progress)
    scaled = calibration_metrics(model, test_tokens, temperature=fit["temperature"], progress=progress)
    if raw["targets"] != scaled["targets"]:
        raise ValueError("raw/scaled metric masking mismatch")
    return {"fit": fit, "raw_test": raw, "scaled_test": scaled}


def repack_student(student, target_names, *, bits):
    """Repack existing factors in place; caller provides an isolated student."""
    if bits not in (8, 16):
        raise ValueError("Stage F compares only FP16 and INT8")
    for name in target_names:
        module = student.get_submodule(name)
        if not isinstance(module, PackedLowRankLinear):
            raise ValueError(f"{name} is not factorized")
        u = module._factor("u", module.out_features, module.rank)
        v = module._factor("v", module.rank, module.in_features)
        packed = PackedLowRankLinear.from_factors(u, v, module.bias, bits=bits).to(u.device)
        parent, child = name.rsplit(".", 1)
        setattr(student.get_submodule(parent), child, packed)
    return student


def storage_accounting(shapes, ranks, *, untouched_parameters):
    """Parameter savings at FP16; actual serialized-byte accounting is separate."""
    if len(shapes) != len(ranks) or untouched_parameters < 0:
        raise ValueError("invalid storage accounting")
    dense = sum(out*inside for out, inside in shapes)
    factors = sum(rank*(out+inside) for (out, inside), rank in zip(shapes, ranks))
    if any(not 0 < rank <= min(shape) for shape, rank in zip(shapes, ranks)):
        raise ValueError("invalid rank")
    return {"rank_fraction_by_target": [rank/min(shape) for shape, rank in zip(shapes, ranks)],
            "dense_target_parameters": dense, "factor_parameters": factors,
            "target_matrix_savings_fraction": 1-factors/dense,
            "whole_model_savings_fraction": (dense-factors)/(dense+untouched_parameters),
            "serialized_bytes_measured_separately": True}


@torch.inference_mode()
def benchmark_inference(model, prompt_tokens, *, progress=None):
    """Batch 1, synchronized 10 warmups/30 samples; greedy cached decode 128.

    Peak memory includes the resident model and temporary KV cache. Each measured
    invocation resets peak stats; timing excludes input construction. INT8 factors
    are dequantized by PackedLowRankLinear: no integer-kernel speedup is implied.
    """
    device = next(model.parameters()).device
    if device.type != "cuda" or len(prompt_tokens) < 512:
        raise ValueError("benchmark requires CUDA and at least 512 prompt tokens")
    rows = []
    for length in (128, 512):
        ids = prompt_tokens[:length].unsqueeze(0).to(device)
        for mode in ("prefill", "decode"):
            # Drop unused calibration/prior-mode allocator pools before warmup.
            # Keep the 10 warmups' pool throughout all 30 measured iterations.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            times, allocated, reserved = [], [], []
            for iteration in range(40):
                if progress:
                    progress(f"timing {mode} length={length} iteration={iteration}")
                if mode == "decode":
                    initial = model(input_ids=ids, use_cache=True)
                    cache = initial.past_key_values
                    token = initial.logits[:, -1].argmax(-1, keepdim=True)
                    del initial
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                if mode == "prefill":
                    output = model(input_ids=ids, use_cache=True)
                else:
                    for _ in range(128):
                        output = model(input_ids=token, past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                        token = output.logits[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-started
                if iteration >= 10:
                    times.append(elapsed)
                    allocated.append(torch.cuda.max_memory_allocated(device))
                    reserved.append(torch.cuda.max_memory_reserved(device))
                del output
                if mode == "decode":
                    del cache, token
            ordered = torch.tensor(times, dtype=torch.float64)
            rows.append({"mode": mode, "prefill_tokens": length, "decode_tokens": 128 if mode == "decode" else 0,
                         "batch_size": 1, "warmups": 10, "measurements": 30, "use_cache": True,
                         "median_seconds": ordered.quantile(.5).item(), "p95_seconds": ordered.quantile(.95).item(),
                         "peak_allocated_bytes": max(allocated), "peak_reserved_bytes": max(reserved)})
    return rows


def _canonical_spec(spec):
    result = dict(spec)
    if result.get("method") == "anchored" and result.get("beta", 0) == 0 and result.get("ridge_fraction", 0) == 0:
        result = {"method": "whitening", "beta": 0.0, "ridge_fraction": 0.0}
    if result["method"] == "whitening":
        result.update(beta=0.0, ridge_fraction=0.0)
    return result


def dataset_content_hash(directory):
    """Hash the actual pinned saved dataset, independent of model tokenization."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted((path for path in directory.rglob("*") if path.is_file()),
                   key=lambda path: path.relative_to(directory).as_posix())
    if not files:
        raise ValueError("empty pinned dataset directory")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8")+b"\0")
        digest.update(sha256_file(path).encode("ascii")+b"\n")
    return digest.hexdigest()


def _selection(config):
    path = Path(config["selection_path"])
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("state") != "frozen" or selection.get("selection_split") != "validation":
        raise ValueError("complete frozen validation selection required")
    if selection["dataset_revision"] != config["dataset_revision"] or selection["precision"] != "float16":
        raise ValueError("selection asset/precision mismatch")
    if selection.get("factor_convention", "unbalanced") != config.get("factor_convention", "unbalanced"):
        raise ValueError("factor convention changed after validation selection")
    solver = _canonical_spec(selection["selected_solver"]["specification"])
    baseline = _canonical_spec(selection["strongest_baseline"]["specification"])
    frozen = {"selection_split": "validation", "source_run_id": path.parent.name,
              "solver": solver, "baseline": baseline, "rank": selection["rank"]}
    frozen_methods(frozen)
    return selection, frozen


def _manifest_calibration_hashes(manifests):
    hashes = manifests.get("calibration_hashes", manifests.get("calibration_token_hashes"))
    if not hashes or set(hashes) != {"11", "23", "37"}:
        raise ValueError("actual calibration token hashes missing")
    return hashes


def _target_recipes(targets):
    required = ("name", "shape", "rank", "factor_parameters")
    if not targets or any(any(key not in target for key in required) for target in targets):
        raise ValueError("complete target recipes required")
    return [{key: target[key] for key in required} for target in targets]


def _baseline_checkpoints(config, selection, frozen, *, manifests, teacher_hash, targets):
    directory = Path(config["baseline_run_path"])
    status = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    if status["state"] != "completed":
        raise ValueError("completed Stage C required")
    prior = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    for key in ("model_revision", "dataset_revision", "rank", "precision", "context", "stride", "seeds"):
        if prior[key] != config[key]:
            raise ValueError(f"Stage C mismatch: {key}")
    if prior.get("factor_convention", "unbalanced") != config.get("factor_convention", "unbalanced"):
        raise ValueError("Stage C factor convention mismatch")
    source_contract = json.loads((directory/"contract.json").read_text(encoding="utf-8"))
    if source_contract["selection_hash"] != sha256_file(config["selection_path"]):
        raise ValueError("Stage C used a different frozen selection")
    if source_contract.get("teacher_hash") != teacher_hash:
        raise ValueError("fresh teacher differs from Stage C teacher")
    for split in ("train", "validation", "test"):
        if source_contract.get("splits", {}).get(split) != manifests[split]:
            raise ValueError(f"Stage C {split} token content changed")
    if _manifest_calibration_hashes(source_contract.get("splits", {})) != _manifest_calibration_hashes(manifests):
        raise ValueError("Stage C calibration token content changed")
    if source_contract["splits"].get("calibration_positions") != manifests["calibration_positions"]:
        raise ValueError("Stage C calibration positions changed")
    recipes = _target_recipes(targets)
    if _target_recipes(source_contract.get("targets")) != recipes:
        raise ValueError("fresh target names/shapes/ranks differ from Stage C")
    entries = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
    lookup = {}
    for label in ("solver", "baseline"):
        for seed in (11, 23, 37):
            matches = [r for r in entries if r["seed"] == seed and _canonical_spec(r["specification"]) == frozen[label]]
            if not matches:
                raise ValueError(f"frozen {label} seed {seed} missing in Stage C")
            entry = matches[0]
            checkpoint = Path(entry["checkpoint"])
            if sha256_file(checkpoint) != entry["metadata"]["sha256"]:
                raise ValueError("source factor checkpoint hash changed")
            if _target_recipes(entry["metadata"].get("targets")) != recipes:
                raise ValueError("source factor checkpoint target recipe mismatch")
            if entry["metadata"].get("factor_convention", "unbalanced") != config.get("factor_convention", "unbalanced"):
                raise ValueError("source checkpoint factor convention mismatch")
            if entry["metadata"].get("precision") != "float16":
                raise ValueError("source checkpoint factor precision mismatch")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if set(state) != {recipe["name"] for recipe in recipes}:
                raise ValueError("source factor checkpoint target coverage mismatch")
            for recipe in recipes:
                value = state[recipe["name"]]
                out, inside = recipe["shape"]
                rank = recipe["rank"]
                if tuple(value["u"].shape) != (out, rank) or tuple(value["v"].shape) != (rank, inside):
                    raise ValueError("source factor checkpoint tensor shape mismatch")
                if any(value[key].dtype != torch.float16 or not torch.isfinite(value[key]).all() for key in ("u", "v")):
                    raise ValueError("source factors contain nonfinite values or mismatched dtype")
                if value.get("bias") is not None and tuple(value["bias"].shape) != (out,):
                    raise ValueError("source checkpoint bias shape mismatch")
            del state
            lookup[label, seed] = entry
    return lookup


def _install_checkpoint(model, path, bits=16):
    """Install onto fresh teacher in place, retaining only one GPU model."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = set(name for name, module in model.named_modules()
                   if name.endswith("self_attn.o_proj") or name.endswith("attn.c_proj"))
    if set(state) != expected:
        raise ValueError("checkpoint attention target coverage mismatch")
    device = next(model.parameters()).device
    for name, values in state.items():
        parent, child = name.rsplit(".", 1)
        layer = PackedLowRankLinear.from_factors(values["u"], values["v"], values["bias"], bits=bits).to(device)
        setattr(model.get_submodule(parent), child, layer)
    return model


def _save_model_state(model, path):
    """Save all weights and explicit packed recipes, preserving tied tensor storage."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    copied, state = {}, {}
    for name, value in model.state_dict().items():
        key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in copied:
            copied[key] = value.detach().cpu()
        state[name] = copied[key]
    recipes = {}
    for name, layer in model.named_modules():
        if isinstance(layer, PackedLowRankLinear):
            local = layer.state_dict()
            if checkpoint_has_dense_weight(local):
                raise ValueError("packed layer registers dense replacement weights")
            recipes[name] = {"in_features": layer.in_features, "out_features": layer.out_features,
                             "rank": layer.rank, "bits": layer.bits}
    torch.save({"state_dict": state, "packed_recipes": recipes,
                "model_config": model.config.to_dict()}, path)
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path), "packed_recipes": recipes}


def restore_model_checkpoint(path, *, device="cpu", model_factory=None):
    """Reconstruct a complete model on CPU, then transfer once to its device.

    model_factory is an optional testing hook accepting saved model_config. The
    production path constructs from the saved config, never a remote snapshot.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if model_factory is None:
        from transformers import AutoConfig, AutoModelForCausalLM
        saved = dict(payload["model_config"])
        model_type = saved.pop("model_type")
        configuration = AutoConfig.for_model(model_type, **saved)
        model = AutoModelForCausalLM.from_config(configuration, torch_dtype=torch.float16)
    else:
        model = model_factory(payload["model_config"])
    for name, recipe in payload["packed_recipes"].items():
        values = payload["state_dict"]
        bias = values.get(name+".bias")
        layer = PackedLowRankLinear(**recipe,
            u=torch.zeros(recipe["out_features"], recipe["rank"]),
            v=torch.zeros(recipe["rank"], recipe["in_features"]), bias=bias)
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, layer)
    model.load_state_dict(payload["state_dict"], strict=True)
    del payload
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def model_probe_digest(model, tokens):
    """Bitwise hashes for prompt logits and one next-token cached forward."""
    device = next(model.parameters()).device
    ids = tokens[:128][None].to(device)
    prompt = model(input_ids=ids, use_cache=True)
    token = prompt.logits[:, -1].argmax(-1, keepdim=True)
    continuation = model(input_ids=token, past_key_values=prompt.past_key_values, use_cache=True)
    def digest(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    return {"prompt_tokens": ids.shape[-1], "prompt_logits_sha256": digest(prompt.logits),
            "cached_next_logits_sha256": digest(continuation.logits)}



def _composition_source_artifacts(config):
    """Validate an optional completed-C boundary manifest before reusing metrics."""
    directory = Path(config["baseline_run_path"])
    declared = config.get("baseline_artifact_integrity_path")
    integrity = Path(declared) if declared else directory/"artifact-integrity.json"
    verified = False
    if declared or integrity.exists():
        manifest = json.loads(integrity.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})
        for filename in ("metrics.json", "contract.json", "teacher.json"):
            if not artifacts.get(filename) or sha256_file(directory/filename) != artifacts[filename]:
                raise ValueError(f"completed Stage C artifact changed: {filename}")
        verified = True
    return {"run_id": directory.name, "metrics_sha256": sha256_file(directory/"metrics.json"),
            "boundary_integrity_verified": verified,
            "integrity_manifest_sha256": sha256_file(integrity) if verified else None}


def _composition_record(label, seed, rank, group, names, entry, source_artifacts,
                        evaluate_fresh, *, validated_source):
    """Only the exact all-target condition can reuse validated completed-C metrics."""
    if not validated_source:
        raise ValueError("fresh teacher/data/target/checkpoint validation is required")
    group = tuple(group)
    if not group or any(i < 0 or i >= len(names) for i in group) or group != tuple(range(group[0], group[-1]+1)):
        raise ValueError("composition group must be a valid consecutive ordered group")
    full_stack = group == tuple(range(len(names)))
    recipe_by_name = {recipe["name"]: recipe for recipe in entry["metadata"]["targets"]}
    selected_recipes = [recipe_by_name[names[index]] for index in group]
    if any(recipe["rank"] != rank for recipe in selected_recipes):
        raise ValueError("composition rank differs from cached factors")
    reused = {}
    if full_stack:
        metrics = copy.deepcopy(entry["metrics"])
        reused = {"reused_from_run": source_artifacts["run_id"], "source_record_id": entry["id"],
                  "source_metrics_sha256": source_artifacts["metrics_sha256"],
                  "source_boundary_integrity_verified": source_artifacts["boundary_integrity_verified"],
                  "source_integrity_manifest_sha256": source_artifacts["integrity_manifest_sha256"],
                  "independent_replication": False,
                  "evaluation_origin": "reused completed Stage C full-stack measurement"}
    else:
        metrics = evaluate_fresh()
        reused = {"evaluation_origin": "fresh complete native test evaluation"}
    if len(metrics["residual_by_block"]) != len(names):
        raise ValueError("source/fresh drift block count mismatch")
    metrics["group_end_drift"] = copy.deepcopy(metrics["residual_by_block"][group[-1]])
    metrics["downstream_drift"] = copy.deepcopy(metrics["residual_by_block"][group[-1]+1:])
    metadata = {"factor_parameters": sum(recipe["factor_parameters"] for recipe in selected_recipes),
                "targets": copy.deepcopy(selected_recipes), "precision": entry["metadata"]["precision"],
                "factor_convention": entry["metadata"].get("factor_convention", "unbalanced"),
                "serialized_bytes": entry["metadata"]["serialized_bytes"] if full_stack else None,
                "storage_scope": "full-stack factor checkpoint" if full_stack else "subgroup bytes not measured"}
    return {"label": label, "specification": copy.deepcopy(entry["specification"]), "seed": seed,
            "rank": rank, "group": list(group), "split": "test", "checkpoint": entry["checkpoint"],
            "checkpoint_sha256": entry["metadata"]["sha256"], "metadata": metadata,
            "metrics": metrics, **reused}


def run_followups(config, stage):
    """Actual bounded offline D/E/F dispatcher; returns 0 only on complete stage.

    Required configuration: selection_path (frozen Stage B), baseline_run_path
    (complete Stage C). Stage E overrides only the predeclared model and rank.
    All subcomparisons persist incrementally; stage caps terminate incomplete work.
    """
    from .paper_benchmark import create_output, environment, write, tensor_digest, command_line
    from .paper_comparisons import (load_assets, target_modules, canonical_weight,
        capture_inputs, fit_checkpoint, student_from_checkpoint, evaluate_diagnostics,
        validate_selection_provenance)
    from .paper_protocol import token_hash, calibration_starts, evaluate
    if stage not in {"composition", "replication", "practical"}:
        raise ValueError("unsupported followup stage")
    config = dict(config)
    if stage == "replication":
        revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
        config.update(model="openai-community/gpt2", model_revision=revision, rank=256,
            model_path=str(Path(config.get("gpt2_model_path", f"~/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/{revision}")).expanduser().resolve()))
        if Path(config["model_path"]).name != revision:
            raise ValueError("GPT-2 pinned revision mismatch")
    output = create_output(config, stage)
    started = time.monotonic()
    results = []
    (output/"stderr.log").touch()
    def progress(message):
        cap = config.get("runtime_remaining_seconds", {}).get(stage, config["stage_gpu_hours"][stage]*3600)
        if time.monotonic()-started > cap:
            raise TimeoutError(f"{stage} cap exceeded; comparison incomplete")
        print(message, flush=True)
        with (output/"stdout.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.monotonic()-started:.3f} {message}\n")
    def persist(result):
        results.append(result)
        write(output/"metrics.json", results)
        return result
    try:
        selection, frozen = _selection(config)
        data_hash = dataset_content_hash(config["dataset_path"])
        if selection.get("dataset_content_hash") != data_hash:
            raise ValueError("pinned dataset disk content changed after validation selection")
        write(output/"frozen-selection.json", selection)
        env = environment(config)
        write(output/"preflight.json", env)
        if not env["ready"]:
            raise RuntimeError(f"offline preflight failed: {env['missing']}")
        if config["precision"] != "float16" or config["seeds"] != [11, 23, 37]:
            raise ValueError("frozen precision/seed protocol mismatch")
        teacher, tokens = load_assets(config)
        manifests = {split: {"tokens": len(value), "hash": token_hash(value)} for split, value in tokens.items()}
        manifests["calibration_positions"] = {str(seed): calibration_starts(len(tokens["train"]), seed) for seed in config["seeds"]}
        manifests["calibration_hashes"] = {str(seed): token_hash(torch.cat([
            tokens["train"][position:position+512] for position in manifests["calibration_positions"][str(seed)]]))
            for seed in config["seeds"]}
        targets = target_modules(teacher)
        names = [name for name, _ in targets]
        shapes = [list(canonical_weight(module).shape) for _, module in targets]
        original = tensor_digest(teacher)
        target_recipes = [{"name": name, "shape": shape, "rank": config["rank"],
                          "factor_parameters": config["rank"]*sum(shape)} for name, shape in zip(names, shapes)]
        if stage != "replication":
            if selection["model_revision"] != config["model_revision"] or selection["rank"] != config["rank"]:
                raise ValueError("frozen selection model/rank changed")
            if selection["validation_hash"] != manifests["validation"]["hash"]:
                raise ValueError("frozen validation tokens changed")
            validate_selection_provenance(selection, config, manifests, original, target_recipes)
        write(output/"token-manifests.json", manifests)
        write(output/"contract.json", {"run_id": output.name, "output_directory": str(output.resolve()),
            "command": command_line(), "environment": env, "configuration": config,
            "splits": manifests, "targets": target_recipes,
            "representation_family": "packed low-rank factors", "group_size": "composition groups" if stage == "composition" else "all attention outputs",
            "teacher_hash": original, "selection_hash": sha256_file(config["selection_path"]),
            "dataset_content_hash": data_hash,
            "baseline": frozen["baseline"], "selection_model_revision": selection["model_revision"],
            "temperature_fit": {"split": "validation", "method": "bounded_grid",
                "grid": config.get("temperature_grid", [math.exp(-math.log(4)+i*math.log(16)/32) for i in range(33)]),
                "boundary_flag": True, "freeze_before_test": True},
            "expected_artifacts": ["config.json", "contract.json", "stdout.log", "stderr.log", "metrics.json", "report.md", "status.json"]})
        write(output/"status.json", {"state": "running"})
        if stage == "composition":
            source = _baseline_checkpoints(config, selection, frozen, manifests=manifests,
                                           teacher_hash=original, targets=target_recipes)
            source_artifacts = _composition_source_artifacts(config)
            preserved = {}
            resume = config.get("composition_resume_path")
            if resume:
                resume = Path(resume)
                manifest = json.loads((resume/"artifact-integrity.json").read_text())
                for filename, digest in manifest["artifacts"].items():
                    if sha256_file(resume/filename) != digest:
                        raise ValueError("preserved composition artifact changed")
                previous = json.loads((resume/"contract.json").read_text())
                current = json.loads((output/"contract.json").read_text())
                for key in ("splits", "targets", "teacher_hash", "selection_hash", "dataset_content_hash", "baseline"):
                    if previous[key] != current[key]:
                        raise ValueError(f"composition continuation mismatch: {key}")
                for row in json.loads((resume/"metrics.json").read_text()):
                    key = (row["label"], row["seed"], tuple(row["group"]))
                    entry = source[row["label"], row["seed"]]
                    if key in preserved or row["checkpoint_sha256"] != entry["metadata"]["sha256"] or row["specification"] != entry["specification"] or row["metrics"]["targets"] != len(tokens["test"])-1:
                        raise ValueError("invalid preserved composition record")
                    preserved[key] = row
            teacher_result = evaluate(teacher, tokens["test"], progress=progress)
            write(output/"teacher.json", teacher_result)
            def execute(label, seed, rank, group, split):
                if split != "test":
                    raise ValueError("composition protocol requires the native test split")
                entry = source[label, seed]
                cached = preserved.get((label, seed, tuple(group)))
                def evaluate_fresh():
                    student = student_from_checkpoint(teacher, entry["checkpoint"], names=[names[i] for i in group])
                    try:
                        return evaluate_diagnostics(student, teacher, tokens[split], config, progress)
                    finally:
                        del student
                        torch.cuda.empty_cache()
                result = _composition_record(label, seed, rank, group, names, entry, source_artifacts,
                    (lambda: copy.deepcopy(cached["metrics"])) if cached else evaluate_fresh, validated_source=True)
                if cached:
                    result.update(evaluation_origin="preserved complete composition measurement",
                        reused_from_run=resume.name, source_metrics_sha256=sha256_file(resume/"metrics.json"),
                        independent_replication=False)
                return persist(result)
            for group in composition_groups(len(targets)):
                for label in ("baseline", "solver"):
                    for seed in config["seeds"]:
                        progress("composition comparison")
                        execute(label, seed, frozen["rank"], group, "test")
            if tensor_digest(teacher) != original:
                raise RuntimeError("composition mutated teacher")
        elif stage == "replication":
            teacher_result = evaluate(teacher, tokens["test"], progress=progress)
            write(output/"teacher.json", teacher_result)
            inputs_cache, checkpoints = {}, {}
            total_parameters = sum(p.numel() for p in teacher.parameters())
            untouched = total_parameters-sum(out*inside for out, inside in shapes)
            def execute(spec, seed, ranks, group, split):
                if spec["method"] == "teacher":
                    return persist({"specification": spec, "seed": seed, "split": split, "metrics": teacher_result})
                if len(set(ranks)) != 1:
                    raise ValueError("per-target rank mismatch; GPT-2 equal shapes expected")
                rank = ranks[0]
                key = (json.dumps(spec, sort_keys=True), seed, rank)
                if key not in checkpoints:
                    if seed not in inputs_cache:
                        inputs_cache[seed] = capture_inputs(teacher, tokens["train"], seed, progress)
                    identifier = f"fit-{len(checkpoints):02d}-seed-{seed}-rank-{rank}-{spec['method']}"
                    fit_config = {**config, "rank": rank}
                    checkpoints[key] = fit_checkpoint(teacher, inputs_cache[seed], fit_config, spec, output/identifier, progress)
                checkpoint, metadata = checkpoints[key]
                student = student_from_checkpoint(teacher, checkpoint)
                metric = evaluate_diagnostics(student, teacher, tokens[split], config, progress)
                result = persist({"specification": spec, "seed": seed, "rank": rank, "split": split,
                    "checkpoint": str(checkpoint.resolve()), "metadata": metadata,
                    "storage": storage_accounting(shapes, ranks, untouched_parameters=untouched), "metrics": metric})
                del student
                torch.cuda.empty_cache()
                return result
            run_replication(frozen, shapes, execute, check_budget=lambda: progress("replication comparison"))
            if tensor_digest(teacher) != original:
                raise RuntimeError("replication mutated teacher")
        else:
            source = _baseline_checkpoints(config, selection, frozen, manifests=manifests,
                                           teacher_hash=original, targets=target_recipes)
            # One GPU model only throughout calibration and inference timing.
            dense_storage = _save_model_state(teacher, output/"dense-teacher.pt")
            teacher_control = calibration_control(teacher, tokens["validation"], tokens["test"], progress=progress,
                freeze_path=output/"teacher-frozen-temperature.json", temperatures=config.get("temperature_grid"))
            teacher_timing = benchmark_inference(teacher, tokens["test"], progress=progress)
            if tensor_digest(teacher) != original:
                raise RuntimeError("calibration/timing mutated dense teacher")
            persist({"label": "teacher", "calibration": teacher_control, "timing": teacher_timing,
                     "whole_checkpoint": dense_storage})
            del targets, teacher
            gc.collect()
            torch.cuda.empty_cache()
            for seed in config["seeds"]:
                for label in ("baseline", "solver"):
                    for bits in (16, 8):
                        progress(f"practical {label} seed={seed} bits={bits}")
                        model, refreshed_tokens = load_assets(config)
                        if any(token_hash(refreshed_tokens[s]) != manifests[s]["hash"] for s in tokens):
                            raise ValueError("token manifests changed on fresh reload")
                        entry = source[label, seed]
                        _install_checkpoint(model, entry["checkpoint"], bits)
                        identifier = f"{label}-seed-{seed}-bits-{bits}"
                        directory = output/identifier
                        directory.mkdir(exist_ok=False)
                        parity = {}
                        for name in names:
                            layer = model.get_submodule(name)
                            probe = torch.ones(1, 4, layer.in_features, device=config["device"], dtype=torch.float16)
                            parity[name] = save_reload_factor(layer, directory/(name+".pt"), probe)
                        del probe
                        control = calibration_control(model, tokens["validation"], tokens["test"], progress=progress,
                            freeze_path=directory/"frozen-temperature.json", temperatures=config.get("temperature_grid"))
                        timing = benchmark_inference(model, tokens["test"], progress=progress)
                        checkpoint = _save_model_state(model, directory/"whole-model.pt")
                        before_probe = model_probe_digest(model, tokens["test"])
                        del layer, model, refreshed_tokens
                        gc.collect()
                        torch.cuda.empty_cache()
                        progress(f"full checkpoint reload {identifier}")
                        restored = restore_model_checkpoint(directory/"whole-model.pt", device=config["device"])
                        after_probe = model_probe_digest(restored, tokens["test"])
                        if before_probe != after_probe:
                            raise RuntimeError("full-model packed checkpoint reload changed probe logits")
                        parity_result = {"exact": True, "before": before_probe, "after": after_probe,
                                         "scope": "128-token prompt logits and one cached next-token forward"}
                        write(directory/"whole-model-reload-parity.json", parity_result)
                        del restored
                        gc.collect()
                        torch.cuda.empty_cache()
                        persist({"label": label, "seed": seed, "bits": bits, "specification": frozen[label],
                            "source_checkpoint": entry["checkpoint"], "source_sha256": entry["metadata"]["sha256"],
                            "factor_reload": parity, "factor_checkpoint_bytes": sum(p["bytes"] for p in parity.values()),
                            "whole_checkpoint": checkpoint, "whole_model_savings_fraction": 1-checkpoint["bytes"]/dense_storage["bytes"],
                            "whole_model_reload_parity": parity_result, "calibration": control, "timing": timing})
            limits = []
            for seed in config["seeds"]:
                baseline = next(r for r in results if r.get("seed") == seed and r["label"] == "baseline" and r["bits"] == 16)
                for candidate in [r for r in results if r.get("seed") == seed and r["label"] == "solver"]:
                    for measured, reference in zip(candidate["timing"], baseline["timing"]):
                        latency = measured["median_seconds"]/reference["median_seconds"]-1
                        memory = measured["peak_allocated_bytes"]/reference["peak_allocated_bytes"]-1
                        limits.append({"seed": seed, "bits": candidate["bits"], "mode": measured["mode"],
                            "prefill_tokens": measured["prefill_tokens"], "median_latency_regression": latency,
                            "peak_allocated_regression": memory,
                            "within_hardware_budget": max(latency, memory) <= config["hardware_regression_limit"]})
            write(output/"hardware-gates.json", limits)
        progress("all declared comparisons completed")
        write(output/"status.json", {"state": "completed", "elapsed_seconds": time.monotonic()-started})
        (output/"report.md").write_text(f"# {stage}\n\nCompleted {len(results)} declared comparisons. "
            "Frozen validation settings retained; no promotion claim until paired uncertainty and all gates are reviewed.\n"
            "Temperature calibration uses a declared bounded grid; INT8 execution dequantizes factors and claims no integer-kernel acceleration.\n", encoding="utf-8")
        return 0
    except Exception as exc:
        state = "incomplete" if isinstance(exc, TimeoutError) else "failed"
        (output/"stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output/"status.json", {"state": state, "error": str(exc), "elapsed_seconds": time.monotonic()-started})
        (output/"report.md").write_text(f"# {stage}\n\n{state}: {exc}\nPartial artifacts preserved; no comparable completion claim.\n", encoding="utf-8")
        print(f"{stage} {state}: {output}: {exc}", file=sys.stderr)
        return 1

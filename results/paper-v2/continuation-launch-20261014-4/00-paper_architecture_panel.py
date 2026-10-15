"""Pinned BF16 attention-only architecture confirmation; offline and budgeted.

This additive runner never dispatches GPT-2. Qwen DeltaNet and LFM convolution
mixers remain dense. Every result is exploratory, with a separate teacher and
token manifest for each architecture; raw cross-tokenizer PPL is not compared.
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
import torch.nn.functional as F

from .compression_core import sha256_file
from .paper_benchmark import command_line, create_output, environment, tensor_digest, write
from .paper_comparisons import Geometry
from .paper_followups import dataset_content_hash
from .paper_math import factors, weighted_factors
from .paper_precision import PrecisionLinear, identity_factors
from .paper_protocol import calibration_starts, token_hash, windows


PINNED = {
    "Qwen/Qwen3.5-0.8B": {"revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "architecture": "qwen3_5_text", "blocks": 24, "indices": [3, 7, 11, 15, 19, 23],
        "suffix": "self_attn.o_proj", "shape": [1024, 2048]},
    "LiquidAI/LFM2.5-1.2B-Instruct": {"revision": "0f604ada3f766f9f257460c4c9f0b5d6f69d431b",
        "architecture": "lfm2", "blocks": 16, "indices": [2, 5, 8, 10, 12, 14],
        "suffix": "self_attn.out_proj", "shape": [2048, 2048]},
}
SEEDS = (11, 23, 37)
VALIDATION_FRONTIER = {"split": "validation", "rank_rule": "min(out,in)//divisor",
    "divisors": [2, 3, 4], "seeds": list(SEEDS), "methods": ["whitening", "anchored"],
    "select_on_test": False}


def matched_rank(shape):
    """Match Gemma's factor/dense fraction 835584/1179648 = 17/24."""
    out, inside = shape
    rank = 17 * out * inside // (24 * (out + inside))
    if not 0 < rank <= min(shape):
        raise ValueError("invalid target shape/rank")
    return rank


def panel_cases(baseline, anchor):
    """Eight confirmation rows and eighteen validation-only frontier rows."""
    primary = [("teacher", 11, {"method": "teacher"}), ("svd", 11, {"method": "svd"})]
    primary += [(label, seed, spec) for seed in SEEDS for label, spec in (("baseline", baseline), ("solver", anchor))]
    cases = [{"label": label, "seed": seed, "specification": copy.deepcopy(spec),
        "split": "test", "rank_divisor": None, "evaluation_role": "frozen confirmation"}
        for label, seed, spec in primary]
    cases += [{"label": label, "seed": seed, "specification": copy.deepcopy(spec),
        "split": "validation", "rank_divisor": divisor, "evaluation_role": "validation-only rank frontier"}
        for divisor in (2, 3, 4) for seed in SEEDS
        for label, spec in (("baseline", baseline), ("solver", anchor))]
    return cases


def case_recipes(primary_recipes, case):
    recipes = copy.deepcopy(primary_recipes)
    for recipe in recipes:
        if case["label"] == "teacher":
            recipe.update(rank=None, factor_parameters=None, representation="dense teacher")
        elif case["rank_divisor"] is not None:
            rank = min(recipe["shape"]) // case["rank_divisor"]
            if rank <= 0:
                raise ValueError("frontier divisor exceeds target dimension")
            recipe.update(rank=rank, factor_parameters=rank*sum(recipe["shape"]))
    return recipes


def record_scope(row):
    return row["label"], row["seed"], row["split"], row["rank_divisor"]


def relative_metrics(measured, reference, split):
    if measured["targets"] != reference["targets"]:
        raise ValueError("teacher-relative metrics require identical split target counts")
    return {"teacher_reference_split": split, "teacher_reference_targets": reference["targets"],
        "teacher_relative_nll_difference": measured["nll"]-reference["nll"],
        "teacher_relative_ppl": measured["ppl"]/reference["ppl"]}


def parameter_storage(recipes, whole_model_parameters, checkpoint_bytes=None):
    dense = sum(recipe["dense_parameters"] for recipe in recipes)
    teacher = all(recipe["rank"] is None for recipe in recipes)
    factors_count = dense if teacher else sum(recipe["factor_parameters"] for recipe in recipes)
    return {"rank_fraction_by_target": [1.0 if recipe["rank"] is None else
            recipe["rank"]/min(recipe["shape"]) for recipe in recipes],
        "dense_target_parameters": dense, "factor_parameters": None if teacher else factors_count,
        "target_matrix_parameter_savings_fraction": 1-factors_count/dense,
        "whole_model_parameter_savings_fraction": (dense-factors_count)/whole_model_parameters,
        "factor_checkpoint_bytes": checkpoint_bytes, "whole_model_serialized_bytes": None,
        "scope": "parameter savings and measured factor file bytes; whole-model bytes not measured"}


def validate_completed_rows(rows, contract, validation_teacher):
    cases = panel_cases(contract["baseline"], contract["selected_solver"])
    wanted = {record_scope(case): case for case in cases}
    observed = {record_scope(row): row for row in rows}
    if len(rows) != 26 or len(observed) != 26 or set(observed) != set(wanted):
        raise ValueError("completed architecture cohort requires eight test and eighteen validation scopes")
    if validation_teacher.get("split") != "validation" or validation_teacher["metrics"]["targets"] != contract["splits"]["validation"]["tokens"]-1:
        raise ValueError("validation teacher reference is incomplete or has the wrong split")
    if (contract["splits"]["validation"].get("hash") and
            validation_teacher.get("token_hash") != contract["splits"]["validation"]["hash"]):
        raise ValueError("validation teacher token provenance mismatch")
    references = {"test": observed[("teacher", 11, "test", None)]["metrics"],
        "validation": validation_teacher["metrics"]}
    for key, row in observed.items():
        case = wanted[key]
        recipes = case_recipes(contract["targets"], case)
        ranks = {recipe["rank"] for recipe in recipes}
        expected_rank = next(iter(ranks)) if len(ranks) == 1 else [recipe["rank"] for recipe in recipes]
        split = case["split"]
        if (row["specification"] != case["specification"] or row["evaluation_role"] != case["evaluation_role"] or
                row["target_recipes"] != recipes or row["rank"] != expected_rank or
                row["metrics"]["targets"] != contract["splits"][split]["tokens"]-1):
            raise ValueError("completed architecture scope/rank/target provenance mismatch")
        expected_relative = relative_metrics(row["metrics"], references[split], split)
        for field, value in expected_relative.items():
            actual = row.get(field)
            if isinstance(value, float):
                if not isinstance(actual, (int, float)) or not math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("teacher-relative metric uses the wrong split/reference")
            elif actual != value:
                raise ValueError("teacher-relative metric uses the wrong split/reference")
        if case["rank_divisor"] is not None and row.get("independent_replication") is not False:
            raise ValueError("validation frontier is not a new independent confirmation")
        expected_storage = parameter_storage(recipes, contract["whole_model_parameters"],
            row.get("metadata", {}).get("serialized_bytes"))
        if row.get("storage") != expected_storage:
            raise ValueError("rank/target/whole-model storage accounting mismatch")
        if case["label"] != "teacher":
            if row["metadata"]["targets"] != recipes or row["metadata"]["factor_parameters"] != sum(recipe["factor_parameters"] for recipe in recipes):
                raise ValueError("frontier checkpoint recipe/parameter provenance mismatch")
            if sha256_file(row["checkpoint"]) != row["metadata"]["sha256"]:
                raise ValueError("completed architecture factor checkpoint changed")
            payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            if (payload.get("targets") != recipes or payload.get("specification") != case["specification"] or
                    payload.get("storage") != "bfloat16" or payload.get("compute") != "bfloat16" or
                    set(payload.get("state", {})) != {recipe["name"] for recipe in recipes}):
                raise ValueError("checkpoint payload recipe/specification provenance mismatch")
            for recipe in recipes:
                state = payload["state"][recipe["name"]]
                out, inside = recipe["shape"]
                rank = recipe["rank"]
                if set(state) != {"left", "right", "bias"} or tuple(state["left"].shape) != (out, rank) or tuple(state["right"].shape) != (rank, inside):
                    raise ValueError("checkpoint frontier factor tensor shape mismatch")
                if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all()
                        for value in state.values() if value is not None):
                    raise ValueError("checkpoint factor precision/finite mismatch")
                if state["bias"] is not None and tuple(state["bias"].shape) != (out,):
                    raise ValueError("checkpoint factor bias shape mismatch")
    return True


def architecture_targets(model):
    kind = model.config.model_type
    if kind not in {"qwen3_5_text", "lfm2"}:
        raise ValueError("only the two declared hybrid text architectures are supported")
    suffix = "self_attn.o_proj" if kind == "qwen3_5_text" else "self_attn.out_proj"
    targets = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    blocks = list(model.model.layers)
    expected = [i for i, value in enumerate(model.config.layer_types) if value == "full_attention"]
    observed = [int(name.split(".layers.")[1].split(".")[0]) for name, _ in targets]
    if observed != expected or len(blocks) != model.config.num_hidden_layers or not targets:
        raise ValueError("attention targets do not match declared decoder block types")
    if any(not isinstance(module, torch.nn.Linear) for _, module in targets):
        raise ValueError("uncompressed attention targets must be Linear [out,in]")
    return targets


def target_recipes(model, *, enforce_pin=True):
    recipes = []
    for name, module in architecture_targets(model):
        shape = list(module.weight.shape)
        rank = matched_rank(shape)
        recipes.append({"name": name, "block_index": int(name.split(".layers.")[1].split(".")[0]),
            "shape": shape, "rank": rank, "factor_parameters": rank * sum(shape),
            "dense_parameters": math.prod(shape), "precision": "bfloat16"})
    if enforce_pin:
        expected = next(value for value in PINNED.values() if value["architecture"] == model.config.model_type)
        if ([r["block_index"] for r in recipes] != expected["indices"] or
                len(model.model.layers) != expected["blocks"] or
                any(r["shape"] != expected["shape"] for r in recipes)):
            raise ValueError("pinned architecture target recipe mismatch")
    return recipes


def validate_config(config):
    if config.get("precision") != "bfloat16" or config.get("seeds") != list(SEEDS):
        raise ValueError("panel requires declared BF16 and three paired seeds")
    if (config.get("context"), config.get("stride")) != (512, 256):
        raise ValueError("full native evaluation context/stride must remain 512/256")
    if config.get("calibration") != {"split": "train", "chunks": 16, "chunk_tokens": 512}:
        raise ValueError("fixed train calibration required")
    if config.get("selection_split") != "validation" or config.get("evaluation_split") != "test":
        raise ValueError("selection/evaluation split mismatch")
    if config.get("validation_frontier", VALIDATION_FRONTIER) != VALIDATION_FRONTIER:
        raise ValueError("validation frontier protocol must remain frozen")
    models = config.get("models", [])
    if len(models) != 2 or {r.get("model") for r in models} != set(PINNED):
        raise ValueError("panel must contain exactly the pinned Qwen and LFM models; GPT-2 excluded")
    if config.get("existing_gemma_reference") is None or config.get("max_models", 4) > 4 or len(models)+1 > config.get("max_models", 4):
        raise ValueError("model cap includes the existing Gemma reference")
    if config.get("stage_gpu_hours", {}).get("replication") != 4 or config.get("model_gpu_hours", 2) != 2:
        raise ValueError("unchanged E cap is four hours, two hours per new architecture")
    if config.get("placement", "paired_gpu") not in {"paired_gpu", "cpu"}:
        raise ValueError("placement must be explicit paired_gpu or cpu; no automatic fallback")
    for record in models:
        expected = PINNED[record["model"]]
        if record.get("model_revision") != expected["revision"] or record.get("architecture") != expected["architecture"]:
            raise ValueError("pinned model/revision/architecture mismatch")
        if Path(record["model_path"]).name != record["model_revision"]:
            raise ValueError("model snapshot path does not match pinned revision")
        if record.get("model_gpu_hours", 2) != 2:
            raise ValueError("each new model has an unchanged two-hour cap")
    return config


def prior_costs(root):
    """Count all preserved Stage-E attempts, including cancelled/failed work."""
    total, by_model = 0.0, {name: 0.0 for name in PINNED}
    for directory in Path(root).iterdir():
        if not directory.is_dir() or not directory.name.startswith(("replication-", "architecture-replication-")):
            continue
        path = directory / "status.json"
        if not path.exists():
            raise RuntimeError(f"prior Stage-E status unresolved: {directory}")
        status = json.loads(path.read_text())
        if status.get("state") in {"running", "preparing"}:
            raise RuntimeError(f"prior Stage-E run still active: {directory}")
        elapsed = status.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise RuntimeError(f"unknown prior Stage-E cost: {directory}")
        total += elapsed
        name = status.get("model")
        if name in by_model:
            by_model[name] += elapsed
    return total, by_model


def completed_model_run(config, record):
    """Recover a complete model cohort after interruption, without refitting it."""
    matches = []
    for directory in Path(config["output_root"]).glob("architecture-replication-*"):
        path = directory/"status.json"
        if not path.exists():
            continue
        status = json.loads(path.read_text())
        if status.get("model") != record["model"] or status.get("state") != "completed":
            continue
        integrity = json.loads((directory/"artifact-integrity.json").read_text())
        required = {"contract.json", "metrics.json", "teacher-audit.json", "readiness.json", "validation-teacher.json"}
        if not required <= set(integrity["artifacts"]):
            raise ValueError("completed architecture integrity manifest lacks required artifacts")
        for name, digest in integrity["artifacts"].items():
            if sha256_file(directory/name) != digest:
                raise ValueError("completed architecture artifact changed")
        contract = json.loads((directory/"contract.json").read_text())
        prior = contract["configuration"]
        expected = {**config, **record}
        for key in ("model", "model_revision", "architecture", "precision", "seeds", "context", "stride",
                    "dataset_revision", "calibration", "selection_split", "evaluation_split"):
            if prior[key] != expected[key]:
                raise ValueError(f"completed architecture confirmation mismatch: {key}")
        if (prior.get("validation_frontier", VALIDATION_FRONTIER) != expected.get("validation_frontier", VALIDATION_FRONTIER) or
                contract.get("validation_frontier") != VALIDATION_FRONTIER):
            raise ValueError("completed architecture validation frontier protocol changed")
        if (contract["selection_hash"] != sha256_file(config["selection_path"]) or
                contract["dataset_content_hash"] != dataset_content_hash(config["dataset_path"])):
            raise ValueError("completed architecture source selection/dataset changed")
        for name, digest in contract["implementation_hashes"].items():
            if sha256_file(name) != digest:
                raise ValueError("architecture implementation changed since complete cohort")
        rows = json.loads((directory/"metrics.json").read_text())
        validate_completed_rows(rows, contract, json.loads((directory/"validation-teacher.json").read_text()))
        matches.append(directory)
    if len(matches) > 1:
        raise ValueError("multiple completed model cohorts require explicit resolution")
    return matches[0] if matches else None


def frozen_specs(config):
    path = Path(config["selection_path"])
    selected = json.loads(path.read_text())
    if selected.get("state") != "frozen" or selected.get("selection_split") != "validation":
        raise ValueError("completed validation-frozen Gemma selection required")
    if json.loads((path.parent/"status.json").read_text()).get("state") != "completed":
        raise ValueError("source regularization stage must be completed")
    anchor = selected["selected_solver"]["specification"]
    baseline = selected["strongest_baseline"]["specification"]
    if anchor != {"method": "anchored", "beta": .03, "ridge_fraction": 0.0}:
        raise ValueError("frozen selected anchor differs from predeclared panel")
    if baseline.get("method") not in {"whitening", "anchored"} or baseline.get("beta", 0) != 0 or baseline.get("ridge_fraction", 0) != 0:
        raise ValueError("frozen strongest component baseline is not unanchored whitening")
    if selected.get("dataset_revision") != config["dataset_revision"] or selected.get("dataset_content_hash") != dataset_content_hash(config["dataset_path"]):
        raise ValueError("raw pinned dataset differs from frozen selection")
    return selected, {"method": "whitening", "beta": 0.0, "ridge_fraction": 0.0}, anchor


def load_model(record):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, Lfm2ForCausalLM
    path = record["model_path"]
    if record["architecture"] == "qwen3_5_text":
        # The explicit text class extracts text_config and applies the installed
        # language_model prefix mapping. Vision and MTP are excluded by the class.
        configuration = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
        model, info = Qwen3_5ForCausalLM.from_pretrained(path, config=configuration,
            local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
    elif record["architecture"] == "lfm2":
        model, info = Lfm2ForCausalLM.from_pretrained(path, local_files_only=True,
            dtype=torch.bfloat16, output_loading_info=True)
    else:
        raise ValueError("unsupported architecture")
    if any(info.get(key) for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")):
        raise ValueError(f"text checkpoint loading mismatch: {info}")
    return model.eval().requires_grad_(False), {key: list(value) for key, value in info.items()}


def load_tokens(record, dataset_path):
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(record["model_path"], local_files_only=True)
    data = load_from_disk(dataset_path)
    if set(data) != {"train", "validation", "test"}:
        raise ValueError("all three native dataset splits required")
    tokens = {split: tokenizer("\n\n".join(data[split]["text"]), add_special_tokens=False,
        return_tensors="pt")["input_ids"][0] for split in data}
    manifests = {split: {"tokens": len(value), "hash": token_hash(value)} for split, value in tokens.items()}
    if len({value["hash"] for value in manifests.values()}) != 3:
        raise ValueError("split hashes must differ")
    positions = {str(seed): calibration_starts(len(tokens["train"]), seed) for seed in SEEDS}
    manifests["calibration_positions"] = positions
    manifests["calibration_token_hashes"] = {seed: token_hash(torch.cat([
        tokens["train"][start:start+512] for start in starts])) for seed, starts in positions.items()}
    return tokens, manifests


@torch.inference_mode()
def capture_calibration(teacher, train, seed, *, device, progress):
    captured = {name: [] for name, _ in architecture_targets(teacher)}
    handles = []
    for name, module in architecture_targets(teacher):
        def hook(_module, args, name=name):
            value = args[0].detach().reshape(-1, args[0].shape[-1])
            if value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
                raise ValueError("calibration must retain finite native BF16 activations")
            captured[name].append(value.cpu())
        handles.append(module.register_forward_pre_hook(hook))
    try:
        for index, start in enumerate(calibration_starts(len(train), seed)):
            progress(f"calibration seed={seed} chunk={index}")
            teacher(input_ids=train[start:start+512][None].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(values).T.contiguous() for name, values in captured.items()}


def fit_factors(teacher, recipes, inputs, spec, directory, *, device, progress, rcond):
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    state = {}
    for recipe in recipes:
        progress(f"fit {spec} {recipe['name']}")
        module = teacher.get_submodule(recipe["name"])
        weight = module.weight.detach().to(device=device, dtype=torch.float32)
        if spec["method"] == "svd":
            left, right = factors(weight, recipe["rank"])
        else:
            left, right = weighted_factors(weight, inputs[recipe["name"]].to(device), recipe["rank"],
                beta=spec["beta"], ridge_fraction=0.0, rcond=rcond)
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError("nonfinite fitted factors")
        state[recipe["name"]] = {"left": left.to(dtype=torch.bfloat16, device="cpu"),
            "right": right.to(dtype=torch.bfloat16, device="cpu"),
            "bias": None if module.bias is None else module.bias.detach().to(dtype=torch.bfloat16, device="cpu")}
        del weight, left, right
    path = directory/"factors.pt"
    torch.save({"targets": recipes, "specification": spec, "state": state,
        "storage": "bfloat16", "compute": "bfloat16"}, path)
    if cuda:
        torch.cuda.synchronize(device)
    metadata = {"factor_parameters": sum(r["factor_parameters"] for r in recipes),
        "serialized_bytes": path.stat().st_size, "sha256": sha256_file(path),
        "factor_tensor_bytes": sum(t.numel()*t.element_size() for value in state.values() for t in value.values() if t is not None),
        "fit_seconds": time.monotonic()-started, "precision": "bfloat16",
        "factor_convention": "unbalanced", "targets": recipes,
        "peak_fitting_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
        "peak_fitting_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}
    write(directory/"metadata.json", metadata)
    return path, metadata


def make_student(teacher, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = {name for name, _ in architecture_targets(teacher)}
    if set(payload["state"]) != expected:
        raise ValueError("factor checkpoint target coverage mismatch")
    student = copy.deepcopy(teacher)
    device = next(teacher.parameters()).device
    for name, values in payload["state"].items():
        if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in values.values() if value is not None):
            raise ValueError("checkpoint precision/finite check failed")
        parent, child = name.rsplit(".", 1)
        setattr(student.get_submodule(parent), child,
            PrecisionLinear(values["left"], values["right"], values["bias"]).to(device))
    return student.eval().requires_grad_(False)


@torch.inference_mode()
def evaluate_panel(student, teacher, tokens, *, device, progress, context=512, stride=256):
    """Full shifted-target quality and drift at every decoder block."""
    residual = [Geometry() for _ in teacher.model.layers]
    local = {name: Geometry() for name, _ in architecture_targets(teacher)}
    refs, candidates, local_refs, local_candidates = {}, {}, {}, {}
    handles = []
    same = student is teacher
    for name, module in architecture_targets(teacher):
        candidate = student.get_submodule(name)
        def hook(_module, args, output, name=name, candidate=candidate):
            local_refs[name] = output.detach()
            local_candidates[name] = output.detach() if same else candidate(args[0]).detach()
        handles.append(module.register_forward_hook(hook))
    if not same:
        for model, storage in ((teacher, refs), (student, candidates)):
            for index, block in enumerate(model.model.layers):
                def hook(_module, _args, output, index=index, storage=storage):
                    storage[index] = (output[0] if isinstance(output, tuple) else output).detach()
                handles.append(block.register_forward_hook(hook))
    total = kl_total = 0.0
    count = block_count = 0
    block_sum = 0.0
    nll_blocks = []
    try:
        for index, (start, end, first) in enumerate(windows(len(tokens), context, stride)):
            progress(f"evaluation window={index} targets={count}", quiet=index % 20 != 0)
            ids = tokens[start:end][None].to(device)
            offset = first-start-1
            reference = teacher(input_ids=ids, use_cache=False).logits[0, offset:-1].float()
            actual = reference if same else student(input_ids=ids, use_cache=False).logits[0, offset:-1].float()
            losses = F.cross_entropy(actual, ids[0, offset+1:], reduction="none")
            ref_log = reference.log_softmax(-1)
            kl = (ref_log.exp()*(ref_log-actual.log_softmax(-1))).sum(-1)
            if not torch.isfinite(losses).all() or not torch.isfinite(kl).all():
                raise FloatingPointError("NaN/Inf full-test metric")
            values = losses.double().cpu().tolist()
            total += sum(values)
            kl_total += kl.double().sum().item()
            count += len(values)
            for value in values:
                block_sum += value
                block_count += 1
                if block_count == 4096:
                    nll_blocks.append({"nll_sum": block_sum, "targets": block_count})
                    block_sum, block_count = 0.0, 0
            if not same:
                for i, geometry in enumerate(residual):
                    geometry.add(refs[i][:, offset:-1], candidates[i][:, offset:-1])
            for name, geometry in local.items():
                geometry.add(local_refs[name][:, offset:-1], local_candidates[name][:, offset:-1])
            refs.clear(); candidates.clear(); local_refs.clear(); local_candidates.clear()
    finally:
        for handle in handles:
            handle.remove()
    if block_count:
        nll_blocks.append({"nll_sum": block_sum, "targets": block_count})
    if count != len(tokens)-1 or count <= 0:
        raise ValueError("full native test target count mismatch")
    drift = ([{"normalized_squared_error": 0.0, "cosine_distance": 0.0,
        "norm_ratio": 1.0, "window_linear_cka": 1.0, "tokens": count} for _ in residual]
        if same else [geometry.result() for geometry in residual])
    return {"nll_sum": total, "targets": count, "nll": total/count, "ppl": math.exp(total/count),
        "logit_kl": kl_total/count, "nll_blocks": nll_blocks,
        "local_output": {name: geometry.result() for name, geometry in local.items()},
        "residual_by_block": drift,
        "primary_drift": sum(row["normalized_squared_error"] for row in drift)/len(drift),
        "drift_scope": "all decoder blocks, including unchanged DeltaNet/convolution blocks"}


@torch.inference_mode()
def readiness_probe(teacher, tokens, tolerance, *, progress):
    """Validation-only copied, W-times-I and SVD full-rank numerical probes."""
    device = next(teacher.parameters()).device
    ids = tokens[:512][None].to(device)
    reference = teacher(input_ids=ids, use_cache=False).logits.float().cpu()
    results = {}
    for variant in ("copied", "identity_bf16", "fullrank_svd_bf16"):
        progress("readiness probe "+variant)
        student = copy.deepcopy(teacher)
        if variant != "copied":
            for name, module in architecture_targets(teacher):
                progress("readiness factor "+name)
                left, right = (identity_factors(module.weight) if variant == "identity_bf16"
                    else factors(module.weight.float(), min(module.weight.shape)))
                parent, child = name.rsplit(".", 1)
                setattr(student.get_submodule(parent), child,
                    PrecisionLinear(left, right, module.bias).to(device))
        actual = student(input_ids=ids, use_cache=False).logits.float().cpu()
        if not torch.isfinite(actual).all():
            raise FloatingPointError("nonfinite readiness logits")
        results[variant] = {"passed": torch.allclose(actual, reference, **tolerance),
            "max_logit_error": (actual-reference).abs().max().item()}
        del student, actual
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"split": "validation", "probe_tokens": ids.shape[-1], "tolerance": tolerance,
        "variants": results, "full_test_readiness": False,
        "classification": "exploratory; probe does not establish full-test full-rank equivalence"}


def run_model(config, record, allowance, prior):
    local = {**config, **record}
    local.update(tokenizer_revision=record["model_revision"], rank="shape-derived by 17/24 factor/dense fraction",
        methods=["svd", "whitening", "anchored"],
        fitting_numerics={"svd": "float32 SVD of native BF16 reference weights",
            "weighted_methods": "float64 covariance eigendecomposition and SVD; factors returned FP32 then stored BF16",
            "calibration": "native BF16 inputs without intermediate FP16 conversion"})
    local["validation_frontier"] = copy.deepcopy(config.get("validation_frontier", VALIDATION_FRONTIER))
    local.pop("gpt2_model_path", None)
    output = create_output(local, "architecture-replication")
    started = time.monotonic()
    rows = []
    write(output/"status.json", {"state": "preparing", "model": record["model"], "elapsed_seconds": 0.0})
    (output/"stderr.log").touch()
    def progress(message, quiet=False):
        elapsed = time.monotonic()-started
        if elapsed >= allowance:
            raise TimeoutError("shared Stage-E/per-model cumulative cap exhausted")
        if not quiet:
            message = f"{elapsed:.3f} {message}"
            print(message, flush=True)
            with (output/"stdout.log").open("a", encoding="utf-8") as handle:
                handle.write(message+"\n")
    def persist(row):
        rows.append(row)
        write(output/"metrics.json", rows)
    try:
        progress("preflight")
        env = environment(local)
        write(output/"preflight.json", env)
        if not env["ready"]:
            raise RuntimeError(f"offline asset/environment preflight failed: {env['missing']}")
        selection, baseline, anchor = frozen_specs(config)
        write(output/"frozen-selection.json", selection)
        teacher, loading = load_model(record)
        recipes = target_recipes(teacher)
        original = tensor_digest(teacher)
        total_parameters = sum(p.numel() for p in teacher.parameters())
        target_parameters = sum(row["dense_parameters"] for row in recipes)
        factor_parameters = sum(row["factor_parameters"] for row in recipes)
        tokens, manifests = load_tokens(record, config["dataset_path"])
        placement = config.get("placement", "paired_gpu")
        # CPU execution is explicit and is never used for GPU speed claims.
        device = torch.device(config["device"] if placement == "paired_gpu" else "cpu")
        teacher = teacher.to(device)
        contract = {"run_id": output.name, "command": command_line(), "environment": env,
            "configuration": local, "splits": manifests, "targets": recipes,
            "teacher_hash": original, "selection_hash": sha256_file(config["selection_path"]),
            "dataset_content_hash": dataset_content_hash(config["dataset_path"]),
            "loading_info": loading, "target_family": "full-attention output only",
            "implementation_hashes": {str(path.resolve()): sha256_file(path) for path in
                (Path(__file__), Path("src/paper_precision.py"), Path("src/paper_math.py"),
                 Path("src/paper_comparisons.py"), Path("src/paper_protocol.py"))},
            "excluded_mixers": "Qwen DeltaNet and LFM short-convolution outputs remain dense",
            "rank_policy": "floor((17/24)*out*in/(out+in)); per-model paired methods share exact ranks",
            "whole_model_parameters": total_parameters, "dense_target_parameters": target_parameters,
            "factor_parameters": factor_parameters,
            "whole_model_parameter_savings_fraction": (target_parameters-factor_parameters)/total_parameters,
            "parameter_savings_excludes_serialization_overhead": True,
            "placement": placement, "execution_device": str(device),
            "prior_stage_e_seconds": prior, "remaining_seconds_at_start": allowance,
            "classification": "exploratory", "baseline": baseline, "selected_solver": anchor,
            "comparison_plan": panel_cases(baseline, anchor),
            "validation_frontier": copy.deepcopy(VALIDATION_FRONTIER),
            "precision_scope": "BF16 native weights/activations and stored factors; weighted fitting FP64, SVD FP32",
            "cross_model_comparison": "teacher-relative only; tokenizer-specific PPL cannot be directly ranked",
            "expected_artifacts": ["stdout.log", "stderr.log", "config.json", "contract.json", "metrics.json", "readiness.json", "validation-teacher.json", "status.json", "report.md", "fit-*/factors.pt"]}
        write(output/"contract.json", contract)
        write(output/"status.json", {"state": "running", "model": record["model"], "elapsed_seconds": time.monotonic()-started})
        write(output/"readiness.json", readiness_probe(teacher, tokens["validation"],
            config.get("identity_tolerance", {"atol": .1, "rtol": .01}), progress=progress))
        references = {"test": evaluate_panel(teacher, teacher, tokens["test"], device=device, progress=progress)}
        cases = panel_cases(baseline, anchor)
        teacher_case = cases[0]
        persist({**teacher_case, "rank": None, "target_recipes": case_recipes(recipes, teacher_case),
            "storage": parameter_storage(case_recipes(recipes, teacher_case), total_parameters),
            "metrics": references["test"], **relative_metrics(references["test"], references["test"], "test")})
        references["validation"] = evaluate_panel(teacher, teacher, tokens["validation"], device=device, progress=progress)
        validation_teacher = {"split": "validation", "metrics": references["validation"],
            "token_hash": manifests["validation"]["hash"], "purpose": "validation frontier reference only"}
        write(output/"validation-teacher.json", validation_teacher)
        inputs_cache = {}
        for index, case in enumerate(cases[1:]):
            seed, spec, split = case["seed"], case["specification"], case["split"]
            current_recipes = case_recipes(recipes, case)
            progress(f"comparison split={split} divisor={case['rank_divisor']} seed={seed} specification={spec}")
            if spec["method"] != "svd" and seed not in inputs_cache:
                inputs_cache[seed] = capture_calibration(teacher, tokens["train"], seed, device=device, progress=progress)
            checkpoint, metadata = fit_factors(teacher, current_recipes, inputs_cache.get(seed), spec, output/f"fit-{index:02d}-seed-{seed}",
                device=device, progress=progress, rcond=config.get("pseudoinverse_rcond", 1e-7))
            student = make_student(teacher, checkpoint)
            measured = evaluate_panel(student, teacher, tokens[split], device=device, progress=progress)
            ranks = {recipe["rank"] for recipe in current_recipes}
            rank = next(iter(ranks)) if len(ranks) == 1 else [recipe["rank"] for recipe in current_recipes]
            persist({**case, "rank": rank, "target_recipes": current_recipes, "checkpoint": str(checkpoint.resolve()),
                "metadata": metadata, "metrics": measured, **relative_metrics(measured, references[split], split),
                "storage": parameter_storage(current_recipes, total_parameters, metadata["serialized_bytes"]),
                **({"independent_replication": False} if case["rank_divisor"] is not None else {})})
            del student
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if tensor_digest(teacher) != original:
            raise RuntimeError("reference teacher mutated")
        validate_completed_rows(rows, contract, validation_teacher)
        progress("all eight test and eighteen validation-frontier comparisons completed")
        write(output/"teacher-audit.json", {"unchanged": True, "tensor_sha256": original})
        if time.monotonic()-started >= allowance:
            raise TimeoutError("Stage-E cap exhausted at final boundary; retain incomplete")
        write(output/"artifact-integrity.json", {"artifacts": {name: sha256_file(output/name)
            for name in ("contract.json", "metrics.json", "teacher-audit.json", "readiness.json", "validation-teacher.json")}})
        write(output/"status.json", {"state": "completed", "model": record["model"],
            "elapsed_seconds": time.monotonic()-started, "records": len(rows), "exploratory": True})
        report = ["# Architecture confirmation", "", "Eight full native-test and eighteen validation-frontier records completed. "
            "Attention-only targets; all decoder blocks contribute drift. BF16 confirmation transfers Gemma-selected "
            "settings without retuning. Compare within-model teacher-relative quality; do not rank raw PPL across tokenizers. "
            "Validation ranks use divisors 2/3/4 and do not retune frozen test settings. Each teacher-relative value "
            "uses the corresponding split's teacher. Readiness is validation-probe-only, with no practical hardware gate, "
            "so results remain exploratory.", "",
            "| Split | Divisor | Rank | Method | Seed | NLL | PPL / same-split teacher PPL | Teacher KL | Mean block drift | Target parameter savings | Whole-model parameter savings | Factor file bytes |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        report += [f"| {row['split']} | {row['rank_divisor']} | {row['rank']} | {row['label']} | {row['seed']} | {row['metrics']['nll']:.9f} | "
            f"{row['teacher_relative_ppl']:.6f} | {row['metrics']['logit_kl']:.9g} | "
            f"{row['metrics']['primary_drift']:.9g} | {row['storage']['target_matrix_parameter_savings_fraction']:.6f} | "
            f"{row['storage']['whole_model_parameter_savings_fraction']:.6f} | {row['storage']['factor_checkpoint_bytes']} |" for row in rows]
        (output/"report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        return 0, output
    except Exception as exc:
        (output/"stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        state = "incomplete" if isinstance(exc, TimeoutError) else "failed"
        write(output/"status.json", {"state": state, "model": record["model"], "error": str(exc),
            "elapsed_seconds": time.monotonic()-started, "records": len(rows), "exploratory": True,
            "budget_limited": isinstance(exc, TimeoutError)})
        (output/"report.md").write_text(f"# Incomplete architecture confirmation\n\n{state}: {exc}\n"
            "Partial records and tracebacks retained. No complete comparison or promotion claim.\n", encoding="utf-8")
        return 1, output


def run_panel(config):
    validate_config(config)
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    output = create_output(config, "architecture-panel")
    ledger = []
    incomplete_models = []
    write(output/"status.json", {"state": "running"})
    try:
        frozen_specs(config)
        for record in config["models"]:
            completed = completed_model_run(config, record)
            if completed is not None:
                ledger.append({"model": record["model"], "run": str(completed.resolve()),
                    "exit_code": 0, "reused_complete_cohort": True, "independent_replication": False})
                write(output/"ledger.json", ledger)
                continue
            total, by_model = prior_costs(root)
            allowance = min(4*3600-total, 2*3600-by_model[record["model"]])
            if allowance <= 0:
                raise TimeoutError("shared Stage-E or per-model cumulative cap exhausted; no continuation")
            code, directory = run_model(config, record, allowance, total)
            ledger.append({"model": record["model"], "run": str(directory.resolve()), "exit_code": code,
                "remaining_seconds_at_start": allowance})
            write(output/"ledger.json", ledger)
            if code:
                status = json.loads((directory/"status.json").read_text())
                if status.get("budget_limited") is True:
                    # The amendment gives independent per-model allocations;
                    # preserve this incomplete cohort, then try the other model
                    # under its own cap and the shared E remainder. Never retry.
                    incomplete_models.append(record["model"])
                else:
                    raise RuntimeError(f"model stage failed; inspect {directory}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if incomplete_models:
            write(output/"status.json", {"state": "incomplete", "exploratory": True,
                "budget_limited_models": incomplete_models, "advance_to_practical": False})
            return 1
        write(output/"status.json", {"state": "completed", "exploratory": True})
        return 0
    except Exception as exc:
        write(output/"status.json", {"state": "incomplete", "error": str(exc), "exploratory": True})
        (output/"stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_architecture_panel_v3.json")
    args = parser.parse_args()
    return run_panel(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    raise SystemExit(main())

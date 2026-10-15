"""Offline BF16/FP8 numerical readiness controls, with a shared Stage-A cap.

FP8 here describes row-scaled storage, decoded to BF16 for computation. A
separate native FP8 GEMM probe cannot establish whole-model FP8 acceleration.
"""
from __future__ import annotations

import argparse
import copy
import gc
import inspect
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .compression_core import sha256_file
from .paper_benchmark import (command_line, create_output, environment,
                              load_config, tensor_digest, write)
from .paper_math import factors
from .paper_protocol import calibration_starts, token_hash, windows


FORMATS = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
VARIANTS = ("teacher_bf16", "copy_bf16", "identity_bf16", "svd_bf16",
            "svd_fp32_compute", "svd_fp8_e4m3", "svd_fp8_e5m2",
            "dense_fp8_e4m3", "dense_fp8_e5m2")


def row_quantize(value, dtype):
    """Finite per-row symmetric FP8 codes and positive FP32 decode scales."""
    value = value.detach().float()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError("finite matrix required")
    maximum = value.abs().amax(1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / torch.finfo(dtype).max,
                        torch.ones_like(maximum))
    # Subnormal scales can otherwise underflow to zero on tiny synthetic inputs.
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    codes = (value / scale).clamp(-torch.finfo(dtype).max,
                                 torch.finfo(dtype).max).to(dtype)
    if not torch.isfinite(codes.float()).all():
        raise FloatingPointError("nonfinite FP8 code")
    return codes, scale


def row_decode(codes, scale, dtype=torch.bfloat16):
    return (codes.float() * scale.float()).to(dtype)


def identity_factors(weight):
    if weight.ndim != 2:
        raise ValueError("matrix required")
    if weight.shape[0] <= weight.shape[1]:
        return torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device), weight.clone()
    return weight.clone(), torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device)


class PrecisionLinear(nn.Module):
    """Full-rank factors or a dense FP8 storage control; no FP16 casts."""
    def __init__(self, left, right=None, bias=None, *, storage="bf16", compute="bf16"):
        super().__init__()
        if storage not in {"bf16", "fp32", *FORMATS} or compute not in {"bf16", "fp32"}:
            raise ValueError("unknown numerical format")
        self.storage, self.compute = storage, compute
        self.factorized = right is not None
        for name, value in (("left", left), ("right", right)):
            if value is None:
                continue
            if storage in FORMATS:
                codes, scale = row_quantize(value, FORMATS[storage])
                self.register_buffer(name, codes)
                self.register_buffer(name + "_scale", scale)
            else:
                self.register_buffer(name, value.detach().clone().to(
                    torch.bfloat16 if storage == "bf16" else torch.float32))
        self.register_buffer("bias", None if bias is None else bias.detach().clone().to(
            torch.float32 if compute == "fp32" else torch.bfloat16))

    def decoded(self, name):
        dtype = torch.float32 if self.compute == "fp32" else torch.bfloat16
        value = getattr(self, name)
        if self.storage in FORMATS:
            return row_decode(value, getattr(self, name + "_scale"), dtype)
        return value.to(dtype)

    def forward(self, x):
        dtype = torch.float32 if self.compute == "fp32" else torch.bfloat16
        value = x.to(dtype)
        if self.factorized:
            value = F.linear(value, self.decoded("right"))
        value = F.linear(value, self.decoded("left"), self.bias)
        return value.to(x.dtype)


@torch.inference_mode()
def evaluate_explicit(model, tokens, *, execution_device, progress=None, context=512, stride=256):
    """Exactly-once shifted targets, including tail; supports meta offload weights."""
    total, count = 0.0, 0
    blocks, block_sum, block_count = [], 0.0, 0
    for index, (start, end, first) in enumerate(windows(len(tokens), context, stride)):
        if progress:
            progress(f"evaluation window={index} targets={count}", quiet=index % 20 != 0)
        ids = tokens[start:end].unsqueeze(0).to(execution_device)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        offset = first - start - 1
        losses = F.cross_entropy(logits[offset:-1].float(), ids[0, offset+1:], reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("NaN/Inf NLL")
        values = losses.double().cpu().tolist()
        total += sum(values)
        count += len(values)
        for value in values:
            block_sum += value
            block_count += 1
            if block_count == 4096:
                blocks.append({"nll_sum": block_sum, "targets": block_count})
                block_sum, block_count = 0.0, 0
    if block_count:
        blocks.append({"nll_sum": block_sum, "targets": block_count})
    if count != len(tokens)-1 or count <= 0:
        raise ValueError("target count mismatch")
    return {"nll_sum": total, "targets": count, "nll": total/count,
            "ppl": math.exp(total/count), "nll_blocks": blocks}


def place_model(model, offload, execution_device):
    if offload == "gpu":
        return model.to(execution_device)
    if offload != "cpu":
        raise ValueError("offload must be gpu or cpu")
    from accelerate import cpu_offload
    return cpu_offload(model, execution_device=torch.device(execution_device), offload_buffers=True)


def native_fp8_probe(device):
    """Record actual installed public scaled_mm support, never substitute results."""
    result = {"scope": "32x32 standalone GEMM, not model execution", "formats": {}}
    for name, dtype in FORMATS.items():
        entry = {"dtype": str(dtype), "api": "torch.nn.functional.scaled_mm"}
        try:
            a = torch.eye(32, device=device).to(dtype)
            b = torch.eye(32, device=device).T.to(dtype)
            scale = torch.ones((), device=device)
            signature = inspect.signature(F.scaled_mm)
            entry["signature"] = str(signature)
            if "scale_recipe_a" in signature.parameters:
                recipe = F.ScalingType.TensorWise
                actual = F.scaled_mm(a, b, scale_a=scale, scale_recipe_a=recipe,
                    scale_b=scale, scale_recipe_b=recipe, output_dtype=torch.bfloat16)
            else:
                actual = F.scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            entry.update({"supported": True, "max_identity_error":
                          (actual.float()-torch.eye(32, device=device)).abs().max().item()})
        except Exception as exc:
            entry.update({"supported": False, "error_type": type(exc).__name__, "error": str(exc)})
        result["formats"][name] = entry
    return result


def _report(output, metrics, state, reason=None):
    lines = ["# BF16/FP8 numerical readiness", "",
             "Single-seed full-rank controls; no compression efficacy or hardware-performance claim.", "",
             "FP8 factors/weights use row-scaled storage decoded to BF16 for GEMM. Native FP8 support is a separate tiny probe.", "",
             f"State: {state}." + (f" {reason}" if reason else ""), "",
             "| Variant | Test NLL | PPL | Probe max error | Probe pass | NLL pass |", "|---|---:|---:|---:|---|---|"]
    for name, record in metrics.items():
        lines.append(f"| {name} | {record['nll']:.9f} | {record['ppl']:.6f} | "
                     f"{record['max_logit_error']:.9g} | {record['identity_passed']} | {record['nll_passed']} |")
    lines += ["", "Reference: untouched BF16 dense teacher. Probe: first 512 validation tokens; atol=0.1, rtol=0.01. "
              "Test: every native WikiText-2 shifted target, context 512 / stride 256; NLL difference tolerance 0.001. "
              "A failed numerical gate is retained and does not prevent the other predeclared controls from being measured.", ""]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")


def run_precision(config, offload="gpu"):
    """Run all predeclared controls or retain an explicit budget-limited partial run."""
    from .paper_study import spent_seconds
    prior = spent_seconds(config["output_root"], "readiness")
    # Root's shared accounting must include this additive prefix before execution.
    if "precision-readiness-" not in inspect.getsource(spent_seconds):
        raise RuntimeError("shared Stage-A accounting must include precision-readiness before launch")
    allowance = config["stage_gpu_hours"]["readiness"]*3600-prior
    output = create_output(config, "precision-readiness")
    started = time.monotonic()
    metrics = {}
    (output/"stderr.log").touch()
    write(output/"status.json", {"state": "preparing", "elapsed_seconds": 0.0})
    def progress(message, quiet=False):
        elapsed = time.monotonic()-started
        if elapsed >= allowance:
            raise TimeoutError("cumulative Stage-A cap exhausted; remaining controls incomplete")
        if not quiet:
            text = f"elapsed={elapsed:.3f}s {message}"
            print(text, flush=True)
            with (output/"stdout.log").open("a", encoding="utf-8") as handle:
                handle.write(text+"\n")
    try:
        progress("initialization")
        if config["identity_tolerance"] != {"atol": 0.1, "rtol": 0.01}:
            raise ValueError("precision diagnosis requires unchanged atol=0.1, rtol=0.01")
        env = environment(config)
        env["numerics"].update({
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "gpu_compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None})
        write(output/"preflight.json", env)
        if not env["ready"]:
            raise RuntimeError("preflight failed: "+str(env["missing"]))
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(11)
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        data = load_from_disk(config["dataset_path"])
        if set(data) != {"train", "validation", "test"}:
            raise ValueError("unexpected splits")
        tokens = {split: tokenizer("\n\n".join(data[split]["text"]), add_special_tokens=False,
                  return_tensors="pt")["input_ids"][0] for split in data}
        manifests = {s: {"tokens": len(t), "sha256": token_hash(t)} for s, t in tokens.items()}
        positions = calibration_starts(len(tokens["train"]), 11)
        manifests["calibration_seed11"] = {"positions": positions,
            "token_sha256": token_hash(torch.cat([tokens["train"][s:s+512] for s in positions])),
            "used_for_fitting": False}
        if len({manifests[s]["sha256"] for s in data}) != 3:
            raise ValueError("split hashes must differ")
        write(output/"token-manifests.json", manifests)
        execution_device = torch.device(config["device"])
        if execution_device.type != "cuda":
            raise ValueError("GPU execution with optional CPU storage offload required")
        teacher = AutoModelForCausalLM.from_pretrained(config["model_path"],
                  local_files_only=True, dtype=torch.bfloat16).eval()
        targets = [(n, m) for n, m in teacher.named_modules() if n.endswith("self_attn.o_proj")]
        if not targets:
            raise ValueError("Gemma attention output projections missing")
        original_hash = tensor_digest(teacher)
        write(output/"teacher.json", {"tensor_sha256": original_hash, "dtype": "bfloat16"})
        contract = {"run_id": output.name, "output_directory": str(output.resolve()),
            "command": command_line(), "environment": env, "configuration": config,
            "seed": 11, "splits": manifests, "variants": VARIANTS, "offload": offload,
            "placement": "CPU weights, CUDA execution via accelerate.cpu_offload(offload_buffers=True)" if offload == "cpu" else "CUDA execution and active weights; pristine reference retained on CPU",
            "targets": [{"name": n, "shape": list(m.weight.shape), "rank": min(m.weight.shape),
                "factor_parameters": min(m.weight.shape)*sum(m.weight.shape)} for n,m in targets],
            "baseline": "untouched dense BF16 teacher and copied BF16 control",
            "factorization": "unbalanced float32 SVD of BF16 reference weights, once per target; decompositions stored in RAM",
            "fp8": "row-wise FP32 decode scales; finite float8 codes; dequantize to BF16 for GEMM; no native FP8 model claim",
            "prior_readiness_seconds": prior, "remaining_seconds_at_start": allowance,
            "budget_accounting": "conservative elapsed wall time including fitting, evaluation, failures and checkpoints",
            "identity_tolerance": config["identity_tolerance"], "nll_tolerance": 0.001,
            "expected_artifacts": ["stdout.log", "stderr.log", "config.json", "contract.json", "metrics.json", "report.md", "checkpoints/", "native-fp8-probe.json"]}
        write(output/"contract.json", contract)
        write(output/"native-fp8-probe.json", native_fp8_probe(execution_device))
        write(output/"status.json", {"state": "running", "elapsed_seconds": time.monotonic()-started})
        decompositions = {}
        for name, module in targets:
            progress("full-rank SVD "+name)
            w = module.weight.detach().to(device=execution_device, dtype=torch.float32)
            left, right = factors(w, min(w.shape))
            decompositions[name] = (left.cpu(), right.cpu())
            del w, left, right
        progress("SVD fitting complete")
        torch.cuda.empty_cache()
        reference = None
        checkpoint_dir = output/"checkpoints"
        checkpoint_dir.mkdir()
        for variant in VARIANTS:
            progress("control started "+variant)
            # The untouched reference remains on CPU for its final hash audit.
            student = copy.deepcopy(teacher)
            if variant not in {"teacher_bf16", "copy_bf16"}:
                for name, original in targets:
                    if variant == "identity_bf16":
                        left, right = identity_factors(original.weight)
                        packed = PrecisionLinear(left, right, original.bias)
                    elif variant.startswith("dense_fp8_"):
                        packed = PrecisionLinear(original.weight, bias=original.bias,
                                                 storage=variant.rsplit("_", 1)[1])
                    else:
                        left, right = decompositions[name]
                        storage = (variant.rsplit("_", 1)[1] if "fp8" in variant
                                   else "fp32" if variant == "svd_fp32_compute" else "bf16")
                        packed = PrecisionLinear(left, right, original.bias, storage=storage,
                            compute="fp32" if variant == "svd_fp32_compute" else "bf16")
                    parent, child = name.rsplit(".", 1)
                    setattr(student.get_submodule(parent), child, packed)
            target_state = {n: {k:v.detach().cpu() for k,v in student.get_submodule(n).state_dict().items()} for n,_ in targets}
            checkpoint = checkpoint_dir/(variant+".pt")
            torch.save({"variant": variant, "target_state": target_state}, checkpoint)
            tensor_bytes = sum(v.numel()*v.element_size() for state in target_state.values() for v in state.values())
            del target_state
            student = place_model(student, offload, execution_device)
            ids = tokens["validation"][:512].unsqueeze(0).to(execution_device)
            with torch.inference_mode():
                actual = student(input_ids=ids, use_cache=False).logits.float().cpu()
            if reference is None:
                reference = actual.clone()
            max_error = (actual-reference).abs().max().item()
            identity = torch.allclose(actual, reference, **config["identity_tolerance"])
            progress(f"probe {variant}: passed={identity} max_error={max_error}")
            measured = evaluate_explicit(student, tokens["test"], execution_device=execution_device,
                                        progress=progress, context=config["context"], stride=config["stride"])
            measured.update({"max_logit_error": max_error, "identity_passed": identity,
                "nll_passed": abs(measured["nll"]-metrics.get("teacher_bf16", measured)["nll"]) <= 0.001,
                "target_tensor_bytes": tensor_bytes, "target_checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint), "checkpoint": str(checkpoint.resolve()),
                "storage_scope": "attention output targets only; full-rank controls, not whole-model compression",
                "compute": "float32 factors with BF16 output" if variant == "svd_fp32_compute" else "bfloat16",
                "fp8_storage": "fp8" in variant})
            metrics[variant] = measured
            write(output/"metrics.json", metrics)
            _report(output, metrics, "running")
            del actual, student
            gc.collect()
            torch.cuda.empty_cache()
            progress("control completed "+variant)
        if tensor_digest(teacher) != original_hash:
            raise RuntimeError("untouched BF16 teacher mutated")
        progress("all predeclared controls completed")
        write(output/"teacher-audit.json", {"unchanged": True, "tensor_sha256": original_hash})
        write(output/"status.json", {"state": "completed", "elapsed_seconds": time.monotonic()-started,
              "all_numerical_gates_passed": all(m["identity_passed"] and m["nll_passed"] for m in metrics.values())})
        _report(output, metrics, "completed")
        return 0
    except Exception as exc:
        elapsed = time.monotonic()-started
        (output/"stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output/"status.json", {"state": "failed", "elapsed_seconds": elapsed,
            "error": str(exc), "completed_variants": list(metrics),
            "missing_variants": [v for v in VARIANTS if v not in metrics], "budget_limited": isinstance(exc, TimeoutError)})
        _report(output, metrics, "incomplete", str(exc))
        print(f"Precision diagnosis stopped; evidence retained in {output}: {exc}", flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_protocol_v2.json")
    parser.add_argument("--offload", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    return run_precision(load_config(args.config), args.offload)


if __name__ == "__main__":
    raise SystemExit(main())

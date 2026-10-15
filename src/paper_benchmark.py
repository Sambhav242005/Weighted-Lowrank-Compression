"""Additive offline protocol-v2 preflight and Stage-A identity controls.

Later stages are intentionally not dispatched before readiness is established.
No model or dataset download is performed by this module.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import importlib.metadata
import json
import platform
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from .compression_core import PackedLowRankLinear, sha256_file
from .paper_math import factors
from .paper_protocol import calibration_starts, evaluate, token_hash


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def command_line():
    return subprocess.list2cmdline(["rtk", *sys.orig_argv])


def physical_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in ["total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended"]]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total_bytes": status.total, "available_bytes": status.available}
    return None


def dependency_errors():
    from packaging.requirements import Requirement
    errors = []
    for owner in ("transformers", "datasets"):
        for raw in importlib.metadata.requires(owner) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                errors.append(f"{owner}: missing {requirement.name}")
                continue
            if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                errors.append(f"{owner} requires {requirement.name}{requirement.specifier}; found {version}")
    return errors


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["protocol"] != "paper-v2" or config["seeds"] != [11, 23, 37]:
        raise ValueError("protocol/seed mismatch")
    if (config["context"], config["stride"]) != (512, 256):
        raise ValueError("context/stride mismatch")
    if config["calibration"] != {"split": "train", "chunks": 16, "chunk_tokens": 512}:
        raise ValueError("calibration mismatch")
    if config["selection_split"] != "validation" or config["evaluation_split"] != "test":
        raise ValueError("split mismatch")
    for key in ["model_path", "dataset_path", "output_root"]:
        config[key] = str(Path(config[key]).expanduser().resolve())
    if Path(config["model_path"]).name != config["model_revision"]:
        raise ValueError("model path/revision mismatch")
    return config


def environment(config):
    missing = []
    incompatibilities = dependency_errors()
    missing.extend(incompatibilities)
    model = Path(config["model_path"])
    if not (model / "config.json").is_file():
        missing.append("pinned model config")
    if not list(model.glob("*.safetensors")):
        missing.append("pinned model weights")
    if not (Path(config["dataset_path"]) / "dataset_dict.json").is_file():
        missing.append("pinned train/validation/test dataset")
    asset_manifest = Path("paper-assets/asset-revisions.json")
    if not asset_manifest.exists() or json.loads(asset_manifest.read_text()).get("wikitext") != config["dataset_revision"]:
        missing.append("dataset revision provenance")
    if not torch.cuda.is_available():
        missing.append("CUDA device")
    sha = subprocess.check_output(["rtk", "git", "rev-parse", "HEAD"], text=True).strip()
    return {"ready": not missing, "missing": missing, "git_revision": sha,
            "dependency_errors": incompatibilities,
            "command": command_line(), "python_executable": sys.executable, "ram": physical_memory(),
            "python": platform.python_version(), "packages": {
                p: importlib.metadata.version(p) for p in ["torch", "transformers", "datasets", "huggingface-hub", "pytest"]},
            "package_inventory": {d.metadata.get("Name", "unknown"): d.version for d in importlib.metadata.distributions()},
            "numerics": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                         "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                         "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                         "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "disk_free_bytes": shutil.disk_usage(config["output_root"] if Path(config["output_root"]).exists() else ".").free,
            "source_hashes": {str(p): sha256_file(p) for p in Path("src").glob("*.py")}}


def create_output(config, stage):
    name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = Path(config["output_root"]) / f"{stage}-{name}"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    return output


def preflight(config):
    output = create_output(config, "preflight")
    env = environment(config)
    write(output / "preflight.json", env)
    write(output / "status.json", {"state": "completed" if env["ready"] else "waiting for a human"})
    print(json.dumps({"output": str(output), **env}, indent=2))
    return 0 if env["ready"] else 2


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def readiness(config):
    output = create_output(config, "readiness")
    started = time.monotonic()

    def progress(message):
        if time.monotonic() - started > config["stage_gpu_hours"]["readiness"] * 3600:
            raise TimeoutError("Stage-A time cap exceeded; comparison incomplete")
        text = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(text, flush=True)
        with (output / "stdout.log").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    try:
        env = environment(config)
        write(output / "preflight.json", env)
        if not env["ready"]:
            write(output / "status.json", {"state": "waiting for a human", "missing": env["missing"]})
            return 2
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        data = load_from_disk(config["dataset_path"])
        if set(data) != {"train", "validation", "test"}:
            raise ValueError("unexpected splits")
        tokens = {s: tokenizer("\n\n".join(data[s]["text"]), add_special_tokens=False,
                               return_tensors="pt")["input_ids"][0] for s in data}
        manifests = {s: {"tokens": len(t), "hash": token_hash(t)} for s, t in tokens.items()}
        if len({m["hash"] for m in manifests.values()}) != 3:
            raise ValueError("split hashes are not distinct")
        manifests["calibration_positions"] = {
            str(seed): calibration_starts(len(tokens["train"]), seed) for seed in config["seeds"]}
        write(output / "token-manifests.json", manifests)
        teacher = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
            torch_dtype=torch.float16).to(config["device"]).eval()
        targets = [(name, module) for name, module in teacher.named_modules() if name.endswith("self_attn.o_proj")]
        if not targets:
            raise ValueError("expected Gemma attention output projections missing")
        contract = {"run_id": output.name, "output_directory": str(output), "command": command_line(),
                    "environment": env, "configuration": config, "splits": manifests,
                    "targets": [{"name": n, "shape": list(m.weight.shape),
                                 "full_rank_parameters": min(m.weight.shape)*sum(m.weight.shape),
                                 "compressed_parameters": config["rank"]*sum(m.weight.shape)} for n, m in targets],
                    "baseline": "untouched dense teacher; identity controls use full rank",
                    "expected_artifacts": ["stdout.log", "stderr.log", "config.json", "metrics.json", "report.md", "contract.json"]}
        write(output / "contract.json", contract)
        original_hash = tensor_digest(teacher)
        write(output / "status.json", {"state": "running"})
        metrics = {}
        reference = None
        ids = tokens["validation"][:512].unsqueeze(0).to(config["device"])
        with torch.inference_mode():
            reference = teacher(input_ids=ids, use_cache=False).logits.float()
        for method in ["teacher", "copied_teacher", "full_rank_factorized"]:
            progress(f"identity control started: {method}")
            student = teacher if method == "teacher" else copy.deepcopy(teacher)
            if method == "full_rank_factorized":
                for name, module in targets:
                    basis = config.get("readiness_factorization", "svd")
                    if basis == "identity":
                        w = module.weight.detach()
                        if w.shape[0] <= w.shape[1]:
                            u = torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
                            v = w
                        else:
                            u = w
                            v = torch.eye(w.shape[1], device=w.device, dtype=w.dtype)
                    elif basis == "svd":
                        u, v = factors(module.weight.detach().float(), min(module.weight.shape))
                    else:
                        raise ValueError("unknown readiness factorization")
                    parent_name, child = name.rsplit(".", 1)
                    packed = PackedLowRankLinear.from_factors(u, v, module.bias, bits=16)
                    setattr(student.get_submodule(parent_name), child, packed.to(config["device"]))
            with torch.inference_mode():
                actual = student(input_ids=ids, use_cache=False).logits.float()
            tolerance = config["identity_tolerance"]
            max_error = (actual-reference).abs().max().item()
            identity = torch.allclose(actual, reference, **tolerance)
            progress(f"identity={identity} max_logit_error={max_error}")
            if not identity:
                raise RuntimeError(f"{method} identity control failed")
            result = evaluate(student, tokens["test"], progress=progress)
            result.update({"max_logit_error": max_error, "identity_passed": identity})
            metrics[method] = result
            write(output / "metrics.json", metrics)
            if method != "teacher":
                del student
                torch.cuda.empty_cache()
        for method, result in metrics.items():
            if abs(result["nll"] - metrics["teacher"]["nll"]) > 0.001:
                raise RuntimeError(f"{method} full-test NLL identity failed")
        if tensor_digest(teacher) != original_hash:
            raise RuntimeError("teacher was mutated")
        write(output / "status.json", {"state": "completed", "elapsed_seconds": time.monotonic()-started})
        (output / "report.md").write_text("# Stage A readiness\n\nIdentity controls passed; no compression efficacy claim.\n"
            "Stages B–F remain pending.\n", encoding="utf-8")
        progress("readiness complete")
        return 0
    except Exception as exc:
        (output / "stderr.log").write_text(traceback.format_exc(), encoding="utf-8")
        write(output / "status.json", {"state": "failed", "error": str(exc)})
        print(f"Stage A failed; evidence retained in {output}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["preflight", "run", "report"])
    parser.add_argument("--config", default="configs/paper_protocol_v2.json")
    parser.add_argument("--stage", choices=["readiness", "regularization", "baselines", "composition", "replication", "practical", "local-core"], default="readiness")
    parser.add_argument("--study", default="local-core")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.operation == "preflight":
        return preflight(config)
    if args.operation == "run":
        if args.stage == "readiness":
            return readiness(config)
        if args.stage == "local-core":
            from .paper_study import run_study
            return run_study(config)
        if args.stage in {"regularization", "baselines"}:
            from .paper_comparisons import run_comparisons
            return run_comparisons(config, args.stage)
        from .paper_followups import run_followups
        return run_followups(config, args.stage)
    if args.study != config["study"]:
        raise ValueError("study mismatch")
    from .paper_reporting import report_study
    return report_study(config, args.study)


if __name__ == "__main__":
    raise SystemExit(main())

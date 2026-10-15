"""Bounded diagnosis of full-rank factor rounding under the frozen protocol.

This records failures without weakening the declared readiness tolerance.
"""
import copy
import json
import time
import traceback
from pathlib import Path

import torch
from torch import nn

from .paper_benchmark import create_output, environment, load_config, tensor_digest, write
from .paper_protocol import evaluate, token_hash
from .compression_core import PackedLowRankLinear


class FloatFactors(nn.Module):
    def __init__(self, u, v, bias):
        super().__init__()
        self.register_buffer("u", u.detach().float().clone())
        self.register_buffer("v", v.detach().float().clone())
        self.register_buffer("bias", None if bias is None else bias.detach().float().clone())

    def forward(self, x):
        result = (x.float() @ self.v.T) @ self.u.T
        if self.bias is not None:
            result = result + self.bias
        return result.to(x.dtype)


def diagnose(config):
    output = create_output(config, "svd-diagnosis")
    started = time.monotonic()
    metrics = {}

    def progress(message):
        # All earlier Stage-A runs together took well below 20 minutes; retain
        # 40 minutes here inside the originally declared one-hour cap.
        if time.monotonic() - started > 2400:
            raise TimeoutError("SVD diagnosis exceeded remaining Stage-A allocation")
        text = f"{time.monotonic()-started:.1f}s {message}"
        print(text, flush=True)
        with (output / "stdout.log").open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    try:
        from datasets import load_from_disk
        from transformers import AutoModelForCausalLM, AutoTokenizer
        env = environment(config)
        if not env["ready"]:
            raise RuntimeError(f"missing assets: {env['missing']}")
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        dataset = load_from_disk(config["dataset_path"])
        tokens = {s: tokenizer("\n\n".join(dataset[s]["text"]), add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0] for s in ["train", "validation", "test"]}
        teacher = AutoModelForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
            dtype=torch.float16).to(config["device"]).eval()
        targets = [(n, m) for n, m in teacher.named_modules() if n.endswith("self_attn.o_proj")]
        teacher_hash = tensor_digest(teacher)
        contract = {"configuration": config, "environment": env,
                    "command": "rtk .venv-cuda/Scripts/python.exe -u -m src.paper_svd_diagnosis",
                    "model_revision": config["model_revision"], "dataset_revision": config["dataset_revision"],
                    "split_hashes": {s: token_hash(t) for s,t in tokens.items()},
                    "teacher_hash": teacher_hash, "seed": 11, "rank": "full rank",
                    "baseline": "untouched FP16 teacher",
                    "targets": [{"name": n, "shape": list(m.weight.shape)} for n,m in targets],
                    "decision": "preserve original logit atol/rtol and NLL identity thresholds",
                    "diagnostic_variants": ["dense_svd_fp32", "dense_svd_fp16", "unbalanced_fp16", "balanced_fp16", "float32_execution"]}
        write(output / "contract.json", contract)
        write(output / "status.json", {"state": "running"})
        progress("model loaded; decompose each full-rank target once")
        decompositions = {}
        for name, module in targets:
            u,s,v = torch.linalg.svd(module.weight.float(), full_matrices=False)
            decompositions[name] = (u.cpu(),s.cpu(),v.cpu())
        probe = tokens["validation"][:512].unsqueeze(0).to(config["device"])
        with torch.inference_mode():
            reference = teacher(input_ids=probe,use_cache=False).logits.float()
        metrics["teacher"] = evaluate(teacher,tokens["test"],progress=progress)
        write(output / "metrics.json", metrics)
        for variant in contract["diagnostic_variants"]:
            progress(f"variant started {variant}")
            student = copy.deepcopy(teacher)
            errors = []
            for name,module in targets:
                u,s,v = [t.to(config["device"]) for t in decompositions[name]]
                left,right = u*s,v
                if variant == "balanced_fp16":
                    left,right = u*s.sqrt(), s.sqrt()[:,None]*v
                parent,child = name.rsplit(".",1)
                if variant.startswith("dense_"):
                    replacement = student.get_submodule(name)
                    reconstructed = left @ right if variant == "dense_svd_fp32" else left.half().float() @ right.half().float()
                    replacement.weight.data.copy_(reconstructed.to(replacement.weight.dtype))
                elif variant == "float32_execution":
                    replacement = FloatFactors(left,right,module.bias).to(config["device"])
                    setattr(student.get_submodule(parent),child,replacement)
                    reconstructed = left @ right
                else:
                    replacement = PackedLowRankLinear.from_factors(left,right,module.bias,bits=16).to(config["device"])
                    setattr(student.get_submodule(parent),child,replacement)
                    reconstructed = replacement.u.float() @ replacement.v.float()
                weight=module.weight.float()
                error=(reconstructed-weight).square().sum()/weight.square().sum()
                errors.append({"name":name,"relative_squared_weight_error":error.item()})
            with torch.inference_mode():
                actual = student(input_ids=probe,use_cache=False).logits.float()
            identity = torch.allclose(actual,reference,**config["identity_tolerance"])
            max_error = (actual-reference).abs().max().item()
            progress(f"probe identity={identity} max_error={max_error}")
            score=evaluate(student,tokens["test"],teacher=teacher,progress=progress)
            score.update({"identity_probe_passed":identity,"max_logit_error":max_error,
                          "relative_weight_errors":errors,
                          "nll_identity_passed":abs(score["nll"]-metrics["teacher"]["nll"])<=0.001})
            metrics[variant]=score
            write(output / "metrics.json",metrics)
            progress(f"variant finished {variant} nll={score['nll']} kl={score['logit_kl']}")
            del student,replacement,actual
            torch.cuda.empty_cache()
        if tensor_digest(teacher)!=teacher_hash:
            raise RuntimeError("teacher mutated")
        passed=[n for n,v in metrics.items() if n!='teacher' and v['identity_probe_passed'] and v['nll_identity_passed']]
        write(output / "status.json",{"state":"completed","elapsed_seconds":time.monotonic()-started,"passed_variants":passed})
        lines=["# Full-rank SVD rounding diagnosis", "", f"Passing declared identity gates: {passed}.", "", "| Variant | NLL | PPL | KL | Max probe logit error | Probe gate | NLL gate |", "|---|---:|---:|---:|---:|---|---|"]
        for n,v in metrics.items():
            if n=='teacher': continue
            lines.append(f"| {n} | {v['nll']:.9f} | {v['ppl']:.6f} | {v['logit_kl']:.9g} | {v['max_logit_error']:.6f} | {v['identity_probe_passed']} | {v['nll_identity_passed']} |")
        lines.extend(["", "This diagnoses full-rank numerical behavior; it is not compression evidence."])
        (output / "report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        progress("diagnosis complete")
        return 0
    except Exception as exc:
        (output / "stderr.log").write_text(traceback.format_exc(),encoding="utf-8")
        write(output / "status.json",{"state":"failed","error":str(exc),"elapsed_seconds":time.monotonic()-started})
        raise


if __name__ == "__main__":
    diagnose(load_config("configs/paper_protocol_v2.json"))

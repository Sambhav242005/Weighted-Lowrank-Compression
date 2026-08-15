"""Evaluate 90% unstructured pruning on cached GPT-2 Small."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer


MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SPARSITY = 0.90
CHUNK = 512


def stream_tokens(dataset: Dataset, tokenizer: GPT2Tokenizer) -> torch.Tensor:
    text = "\n".join(row["text"] for row in dataset if row["text"])
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)


def perplexity(model, tokens: torch.Tensor, device: str) -> float:
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.numel() - 1, CHUNK):
            batch = tokens[start : start + CHUNK].to(device)
            if batch.numel() < 2:
                continue
            # Transformers performs the causal one-token shift internally.
            loss = model(batch[None, :], labels=batch[None, :]).loss
            count = batch.numel() - 1
            total_nll += float(loss.item()) * count
            total_tokens += count
    return math.exp(total_nll / total_tokens)


def block_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if name.startswith("transformer.h.") and name.endswith(("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"))
    }


def collect_input_rms(model, tokens: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []
    for name, module in block_modules(model).items():
        def hook(_module, args, key=name):
            x = args[0].detach().float()
            value = x.square().sum(dim=tuple(range(x.ndim - 1))).cpu()
            sums[key] = sums.get(key, torch.zeros_like(value)) + value
            counts[key] = counts.get(key, 0) + x.numel() // x.shape[-1]
        handles.append(module.register_forward_pre_hook(hook))
    model.eval()
    with torch.no_grad():
        for start in range(0, min(tokens.numel() - 1, 16 * CHUNK), CHUNK):
            batch = tokens[start : start + CHUNK].to(device)
            if batch.numel() > 1:
                model(batch[None, :])
    for handle in handles:
        handle.remove()
    return {name: (sums[name] / counts[name]).sqrt() for name in sums}


def apply_pruning(model, method: str, input_rms: dict[str, torch.Tensor] | None) -> dict:
    total = kept = 0
    for name, module in block_modules(model).items():
        weight = module.weight.data
        score = weight.abs()
        if method == "activation_aware":
            # GPT-2 Conv1D stores [in_features, out_features].
            score = score * input_rms[name].to(weight.device)[:, None]
        keep_count = max(1, int(round((1.0 - SPARSITY) * score.numel())))
        threshold = torch.topk(score.reshape(-1), keep_count, sorted=False).values.min()
        mask = score >= threshold
        module.weight.data.mul_(mask)
        total += weight.numel()
        kept += int(mask.sum().item())
    return {"target_parameters": total, "nonzero_parameters": kept, "sparsity": 1.0 - kept / total}


def sparse_bytes(model) -> dict:
    total_dense = total_sparse = 0
    for module in block_modules(model).values():
        weight = module.weight.data
        nonzero = int(torch.count_nonzero(weight).item())
        total_dense += weight.numel() * 2
        # fp16 value + uint16 column index per nonzero, plus uint32 row pointers.
        rows = weight.shape[0]
        total_sparse += nonzero * 4 + (rows + 1) * 4
    return {"dense_fp16_bytes": total_dense, "csr_fp16_uint16_bytes": total_sparse, "sparse_storage_ratio": total_dense / total_sparse}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--train-arrow", type=Path, required=True)
    parser.add_argument("--test-arrow", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = GPT2Tokenizer.from_pretrained(str(args.snapshot), local_files_only=True)
    train = stream_tokens(Dataset.from_file(str(args.train_arrow)), tokenizer)
    test = stream_tokens(Dataset.from_file(str(args.test_arrow)), tokenizer)
    result = {
        "run_id": args.out.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "device": args.device,
        "sparsity_target": SPARSITY,
        "calibration_tokens": min(train.numel(), 16 * CHUNK),
        "test_tokens": int(test.numel()),
    }
    baseline = GPT2LMHeadModel.from_pretrained(str(args.snapshot), local_files_only=True).to(args.device)
    result["baseline_ppl"] = perplexity(baseline, test, args.device)
    result["baseline_bytes"] = sparse_bytes(baseline)
    del baseline
    torch.cuda.empty_cache()
    for method in ("magnitude", "activation_aware"):
        model = GPT2LMHeadModel.from_pretrained(str(args.snapshot), local_files_only=True).to(args.device)
        input_rms = collect_input_rms(model, train, args.device) if method == "activation_aware" else None
        pruning = apply_pruning(model, method, input_rms)
        result[method] = {"pruning": pruning, "ppl": perplexity(model, test, args.device), "bytes": sparse_bytes(model)}
        del model
        torch.cuda.empty_cache()
        (args.out / "progress.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

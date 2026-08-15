"""Pinned first Track-A Gemma run: weighted low-rank versus matched SVD.

This runner is intentionally standalone and writes a timestamped result folder.
It uses only already-cached local model and Arrow dataset snapshots.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compression_core import fit_activation_weighted_low_rank


MODEL_DIR = Path(
    r"C:\Users\NPC\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
DATA_DIR = Path(
    r"C:\Users\NPC\.cache\huggingface\datasets\wikitext\wikitext-2-raw-v1\0.0.0\b08601e04326c79dfdd32d625aee71d232d685c3"
)
SEEDS = (11, 23, 37)
BETA = 0.1
RANK_DIVISOR = 3
CHUNK = 512
DEVICE = torch.device("cuda")


def stream_tokens(tokenizer, dataset: Dataset) -> torch.Tensor:
    text = "\n\n".join(row["text"] for row in dataset)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]


def calibration_chunks(tokens: torch.Tensor, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randperm(max(1, tokens.numel() // CHUNK), generator=generator)[:16] * CHUNK
    chunks = [tokens[int(start) : int(start) + CHUNK] for start in starts]
    return [chunk for chunk in chunks if chunk.numel() == CHUNK]


def capture(module, model, chunks):
    inputs, outputs = [], []

    def hook(_module, args, output):
        inputs.append(args[0].detach().reshape(-1, args[0].shape[-1]).T.float().cpu())
        outputs.append(output.detach().reshape(-1, output.shape[-1]).T.float().cpu())

    handle = module.register_forward_hook(hook)
    with torch.inference_mode():
        for chunk in chunks:
            model(input_ids=chunk.unsqueeze(0).to(DEVICE))
    handle.remove()
    return torch.cat(inputs, dim=1), torch.cat(outputs, dim=1)


def perplexity(model, tokenizer, tokens: torch.Tensor) -> float:
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, tokens.numel() - CHUNK, CHUNK):
            ids = tokens[start : start + CHUNK].unsqueeze(0).to(DEVICE)
            loss = model(input_ids=ids, labels=ids).loss
            count = ids.shape[1] - 1
            total_loss += float(loss) * count
            total_tokens += count
    return float(np.exp(total_loss / total_tokens))


def fit_svd(weight: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vt = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def replace_all(model, teacher, chunks, method: str, rank: int):
    for teacher_block, student_block in zip(teacher.model.layers, model.model.layers):
        teacher_proj = teacher_block.self_attn.o_proj
        student_proj = student_block.self_attn.o_proj
        weight = teacher_proj.weight.detach().float().cpu()
        if method == "svd":
            fitted = fit_svd(weight, rank)
        else:
            x, y = capture(teacher_proj, teacher, chunks)
            bias = teacher_proj.bias.detach().float().cpu() if teacher_proj.bias is not None else None
            fitted = fit_activation_weighted_low_rank(
                weight, x, y, rank, beta=BETA, bias=bias, ridge_fraction=0.01
            )
        student_proj.weight.data.copy_(fitted.to(student_proj.weight.dtype).to(DEVICE))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; use .venv-cuda")
    run_id = datetime.now(timezone.utc).strftime("track-a-gemma-%Y%m%d-%H%M%S")
    output_dir = Path("results") / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    def progress(message: str):
        print(message, flush=True)
        with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    progress(f"run_started {run_id}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    train = Dataset.from_file(str(DATA_DIR / "wikitext-train.arrow"))
    test = Dataset.from_file(str(DATA_DIR / "wikitext-test.arrow"))
    train_tokens = stream_tokens(tokenizer, train)
    test_tokens = stream_tokens(tokenizer, test)
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to(DEVICE).eval()
    rank = teacher.config.hidden_size // RANK_DIVISOR
    progress(f"model_loaded rank={rank} train_tokens={train_tokens.numel()} test_tokens={test_tokens.numel()}")
    baseline = perplexity(teacher, tokenizer, test_tokens)
    progress(f"baseline_complete ppl={baseline}")
    metrics = {"run_id": run_id, "baseline_ppl": baseline, "rank": rank, "beta": BETA, "seeds": {}}
    (output_dir / "config.json").write_text(json.dumps({
        "model_snapshot": str(MODEL_DIR), "dataset_snapshot": str(DATA_DIR),
        "seeds": SEEDS, "rank": rank, "beta": BETA, "chunk": CHUNK,
        "protocol": "full contiguous WikiText-2 test stream",
    }, indent=2))
    for seed in SEEDS:
        chunks = calibration_chunks(train_tokens, seed)
        metrics["seeds"][str(seed)] = {}
        for method in ("svd", "weighted"):
            started = time.time()
            progress(f"method_started seed={seed} method={method}")
            student = copy.deepcopy(teacher)
            replace_all(student, teacher, chunks, method, rank)
            score = perplexity(student, tokenizer, test_tokens)
            metrics["seeds"][str(seed)][method] = {
                "ppl": score,
                "delta_pct": (score - baseline) / baseline * 100,
                "elapsed_sec": time.time() - started,
                "calibration_chunks": len(chunks),
            }
            del student
            torch.cuda.empty_cache()
            progress(f"method_complete seed={seed} method={method} ppl={score}")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    progress("run_complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

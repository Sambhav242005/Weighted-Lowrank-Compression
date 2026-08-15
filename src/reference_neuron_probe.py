"""Probe whether rows of weight matrices can be encoded from reference rows.

This is an exploratory structural test, not a model compressor.  It reports
the best single reference row, normalized residual energy, and an int4
residual byte estimate with all side information included.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def reference_fit(weight: torch.Tensor, k: int, *, ridge: float = 1e-8) -> dict:
    """Fit W ~= A @ R + b and choose k rows greedily."""
    w = weight.detach().float().cpu()
    if w.ndim != 2:
        raise ValueError("weight must be a matrix")
    n, d = w.shape
    centered = w - w.mean(dim=1, keepdim=True)
    norms = centered.square().sum(dim=1)
    k = min(k, n)
    selected: list[int] = []
    residual_for_selection = centered.clone()
    for _ in range(k):
        index = int(residual_for_selection.square().sum(dim=1).argmax().item())
        selected.append(index)
        basis = centered[selected].T
        coefficients = torch.linalg.lstsq(basis, centered.T).solution
        residual_for_selection = (centered.T - basis @ coefficients).T
        residual_for_selection[selected] = 0
    references = w[selected]
    # Include an affine intercept so the representation is not penalized by
    # row-wise offsets unrelated to the reference directions.
    design = torch.cat((references.T, torch.ones(d, 1)), dim=1)
    gram = design.T @ design + ridge * torch.eye(k + 1)
    coefficients = w @ design @ torch.linalg.inv(gram)
    prediction = coefficients @ design.T
    residual = w - prediction
    residual_ratio = float(residual.square().sum().item() / w.square().sum().clamp_min(1e-12).item())
    singular_values = torch.linalg.svdvals(w)
    rank_k_energy = float(singular_values[k:].square().sum().item() / w.square().sum().clamp_min(1e-12).item())
    corr = (centered @ centered[selected].T) / torch.sqrt(
        norms[:, None] * norms[selected][None, :]
    ).clamp_min(1e-12)
    baseline_bytes = n * d * 2
    # Reference fp16 + two fp16 coefficients per row + signed int4 residual
    # plus one fp16 scale per 64-value group.
    group = 64
    groups = n * math.ceil(d / group)
    packed_bytes = k * d * 2 + n * (k + 1) * 2 + math.ceil(n * d / 2) + groups * 2
    return {
        "reference_count": k,
        "rows": n,
        "columns": d,
        "reference_rows": selected,
        "mean_abs_corr_to_references": float(corr.abs().mean().item()),
        "best_residual_energy_ratio": residual_ratio,
        "matched_rank_k_svd_residual_energy_ratio": rank_k_energy,
        "fp16_baseline_bytes": baseline_bytes,
        "reference_plus_fp16_coeff_plus_int4_residual_bytes": packed_bytes,
        "estimated_storage_ratio": baseline_bytes / packed_bytes,
        "residual_mean_abs": float(residual.abs().mean().item()),
        "residual_std": float(residual.std().item()),
    }


def random_probe(seed: int, rows: int, columns: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    cases = {
        "iid_gaussian": torch.randn(rows, columns, generator=generator),
        "shared_reference": None,
        "global_rank8": None,
    }
    base = torch.randn(columns, generator=generator)
    cases["shared_reference"] = torch.randn(rows, 1, generator=generator) * 0.2 * base + base + torch.randn(rows, columns, generator=generator) * 0.02
    cases["global_rank8"] = torch.randn(rows, 8, generator=generator) @ torch.randn(8, columns, generator=generator) + torch.randn(rows, columns, generator=generator) * 0.05
    return {name: {str(k): reference_fit(matrix, k) for k in (1, 4, 8, 16)} for name, matrix in cases.items()}


def model_probe(snapshot: Path, device: str) -> dict:
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(str(snapshot), local_files_only=True).to(device).eval()
    rows = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and ".h." in name:
                rows[name] = {str(k): reference_fit(parameter, k) for k in (1, 4, 8, 16)}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("random", "gpt2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {
        "run_id": args.out.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "mode": args.mode,
        "seed": args.seed,
    }
    if args.mode == "random":
        result["matrices"] = random_probe(args.seed, args.rows, args.columns)
    else:
        if args.snapshot is None:
            raise SystemExit("--snapshot is required for --mode gpt2")
        result["model_snapshot"] = str(args.snapshot)
        result["device"] = args.device
        result["matrices"] = model_probe(args.snapshot, args.device)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

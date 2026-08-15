"""Shared, serializable compression primitives.

This module is deliberately independent from the legacy experiment scripts.  A
packed layer stores factors and quantization metadata only; it never registers a
materialized dense replacement weight.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class RunContract:
    run_id: str
    model: str
    model_revision: str
    tokenizer_revision: str
    dataset: str
    dataset_revision: str
    split_hashes: Mapping[str, str]
    seeds: tuple[int, ...]
    git_sha: str | None
    command: str
    package_versions: Mapping[str, str]
    device: str
    budget_bytes: int | None = None
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        model: str,
        model_revision: str,
        tokenizer_revision: str,
        dataset: str,
        dataset_revision: str,
        split_hashes: Mapping[str, str],
        seeds: tuple[int, ...] | list[int],
        command: str,
        device: str | None = None,
        budget_bytes: int | None = None,
        package_versions: Mapping[str, str] | None = None,
    ) -> "RunContract":
        versions = dict(package_versions or {})
        versions.setdefault("python", platform.python_version())
        versions.setdefault("torch", torch.__version__)
        return cls(
            run_id=run_id,
            model=model,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            dataset=dataset,
            dataset_revision=dataset_revision,
            split_hashes=dict(split_hashes),
            seeds=tuple(seeds),
            git_sha=_git_sha(),
            command=command,
            package_versions=versions,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            budget_bytes=budget_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ModuleRecipe:
    module: str
    representation: str
    rank: int | None = None
    bits: int | None = None
    group_size: int | None = None
    residual_bits: int | None = None
    metadata_bytes: int = 0


@dataclass(frozen=True)
class CompressionRecipe:
    name: str
    modules: tuple[ModuleRecipe, ...]
    total_serialized_byte_budget: int | None = None

    def serialized_bytes(self, artifacts: Mapping[str, int]) -> int:
        total = sum(int(artifacts[name]) for name in artifacts)
        if self.total_serialized_byte_budget is not None and total > self.total_serialized_byte_budget:
            raise ValueError(
                f"recipe exceeds byte budget: {total} > {self.total_serialized_byte_budget}"
            )
        return total


def _pack_unsigned(values: Tensor, bits: int) -> Tensor:
    if bits == 8:
        return values.to(torch.uint8).contiguous()
    if bits != 4:
        raise ValueError("only 4-bit and 8-bit packing are supported")
    flat = values.to(torch.uint8).flatten()
    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)])
    return (flat[0::2] | (flat[1::2] << 4)).contiguous()


def _unpack_unsigned(packed: Tensor, count: int, bits: int) -> Tensor:
    if bits == 8:
        return packed.flatten()[:count]
    if bits != 4:
        raise ValueError("only 4-bit and 8-bit packing are supported")
    flat = packed.flatten()
    output = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=flat.device)
    output[0::2] = flat & 0x0F
    output[1::2] = flat >> 4
    return output[:count]


def _quantize_rows(values: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    levels = (1 << bits) - 1
    values = values.float()
    minimum = values.amin(dim=1, keepdim=True)
    maximum = values.amax(dim=1, keepdim=True)
    scales = ((maximum - minimum) / levels).clamp_min(1e-12)
    codes = torch.round((values - minimum) / scales).clamp(0, levels).to(torch.uint8)
    # Store [scale, minimum] per row.  This is explicit in the checkpoint.
    return _pack_unsigned(codes, bits), torch.cat([scales, minimum], dim=1)


class PackedLowRankLinear(nn.Module):
    """A low-rank linear layer with packed factor storage.

    The checkpoint contains packed U/V codes, row scales/minima, and metadata.
    Dequantized factors are temporary forward values; no dense U@V tensor is
    stored or registered.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bits: int = 16,
        bias: Tensor | None = None,
        u: Tensor | None = None,
        v: Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0 < rank <= min(in_features, out_features):
            raise ValueError("rank must be positive and fit the matrix dimensions")
        if bits not in (4, 8, 16):
            raise ValueError("bits must be 4, 8, or 16")
        self.in_features, self.out_features, self.rank, self.bits = (
            in_features, out_features, rank, bits
        )
        if u is None or v is None:
            raise ValueError("u and v are required")
        if tuple(u.shape) != (out_features, rank) or tuple(v.shape) != (rank, in_features):
            raise ValueError("factor shapes do not match layer dimensions")
        self._store_factor("u", u, bits)
        self._store_factor("v", v, bits)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    def _store_factor(self, name: str, value: Tensor, bits: int) -> None:
        if bits == 16:
            self.register_buffer(name, value.detach().to(torch.float16).contiguous())
            return
        packed, params = _quantize_rows(value, bits)
        self.register_buffer(name, packed)
        self.register_buffer(f"{name}_params", params.to(torch.float32))

    def _factor(self, name: str, rows: int, cols: int) -> Tensor:
        packed = getattr(self, name)
        if self.bits == 16:
            return packed.to(dtype=torch.float32)
        codes = _unpack_unsigned(packed, rows * cols, self.bits).reshape(rows, cols).float()
        params = getattr(self, f"{name}_params")
        return codes * params[:, :1] + params[:, 1:2]

    def forward(self, x: Tensor) -> Tensor:
        v = self._factor("v", self.rank, self.in_features)
        u = self._factor("u", self.out_features, self.rank)
        result = torch.matmul(torch.matmul(x, v.transpose(0, 1)), u.transpose(0, 1))
        return result + self.bias.to(result.dtype) if self.bias is not None else result

    @classmethod
    def from_factors(
        cls, u: Tensor, v: Tensor, bias: Tensor | None = None, *, bits: int = 16
    ) -> "PackedLowRankLinear":
        return cls(v.shape[1], u.shape[0], u.shape[1], bits=bits, bias=bias, u=u, v=v)

    def serialized_bytes(self) -> int:
        total = sum(t.numel() * t.element_size() for t in self.buffers())
        return int(total)


def checkpoint_has_dense_weight(state_dict: Mapping[str, Tensor]) -> bool:
    """Return whether a packed checkpoint contains a suspicious dense W tensor."""
    dense_names = {"weight", "dense_weight", "reconstructed_weight", "W"}
    return any(name.rsplit(".", 1)[-1] in dense_names for name in state_dict)


def fit_activation_weighted_low_rank(
    weight: Tensor,
    inputs: Tensor,
    outputs: Tensor,
    rank: int,
    *,
    beta: float = 0.1,
    bias: Tensor | None = None,
    ridge_fraction: float = 0.01,
) -> Tensor:
    """Fit ``W_hat`` for ``Y ~= W_hat X`` in an activation-weighted norm.

    Inputs and outputs are column-major calibration matrices with shapes
    ``[in_features, tokens]`` and ``[out_features, tokens]``.  The returned
    tensor has the same orientation as ``weight`` (``[out_features, in_features]``).
    The ridge anchor keeps the fit well-defined and makes the reference weight
    explicit rather than silently changing the target when calibration is rank
    deficient.
    """
    if inputs.ndim != 2 or outputs.ndim != 2 or weight.ndim != 2:
        raise ValueError("weight, inputs, and outputs must be rank-2 tensors")
    if inputs.shape[1] != outputs.shape[1] or tuple(weight.shape) != (
        outputs.shape[0], inputs.shape[0]
    ):
        raise ValueError("weight and calibration matrix shapes are inconsistent")
    if not 0 < rank <= min(weight.shape):
        raise ValueError("rank must be positive and fit the matrix dimensions")
    if beta < 0 or ridge_fraction < 0:
        raise ValueError("beta and ridge_fraction must be non-negative")

    w = weight.float()
    x = inputs.float()
    y = outputs.float()
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != y.shape[0]:
            raise ValueError("bias must have one value per output feature")
        y = y - bias.float().unsqueeze(1)
    n = x.shape[1]
    eye = torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    gram = x @ x.T + beta * n * eye
    cross = y @ x.T + beta * n * w

    gram_evals, gram_vectors = torch.linalg.eigh(gram)
    ridge = ridge_fraction * gram_evals.clamp_min(0).mean()
    regularized = gram_evals.clamp_min(0) + ridge
    inverse = gram_vectors @ torch.diag(regularized.reciprocal()) @ gram_vectors.T
    gram_half = gram_vectors @ torch.diag(regularized.sqrt()) @ gram_vectors.T
    full_fit = cross @ inverse
    left, _, _ = torch.linalg.svd(full_fit @ gram_half, full_matrices=False)
    basis = left[:, :rank]
    return basis @ (basis.T @ full_fit)

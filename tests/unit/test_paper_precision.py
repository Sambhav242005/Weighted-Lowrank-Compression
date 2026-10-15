import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.paper_precision import (FORMATS, PrecisionLinear, evaluate_explicit,
                                 identity_factors, place_model, row_decode,
                                 row_quantize)


@pytest.mark.parametrize("dtype", list(FORMATS.values()))
def test_row_fp8_finite_zero_rows_and_decode(dtype):
    value = torch.tensor([[0., 0., 0.], [1., -2., 4.],
                          [1e-20, -2e-20, 4e-20], [1e6, -2e6, 4e6]])
    codes, scale = row_quantize(value, dtype)
    assert codes.dtype == dtype
    assert scale.dtype == torch.float32 and scale.shape == (4, 1)
    assert torch.isfinite(codes.float()).all() and (scale > 0).all()
    assert torch.equal(codes.float()[0], torch.zeros(3))
    decoded = row_decode(codes, scale, torch.float32)
    assert torch.isfinite(decoded).all()
    assert torch.allclose(decoded, value, rtol=.13, atol=1e-25)


@pytest.mark.parametrize("invalid", [torch.tensor([1.]), torch.tensor([[float("nan")]]),
                                     torch.tensor([[float("inf")]])])
def test_row_fp8_rejects_invalid(invalid):
    with pytest.raises(ValueError):
        row_quantize(invalid, torch.float8_e4m3fn)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (4, 4)])
def test_rectangular_identity_basis_orientation(shape):
    weight = torch.arange(math.prod(shape), dtype=torch.bfloat16).reshape(shape)/8
    left, right = identity_factors(weight)
    assert left.shape == (shape[0], min(shape))
    assert right.shape == (min(shape), shape[1])
    x = torch.arange(2*shape[1], dtype=torch.bfloat16).reshape(2, shape[1])/16
    packed = PrecisionLinear(left, right)
    assert torch.equal(packed(x), F.linear(x, weight))
    assert packed(x).dtype == torch.bfloat16
    assert packed.left.dtype == torch.bfloat16
    assert packed.right.dtype == torch.bfloat16


def test_bf16_factors_never_round_through_fp16():
    # BF16 range extends beyond FP16; values stay finite through BF16 GEMMs.
    left = torch.tensor([[100000., 0.], [0., 100000.]], dtype=torch.bfloat16)
    right = torch.eye(2, dtype=torch.bfloat16)
    module = PrecisionLinear(left, right)
    value = module(torch.ones(1, 2, dtype=torch.bfloat16))
    assert torch.isfinite(value).all()
    assert torch.equal(value, left.diag().unsqueeze(0))
    assert all(t.dtype == torch.bfloat16 for t in module.buffers())


def test_fp32_factor_compute_returns_bf16_outputs():
    g = torch.Generator().manual_seed(5)
    left, right = torch.randn(5, 3, generator=g), torch.randn(3, 7, generator=g)
    bias = torch.randn(5, generator=g)
    x = torch.randn(4, 7, generator=g).bfloat16()
    module = PrecisionLinear(left, right, bias, storage="fp32", compute="fp32")
    expected = F.linear(F.linear(x.float(), right), left, bias).bfloat16()
    assert torch.equal(module(x), expected)
    assert module(x).dtype == torch.bfloat16
    assert module.left.dtype == torch.float32


@pytest.mark.parametrize("storage", list(FORMATS))
@pytest.mark.parametrize("factorized", [True, False])
def test_fp8_storage_executes_decoded_bf16_without_dense_factor_product(storage, factorized):
    g = torch.Generator().manual_seed(13)
    left = torch.randn(5, 3, generator=g)
    right = torch.randn(3, 7, generator=g) if factorized else None
    bias = torch.randn(5, generator=g)
    x = torch.randn(4, 7 if factorized else 3, generator=g).bfloat16()
    module = PrecisionLinear(left, right, bias, storage=storage)
    intermediate = F.linear(x, module.decoded("right")) if factorized else x
    expected = F.linear(intermediate, module.decoded("left"), bias.bfloat16())
    assert torch.equal(module(x), expected)
    assert module.left.dtype == FORMATS[storage]
    assert module.decoded("left").dtype == torch.bfloat16
    assert not hasattr(module, "weight")
    assert all(t.dtype != torch.float16 for t in module.buffers())


class ToyModel(nn.Module):
    def __init__(self, meta=False):
        super().__init__()
        self.marker = nn.Parameter(torch.empty((), device="meta" if meta else "cpu"))
        self.seen = []

    def forward(self, input_ids, use_cache=False):
        assert input_ids.device.type == "cpu"
        self.seen.append(input_ids[0].tolist())
        # Local-position-dependent confidence makes overlapping window ownership
        # affect NLL, so incorrect overlap scoring changes the expected result.
        confidence = torch.arange(input_ids.shape[1]).float()/50 + .3
        logits = torch.zeros(*input_ids.shape, 7)
        logits.scatter_(-1, ((input_ids+1) % 7).unsqueeze(-1), confidence[None,:,None])
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("length", [2, 17, 513, 777, 8195])
def test_evaluation_exact_target_ownership_tail_and_meta_device(length):
    tokens = torch.arange(length) % 7
    model = ToyModel(meta=True)
    measured = evaluate_explicit(model, tokens, execution_device="cpu")
    # Independently assign each target to its earliest context window.
    expected = []
    for target in range(1, length):
        window_start = max(0, math.ceil((target+1-512)/256))*256
        confidence = (target-window_start-1)/50+.3
        expected.append(math.log(math.exp(confidence)+6)-confidence)
    assert measured["targets"] == length-1
    assert sum(b["targets"] for b in measured["nll_blocks"]) == length-1
    assert measured["nll_sum"] == pytest.approx(sum(expected), rel=2e-6, abs=1e-5)
    assert measured["nll_blocks"][-1]["targets"] <= 4096
    assert model.seen[-1][-1] == tokens[-1].item()


def test_cpu_offload_hook_evaluation_uses_explicit_device():
    # Exercise Accelerate's actual meta-parameter offload path without CUDA.
    from accelerate import cpu_offload
    model = nn.Sequential(nn.Linear(4, 3, dtype=torch.bfloat16))
    original = model(torch.ones(1, 4, dtype=torch.bfloat16))
    hooked = cpu_offload(model, execution_device=torch.device("cpu"), offload_buffers=True)
    assert next(hooked.parameters()).device.type == "meta"
    assert torch.equal(hooked(torch.ones(1, 4, dtype=torch.bfloat16)), original)


def test_place_model_passes_offload_buffers(monkeypatch):
    import accelerate
    captured = {}
    def fake(model, **kwargs):
        captured.update(kwargs)
        return model
    monkeypatch.setattr(accelerate, "cpu_offload", fake)
    model = ToyModel()
    assert place_model(model, "cpu", "cuda") is model
    assert captured == {"execution_device": torch.device("cuda"), "offload_buffers": True}

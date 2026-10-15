from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.paper_math import weighted_factors
from src.paper_protocol import calibration_starts, evaluate, windows
from src.compression_core import PackedLowRankLinear, fit_activation_weighted_low_rank


@pytest.mark.parametrize("length", [2, 257, 512, 513, 768, 1001])
def test_every_target_scored_once_including_partial_window(length):
    targets = [t for _, end, first in windows(length) for t in range(first, end)]
    assert targets == list(range(1, length))


def test_full_rank_anchor_recovers_teacher_and_ridge_matches_legacy():
    torch.manual_seed(19)
    w, x = torch.randn(5, 7), torch.randn(7, 31)
    u, v = weighted_factors(w, x, 5, beta=0.1)
    assert torch.allclose(u @ v, w, atol=1e-5)
    u, v = weighted_factors(w, x, 3, beta=0.1, ridge_fraction=0.01)
    legacy = fit_activation_weighted_low_rank(w, x, w @ x, 3)
    assert torch.allclose(u @ v, legacy, atol=2e-5, rtol=2e-5)


def test_rank_deficient_pseudoinverse_is_finite_and_preserves_observed_outputs():
    w, x = torch.randn(4, 6), torch.zeros(6, 20)
    x[0] = torch.randn(20)
    u, v = weighted_factors(w, x, 4)
    assert torch.isfinite(u).all() and torch.isfinite(v).all()
    assert torch.allclose(u @ v @ x, w @ x, atol=1e-5)


def test_calibration_is_paired_and_nonoverlapping():
    starts = calibration_starts(512*32, 11)
    assert starts == calibration_starts(512*32, 11)
    assert len(set(starts)) == 16
    assert all(s % 512 == 0 for s in starts)


def test_packed_half_execution_and_reload():
    u, v = torch.randn(5, 3), torch.randn(3, 7)
    layer = PackedLowRankLinear.from_factors(u, v)
    restored = PackedLowRankLinear.from_factors(torch.zeros_like(u), torch.zeros_like(v))
    restored.load_state_dict(layer.state_dict())
    x = torch.randn(2, 7).half()
    result = layer(x)
    assert result.dtype == torch.float16
    assert torch.equal(result, restored(x))


def test_evaluation_accumulates_all_windows_and_shifted_targets():
    class Uniform(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, use_cache):
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 7))

    result = evaluate(Uniform(), torch.arange(1001) % 7)
    assert result["targets"] == 1000
    assert result["ppl"] == pytest.approx(7, rel=1e-6)
    assert sum(b["targets"] for b in result["nll_blocks"]) == 1000

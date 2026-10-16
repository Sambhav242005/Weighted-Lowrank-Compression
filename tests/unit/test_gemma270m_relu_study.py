"""Offline mathematical/coverage checks for the additive ReLU study."""
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gemma270m_relu_study import Bottleneck, budget_rank, chunk_metrics, windows


@pytest.mark.parametrize("nonlinear", [False, True])
def test_budget_shape_dtype_and_serialization(nonlinear):
    r = budget_rank(1024, 640, nonlinear)
    torch.manual_seed(11)
    m = Bottleneck(1024, 640, r, nonlinear)
    x = torch.randn(2, 3, 1024)
    y = m(x)
    assert y.shape == (2, 3, 640) and y.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in m.parameters()) <= 1024 * 640 * .7
    b = io.BytesIO()
    torch.save(m.state_dict(), b)
    assert b.tell() > 0
    b.seek(0)
    copy = Bottleneck(1024, 640, r, nonlinear)
    copy.load_state_dict(torch.load(b, weights_only=True))
    assert torch.equal(copy(x), y)


@pytest.mark.parametrize("length", [2, 127, 128, 256, 257, 384, 513, 1000])
def test_strided_coverage_including_tail(length):
    tokens = torch.arange(length)
    targets = []
    for start, end, first in windows(tokens):
        targets.extend(range(start + first, end))
    assert targets == list(range(1, length))


def test_kl_identity_and_nonfinite_guard():
    logits = torch.randn(8, 10)
    labels = torch.arange(8)
    nll, kl = chunk_metrics(logits, labels, logits.clone())
    assert nll > 0 and abs(kl) < 1e-7
    with pytest.raises(FloatingPointError):
        chunk_metrics(torch.full_like(logits, float("nan")), labels)


def test_known_linear_map_and_invalid_width():
    m = Bottleneck(4, 3, 2)
    x = torch.randn(10, 4)
    assert torch.allclose(m(x), x @ (m.up.weight @ m.down.weight).T, atol=1e-6)
    with pytest.raises(ValueError):
        Bottleneck(4, 3, 0)


def test_tiny_gemma_replacement_and_teacher_preservation():
    from copy import deepcopy
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    from gemma270m_relu_study import install
    cfg = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=16, sliding_window=32, layer_types=["sliding_attention", "full_attention"])
    cfg._attn_implementation = "eager"
    teacher = Gemma3ForCausalLM(cfg).eval()
    student = deepcopy(teacher)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    replacements = [Bottleneck(32, 32, 11, True) for _ in range(2)]
    install(student, replacements, "cpu")
    ids = torch.randint(0, 128, (1, 16))
    with torch.inference_mode():
        expected, actual = teacher(ids).logits, student(ids).logits
    assert expected.shape == actual.shape == (1, 16, 128)
    assert torch.isfinite(actual).all() and not torch.equal(expected, actual)
    assert all(torch.equal(v, before[k]) for k, v in teacher.state_dict().items())

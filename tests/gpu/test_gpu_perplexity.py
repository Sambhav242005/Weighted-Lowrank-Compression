"""GPU-only checks of shift, chunking, tails and teacher-forced cached scoring."""
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
spec = importlib.util.spec_from_file_location("gpu_ppl", Path(__file__).resolve().parents[2] / "docs/evaluate_gpu_perplexity_20261016.py")
ppl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppl)


def test_chunked_nll_shift_matches_direct_cross_entropy_and_tail_coverage():
    torch.manual_seed(11)
    hidden = torch.randn(1, 35, 8, device="cuda")
    head = torch.nn.Linear(8, 17, bias=False, device="cuda")
    ids = torch.randint(0, 17, (1, 35), device="cuda")
    with torch.inference_mode():
        actual, count = ppl.score_positions(head, hidden, ids, 5, chunk=7)
        expected = F.cross_entropy(head(hidden[:, 4:-1]).reshape(-1, 17), ids[:, 5:].reshape(-1), reduction="sum")
    assert count == 30 and abs(actual - float(expected)) < 1e-4
    tokens = torch.arange(513, device="cuda")
    covered = [i for start, end, first in ppl.windows(tokens) for i in range(start + first, end)]
    assert covered == list(range(1, 513))


def test_cached_teacher_forcing_matches_full_sequence_for_tiny_gemma():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=8, sliding_window=32, layer_types=["sliding_attention", "full_attention"])
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval().cuda()
    ids = torch.randint(0, 32, (1, 24), device="cuda")
    with torch.inference_mode():
        hidden = model.model(ids, use_cache=False).last_hidden_state
        expected, _ = ppl.score_positions(model.lm_head, hidden, ids, 8)
        h = model.model(ids[:, :8], use_cache=True)
        cache, actual = h.past_key_values, 0.
        for pos in range(8, 24):
            logits = model.lm_head(h.last_hidden_state[:, -1:]).reshape(1, 32)
            nll, _ = ppl.chunk_metrics(logits, ids[:, pos])
            actual += nll
            if pos < 23:
                h = model.model(ids[:, pos:pos + 1], past_key_values=cache, use_cache=True)
                cache = h.past_key_values
    assert abs(actual - expected) < 1e-4

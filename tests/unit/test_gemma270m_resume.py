"""CPU-only equivalence of interrupted-prefix recovery to uninterrupted quality evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gemma270m_relu_study import Bottleneck, Budget, quality
from gemma270m_resume import resume_quality, validate_prefix


def test_recovered_quality_matches_uninterrupted(tmp_path, monkeypatch):
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
    torch.set_num_threads(2)
    torch.manual_seed(37)
    cfg = Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=8, sliding_window=32, layer_types=["sliding_attention", "full_attention"])
    cfg._attn_implementation = "eager"
    model = Gemma3ForCausalLM(cfg).eval()
    dense = [layer.self_attn.o_proj for layer in model.model.layers]
    variants = {"relu_seed11": [Bottleneck(16, 16, 5, True).eval() for _ in range(2)]}
    tokens = torch.randint(0, 32, (300,))
    complete, partial, resumed = [tmp_path / p for p in ("complete", "partial", "resumed")]
    for p in (complete, partial, resumed):
        p.mkdir()
    expected = quality(model, dense, variants, tokens, complete, Budget(60))
    quality(model, dense, variants, tokens[:256], partial, Budget(60))
    checkpoint = json.loads((partial / "quality-progress.json").read_text())
    checkpoint["expected_windows"] = 2
    before = deepcopy(checkpoint)
    actual = resume_quality(model, dense, variants, tokens, resumed, Budget(60), checkpoint)
    assert actual == expected
    assert checkpoint == before
    rows = [json.loads(line) for line in (resumed / "quality-windows.jsonl").read_text().splitlines()]
    assert all(row["window"] == 1 for row in rows)
    checkpoint["aggregates"]["dense"]["count"] -= 1
    with pytest.raises(ValueError, match="counts"):
        validate_prefix(checkpoint, tokens, ["dense", *variants])

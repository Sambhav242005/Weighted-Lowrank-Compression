"""Read-only runner audit; tiny random CPU model, no downloads/GPU use."""
from pathlib import Path
import json
import uuid
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from src.paper_comparisons import canonical_weight, target_modules
from src.compression_core import PackedLowRankLinear
from src.paper_followups import _save_model_state, restore_model_checkpoint, model_probe_digest

torch.set_num_threads(1)
torch.manual_seed(41)
model = GPT2LMHeadModel(GPT2Config(n_embd=16, n_layer=2, n_head=2,
    vocab_size=32, bos_token_id=1, eos_token_id=2, n_positions=256,
    resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).half().eval()
tokens = torch.randint(0, 32, (140,))
targets = target_modules(model)
assert len(targets) == 2
for name, module in targets:
    assert module.__class__.__name__ == "Conv1D"
    x = torch.randn(2, 3, 16).half()
    canonical = canonical_weight(module)
    direct = x @ canonical.T + module.bias
    assert torch.equal(module(x), direct), "GPT2 transpose/bias mismatch"

root = Path("results/paper-v2") / ("cpu-adapter-audit-20261014-" + uuid.uuid4().hex[:8])
root.mkdir(exist_ok=False)
rows = []
for bits in (16, 8):
    for name, _ in targets:
        layer = PackedLowRankLinear.from_factors(torch.randn(16, 8).half(),
            torch.randn(8, 16).half(), torch.randn(16).half(), bits=bits)
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, layer)
    before = model_probe_digest(model, tokens)
    checkpoint = root / f"whole-bits-{bits}.pt"
    storage = _save_model_state(model, checkpoint)
    restored = restore_model_checkpoint(checkpoint, device="cpu")
    after = model_probe_digest(restored, tokens)
    assert before == after, f"bits={bits} whole-model reload changed cached logits"
    assert restored.transformer.wte.weight.data_ptr() == restored.lm_head.weight.data_ptr()
    rows.append({"bits": bits, "exact_prompt_and_cached_parity": True,
        "tied_embedding_storage_preserved": True, "bytes": storage["bytes"]})
report = {"state": "completed", "device": "cpu", "synthetic": True,
    "gpt2_conv1d_transpose_and_bias": True, "rows": rows}
(root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"directory": str(root.resolve()), **report}))

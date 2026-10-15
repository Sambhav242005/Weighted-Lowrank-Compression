"""No-network, CPU/meta audit of selected hybrid-model adapters."""
import json
import uuid
from pathlib import Path
from fractions import Fraction

import torch
import transformers
from transformers import (Qwen3_5Config, Qwen3_5TextConfig,
    Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Lfm2Config, Lfm2ForCausalLM)
from src.paper_comparisons import target_modules, block_modules

torch.set_num_threads(1)
torch.manual_seed(7)
root = Path("results/paper-v2") / ("cpu-hybrid-adapter-audit-20261014-" + uuid.uuid4().hex[:8])
root.mkdir(exist_ok=False)
selected = json.loads(Path("paper-assets/architecture-selection-20261014.json").read_text())
rows = []
for record in selected["records"]:
    qwen = record["model"].startswith("Qwen/")
    configuration = (Qwen3_5TextConfig(**record["config"]["text_config"]) if qwen
                     else Lfm2Config(**record["config"]))
    cls = Qwen3_5ForCausalLM if qwen else Lfm2ForCausalLM
    with torch.device("meta"):
        model = cls(configuration)
    suffix = "self_attn.o_proj" if qwen else "self_attn.out_proj"
    targets = [(name, module) for name, module in model.named_modules() if name.endswith(suffix)]
    expected = [i for i, kind in enumerate(configuration.layer_types) if kind == "full_attention"]
    blocks = block_modules(model)
    assert len(targets) == 6 and len(blocks) == configuration.num_hidden_layers
    recipes = []
    for name, module in targets:
        index = int(name.split(".layers.")[1].split(".")[0])
        assert index in expected
        out, inside = module.weight.shape
        rank = (17 * out * inside) // (24 * (out + inside))
        recipes.append({"name": name, "block_index": index, "shape": [out, inside],
            "rank": rank, "factor_parameters": rank * (out + inside),
            "factor_to_dense_parameter_fraction": float(Fraction(rank * (out + inside), out * inside))})
    try:
        existing = target_modules(model)
        current_status = {"count": len(existing)}
    except ValueError as exc:
        current_status = {"error": str(exc)}
    rows.append({"model": record["model"], "revision": record["revision"],
        "decoder_blocks": len(blocks), "attention_block_indices": expected,
        "target_recipes": recipes, "current_adapter": current_status})
    del model

# Save a genuinely composite, tiny random checkpoint, then exercise installed
# from_pretrained config extraction and language_model prefix conversion.
text = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
    num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
    linear_num_value_heads=2, linear_conv_kernel_dim=4, tie_word_embeddings=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
        "partial_rotary_factor": .25, "mrope_section": [1, 1, 0], "mrope_interleaved": True},
    layer_types=["linear_attention"] * 3 + ["full_attention"])
composite = Qwen3_5Config(text_config=text.to_dict(), vision_config={
    "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
    "out_hidden_size": 16, "num_position_embeddings": 16, "patch_size": 2,
    "spatial_merge_size": 1, "temporal_patch_size": 1, "deepstack_visual_indexes": []},
    tie_word_embeddings=True)
whole = Qwen3_5ForConditionalGeneration(composite).eval()
checkpoint = root / "tiny-composite-checkpoint"
whole.save_pretrained(checkpoint)
loaded, info = Qwen3_5ForCausalLM.from_pretrained(checkpoint, local_files_only=True,
    output_loading_info=True)
assert not info["missing_keys"], info
assert not info["mismatched_keys"], info
for name, tensor in loaded.state_dict().items():
    source_name = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
    assert torch.equal(tensor, whole.state_dict()[source_name]), name
ids = torch.randint(0, 32, (1, 8))
with torch.inference_mode():
    full_logits = whole(input_ids=ids, use_cache=False).logits
    text_logits = loaded(input_ids=ids, use_cache=False).logits
assert torch.equal(full_logits, text_logits)
report = {"state": "completed", "transformers": transformers.__version__, "device": "cpu/meta",
    "network": False, "real_checkpoint_weights_loaded": False, "models": rows,
    "qwen_tiny_multimodal_to_text": {"missing_keys": list(info["missing_keys"]),
        "unexpected_keys": list(info["unexpected_keys"]), "mismatched_keys": list(info["mismatched_keys"]),
        "all_text_weights_equal": True, "text_logits_equal": True}}
(root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({"directory": str(root.resolve()), **report}))

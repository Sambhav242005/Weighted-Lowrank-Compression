"""
Storage accounting: low-rank weighted fit vs quantization (Qwen2.5-7B-Instruct)
================================================================================
Pure computation, no GPU. Answers: at the quality levels we measured, how
much storage does the low-rank method actually save vs standard quantization?

Conventions:
  - one-sided kx rank convention: r = min(in, out) // k
  - low-rank stored in FACTORED form U[m,r] @ V[r,n] (the only form that
    saves storage; storing the rank-r reconstruction dense saves nothing)
  - fp16 = 2 bytes/param; int b = b bits/param (+ GPTQ-style group scale
    overhead ~0.4 bit for int4)

Qwen2.5-7B config: hidden 3584, inter 18944, heads 28, kv_heads 4,
head_dim 128, 28 layers, vocab 152064.
"""
import json
from pathlib import Path

H, I, KV, V, L = 3584, 18944, 512, 152064, 28

LAYERS = {
    "q_proj": (H, H), "k_proj": (KV, H), "v_proj": (KV, H), "o_proj": (H, H),
    "gate_proj": (H, I), "up_proj": (H, I), "down_proj": (I, H),
}
per_layer = {k: m * n for k, (m, n) in LAYERS.items()}
layer_params = sum(per_layer.values())
total_params = layer_params * L + V * H          # embed (tied lm_head)
o_proj_params = per_layer["o_proj"] * L
attn_params = (per_layer["q_proj"] + per_layer["k_proj"] +
               per_layer["v_proj"] + per_layer["o_proj"]) * L

def lowrank_bytes(m, n, k, bytes_per=2):
    """Factored storage for rank min(m,n)//k approximation."""
    r = min(m, n) // k
    return r * (m + n) * bytes_per

def quant_bytes(params, bits):
    return params * bits / 8

MODEL_FP16 = total_params * 2

report = {
    "model_params_B": round(total_params / 1e9, 3),
    "model_fp16_GB": round(MODEL_FP16 / 1e9, 3),
    "o_proj_share_of_params_pct": round(100 * o_proj_params / total_params, 2),
    "attention_share_of_params_pct": round(100 * attn_params / total_params, 2),
}

# ---- our measured points (o_proj only, factored fp16) -------------------
ours = {}
for k in (3, 4, 6, 8):
    m, n = LAYERS["o_proj"]
    orig = o_proj_params * 2
    new = lowrank_bytes(m, n, k) * L
    ratio = orig / new
    saved = (orig - new) / MODEL_FP16
    ours[f"rank_{k}x"] = {
        "per_matrix_storage_ratio": round(ratio, 2),
        "whole_model_savings_pct": round(100 * saved, 2),
    }
# measured PPL deltas (o_proj-only scope)
ours["rank_3x"]["measured_ppl_delta"] = {"gpt2": "+3.58%", "qwen7b": "+3.84%"}
ours["rank_4x"]["measured_ppl_delta"] = {"gpt2": "+17.94%"}
report["lowrank_o_proj_only"] = ours

# ---- quantization tiers (ALL weights) ------------------------------------
quant = {}
for name, bits, ppl in (("int8", 8, "~0-0.1% (7B, well established)"),
                        ("int4_gptq_awq", 4.4, "~0.3-1% at 7B (literature)"),
                        ("int3", 3.25, "~1-3% at 7B (literature)"),
                        ("int2", 2.5, "severe degradation")):
    qb = quant_bytes(total_params, bits)
    quant[name] = {
        "bits_per_weight": bits,
        "whole_model_storage_ratio": round(16 / bits, 2),
        "model_size_GB": round(qb / 1e9, 2),
        "typical_ppl_cost": ppl,
        "scope": "ALL weights",
    }
report["quantization"] = quant

# ---- what rank would match quant tiers on o_proj alone --------------------
# per-matrix storage ratio for square dxd at rank-k convention = k/2
report["equivalence_note"] = (
    "Factored low-rank at one-sided kx rank gives k/2 x storage per square "
    "matrix. Matching int4 (4x) on o_proj alone needs k=8 (rank 448), which "
    "measured +94% PPL on GPT-2 -- unreachable. Matching int8 (2x) needs "
    "k=4, which measured +17.94%. Quantization reaches better PPL at 4x the "
    "whole-model ratio; low-rank reaches +3.84% at 1.6% whole-model savings."
)
report["combined_paths"] = [
    "Quantize the low-rank factors (int4 U,V): 3x-rank o_proj -> ~6x per-matrix",
    "SVDQuant-style: fp16 low-rank branch for outliers + int4 residual -> ~3.5x whole model with quality above plain int4 (arXiv 2411.05006)",
]

print(json.dumps(report, indent=2))
out = Path("results/storage_vs_quantization.json")
out.write_text(json.dumps(report, indent=2))
print(f"Saved {out}")

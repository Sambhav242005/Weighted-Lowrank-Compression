"""
Layer 0 Investigation
=====================
Why is layer 0 so sensitive to compression?

Hypotheses:
1. Layer 0 has a different singular value spectrum (more evenly distributed)
2. Layer 0 operates on raw embeddings (different geometry than hidden states)
3. Layer 0's error propagates exponentially through the network

Test: Compare spectra, measure error amplification rate per layer.
"""

import torch, numpy as np, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

# Load model
print("Loading GPT-2 Small...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ============================================================
# 1. Singular value spectra comparison
# ============================================================
print("\n" + "=" * 70)
print("SINGULAR VALUE SPECTRA")
print("=" * 70)

spectra = {}
for i in range(12):
    W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    
    # Effective rank (entropy-based)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    effective_rank = torch.exp(entropy)
    
    # Percentage of variance captured by top-k components
    k_values = [16, 32, 64, 128, 256]
    var_captured = {}
    for k in k_values:
        if k <= len(cumvar):
            var_captured[f"top{k}"] = float(cumvar[k-1] * 100)
    
    spectra[f"layer{i}"] = {
        "shape": list(W.shape),
        "spectral_norm": float(S[0]),
        "frobenius_norm": float(torch.sqrt((S ** 2).sum())),
        "effective_rank": float(effective_rank),
        "entropy": float(entropy),
        "var_captured": var_captured,
        "top5_singular_values": S[:5].tolist(),
        "bottom5_singular_values": S[-5:].tolist(),
    }
    
    print(f"  Layer {i:2d}: shape={list(W.shape)}, "
          f"spectral_norm={S[0]:.4f}, "
          f"eff_rank={effective_rank:.1f}, "
          f"top64={var_captured.get('top64', 0):.1f}%, "
          f"top128={var_captured.get('top128', 0):.1f}%")

# ============================================================
# 2. Error amplification rate
# ============================================================
print("\n" + "=" * 70)
print("ERROR AMPLIFICATION RATE")
print("=" * 70)

# For each layer, replace with SVD(r=128) and measure error at each subsequent layer
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
eval_texts = eval_texts[:30]

rank = 128
amplification = {}

for target_layer in [0, 1, 5, 10]:
    print(f"\n  Compressing layer {target_layer}...")
    
    # Collect original hidden states
    orig_hidden = {}
    def make_hook_orig(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            orig_hidden[name] = x.detach().cpu()
        return hook_fn
    
    hooks = []
    for i in range(12):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook_orig(f"layer{i}"))
        hooks.append(hook)
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    # Build compressed model
    import copy
    model_sub = copy.deepcopy(model)
    W_orig = model_sub.transformer.h[target_layer].attn.c_proj.weight.data.cpu().float()
    U, S, Vt = torch.linalg.svd(W_orig, full_matrices=False)
    W_sub = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]).to(DEVICE)
    model_sub.transformer.h[target_layer].attn.c_proj.weight.data = W_sub
    
    # Collect compressed hidden states
    sub_hidden = {}
    def make_hook_sub(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            sub_hidden[name] = x.detach().cpu()
        return hook_fn
    
    hooks_sub = []
    for i in range(12):
        hook = model_sub.transformer.h[i].attn.c_proj.register_forward_hook(make_hook_sub(f"layer{i}"))
        hooks_sub.append(hook)
    
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            _ = model_sub(**inputs)
    
    for h in hooks_sub:
        h.remove()
    
    # Measure error at each layer
    errors = []
    for i in range(12):
        key = f"layer{i}"
        if key in orig_hidden and key in sub_hidden:
            h_orig = orig_hidden[key]
            h_sub = sub_hidden[key]
            mse = ((h_orig.float() - h_sub.float()) ** 2).mean().item()
            cos = torch.nn.functional.cosine_similarity(
                h_orig.float().flatten(), h_sub.float().flatten(), dim=0
            ).item()
            errors.append({"layer": i, "mse": mse, "cosine": cos})
    
    amplification[f"layer{target_layer}"] = errors
    
    print(f"    Error at each layer boundary:")
    for e in errors:
        marker = " <-- compressed" if e["layer"] == target_layer else ""
        print(f"      Layer {e['layer']:2d}: MSE={e['mse']:.6f}  cos={e['cosine']:.4f}{marker}")
    
    del model_sub
    torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# 3. Summary
# ============================================================
print("\n" + "=" * 70)
print("LAYER 0 INVESTIGATION SUMMARY")
print("\n" + "=" * 70)

# Find layer with most even spectrum (hardest to compress)
eff_ranks = {k: v["effective_rank"] for k, v in spectra.items()}
most_even = max(eff_ranks, key=eff_ranks.get)
most_peaky = min(eff_ranks, key=eff_ranks.get)

print(f"\n  Most even spectrum (hardest to compress): {most_even} (eff_rank={eff_ranks[most_even]:.1f})")
print(f"  Most peaky spectrum (easiest to compress): {most_peaky} (eff_rank={eff_ranks[most_peaky]:.1f})")

# Error amplification
print(f"\n  Error amplification patterns:")
for source, errors in amplification.items():
    source_idx = int(source.replace("layer", ""))
    source_error = next(e for e in errors if e["layer"] == source_idx)["mse"]
    final_error = errors[-1]["mse"]
    amp_factor = final_error / (source_error + 1e-10)
    print(f"    {source}: initial MSE={source_error:.6f}, final MSE={final_error:.6f}, amplification={amp_factor:.1f}x")

# Save results
output = {
    "spectra": spectra,
    "amplification": amplification,
}

output_path = Path("results/layer0_investigation.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")

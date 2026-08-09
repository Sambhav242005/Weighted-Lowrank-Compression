"""
Phase 3: Cumulative replacement of all W_O layers.
Tests error compounding through the residual stream.
"""

import sys, json, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from layer_extraction import extract_gpt2_layers, get_model_config
from representations import fit_svd_at_threshold, fit_fourier_at_threshold, fit_low_rank_product
from baseline_eval import get_baseline, compute_perplexity

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# Baseline
baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20, device=DEVICE)
print(f"Baseline PPL: {baseline.perplexity:.2f}\n")

# Extract weights
weights, _ = extract_gpt2_layers("gpt2")
config = get_model_config("gpt2")
n_layers = config['n_layer']

# ============================================================
# Test 1: Progressive layer replacement (1, 2, 4, 8, 12 layers)
# Using best representation from Phase 2: lowrank_r128
# ============================================================
print("=" * 70)
print("TEST 1: Progressive replacement with lowrank_r128 (3x compression)")
print("=" * 70)

progressive_results = []

for n_replace in [1, 2, 4, 8, 12]:
    print(f"\nReplacing {n_replace}/{n_layers} W_O layers...")
    
    model_copy = copy.deepcopy(model)
    
    for i in range(n_replace):
        layer_name = f"layer{i}.attn.W_O"
        weight = weights[layer_name].tensor
        
        # Fit low-rank at rank 128
        rep = fit_low_rank_product(weight, rank=128, name="lowrank", device=DEVICE, steps=500)
        recon = rep.reconstruct().to(DEVICE)
        
        # Substitute
        block = model_copy.transformer.h[i]
        block.attn.c_proj.weight.data = recon.float()
    
    # Evaluate
    ppl = compute_perplexity(model_copy, tokenizer, eval_texts, max_length=256, device=DEVICE)
    delta = ppl - baseline.perplexity
    delta_pct = (delta / baseline.perplexity) * 100
    
    progressive_results.append({
        "n_replaced": n_replace,
        "perplexity": ppl,
        "delta": delta,
        "delta_pct": delta_pct,
        "compression": 3.0,  # each layer 3x
    })
    
    print(f"  PPL: {baseline.perplexity:.2f} -> {ppl:.2f} ({delta_pct:+.2f}%)")
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Test 2: Same compression, different representations
# Replace ALL 12 W_O layers
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: All 12 layers, different representations")
print("=" * 70)

repr_configs = [
    ("SVD 99%", lambda w: fit_svd_at_threshold(w, 0.99, device=DEVICE)),
    ("SVD 95%", lambda w: fit_svd_at_threshold(w, 0.95, device=DEVICE)),
    ("Fourier 99%", lambda w: fit_fourier_at_threshold(w, 0.99, device=DEVICE)),
    ("LowRank r128", lambda w: fit_low_rank_product(w, rank=128, device=DEVICE, steps=500)),
    ("LowRank r256", lambda w: fit_low_rank_product(w, rank=256, device=DEVICE, steps=500)),
    ("LowRank r64", lambda w: fit_low_rank_product(w, rank=64, device=DEVICE, steps=500)),
]

all_layers_results = []

for repr_name, fit_fn in repr_configs:
    print(f"\n--- {repr_name} ---")
    model_copy = copy.deepcopy(model)
    
    total_orig = 0
    total_comp = 0
    
    for i in range(n_layers):
        layer_name = f"layer{i}.attn.W_O"
        weight = weights[layer_name].tensor
        rep = fit_fn(weight)
        recon = rep.reconstruct().to(DEVICE)
        
        block = model_copy.transformer.h[i]
        block.attn.c_proj.weight.data = recon.float()
        
        total_orig += rep.original_params
        total_comp += rep.n_params
    
    ppl = compute_perplexity(model_copy, tokenizer, eval_texts, max_length=256, device=DEVICE)
    delta_pct = ((ppl - baseline.perplexity) / baseline.perplexity) * 100
    overall_compression = total_orig / total_comp
    
    all_layers_results.append({
        "repr": repr_name,
        "compression": overall_compression,
        "perplexity": ppl,
        "delta_pct": delta_pct,
    })
    
    print(f"  Compression: {overall_compression:.2f}x | PPL: {ppl:.2f} ({delta_pct:+.2f}%)")
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Test 3: Adaptive compression - more on robust layers, less on sensitive
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Adaptive compression (aggressive on late layers, conservative on early)")
print("=" * 70)

# Layer sensitivity from Phase 2: layer0 most sensitive, layer2 least
# Extrapolate: later layers are more robust
adaptive_configs = [
    ("Uniform 3x", [128] * 12),
    ("Adaptive A", [256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64]),
    ("Adaptive B", [256, 256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 64]),
    ("Aggressive", [128, 64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 32]),
]

adaptive_results = []

for config_name, ranks in adaptive_configs:
    print(f"\n--- {config_name} ---")
    model_copy = copy.deepcopy(model)
    
    total_orig = 0
    total_comp = 0
    
    for i in range(n_layers):
        layer_name = f"layer{i}.attn.W_O"
        weight = weights[layer_name].tensor
        rank = ranks[i]
        
        rep = fit_low_rank_product(weight, rank=rank, device=DEVICE, steps=500)
        recon = rep.reconstruct().to(DEVICE)
        
        block = model_copy.transformer.h[i]
        block.attn.c_proj.weight.data = recon.float()
        
        total_orig += rep.original_params
        total_comp += rep.n_params
    
    ppl = compute_perplexity(model_copy, tokenizer, eval_texts, max_length=256, device=DEVICE)
    delta_pct = ((ppl - baseline.perplexity) / baseline.perplexity) * 100
    overall_compression = total_orig / total_comp
    
    adaptive_results.append({
        "config": config_name,
        "ranks": ranks,
        "compression": overall_compression,
        "perplexity": ppl,
        "delta_pct": delta_pct,
    })
    
    print(f"  Compression: {overall_compression:.2f}x | PPL: {ppl:.2f} ({delta_pct:+.2f}%)")
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Save all results
# ============================================================
results = {
    "baseline_ppl": baseline.perplexity,
    "progressive": progressive_results,
    "all_layers": all_layers_results,
    "adaptive": adaptive_results,
}

with open(output_path / "phase3_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3 SUMMARY")
print("=" * 70)

print(f"\nBaseline PPL: {baseline.perplexity:.2f}\n")

print("Progressive Replacement (LowRank r128, 3x per layer):")
print(f"  {'Layers':<10} {'PPL':<10} {'Delta':<10}")
for r in progressive_results:
    print(f"  {r['n_replaced']:<10} {r['perplexity']:<10.2f} {r['delta_pct']:>+8.2f}%")

print("\nAll 12 Layers - Representation Comparison:")
print(f"  {'Repr':<20} {'Compression':<14} {'PPL':<10} {'Delta'}")
for r in all_layers_results:
    print(f"  {r['repr']:<20} {r['compression']:<14.2f}x {r['perplexity']:<10.2f} {r['delta_pct']:>+8.2f}%")

print("\nAdaptive Compression:")
print(f"  {'Config':<20} {'Compression':<14} {'PPL':<10} {'Delta'}")
for r in adaptive_results:
    print(f"  {r['config']:<20} {r['compression']:<14.2f}x {r['perplexity']:<10.2f} {r['delta_pct']:>+8.2f}%")

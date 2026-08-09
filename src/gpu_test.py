"""Minimal GPU test: SVD on layer0.W_O, full model inference on GPU."""

import sys, torch
sys.path.insert(0, '.')

DEVICE = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)}")

from layer_extraction import extract_gpt2_layers
from representations import fit_svd_at_threshold
from substitution import test_substitution
from baseline_eval import get_baseline

# Baseline on GPU
baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20, device=DEVICE)
model = model.to(DEVICE)
print(f"Baseline PPL: {baseline.perplexity:.2f}")

# SVD on layer0.W_O
weights, _ = extract_gpt2_layers("gpt2")
layer_name = "layer0.attn.W_O"
weight = weights[layer_name].tensor

print(f"\nFitting SVD (99%) on {layer_name}...")
svd_rep = fit_svd_at_threshold(weight, variance_threshold=0.99)
print(f"  Rank: {svd_rep.metadata['rank']}, compression: {svd_rep.compression_ratio:.2f}x")

print("Testing substitution (GPU inference)...")
result = test_substitution(
    original_model=model,
    layer_name=layer_name,
    representation=svd_rep,
    tokenizer=tokenizer,
    eval_texts=eval_texts,
    baseline_perplexity=baseline.perplexity,
    max_length=256,
    device=DEVICE,
)

print(f"\nDone! PPL: {baseline.perplexity:.2f} -> {result.perplexity:.2f} ({result.perplexity_delta_pct:+.2f}%)")

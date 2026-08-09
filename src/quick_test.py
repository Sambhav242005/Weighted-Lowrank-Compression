"""Quick test: single layer, single representation, to verify the pipeline works."""

import sys
sys.path.insert(0, '.')

import torch
from layer_extraction import extract_gpt2_layers
from representations import fit_svd_at_threshold, fit_fourier_at_threshold, fit_hypernetwork
from substitution import test_substitution
from baseline_eval import get_baseline

# Baseline
print("Getting baseline...")
baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20)
print(f"Baseline PPL: {baseline.perplexity:.2f}")

# Extract
print("\nExtracting weights...")
weights, _ = extract_gpt2_layers("gpt2")

# Test layer0.attn.W_O with SVD only
layer_name = "layer0.attn.W_O"
weight = weights[layer_name].tensor
print(f"\nTesting {layer_name}, shape={weight.shape}")

# Fit SVD at 99% variance
print("Fitting SVD (99% variance)...")
svd_rep = fit_svd_at_threshold(weight, variance_threshold=0.99)
print(f"  Rank: {svd_rep.metadata['rank']}, compression: {svd_rep.compression_ratio:.2f}x")

# Test substitution
print("Testing substitution...")
result = test_substitution(
    original_model=model,
    layer_name=layer_name,
    representation=svd_rep,
    tokenizer=tokenizer,
    eval_texts=eval_texts,
    baseline_perplexity=baseline.perplexity,
    max_length=256,
)

print(f"\nDone! PPL delta: {result.perplexity_delta_pct:+.2f}%")

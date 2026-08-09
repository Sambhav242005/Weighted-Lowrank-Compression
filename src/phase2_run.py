"""Phase 2: All representations on GPU."""

import sys, json, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from layer_extraction import extract_gpt2_layers, get_model_config
from representations import fit_svd_at_threshold, fit_fourier_at_threshold, fit_hypernetwork, fit_low_rank_product
from substitution import test_substitution
from baseline_eval import get_baseline

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# Baseline on GPU
baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20, device=DEVICE)
print(f"Baseline PPL: {baseline.perplexity:.2f}")

# Extract weights
weights, _ = extract_gpt2_layers("gpt2")
config = get_model_config("gpt2")

# Test first 3 W_O layers
target_layers = [f"layer{i}.attn.W_O" for i in range(3)]
all_results = []

for layer_name in target_layers:
    print(f"\n{'='*60}")
    print(f"Layer: {layer_name}")
    weight = weights[layer_name].tensor

    representations = []

    for vt in [0.90, 0.95, 0.99]:
        representations.append(fit_svd_at_threshold(weight, variance_threshold=vt, name="svd", device=DEVICE))
    for vt in [0.90, 0.95, 0.99]:
        representations.append(fit_fourier_at_threshold(weight, variance_threshold=vt, name="fourier", device=DEVICE))
    for hd in [32, 64]:
        representations.append(fit_hypernetwork(weight, hidden_dim=hd, name="hypernet", device=DEVICE, steps=1000))
    for rank in [64, 128, 256]:
        representations.append(fit_low_rank_product(weight, rank=rank, name="lowrank", device=DEVICE, steps=500))

    for rep in representations:
        try:
            result = test_substitution(
                original_model=model, layer_name=layer_name, representation=rep,
                tokenizer=tokenizer, eval_texts=eval_texts,
                baseline_perplexity=baseline.perplexity, max_length=256, device=DEVICE,
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR {rep.name}: {e}")

# Save
data = {
    "baseline_ppl": baseline.perplexity,
    "results": [{
        "layer": r.layer_name, "repr": r.representation,
        "ratio": r.compression_ratio, "frob": r.relative_frobenius_error,
        "ppl": r.perplexity, "ppl_pct": r.perplexity_delta_pct, "top1": r.top1_agreement,
    } for r in all_results]
}
with open(output_path / "phase2_results.json", "w") as f:
    json.dump(data, f, indent=2)

# Summary
print(f"\n{'='*70}")
print(f"PHASE 2 SUMMARY — Baseline PPL: {baseline.perplexity:.2f}")
print(f"{'='*70}")
for layer in target_layers:
    lr = [r for r in all_results if r.layer_name == layer]
    print(f"\n{layer}:")
    print(f"  {'Repr':<22} {'Comp':<8} {'Frob':<10} {'PPL':<8} {'Delta':<10} {'Top1'}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    for r in sorted(lr, key=lambda x: x.compression_ratio, reverse=True):
        print(f"  {r.representation:<22} {r.compression_ratio:<8.2f}x {r.relative_frobenius_error:<10.4f} "
              f"{r.perplexity:<8.2f} {r.perplexity_delta_pct:>+8.2f}%  {r.top1_agreement:<8.4f}")

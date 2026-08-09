"""
Phase 2: Representation fitting and functional substitution.
Tests multiple representations on W_O layers (most compressible).
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from layer_extraction import extract_gpt2_layers, get_model_config
from representations import (
    fit_svd, fit_svd_at_threshold,
    fit_fourier, fit_fourier_at_threshold,
    fit_hypernetwork, fit_low_rank_product,
)
from substitution import test_substitution
from baseline_eval import get_baseline


def run_phase2(
    model_name: str = "gpt2",
    output_dir: str = "results",
    variance_thresholds: list = [0.90, 0.95, 0.99],
    target_layers: list = None,  # None = all W_O layers
):
    """Run Phase 2: representation fitting and testing."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("PHASE 2: REPRESENTATION FITTING & FUNCTIONAL SUBSTITUTION")
    print("=" * 70)
    
    # Get baseline
    print("\n[1/4] Computing baseline model metrics...")
    baseline, model, tokenizer, eval_texts = get_baseline(model_name)
    
    # Extract weights
    print("\n[2/4] Extracting weights...")
    weights, _ = extract_gpt2_layers(model_name)
    config = get_model_config(model_name)
    
    # Target layers: W_O is most compressible
    if target_layers is None:
        target_layers = [f"layer{i}.attn.W_O" for i in range(config['n_layer'])]
    
    # Test each target layer
    all_results = []
    
    print(f"\n[3/4] Testing {len(target_layers)} layers...")
    
    for layer_name in target_layers:
        print(f"\n--- {layer_name} ---")
        weight = weights[layer_name].tensor
        
        # Generate representations at different budgets
        representations = []
        
        for vt in variance_thresholds:
            # SVD
            svd_rep = fit_svd_at_threshold(weight, variance_threshold=vt, name="svd")
            representations.append(svd_rep)
            
            # Fourier
            fourier_rep = fit_fourier_at_threshold(weight, variance_threshold=vt, name="fourier")
            representations.append(fourier_rep)
        
        # Hypernetwork at different sizes
        for hidden_dim in [32, 64, 128]:
            hyp_rep = fit_hypernetwork(weight, hidden_dim=hidden_dim, name="hypernet")
            representations.append(hyp_rep)
        
        # Low-rank product at different ranks
        for rank in [64, 128, 256]:
            lr_rep = fit_low_rank_product(weight, rank=rank, name="lowrank")
            representations.append(lr_rep)
        
        # Test each representation
        for rep in representations:
            try:
                result = test_substitution(
                    original_model=model,
                    layer_name=layer_name,
                    representation=rep,
                    tokenizer=tokenizer,
                    eval_texts=eval_texts,
                    baseline_perplexity=baseline.perplexity,
                    max_length=256,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR testing {rep.name}: {e}")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    save_results(all_results, baseline, output_path)
    
    return all_results, baseline


def save_results(results, baseline, output_path):
    """Save Phase 2 results as report and JSON."""
    # JSON
    data = {
        "baseline": {
            "perplexity": baseline.perplexity,
            "top1_accuracy": baseline.top1_accuracy,
            "top5_accuracy": baseline.top5_accuracy,
            "logit_entropy": baseline.logit_entropy,
        },
        "results": [{
            "layer": r.layer_name,
            "repr": r.representation,
            "orig_params": r.original_params,
            "comp_params": r.compressed_params,
            "compression": r.compression_ratio,
            "frob_error": r.relative_frobenius_error,
            "perplexity": r.perplexity,
            "ppl_delta_pct": r.perplexity_delta_pct,
            "top1_agreement": r.top1_agreement,
            "kl_div": r.logit_divergence.get("kl_orig_to_sub", 0),
        } for r in results],
    }
    
    with open(output_path / "phase2_results.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Markdown report
    report = []
    report.append("# Phase 2: Representation Comparison Report\n\n")
    report.append(f"**Baseline perplexity:** {baseline.perplexity:.2f}\n")
    report.append(f"**Baseline top-1 accuracy:** {baseline.top1_accuracy:.4f}\n\n")
    
    # Group by layer
    layers = sorted(set(r.layer_name for r in results))
    
    report.append("## Perplexity Impact by Layer and Representation\n\n")
    report.append("| Layer | Representation | Compression | Frobenius Error | Perplexity | PPL Delta% | Top-1 Agreement |\n")
    report.append("|-------|---------------|-------------|-----------------|------------|------------|------------------|\n")
    
    for layer in layers:
        layer_results = [r for r in results if r.layer_name == layer]
        # Sort by compression ratio
        layer_results.sort(key=lambda r: r.compression_ratio, reverse=True)
        
        for r in layer_results:
            report.append(f"| {r.layer_name} | {r.representation} | "
                         f"{r.compression_ratio:.2f}x | {r.relative_frobenius_error:.4f} | "
                         f"{r.perplexity:.2f} | {r.perplexity_delta_pct:+.2f}% | "
                         f"{r.top1_agreement:.4f} |\n")
        report.append("\n")
    
    # Pareto analysis
    report.append("## Pareto Front: Compression vs Perplexity\n\n")
    
    # Find best representation at each compression level
    all_sorted = sorted(results, key=lambda r: r.compression_ratio, reverse=True)
    
    report.append("| Compression | Best Representation | PPL Delta% | Frobenius Error |\n")
    report.append("|-------------|--------------------|------------|-----------------|")
    
    # Bin by compression ratio
    bins = [(4.0, 5.0), (3.0, 4.0), (2.0, 3.0), (1.5, 2.0), (1.0, 1.5)]
    for low, high in bins:
        in_bin = [r for r in results if low <= r.compression_ratio < high]
        if in_bin:
            # Best = lowest perplexity delta
            best = min(in_bin, key=lambda r: abs(r.perplexity_delta_pct))
            report.append(f"\n| {low:.1f}-{high:.1f}x | {best.representation} "
                         f"(on {best.layer_name}) | {best.perplexity_delta_pct:+.2f}% | "
                         f"{best.relative_frobenius_error:.4f} |")
    
    report.append("\n\n")
    
    # Summary
    report.append("## Key Findings\n\n")
    
    # Best overall
    best_compression = max(results, key=lambda r: r.compression_ratio)
    best_preservation = min(results, key=lambda r: abs(r.perplexity_delta_pct))
    
    report.append(f"- **Highest compression:** {best_compression.representation} on "
                 f"{best_compression.layer_name} at {best_compression.compression_ratio:.2f}x "
                 f"(PPL delta: {best_compression.perplexity_delta_pct:+.2f}%)\n")
    report.append(f"- **Best preservation:** {best_preservation.representation} on "
                 f"{best_preservation.layer_name} at {best_preservation.compression_ratio:.2f}x "
                 f"(PPL delta: {best_preservation.perplexity_delta_pct:+.2f}%)\n")
    
    # Group analysis
    report.append("\n### Representation Family Comparison\n\n")
    families = {}
    for r in results:
        family = r.representation.split('_')[0]
        if family not in families:
            families[family] = []
        families[family].append(r)
    
    report.append("| Family | Avg Compression | Avg PPL Delta% | Avg Frobenius Error |\n")
    report.append("|--------|----------------|----------------|--------------------|\n")
    
    for family, reps in sorted(families.items()):
        avg_comp = np.mean([r.compression_ratio for r in reps])
        avg_ppl = np.mean([r.perplexity_delta_pct for r in reps])
        avg_frob = np.mean([r.relative_frobenius_error for r in reps])
        report.append(f"| {family} | {avg_comp:.2f}x | {avg_ppl:+.2f}% | {avg_frob:.4f} |\n")
    
    with open(output_path / "phase2_report.md", "w") as f:
        f.writelines(report)
    
    print(f"Results saved to {output_path}/")
    print(f"  - phase2_report.md")
    print(f"  - phase2_results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    
    run_phase2(args.model, args.output)

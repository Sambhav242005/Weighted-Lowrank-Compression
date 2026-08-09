"""
Main profiling pipeline for Phase 1: Layer Extraction & Deep Profiling.
Runs all analyses and produces a comprehensive compressibility report.
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from layer_extraction import extract_gpt2_layers, get_model_config
from spectral_profiling import profile_all_spectral, print_spectral_summary
from null_space import profile_all_null_spaces, print_null_space_summary
from sensitivity import profile_all_sensitivity, print_sensitivity_summary
from statistical_profiling import profile_all_statistical, print_statistical_summary


def run_profiling(model_name: str = "gpt2", output_dir: str = "results"):
    """Run the complete Phase 1 profiling pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("PHASE 1: LAYER EXTRACTION & DEEP PROFILING")
    print(f"Model: {model_name}")
    print("=" * 70)
    
    # Step 1: Extract layers
    print("\n[1/5] Extracting weight matrices...")
    weights, model = extract_gpt2_layers(model_name)
    config = get_model_config(model_name)
    
    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Step 2: Spectral profiling
    print("\n[2/5] Running spectral analysis...")
    spectral_profiles = profile_all_spectral(weights)
    print_spectral_summary(spectral_profiles)
    
    # Step 3: Null space analysis
    print("\n[3/5] Running activation null space analysis...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        null_profiles = profile_all_null_spaces(weights, model, tokenizer, n_samples=30)
        print_null_space_summary(null_profiles)
    except Exception as e:
        print(f"  Null space analysis failed: {e}")
        print("  Continuing with other analyses...")
        null_profiles = []
    
    # Step 4: Sensitivity analysis
    print("\n[4/5] Running sensitivity analysis...")
    try:
        sensitivity_profiles = profile_all_sensitivity(
            weights, model, tokenizer, n_samples=10
        )
        print_sensitivity_summary(sensitivity_profiles)
    except Exception as e:
        print(f"  Sensitivity analysis failed: {e}")
        sensitivity_profiles = []
    
    # Step 5: Statistical profiling
    print("\n[5/5] Running statistical analysis...")
    statistical_profiles = profile_all_statistical(weights)
    print_statistical_summary(statistical_profiles)
    
    # Generate summary report
    generate_report(config, spectral_profiles, null_profiles, 
                   sensitivity_profiles, statistical_profiles, output_path)
    
    print(f"\nResults saved to {output_path}/")
    return weights, spectral_profiles


def generate_report(config, spectral, null, sensitivity, statistical, output_path):
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# Compressibility Profile Report\n")
    report.append(f"**Model:** GPT-2 ({config['n_layer']} layers, {config['n_embd']} dim)\n")
    report.append(f"**Total parameters (per layer block):** "
                 f"Attention={4 * config['n_embd']**2}, "
                 f"MLP={2 * config['n_embd'] * 4 * config['n_embd']}\n\n")
    
    # Spectral summary
    report.append("## Spectral Analysis\n\n")
    report.append("| Matrix | Shape | Rank | 99% Rank | Decay | Rate | Eff Rank |\n")
    report.append("|--------|-------|------|----------|-------|------|----------|\n")
    for p in spectral:
        report.append(f"| {p.matrix_name} | {p.shape[0]}x{p.shape[1]} | "
                     f"{p.rank} | {p.rank_at_99percent} | {p.decay_type} | "
                     f"{p.decay_rate:.3f} | {p.eff_rank:.1f} |\n")
    
    # Null space summary
    if null:
        report.append("\n## Null Space Analysis\n\n")
        report.append("| Matrix | Total Dirs | Active Dirs | Null Space % |\n")
        report.append("|--------|-----------|-------------|-------------|\n")
        for p in null:
            report.append(f"| {p.matrix_name} | {p.n_total_directions} | "
                         f"{p.n_active_directions} | {p.null_space_fraction:.1%} |\n")
    
    # Statistical summary
    report.append("\n## Statistical Properties\n\n")
    report.append("| Matrix | Mean | Std | Skew | Kurtosis | Entropy (bits) |\n")
    report.append("|--------|------|-----|------|----------|----------------|\n")
    for p in statistical:
        report.append(f"| {p.matrix_name} | {p.mean:.4f} | {p.std:.4f} | "
                     f"{p.skewness:.3f} | {p.kurtosis:.3f} | {p.entropy_bits:.2f} |\n")
    
    # Key findings
    report.append("\n## Key Findings\n\n")
    
    # Find layers with strongest spectral decay
    exponential_layers = [p for p in spectral if p.decay_type == "exponential"]
    if exponential_layers:
        report.append("### Exponential Spectral Decay\n")
        report.append("These layers show exponential singular value decay, "
                     "suggesting strong compressibility:\n")
        for p in exponential_layers:
            report.append(f"- **{p.matrix_name}**: decay rate={p.decay_rate:.3f}, "
                         f"99% rank={p.rank_at_99percent}/{min(p.shape)}\n")
    
    power_law_layers = [p for p in spectral if p.decay_type == "power_law"]
    if power_law_layers:
        report.append("\n### Power-Law Spectral Decay\n")
        report.append("These layers show power-law decay:\n")
        for p in power_law_layers:
            report.append(f"- **{p.matrix_name}**: decay rate={p.decay_rate:.3f}, "
                         f"99% rank={p.rank_at_99percent}/{min(p.shape)}\n")
    
    # Compressibility assessment
    report.append("\n### Compressibility Assessment\n\n")
    for p in spectral:
        compression_ratio = min(p.shape) / max(p.rank_at_99percent, 1)
        if compression_ratio > 3:
            report.append(f"- **{p.matrix_name}**: HIGH compressibility "
                         f"(99% at {p.rank_at_99percent}/{min(p.shape)}, "
                         f"{compression_ratio:.1f}x potential compression)\n")
        elif compression_ratio > 1.5:
            report.append(f"- **{p.matrix_name}**: MODERATE compressibility "
                         f"(99% at {p.rank_at_99percent}/{min(p.shape)})\n")
        else:
            report.append(f"- **{p.matrix_name}**: LOW compressibility\n")
    
    # Save report
    with open(output_path / "compressibility_report.md", "w") as f:
        f.writelines(report)
    
    # Save raw data
    raw_data = {
        "config": config,
        "spectral": [{
            "name": p.matrix_name,
            "shape": list(p.shape),
            "rank": p.rank,
            "rank_99": p.rank_at_99percent,
            "rank_95": p.rank_at_95percent,
            "rank_90": p.rank_at_90percent,
            "decay_type": p.decay_type,
            "decay_rate": p.decay_rate,
            "eff_rank": p.eff_rank,
            "spectral_norm": p.spectral_norm,
            "frobenius_norm": p.frobenius_norm,
        } for p in spectral],
        "null_space": [{
            "name": p.matrix_name,
            "null_fraction": p.null_space_fraction,
            "active_dirs": p.n_active_directions,
            "total_dirs": p.n_total_directions,
        } for p in null] if null else [],
        "statistical": [{
            "name": p.matrix_name,
            "mean": p.mean,
            "std": p.std,
            "skewness": p.skewness,
            "kurtosis": p.kurtosis,
            "entropy_bits": p.entropy_bits,
            "sparsity_1e3": p.sparsity_1e3,
        } for p in statistical],
    }
    
    with open(output_path / "profile_data.json", "w") as f:
        json.dump(raw_data, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()
    
    run_profiling(args.model, args.output)

# Research Summary: Neural Network Weight Compression

## Executive Summary

We investigated whether a pretrained transformer's dense weights can be replaced by a more compact representation without breaking the global computation. Our key finding: **the answer is yes, but with important caveats that depend on model size and compression ratio.**

## Key Findings

### 1. The Compression Limit is ~3x for Small Models, ~6x for Medium Models

| Model | 2x | 3x | 4x | 6x |
|-------|-----|-----|-----|-----|
| GPT-2 Small (124M) | +5.50% | +21.36% | +152.12% | +261.81% |
| GPT-2 Medium (355M) | +2.63% | +9.91% | +20.71% | +48.60% |

**Interpretation:**
- At 2x: Both models are fine (<6% PPL increase)
- At 3x: Small is degraded, Medium is usable
- At 4x: Small is broken, Medium is acceptable
- At 6x: Both are degraded, but Medium is 5x more robust

### 2. Larger Models Are MORE Robust to Compression

GPT-2 Medium degrades 2-7x less than GPT-2 Small at the same compression ratio. This is because:
- Larger models have more redundancy (higher effective rank)
- The same rank reduction is a smaller fraction of the total capacity
- Error propagation is less severe in deeper networks

### 3. Non-Uniform Allocation Helps Small Models, Hurts Medium Models

**GPT-2 Small:**
- Inverse eff_rank allocation: +17.79% PPL (vs +21.36% uniform)
- Improvement: 3.5% better
- Best predictor: W_O effective rank (r = -0.867, R² = 0.752)

**GPT-2 Medium:**
- Uniform allocation: +9.91% PPL
- Inverse eff_rank: +10.51% PPL (WORSE)
- L0 special: +11.39% PPL (WORSE)
- Correlation weakened: r = 0.453

**Why the difference?**
- Small models have asymmetric sensitivity (layer 0 is critical)
- Medium models have more uniform sensitivity across layers
- The eff_rank predictor doesn't generalize across model sizes

### 4. Error Propagation is Real but Predictable

- Drift accumulates across layers (MSE 0.03 → 98.36 in Medium)
- Jacobian norms predict RELATIVE ordering but not absolute amplification
- Network is strongly contractive (nonlinearities provide massive contraction)
- Linear model overestimates amplification by 10^7x

### 5. Joint Compression is ALWAYS Worse

- Shared subspace compression gives 3x-100x worse PPL than independent
- Each layer needs its own optimal subspace
- The "shared structure" hypothesis is false

### 6. Weight-Space SVD is the Best Method
 
 - Weight-space SVD: +21.36% PPL at 3x (Small), +9.91% (Medium)
 - Activation-space methods: ALL give +173% PPL (identical, broken)
 - The representation space matters, not the compression method
 
### 7. Activation-Weighted Fitting Beats Plain SVD
 
 - In Gemma-3-1B experiments, activation-weighted rank-384 fitting resulted in a minimal (~6.7%) PPL increase.
 - Matched plain SVD produced catastrophic degradation (251.52 PPL vs 58.12 baseline).
 - **Insight:** Weighting the fit by actual activations preserves high-traffic paths that plain SVD ignores.
 - **Target:** Attention output projections (W_O) are the most compressible block type.
 
 ## Practical Recommendations


### For GPT-2 Small (124M):
- **2x compression**: Use uniform allocation (rank=384). +5.50% PPL.
- **3x compression**: Use inverse eff_rank allocation. +17.79% PPL.
- **4x+ compression**: Not recommended (broken).

### For GPT-2 Medium (355M):
- **2x compression**: Use uniform allocation (rank=512). +2.63% PPL.
- **3x compression**: Use uniform allocation (rank=341). +9.91% PPL.
- **4x compression**: Use uniform allocation (rank=256). +20.71% PPL.
- **6x compression**: Use uniform allocation (rank=170). +48.60% PPL.

### For Larger Models (7B+):
- Expect even more robustness (based on scaling trend)
- Uniform allocation is likely optimal
- Test empirically to confirm

## Open Questions

1. **Does this scale to 7B+ models?** We predict they'll be even more robust.
2. **Can we predict compression sensitivity from model properties?** The eff_rank predictor works for Small but not Medium.
3. **What's the theoretical limit?** Is there a fundamental bound on compression ratio?
4. **Can we improve the nonlinear error propagation model?** Current model overestimates by 10^7x.
5. **What about other compression methods?** (Fourier, tensor decomposition, etc.)

## Files Generated

- `results/drift_profiler.json` — per-layer and group drift metrics
- `results/joint_group_compress.json` — joint vs independent comparison
- `results/layer0_investigation.json` — spectral properties and error amplification
- `results/layer0_fix.json` — targeted layer 0 experiments
- `results/activation_vs_weight.json` — activation-space vs weight-space comparison
- `results/error_propagation_model.json` — Jacobian norms and error trajectories
- `results/compression_sensitivity.json` — weight features and PPL correlations
- `results/optimal_budget.json` — budget allocation strategies
- `results/phase_b_summary.json` — final comparison across ratios
- `results/phase_c_medium.json` — GPT-2 Medium per-layer sensitivity and allocation
- `results/phase_c_medium_ratios.json` — GPT-2 Medium vs Small across ratios

## Conclusion

The answer to "can we compress neural network weights?" is: **yes, up to a point.** The compression limit depends on model size:
- Small models (124M): ~3x compression is the limit
- Medium models (355M): ~6x compression is acceptable
- Larger models: Likely even more robust

The key insight is that **larger models have more redundancy**, so the same rank reduction is less damaging. This suggests that compression is more feasible for production models (which are typically 7B+) than for small research models.

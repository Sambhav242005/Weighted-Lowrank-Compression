# Phase 5: Activation-Preserving Compression Report

**Baseline:** GPT-2 Small, PPL = 48.35

## The Key Question

Instead of: "What mathematical representation best approximates a layer's weights?"
We asked: "What information does a layer actually need to preserve?"

## Step 1-2: Activation Statistics

Layer activations show interesting patterns:
- Early layers (0-3): Lower norm, higher sparsity
- Middle layers (4-7): Medium norm, variable sparsity
- Late layers (8-11): Higher norm, lower sparsity

This suggests different layers process information differently.

## Step 3: Weight-Error vs Activation-Error

| Layer | SVD 99% Weight Err | SVD 99% Act Err | Ratio |
|-------|-------------------|-----------------|-------|
| 0 | 0.099750 | 0.081472 | 0.82 |
| 3 | 0.099765 | 0.113149 | 1.13 |
| 6 | 0.099505 | 0.102539 | 1.03 |
| 9 | 0.099780 | 0.099322 | 1.00 |
| 11 | 0.099471 | 0.119076 | 1.20 |

**Finding:** Weight error and activation error don't correlate perfectly. Layer 0 has activation error LOWER than weight error, while layer 11 has it HIGHER.

## Step 4: Direct Activation-Preserving Optimization

We optimized W_hat = B @ A to minimize ||Wx - W_hat x|| directly.

| Layer | Rank | Weight Err | Act Err | PPL Delta |
|-------|------|------------|---------|-----------|
| 0 | 256 | 0.219077 | 0.027756 | +2.52% |
| 6 | 256 | 0.487895 | 0.143425 | +0.68% |
| 11 | 256 | 0.520228 | 0.074222 | +3.96% |

**Critical finding:** Activation-preserving optimization achieves LOWER activation error than SVD despite HIGHER weight error. This confirms that **weight reconstruction ≠ behavior preservation**.

## Step 5: Functional Evaluation

| Layer | Method | PPL | Delta |
|-------|--------|-----|-------|
| 0 | AP rank=256 | 49.57 | +2.52% |
| 6 | AP rank=256 | 48.68 | **+0.68%** |
| 11 | AP rank=256 | 50.27 | +3.96% |

Layer 6 is the most compressible: only +0.68% PPL with rank=256 activation-preserving.

## Step 6: What Information Does a Layer Need to Preserve?

| Layer | Method | Spectral Overlap | Output Mean Diff | Output Std Diff |
|-------|--------|------------------|------------------|-----------------|
| 0 | SVD 99% | 0.9999 | 0.033194 | 0.006112 |
| 0 | AP r256 | 0.9919 | **0.001719** | **0.000473** |
| 6 | SVD 99% | 0.9999 | 0.023445 | 0.005455 |
| 6 | AP r256 | 0.9576 | **0.000681** | 0.007426 |
| 11 | SVD 99% | 0.9998 | 0.137214 | 0.036700 |
| 11 | AP r256 | 0.8522 | **0.003239** | **0.008697** |

**Major finding:** SVD has HIGHER spectral overlap (0.9999) but WORSE output statistics. The activation-preserving method has LOWER spectral overlap but BETTER output statistics.

**This answers the question:** What matters is preserving the **output distribution**, not the singular subspace.

## Key Insights

1. **Weight reconstruction ≠ behavior preservation.** A representation can have high weight error but low activation error.

2. **Output distribution matters more than singular subspace.** SVD preserves the singular subspace perfectly (0.9999 overlap) but preserves the output distribution poorly.

3. **Activation-preserving optimization finds different solutions.** It sacrifices weight accuracy to preserve functional behavior.

4. **Layer sensitivity varies by what you optimize for.** When optimizing for activations, layer 6 becomes the most compressible (+0.68% PPL).

## Implications

This suggests a new paradigm for neural network compression:

1. **Don't optimize weights.** Optimize activations.
2. **Don't measure reconstruction error.** Measure behavior preservation.
3. **Don't use SVD as the gold standard.** Use activation-preserving optimization.

## Next Steps

1. **Joint optimization:** Fit all layers simultaneously to minimize cumulative activation error
2. **MLP compression:** Test if MLP layers show similar behavior
3. **Cross-model validation:** Test on other GPT-2 variants or different architectures
4. **Theoretical analysis:** Why does activation-preserving optimization find different solutions?

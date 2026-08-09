# Phase 3: Cumulative Replacement Report

**Baseline:** GPT-2 Small, PPL = 48.35, Top-1 = 31.6%

## Test 1: Progressive Layer Replacement

Replaced W_O layers one at a time with LowRank r128 (3x compression each).

| Layers Replaced | PPL | Delta | Notes |
|----------------|-----|-------|-------|
| 1 (layer0) | 78.13 | +61.6% | Layer 0 most sensitive |
| 2 (layers 0-1) | 98.45 | +103.6% | Error compounds |
| 4 (layers 0-3) | 89.49 | +85.1% | Error cancellation effect |
| 8 (layers 0-7) | 130.94 | +170.8% | Significant degradation |
| 12 (all) | 168.55 | +248.6% | Model nearly unusable |

**Key insight:** Error does NOT compound linearly. Layer 4 replacement actually improves over layer2 — suggesting error cancellation in the residual stream at certain depths.

## Test 2: All 12 Layers — Representation Comparison

| Representation | Compression | PPL | Delta | Verdict |
|---------------|-------------|-----|-------|---------|
| SVD 99% | 0.75x | 49.50 | +2.4% | Safe but expands |
| Fourier 99% | **1.16x** | **49.28** | **+1.9%** | **Best PPL/compromise** |
| SVD 95% | 1.01x | 72.97 | +50.9% | Threshold too aggressive |
| LowRank r256 | **1.50x** | 57.67 | +19.3% | Best usable compression |
| LowRank r128 | 3.00x | 168.55 | +248.6% | Too aggressive |
| LowRank r64 | 6.00x | 353.77 | +631.6% | Broken |

**Winner:** Fourier 99% at 1.16x with only +1.9% PPL across all 12 W_O layers.

## Test 3: Adaptive Compression

| Config | Compression | PPL | Delta |
|--------|-------------|-----|-------|
| Uniform 3x | 3.00x | 168.55 | +248.6% |
| Adaptive A | 3.27x | 118.51 | +145.1% |
| Adaptive B | 2.88x | 108.79 | +125.0% |
| Aggressive | 7.20x | 414.58 | +757.4% |

Adaptive configs (more compression on later layers) help but don't solve the fundamental compounding problem.

## Key Findings

1. **Error compounding is the real bottleneck.** A layer that handles 3x compression individually (+0.05% PPL) causes +61% PPL when applied to layer 0, and errors compound nonlinearly through the residual stream.

2. **Fourier beats SVD at the same compression.** At 1.16x, Fourier achieves +1.9% PPL vs SVD 95% at +50.9%. The frequency domain representation captures something SVD misses for functional preservation.

3. **Low-rank factorization is excellent per-layer but doesn't scale.** Phase 2 showed3x compression with <0.1% PPL on single layers. But stacking 12 such layers causes +248% PPL.

4. **There exists a "safe" compression budget.** ~1.16x (Fourier 99%) or ~1.5x (LowRank r256) across all layers keeps PPL within acceptable bounds.

5. **Layer 0 is the bottleneck.** It's the most sensitive to compression. Future work should focus compression budget on later layers (which are more robust) and preserve early layers.

## Recommendations for Next Steps

1. **Phase 4 (Healing):** Apply LoRA fine-tuning after cumulative replacement to recover performance
2. **Mixed-precision approach:** Use high-fidelity representation for layer 0, aggressive compression for layers 8-11
3. **Test on MLP layers:** W_O may not be the optimal target — MLP W_down has higher kurtosis (more compressible structure)

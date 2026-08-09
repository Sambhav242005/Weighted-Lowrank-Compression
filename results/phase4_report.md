# Phase 4: Healing via LoRA Fine-Tuning Report

**Baseline:** GPT-2 Small, PPL = 48.35

## Test 1: LoRA Healing After Full Replacement

All 12 W_O layers replaced with LowRank r128 (3x compression). Then LoRA adapters added and trained for 50 steps on WikiText-2.

| LoRA Rank | Params | PPL Before | PPL After | Recovery |
|-----------|--------|------------|-----------|----------|
| 4 | 73K | 168.55 | 153.67 | 12.4% |
| 8 | 147K | 168.55 | 136.71 | 26.5% |
| 16 | 295K | 168.55 | 132.04 | 30.4% |

**Finding:** LoRA can partially recover lost performance, but recovery is limited (~30% max). The compression damage is too severe for adapters to fully compensate.

## Test 2: Mixed-Precision (Preserve Early Layers)

Keep early layers untouched, compress later layers with LowRank r128.

| Config | Compressed | PPL | Delta |
|--------|------------|-----|-------|
| Preserve L0-L3 | 8/12 | 70.38 | +45.6% |
| Preserve L0-L5 | 6/12 | 66.01 | +36.5% |
| Preserve L0-L7 | 4/12 | 61.71 | +27.6% |
| Preserve L0-L1, L10-L11 | 8/12 | 60.79 | +25.7% |

**Winner:** Preserving layers 0-1 and 10-11 (boundary layers) achieves +25.7% PPL while compressing 8/12 layers. This is 4x better than full compression (+248.6%).

## Test 3: Mixed-Precision + LoRA

Surprisingly, LoRA healing on mixed-precision model made performance WORSE (70.38 → 92.58 PPL). Possible causes:
1. Overfitting on small WikiText-2 subset (100 texts)
2. LoRA interference with partially-compressed layers
3. Learning rate too high for fine-tuning

## Key Findings

1. **LoRA healing has limited effectiveness.** Even rank-16 adapters only recover 30% of lost performance after 3x compression across all layers.

2. **Mixed-precision is the real solution.** Preserving early layers (0-1) and late layers (10-11) while compressing middle layers achieves reasonable compression with much better preservation.

3. **Layer sensitivity varies dramatically.** Layer 0 is the bottleneck — compressing it causes +61% PPL alone. Later layers are more robust.

4. **Error compounding is nonlinear.** 4-layer replacement (85% PPL) is actually better than 2-layer (103% PPL), suggesting error cancellation in the residual stream.

## Recommendations

1. **Do not replace all layers.** Focus compression on layers 4-9 which are more robust.

2. **Preserve boundaries.** Layers 0-1 (input) and 10-11 (output) should use high-fidelity representations.

3. **LoRA healing needs more data.** The 100-text WikiText-2 subset was insufficient. Full dataset training may help.

4. **Consider MLP compression.** W_O layers may not be the optimal target — MLP W_down has higher kurtosis (more compressible structure).

## Final Compression Budget

For ~1.5x overall compression with <30% PPL increase:
- Layers 0-1: Original weights (no compression)
- Layers 2-9: LowRank r256 (1.5x compression)
- Layers 10-11: Original weights (no compression)

This achieves 1.5x compression on 8/12 layers while preserving critical boundary layers.

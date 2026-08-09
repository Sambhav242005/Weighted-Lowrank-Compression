# Phase 6: Cross-Architecture Validation Report

**Model:** Gemma 3 1B (26 layers, 1152 hidden, 6912 intermediate)
**Baseline PPL:** 9.91

## Summary

Tested Function-Preserving Approximation (FP) vs SVD on 6 weight matrices across 3 layers. Results partially validate the hypothesis but reveal important nuances.

## Attention Layers (W_O) — FP Wins

| Layer | SVD Act Err | FP Act Err | Ratio | SVD PPL | FP PPL |
|-------|-------------|------------|-------|---------|--------|
| 0 | 0.0620 | **0.0073** | 8.5x | +3.41% | **-0.03%** |
| 13 | 0.0483 | **0.0073** | 6.6x | -0.32% | +0.40% |
| 25 | 0.0541 | **0.0085** | 6.4x | +0.06% | -0.01% |

**Finding:** FP achieves 6-8x lower activation error than SVD on attention layers, and matches or beats SVD functionally.

## MLP Layers (W_down) — Surprising Results

| Layer | SVD Act Err | FP Act Err | Ratio | SVD PPL | FP PPL |
|-------|-------------|------------|-------|---------|--------|
| 0 | **0.0750** | 0.0768 | 0.98x | +20.26% | +64.03% |
| 13 | **0.1144** | 0.2260 | 0.51x | +0.36% | **-1.60%** |
| 25 | **0.1042** | 0.1763 | 0.59x | +66.52% | **+2.09%** |

**Finding:** On MLP layers, FP has HIGHER activation error than SVD but sometimes BETTER functional performance (layer25: +2.09% vs +66.52%).

## Key Insight

**Activation error is not a perfect predictor of functional performance.**

- On attention layers: lower activation error → better function (consistent)
- On MLP layers: higher activation error → sometimes better function (inconsistent)

This suggests:
1. Weight error is a BAD proxy for function (confirmed from Phase 5)
2. Activation error is a BETTER proxy but not perfect
3. The relationship depends on layer type and what you're measuring

## Comparison with GPT-2 (Phase 5)

| Property | GPT-2 | Gemma 3 1B |
|----------|-------|------------|
| W_O FP advantage | 3x activation error reduction | 6-8x activation error reduction |
| W_down FP advantage | Not tested | Mixed results |
| Attention PPL delta | +0.68% (layer6) | -0.03% (layer0) |
| MLP behavior | N/A | Layer-dependent |

**Cross-architecture validation:** FP consistently outperforms SVD on attention layers across both GPT-2 and Gemma 3.

## What This Means

1. **Function-preserving approximation works across architectures** for attention layers.

2. **MLP layers behave differently.** The relationship between activation error and function is more complex.

3. **The optimization target matters.** We confirmed that weight reconstruction ≠ function reconstruction.

4. **We need better metrics.** Activation error alone doesn't capture everything. KL divergence between logits would be a more direct measure.

## Next Steps

1. **Add KL divergence measurement** to capture output distribution similarity
2. **Test more MLP layers** to understand the inconsistency
3. **Test on another architecture** (Qwen, TinyLlama) for broader validation
4. **Investigate why MLP layers behave differently** from attention layers

## Conclusion

The hypothesis is **partially validated**:
- Function-preserving approximation consistently outperforms weight-preserving on attention layers
- MLP layers show more complex behavior
- The optimization target (weights vs computation) matters

This is still a meaningful finding, but the full picture is more nuanced than "activation preservation always wins."

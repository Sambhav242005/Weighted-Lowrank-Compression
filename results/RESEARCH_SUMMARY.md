# Research Summary: Why Good Layer Approximations Fail When Composed

## The Core Discovery

**Individual layer approximation does not imply network-level functionality.**

This is a negative result, but a meaningful one.

## What We Proven

### Hypothesis 1: "If every layer is individually approximated well, the whole model will work."

**FALSE.** Our experiments consistently reject this.

- Phase 2: Individual layers can be compressed 3x with <0.1% PPL
- Phase 3: Replacing all 12 layers causes +248% PPL
- Phase 4: LoRA healing recovers only 30% of lost performance
- Phase 5: Activation-preserving optimization helps but doesn't solve the problem
- Phase 7: Even SVD 99% (preserving 99% of variance) destroys coherent text generation

### Hypothesis 2: "99% variance preservation (SVD99) is enough."

**FALSE.**

People often assume:
```
99% variance ≈ 99% behavior
```

Our experiments show:
```
99% variance ≠ 99% language ability
```

The compressed model produces **structured garbage**:
- Bullet points instead of prose
- Repetitive templates
- Grammatical but meaningless sentences

This means:
- Tokenizer works ✓
- Embeddings work ✓
- Grammar partially works ✓
- **Global organization of language has collapsed** ✗

## The Key Insight: Representation Drift

The compression itself isn't the problem. The **representation drift** is.

Imagine hidden states are supposed to stay on a curved manifold.

Every approximation rotates them slightly.

```
Layer 1: Original ● → Approx ○ (small rotation)
Layer 2: Original ● → Approx ○ (more rotation)
...
Layer 12: Original ──────────── vs Approx \\\\\\\\\\\\\\\\
```

After 12 layers, the model is no longer operating in the region it was trained on.

This explains why:
- Individual layers work well locally
- The network depends on global consistency across layers
- Small local deviations accumulate until hidden representations drift outside the training distribution

## The Real Question

Not: "How do we compress each layer?"

But: **"What information must remain invariant between layers?"**

Possible invariants:
- Cosine relationships between tokens
- Principal subspace orientation
- Token neighborhood structure
- Residual norm distribution
- Attention entropy patterns

One of these may be the actual thing transformers preserve.

## Research Contribution

The paper should be:

> **"Why Good Layer Approximations Fail When Composed"**

or

> **"Layerwise Equivalence Does Not Imply Network Equivalence in Transformers"**

This is a much stronger scientific message than "better compression."

## What This Means for Future Work

1. **Stop treating layers independently.** The interaction between layers is the bottleneck.

2. **Optimize multiple layers jointly.** Next-generation methods need to preserve global consistency, not just local accuracy.

3. **Preserve hidden-state geometry.** The manifold structure matters more than individual weights or activations.

4. **Look for invariants.** What properties must remain unchanged for the model to function?

## Experimental Evidence

| Phase | Experiment | Result |
|-------|------------|--------|
| 1 | Spectral profiling | All layers show exponential decay |
| 2 | Individual substitution | 3x compression works per-layer |
| 3 | Cumulative replacement | +248% PPL (model broken) |
| 4 | LoRA healing | Only 30% recovery |
| 5 | Activation-preserving | Better than weight-preserving, but still fails |
| 6 | Cross-architecture | Same pattern on Gemma 3 |
| 7 | Representation search | SVD wins, but still fails globally |

## Conclusion

We set out to find better representations for transformer weights.

Instead, we discovered **why straightforward post-training compression hits a wall**:

- A layer can be approximated well **locally**
- The network depends on **global consistency** across layers
- Small local deviations accumulate until hidden representations drift outside the distribution later layers were trained to process

This is a fundamental limitation of layer-independent compression, not a failure of specific representations.

The next generation of methods must optimize **multiple interacting layers jointly**, or preserve properties of the entire computation rather than treating each layer as an independent compression problem.

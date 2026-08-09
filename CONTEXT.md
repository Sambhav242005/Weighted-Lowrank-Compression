# Domain Model: What Do Weights Actually Represent

## Core Question

Not "how to compress weights" but **"what mathematical object do the weights actually represent?"**

The journey: Weights → Layer → Function → Geometry → Information

## Canonical Terms

### Representation
**Definition:** The mathematical object that replaces a dense weight matrix.
**Examples:** Low-rank factors (U, S, V), polynomial coefficients, hypernetwork parameters, implicit neural representation weights, spectral basis coefficients.
**NOT:** The weight matrix itself. A representation is an *alternative* encoding that can reconstruct or replace the original.

### Invariant
**Definition:** A property of the hidden-state stream that must be preserved across layers for the model to function.
**Candidates:** Cosine geometry between tokens, principal subspace orientation, residual norm distribution, attention entropy, token neighborhood structure, logit distribution shape.
**Why it matters:** If we know the invariants, we know what the weights *actually compute* — not what numbers they store.

### Drift
**Definition:** The accumulation of hidden-state deviation across successive compressed layers.
**Formal:** For teacher hidden states h_1, ..., h_L and student hidden states h'_1, ..., h'_L, drift at layer l is some distance metric d(h_l, h'_l) that grows (often nonlinearly) with l.
**Key finding:** Drift is the reason layer-independent compression fails. Small local errors compound until the hidden state exits the training distribution.

### Composition
**Definition:** Whether layer-level approximations survive stacking through the full network.
**Key result:** Layerwise equivalence does NOT imply network equivalence. Individual layer compression works; cumulative replacement fails.
**Implication:** The research problem is not "find a better per-layer approximation" but "find what must be preserved globally."

### Manifold
**Definition:** The geometric structure of the hidden-state space as data flows through the network.
**Connection to drift:** Each layer maps the input manifold to an output manifold. Compression that distorts this mapping causes the hidden states to leave the manifold later layers expect.
**Connection to invariants:** The invariants ARE the manifold properties. What the weights "represent" is the manifold transformation.

### Function
**Definition:** What a layer *does* to its input, abstracted from the specific weight values.
**Key insight:** Two different weight matrices can implement the same function. The function is the thing; the weights are one realization.
**Connection to representation:** A good representation captures the function, not the weights.

### Information
**Definition:** The meaning-bearing structure that survives the transformation through the network.
**Ultimate question:** What information must remain invariant for the model to produce coherent output?
**Connection to geometry:** Information is geometric. The weights encode a transformation of information-geometric structure.

## The Hypothesis Space

| Hypothesis | Implication | Status |
|-----------|-------------|--------|
| Weights are the thing | Compression is fundamentally limited | Rejected by evidence |
| Functions are the thing | Find functional representations | Partially supported (Phase 5) |
| Geometry is the thing | Preserve manifold structure | Under investigation |
| Information is the thing | Preserve invariants | The real question |

## Evidence Summary

1. **Phase 1-4:** Per-layer compression works; cumulative fails (+248% PPL)
2. **Phase 5:** Activation-preserving > weight-preserving, but still fails globally
3. **Phase 6:** Same pattern on Gemma 3 — architecture-independent
4. **Core insight:** Representation drift accumulates. The weights aren't the thing; the information geometry is the thing.

## What's Next

The project must answer: **What is the minimal mathematical description that captures the information-geometric structure of what a transformer learns — independent of the weight representation it's currently stored in?**

## Related Work (Literature)

### Implicit Neural Representations
- SINR (Jayasundara et al., 2025): Weight spaces of INRs follow Gaussian distributions; sparse coding in weight space achieves compression
- CoINR: Compressed INRs across modalities
- INR compression (Strümpler et al., 2021): First comprehensive INR compression pipeline

### Weight Generation
- DeepWeightFlow (2025): Flow matching for generating neural network weights; canonicalization helps for high-dimensional weights
- HyperNetworks (Ha et al., 2017): Meta-networks that generate weights for target networks
- SANE (Schürholt et al., 2024): Kernel density sampling of hypernetwork latents for complete weight generation

### Manifold Geometry
- "Walking the Weight Manifold" (2025): Topological approach to weight manifolds; even simple parametric manifolds can capture task structure
- nGPT (2024): Normalized transformers with representation learning on the hypersphere
- "Different Layers, Different Manifolds" (ICML 2026): Module-wise weight-space geometry in transformers
- Intrinsic dimension of data representations (Ansuini et al., 2019): ID evolves through layers

### Loss Landscape Geometry
- Fort & Jastrzębski (2019): Loss landscape as high-dimensional wedges; intrinsic dimension connects to optimization
- Draxler et al. (2018): Essentially no barriers in neural network energy landscape

### Residual Weight Sharing
- ResidualTransformer (Wang & Li, 2024): Shared full-rank component + unique low-rank residuals across layers

## Open Questions for the Project

1. Can we measure the intrinsic dimension of the hidden-state manifold at each layer?
2. Does drift correlate with intrinsic dimension change?
3. Can we find a manifold parametrization that stays invariant under compression?
4. Is there a "weight manifold" similar to the loss landscape geometry?
5. Can hypernetwork-style weight generation capture the function without storing the weights?

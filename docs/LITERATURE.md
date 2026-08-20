# Literature: Weight Representation & Information Geometry

## Key Papers Found (August 2026)

### 1. Weight Space Geometry
- **"Walking the Weight Manifold" (2025)** — Topological approach to conditioning via weight manifolds. Shows even simple parametric manifolds (lines, ellipses) can capture task structure. Uses metric tensor for optimization.
- **"Different Layers, Different Manifolds" (ICML 2026)** — Module-wise weight-space geometry in transformers. Different layers live on different manifolds.
- **nGPT (2024)** — Normalized transformers: all vectors on unit hypersphere. Matrix-vector multiplications become cosine similarities. Representation learning on hypersphere → more stable training.

### 2. Implicit Neural Representations
- **SINR (Jayasundara et al., 2025)** — Weight spaces of INRs follow Gaussian distributions across modalities. Sparse coding in weight space via compressed sensing. Random sensing matrix controlled by seed.
- **CoINR** — Compressed INRs, extension of SINR.
- **INR Compression (Strümpler et al., 2021)** — First comprehensive INR compression pipeline with quantization, retraining, entropy coding.

### 3. Weight Generation
- **DeepWeightFlow (2025)** — Flow matching for generating neural network weights. Canonicalization helps for high-dimensional weights. PCA as effective compression strategy for O(100M) params.
- **SANE (Schürholt et al., 2024)** — Kernel density sampling of hypernetwork latents for complete weight generation, layer-wise autoregressive.
- **HyperNetworks (Ha et al., 2017)** — Original: meta-networks generating target network weights.
- **"Brief Review of Hypernetworks" (Chauhan et al., 2024)** — Comprehensive survey: weight compression via hypernets, task-conditioned generation.

### 4. Loss Landscape & Intrinsic Dimension
- **Fort & Jastrzębski (2019)** — Loss landscape as high-dimensional wedges. Intrinsic dimension connects to optimization trajectory.
- **"Intrinsic Dimension of Data Representations" (Ansuini et al., 2019)** — ID evolves through layers; fundamental geometric property of representations.
- **"Intrinsic Dimension of Neural Network Ensembles" (2025)** — Manifolds of ensembles; lower ID → more susceptible to perturbations.

### 5. Residual Weight Structure
- **ResidualTransformer (Wang & Li, 2024)** — Shared full-rank component + unique low-rank residuals across layers. Weights don't differ much across consecutive layers.

### 6. Circuits/Interpretability
- **"When Models Manipulate Manifolds" (Transformer Circuits, 2025)** — Geometry of a counting task; manifold parametrization reduces interpretation burden.

## Cross-Layer Error Accumulation & Compensation (added 2026-08-09)

These papers directly address our drift/composition problem. **Correction to the gaps above:** Gap #2 ("no systematic drift profiling exists") is partially false — drift *has* been profiled for quantization, and there are working fixes. Nobody has done it for *low-rank representation replacement*, which keeps our niche but narrows the framing.

### 1. QEP: Quantization Error Propagation (Arai & Ichikawa, arXiv 2504.09629, NeurIPS 2025)
- **Directly our problem, for quantization.** Shows error grows ~exponentially with depth even through full-precision layers downstream of quantized ones (their Fig. 1, LLaMA2-7B, first 10 blocks quantized).
- **Root cause identified:** standard layer-wise objective `min ||W_l X_l - W_hat_l X_l||` fits each layer on *teacher* inputs, but at inference the layer receives *drifted* inputs `X_hat_l`.
- **Fix:** change objective to `min ||W_l X_l - W_hat_l X_hat_l||` — fit each layer sequentially against the student's own upstream outputs. Closed-form continuous optimum: `W_l* = W_l + alpha_l * W_l * delta_l * X_hat_l^T * H_hat_l^{-1}` where `delta_l = X_l - X_hat_l` is the accumulated upstream error; then quantize/fit against the corrected weight. Tunable `alpha_l in [0,1]` controls overfitting (alpha=0 recovers standard layer-wise).
- **Results:** consistent PPL gains across RTN/GPTQ/AWQ on LLaMA2-7B/13B/70B; dramatic at 2-bit (e.g. RTN INT2g128 7B: 4270 -> 35 PPL). Negligible runtime overhead; orthogonal to any layer-wise PTQ method.

### 2. EoRA: Training-free Eigenspace Low-Rank Compensation (Liu et al., NVIDIA, arXiv 2410.21271, ICLR 2026)
- Instead of changing the fit, **adds a low-rank residual path** `W_hat X + B A X` per layer, where `B A` approximates the compression error `Delta W = W - W_hat` *in the eigenspace of input activations* (PCA of `X X^T`, project `Delta W` there, truncate SVD, project back).
- Key theorem: in that eigenspace, dropping the smallest projected singular value is provably optimal for the layer-wise *functional* loss `||Delta W X - B A X||_F` — unlike naive SVD of `Delta W`.
- Training-free, minutes, no gradients; also serves as better LoRA initialization for fine-tuning.
- **Tradeoff for us:** compensation rank consumes part of the compression budget.

### 3. BRECQ: Block Reconstruction (Li et al., ICLR 2021)
- Second-order error analysis: layer-wise objective misses cross-layer terms; end-to-end overfits. Reconstruction at **block granularity (1-4 transformer blocks)** balances cross-layer dependency vs generalization.
- Note: BRECQ-style joint fitting optimizes *independent* representations jointly against block output — unlike our failed joint compression, which forced a *shared input subspace* (a harmful architectural constraint, per THREAD.md Phase A).

### 4. Context: in-layer sequential compensation
- **GPTQ (Frantar et al., 2022)** / **SparseGPT (Frantar & Alistarh, 2023)**: compensate error *within* one layer sequentially via approximate second-order (Hessian) updates. Explains why they survive ratios where naive RTN/SVD dies — but they still use teacher inputs, so cross-layer drift remains (which is exactly what QEP patches).

### Implication for this project
Our Phase 5 "activation-preserving" fit was already a step toward these methods but kept teacher inputs. The missing variable is **conditioning each layer's fit on the drifted inputs it will actually receive** (QEP idea) — untested in this repo for low-rank replacement.

## Gaps in Literature (What Your Project Uniquely Addresses)

1. **Composition failure**: Most work studies individual layers or individual networks. Your project studies what happens when approximations stack.
2. **Drift measurement**: No systematic drift-vs-depth profiling across *low-rank/alternative-representation* compression methods exists (quantization drift was profiled by QEP, arXiv 2504.09629).
3. **Invariant identification**: Which geometric properties actually matter for global behavior is unknown.
4. **Cross-architecture invariance**: Same drift pattern on GPT-2 and Gemma 3 suggests something deeper, but no one has isolated what.
5. **Post-training representation**: Most work either trains from scratch or compresses per-layer. Joint post-training representation fitting is unexplored.

## Key Insight for Your Project

The literature shows that:
- Weight spaces have structure (Gaussian distributions, manifold geometry)
- This structure can be exploited for compression (SINR, hypernets)
- But NOBODY has shown that this structure survives composition across layers
- Your drift finding is the missing piece: the structure exists locally but doesn't compose globally

This positions your work as: **"We know weight spaces have structure. We now know why that structure doesn't compose. Here's what must be preserved instead."**

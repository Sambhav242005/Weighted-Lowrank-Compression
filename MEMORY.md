# Memory: ResearchCompression Findings

## Status: Phase A+B+C Complete — Scaling Trend Confirmed (2026-08-08)

### Core Discovery (Confirmed & Extended)
Individual layer approximation does NOT imply network-level functionality.
Representation drift accumulates across layers until hidden states exit training distribution.

### Phase A Key Findings (2026-08-08)

**Layer 0 is the bottleneck** — not because of its spectrum (it has the most PEAKY spectrum, eff_rank=353.7), but because its error PROPAGATES through all 12 layers and AMPLIFIES (MSE 0.018 at L1 → 0.031 at L11).

**Composition is superlinear** — L0-L7 (+179.90%) > L0-L3 (+92.65%) + L4-L7 (+9.60%). Errors compound faster than the sum of individual errors.

**Shared structure is a myth** — joint compression with shared input subspace is ALWAYS worse than independent (by 3x-100x). Each layer has its own independent transformation.

**The compression limit is ~2-3x** — at rank=256-384, SVD achieves +5-20% PPL. At rank=128 (6x), no method works.

**Weight-space SVD is the best method** — activation-space approaches fail because they project onto a fixed subspace rather than learning a new mapping.

### Key Numbers (Phase A)
- Baseline PPL: 56.47
- Per-layer rank=128: layers 1-10 all <4% PPL delta
- Layer 0 rank=128: +65.52% PPL (catastrophic)
- All 12 layers rank=128: +261.81% PPL
- Budget-neutral best (L0=384, others=103): +89.46% PPL at 6x compression
- Weight SVD rank=384: +5.50% PPL (acceptable at 2x)
- Weight SVD rank=256: +21.36% PPL (acceptable at 3x)

### Error Propagation Patterns
- Layer 0 compressed: error AMPLIFIES (312Mx amplification)
- Layer 1 compressed: error DECAYS (0.00086 → 0.00045)
- Layer 5 compressed: error DECAYS (0.00177 → 0.00079)
- Layer 10 compressed: error stays local (0.005 at L11 only)

### Spectral Properties
- Layer 0: eff_rank=353.7 (most peaky, easiest to compress)
- Layer 10: eff_rank=607.0 (most even, hardest to compress)
- Paradox: easiest to compress causes most damage (because of propagation position)

### What the Literature Shows
- Weight spaces have structure (Gaussian, manifold geometry)
- This structure CAN be exploited (SINR, hypernets)
- But NOBODY has shown structure survives composition
- Your drift finding is the missing piece

### The Real Question (Reframed)
Original: "What is the minimal mathematical description that captures the information-geometric structure?"
Reframed: "Why does error propagation make compression non-composable, and can we predict which layers will cause the most damage?"

### Open Directions
1. Model error propagation mathematically — can we predict amplification rate?
2. Predict compression sensitivity before compressing
3. Test on larger models (GPT-2 Medium/Large, Gemma 3)
4. Explore non-SVD methods (Fourier, low-rank product, tensor decomposition)
5. Investigate layer 11 anomaly (MSE=0 but PPL+5.48%)

## Phase B Findings (2026-08-08)

### Sensitivity Predictor
- Best predictor: W_O effective rank (r = -0.867, R² = 0.752)
- Lower eff_rank → higher compression damage
- Layer 0: eff_rank=353.7 (lowest) → +65.52% PPL
- Layer 10: eff_rank=607.0 (highest) → +3.90% PPL

### Error Propagation Model
- Jacobian norms predict RELATIVE ordering but not absolute amplification
- Network is strongly contractive (nonlinearities provide massive contraction)
- Linear model overestimates amplification by 10^7x

### Optimal Allocation Results

| Ratio | Best Strategy | PPL Delta | Improvement |
|-------|--------------|-----------|-------------|
| 2x | Uniform (384) | +5.50% | — |
| 3x | Inverse eff_rank | +17.79% | 3.5% better |
| 4x | L0 special | +33.62% | 118% better |
| 6x | L0 special | +106.95% | 155% better |

### Key Insight
Non-uniform allocation matters MORE at higher compression ratios. At 2x it doesn't matter. At 4x+ it's the difference between usable and broken.

### Practical Recommendation
- At 2x compression: uniform allocation is fine
- At 3x compression: use inverse eff_rank allocation
- At 4x+ compression: give layer 0 special treatment (rank=384, others reduced)
- At 6x compression: model is still broken (+107%) — this is beyond the compression limit

## Phase C Findings (2026-08-08)

### Key Finding: Larger Models Are MORE Robust

| Ratio | Medium PPL | Small PPL | Robustness |
|-------|-----------|-----------|------------|
| 2x | +2.63% | +5.50% | 2.1x |
| 3x | +9.91% | +21.36% | 2.2x |
| 4x | +20.71% | +152.12% | 7.3x |
| 6x | +48.60% | +261.81% | 5.4x |

### Why Larger Models Are More Robust
- More redundancy (higher effective rank: 737 vs 354 for layer 0)
- Same rank reduction is smaller fraction of capacity
- Error propagation less severe in deeper networks

### Non-Uniform Allocation Doesn't Help Medium Models
- Small: inverse eff_rank gives 3.5% improvement at 3x
- Medium: uniform is BEST — non-uniform hurts performance
- Correlation weakened: r=0.453 (vs -0.867 for Small)

### Practical Implications
- GPT-2 Small: max usable compression is ~3x
- GPT-2 Medium: max usable compression is ~4-6x
- Larger models (7B+): likely even more robust
- Non-uniform allocation: only helps small models

### Open Directions
1. Test on 7B+ models — LLaMA, Mistral, Gemma
2. Develop model-size-aware compression strategy
3. Investigate why eff_rank predictor doesn't generalize
4. Test on different architectures (LLaMA, Mistral, etc.)
5. Build practical compression tool with adaptive allocation

## Phase C Part 3: GPT-2 Large Findings (2026-08-08)

### Scaling Trend Confirmed

| Model | Params | eff_rank | 3x PPL | 6x PPL |
|-------|--------|----------|--------|--------|
| Small | 124M | 354 | +21.36% | +261.81% |
| Medium | 355M | 737 | +9.91% | +48.60% |
| Large | 774M | 952 | +7.45% | +33.71% |

### Key Finding: It's a Scaling Issue, NOT Fundamental

The "broken" behavior in Small models is because they lack redundancy:
- Small: eff_rank ~354 (low redundancy) → can't survive compression
- Medium: eff_rank ~737 (medium redundancy) → tolerates 4x
- Large: eff_rank ~952 (high redundancy) → tolerates 6x

### Practical Compression Limits

| Model Size | Max Usable | Notes |
|------------|-----------|-------|
| 124M | ~3x | Layer 0 critical, needs non-uniform allocation |
| 355M | ~4x | Uniform allocation optimal |
| 774M | ~6x | Very robust, uniform allocation |
| 7B+ | ~6-8x (predicted) | Production models likely even more robust |

### Hardware
- NVIDIA RTX 4070 Ti (CUDA 13.2)
- Python 3.14, torch 2.13.0+cu132
- venv: `.venv-cuda/`

### Files
- `THREAD.md` — full experiment log
- `results/drift_profiler.json` — per-layer and group drift metrics
- `results/joint_group_compress.json` — joint vs independent comparison
- `results/layer0_investigation.json` — spectral properties and error amplification
- `results/layer0_fix.json` — targeted layer 0 experiments
- `results/activation_vs_weight.json` — activation-space vs weight-space comparison

## Drift-Aware Weighted Fit Breakthrough (2026-08-09)

- 3x full-stack c_proj (rank 256, 12 layers): plain SVD +21.34% ->
  **activation-weighted closed-form fit +3.58%** (alpha=0, beta=0.1).
- Formula: M* = C(G + beta·n·I)^-1, then rank-r truncation in G-weighted
  norm; G = X X^T from WikiText-2 TRAIN calibration (64x512 tokens).
- Critical bugs fixed: (1) Conv1D bias must be subtracted from captured
  outputs before fitting (module keeps its bias); (2) activation-weighted
  fit needs the weight-space anchor or it is ill-posed (+8000%+ blowups);
  (3) alternating subspace iteration diverges — use closed form.
- Drift-awareness (student inputs, alpha=1) did NOT help yet: alpha=0 wins
  by ~1pp at every matched beta. QEP-style recapture is not the lever.
- Artifacts: `src/drift_aware_svd.py`, `results/drift_aware_svd.json`,
  `results/drift_aware_svd_sweep2.json`; generality test:
  `src/drift_aware_extended.py` -> `results/drift_aware_extended.json`.

### Generality (same session)
- Weighted fit beats plain SVD in EVERY matrix family tested (5-25x lower
  delta): GPT-2 mlp.c_proj (+76.65 vs +546.12), mlp.c_fc (+2149 vs +7545),
  attn.c_attn (+37.17 vs +807.70), attn full (+49.11 vs +1072.37).
- Gemma 3 1B o_proj 3x: SVD +71.86% vs weighted **-4.45%** (better than
  baseline; verification on 200 texts pending in
  `results/weighted_gemma_oproj_200.json`).
- Compressible family at 3x = attention output projections; MLP matrices
  lack redundancy at these model sizes even with the better fit.

### Ratio frontier (GPT-2 c_proj, weighted vs SVD)
- 2x: +0.10% vs +5.50% | 3x: +3.58% vs +21.34% | 4x: +17.94% vs +152.13%
  | 6x: +53.38% vs +261.86% | 8x: +94.18% vs +374.49%
- Recovers a full compression tier: weighted 4x beats SVD 3x.

### 7B validation (Qwen2.5-7B-Instruct o_proj 3x, fp16, 2026-08-09)
- Baseline 12.29 | plain SVD +26.31% | weighted b=0.1 **+3.84%**.
- Pre-registered decision rule (<=10% supports / >=20% falsifies): **SUPPORTS**.
- Redundancy hypothesis holds at 7B; weighted-vs-SVD gap consistent with
  GPT-2 Small and Gemma-1B. Weights swapped via safetensors reads +
  set_module_tensor_to_device (offloaded params live on meta device).
- Disjoint-calibration robustness: +4.94% (vs +3.58%).
- Gemma o_proj 3x verified on 200 texts: **-7.34% vs baseline**
  (59.75 -> 55.37) -- weighted fit acts as data-aware denoising.

### Storage vs quantization + factor quantization (2026-08-09)
- Accounting: "kx" is a RANK convention; factored storage U[d,r]+V[r,d] at
  r=d/k gives only k/2 x per square matrix (3x-rank = 1.5x storage).
- Factor quantization (Qwen-7B o_proj, rank 1194, symmetric per-row):
  fp16 factors 1.5x/matrix +3.86% | **int8 factors 3.0x/matrix +3.80%
  (LOSSLESS -- per-matrix storage doubles at zero quality cost)** |
  int4 factors 6.0x/matrix +8.22% (still <=10% gate).
  results/weighted_qwen7b_factor_quant.json
- Per-matrix: original o_proj 25.7MB -> int8 factors 8.6MB (1/3 the space).
  But o_proj = 5.09% of Qwen-7B params -> whole model only 14.14 -> ~13.66GB
  (~3.4% saved).
- Honest verdict: low-rank alone CANNOT beat quantization for storage --
  int8 quant = 2x whole model at ~0-0.1% PPL, int4 (GPTQ/AWQ) = ~3.6x whole
  model at ~0.3-1% PPL. Our value = quality-at-rank + denoising insight.
- Path forward = combination: quantized factors (proven lossless at int8),
  SVDQuant-style low-rank outlier branch + int4 residual (~3.5x whole model
  above plain int4 quality, arXiv 2411.05007).
  results/storage_vs_quantization.json

### Hybrid residual split measured (2026-08-09)
- src/hybrid_residual_split.py: W_hat = UV + Q_int4(W-UV) on all 28 o_proj,
  branch rank 256 fp16, residual symmetric int4 group=128.
- pure int4 +0.78% (3.88x/matrix) | hybrid SVD branch +0.60% | hybrid
  weighted branch **+0.56%** (2.50x/matrix) -- weighted branch beats both
  controls, but margins are tiny: int4 alone is nearly free on o_proj.
- New data point: o_proj is easy for BOTH low-rank and quantization.
  Hybrid payoff should live in outlier-heavy families (MLPs = 80.7% of
  params, where plain low-rank fails). Next = per-family routing; MLP hybrid
  is the open GAP. results/hybrid_residual_split.json

## Paper Improvements (2026-08-15)

### "Improve on the Teacher" Concept Clarified
The paper now clearly explains why compression can improve on the teacher:

1. **The Problem**: Transformer weight matrices contain "dead directions" that are never excited by natural text. These directions carry no signal but contribute to weight-space capacity for spurious logits (incorrect predictions).

2. **The Solution**: Activation-weighted low-rank fitting removes these noisy directions by:
   - Using activation statistics to identify which weight directions are actually used
   - Truncating directions that are never excited by calibration data
   - This is analogous to regularization in ML — removing noisy parameters improves generalization

3. **Evidence from Gemma-3-1B**:
   - Teacher PPL: 70.40 → Compressed PPL: 67.47 (−4.45%)
   - Top-1 agreement: only 74.4% (models disagree on 1 in 4 tokens)
   - Entropy increases 12% (less overconfident)
   - NLL decreases (better calibration)

4. **Key Insight**: The compressed model is NOT a copy of the teacher — it's a denoised version that removes weight directions causing overconfident, incorrect predictions.

### Paper Structure Improved
- Abstract now includes a table for quick comparison
- Contributions section clarified with numbered list
- Section 6 expanded with dedicated "Why Compression Can Improve on the Teacher" subsection
- Storage accounting section improved with clearer verdict

### Current Status
- Paper is clearer and easier to understand
- "Improve on the Teacher" concept is now well-explained
- All key findings are properly documented
- Ready for final review and submission preparation


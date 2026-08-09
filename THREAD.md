# Research Thread

## Status: Phase A+B+C Complete — Scaling Trend Confirmed

**Date started:** 2026-08-07
**Last updated:** 2026-08-08
**Current phase:** Phase C — GPT-2 Large Validation Complete

## Run Contract

- **Model:** GPT-2 Small (12 layers, 768 hidden, 12 heads)
- **Dataset:** WikiText-2 Raw V1 (50 eval texts)
- **Device:** NVIDIA RTX 4070 Ti (CUDA 13.2)
- **Rank:** 128 (default), varied in targeted experiments
- **Baseline PPL:** 56.47

## Phase A: Drift Profiling Results

### Experiment 1: Per-Layer Independent Compression (rank=128)

| Layer | PPL Delta | MSE | Cosine | Classification |
|-------|-----------|-----|--------|----------------|
| 0 | +65.52% | 0.031 | 0.942 | catastrophic |
| 1 | -0.14% | 0.00045 | 0.999 | partial |
| 2 | -0.06% | 0.00061 | 0.999 | partial |
| 3 | +0.57% | 0.00071 | 0.999 | failure |
| 4 | +1.19% | 0.00066 | 0.999 | failure |
| 5 | +1.93% | 0.00079 | 0.999 | failure |
| 6 | +0.66% | 0.00090 | 0.998 | failure |
| 7 | +1.58% | 0.00147 | 0.997 | failure |
| 8 | +2.30% | 0.00158 | 0.997 | failure |
| 9 | +3.05% | 0.00473 | 0.991 | failure |
| 10 | +3.90% | 0.00529 | 0.990 | failure |
| 11 | +5.48% | 0.000 | 1.000 | failure |

**Key finding:** Layer 0 is catastrophic. Layers 1-10 are individually fine (<4%).

### Experiment 2: Group Compression (independent per group)

| Group | Size | PPL Delta | Final MSE | Final Cosine |
|-------|------|-----------|-----------|--------------|
| L0-L1 | 2 | +110.59% | 0.053 | 0.905 |
| L2-L3 | 2 | +1.29% | 0.002 | 0.997 |
| L4-L5 | 2 | +3.89% | 0.002 | 0.997 |
| L6-L7 | 2 | +2.93% | 0.003 | 0.995 |
| L8-L9 | 2 | +6.80% | 0.008 | 0.985 |
| L10-L11 | 2 | +13.08% | 0.005 | 0.990 |
| L0-L3 | 4 | +92.65% | 0.039 | 0.927 |
| L4-L7 | 4 | +9.60% | 0.006 | 0.989 |
| L8-L11 | 4 | +29.07% | 0.018 | 0.966 |
| L0-L7 | 8 | +179.90% | 0.066 | 0.886 |
| L0-L11 | 12 | +261.81% | 0.082 | 0.852 |

**Key finding:** Composition is SUPERLINEAR. L0-L7 (+179.90%) > L0-L3 (+92.65%) + L4-L7 (+9.60%).

### Experiment 3: Joint Group Compression (shared input subspace)

| Group | Size | Joint PPL Delta | Independent PPL Delta | Joint Worse? |
|-------|------|-----------------|----------------------|--------------|
| L0 | 1 | +6751.82% | +65.52% | YES (103x) |
| L1 | 1 | +1.84% | -0.14% | YES |
| L2-L3 | 2 | +13.12% | +1.29% | YES (10x) |
| L4-L5 | 2 | +11.70% | +3.89% | YES (3x) |
| L6-L7 | 2 | +14.30% | +2.93% | YES (5x) |
| L8-L9 | 2 | +23.82% | +6.80% | YES (3.5x) |
| L10-L11 | 2 | +38.22% | +13.08% | YES (3x) |
| L0-L3 | 4 | +5091.65% | +92.65% | YES (55x) |
| L4-L7 | 4 | +49.49% | +9.60% | YES (5x) |
| L8-L11 | 4 | +119.39% | +29.07% | YES (4x) |
| L0-L7 | 8 | +4668.99% | +179.90% | YES (26x) |

**Key finding:** Joint compression with shared structure is ALWAYS worse than independent. Shared input subspace is a harmful constraint.

### Experiment 4: Layer 0 Investigation

**Spectral properties:**
- Layer 0: eff_rank=353.7 (most PEAKY, easiest to compress spectrally)
- Layer 10: eff_rank=607.0 (most EVEN, hardest to compress spectrally)

**Error propagation:**
- Layer 0 compressed: error AMPLIFIES (MSE 0.018 at L1 → 0.031 at L11)
- Layer 1 compressed: error DECAYS (MSE 0.00086 at L2 → 0.00045 at L11)
- Layer 5 compressed: error DECAYS (MSE 0.00177 at L6 → 0.00079 at L11)
- Layer 10 compressed: error stays local (MSE 0.005 at L11 only)

**Key finding:** Layer 0's sensitivity is NOT about its spectrum — it's about ERROR PROPAGATION POSITION. Layer 0's error traverses 12 layers and amplifies.

### Experiment 5: Targeted Layer 0 Fix

**Test 1: Higher rank for layer 0**
- L0=128, others=128: +261.81%
- L0=256, others=128: +94.18%
- L0=384, others=128: +61.53%
- L0=512, others=128: +60.62%
- L0=768, others=128: +60.18%

**Test 2: Layer 0 exact, compress others**
- L0=exact, others=64: +205.80%
- L0=exact, others=32: +459.27%
- L0=exact, others=16: +827.86%

**Test 3: Budget-neutral (6x compression)**
- Uniform rank=128: +261.81%
- L0=256, others=115: +109.02%
- L0=384, others=103: +89.46% (BEST)
- L0=exact, others=98: +95.84%

**Key finding:** Even optimal allocation at 6x compression gives +89% PPL. The fundamental limit is the compression ratio, not the allocation.

### Experiment 6: Activation-Space vs Weight-Space

| Method | Rank | PPL Delta | Notes |
|--------|------|-----------|-------|
| Weight SVD | 128 | +261.81% | Baseline |
| Weight SVD | 256 | +21.36% | Good |
| Weight SVD | 384 | +5.50% | Best |
| Activation projection | 128-384 | +173.30% | All same (broken) |
| Dual-space | 128 | +604.63% | Worse |
| Activation-guided | 128 | +10328.80% | Catastrophic |

**Key finding:** Weight-space SVD at rank=384 (+5.50% PPL) is the best method. Activation-space approaches fail because they project onto a fixed subspace rather than learning a new mapping.

## Synthesis: What We Learned

### The Real Findings

1. **Layer 0 is the bottleneck** — not because of its spectrum, but because its error propagates through all 12 layers and amplifies.

2. **Composition is superlinear** — errors compound faster than the sum of individual errors. This is the fundamental reason why per-layer compression doesn't compose.

3. **Shared structure is a myth** — joint compression with shared input subspace is ALWAYS worse than independent. Each layer has its own independent transformation.

4. **The compression limit is ~2-3x** — at rank=256-384, SVD achieves +5-20% PPL. At rank=128 (6x), no method works.

5. **Weight-space SVD is the best method** — activation-space approaches fail because they project onto a fixed subspace.

### Implications for the Research Question

**Original question:** "What is the minimal mathematical description that captures the information-geometric structure of what a transformer learns?"

**Answer from experiments:** The information geometry is layer-specific and error-propagation-sensitive. There is no global shared structure. The minimal description is per-layer SVD at rank=256-384, which achieves 2-3x compression with <20% PPL degradation.

**Reframed question:** "Why does error propagation make compression non-composable, and can we predict which layers will cause the most damage?"

### Next Steps

1. **Test on GPT-2 Large (774M)** — validate scaling trend continues
2. **Test on 7B+ models** — LLaMA, Mistral, Gemma
3. **Develop model-size-aware compression** — adaptive allocation based on model properties
4. **Investigate why eff_rank predictor doesn't generalize** — what's the right predictor for larger models?
5. **Build practical compression tool** — adaptive SVD with model-size-aware allocation

## Rejected Candidates

| ID | Date | Why rejected | Evidence |
|----|------|-------------|----------|
| failure-drift-20260808-0023 | 2026-08-08 | Drift is NOT sublinear — it's superlinear | L0-L7 (+179.90%) > L0-L3 + L4-L7 |
| failure-asymmetry-20260808-0023 | 2026-08-08 | Asymmetry confirmed but not the root cause | Layer 0 sensitivity is about propagation, not spectrum |

## Phase B: Optimal Budget Allocation Results

### Sensitivity Predictor
- **Best predictor**: W_O effective rank (r = -0.867, R² = 0.752)
- Lower eff_rank → higher compression damage
- Layer 0: eff_rank=353.7 (lowest) → +65.52% PPL
- Layer 10: eff_rank=607.0 (highest) → +3.90% PPL

### Error Propagation Model
- Jacobian norms predict RELATIVE ordering but not absolute amplification
- Network is strongly contractive (nonlinearities provide massive contraction)
- Linear model overestimates amplification by 10^7x

### Optimal Allocation Results

| Ratio | Best Strategy | PPL Delta | Ranks |
|-------|--------------|-----------|-------|
| 2x | Uniform (384) | +5.50% | [384×12] |
| 3x | Inverse eff_rank | +17.79% | [380,274,249,245,251,239,250,242,246,231,221,238] |
| 4x | L0 special | +33.62% | [384,175,175,175,175,175,175,175,175,175,175,175] |
| 6x | L0 special | +106.95% | [256,117,117,117,117,117,117,117,117,117,117,117] |

### Key Insight
Non-uniform allocation matters MORE at higher compression ratios. At 2x it doesn't matter. At 4x+ it's the difference between usable and broken.

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
- `research-loop/candidates/` — 2 candidate experiments (rejected)

## Phase C: GPT-2 Medium Validation Results

### Model Comparison
- GPT-2 Small: 124M params, 12 layers, 768 hidden
- GPT-2 Medium: 355M params, 24 layers, 1024 hidden

### Key Finding: Larger Models Are MORE Robust

| Ratio | Medium PPL Delta | Small PPL Delta | Robustness Factor |
|-------|-----------------|-----------------|-------------------|
| 2x | +2.63% | +5.50% | 2.1x |
| 3x | +9.91% | +21.36% | 2.2x |
| 4x | +20.71% | +152.12% | 7.3x |
| 6x | +48.60% | +261.81% | 5.4x |

### Per-Layer Sensitivity (rank=128)
- GPT-2 Small: Layer 0 = +65.52%, Layer 11 = +5.48%
- GPT-2 Medium: Layer 0 = -0.18%, Layer 23 = +1.79%
- Larger models have more redundancy → less sensitive to compression

### Optimal Allocation
- GPT-2 Small: Inverse eff_rank allocation beats uniform by 3.5% at 3x
- GPT-2 Medium: Uniform allocation is BEST — non-uniform hurts performance
- Correlation weakened: r=0.453 (vs -0.867 for Small)

### Drift Profile (rank=128)
- Layer 0: cos=0.9989, MSE=0.031
- Layer 23: cos=0.9519, MSE=98.360
- Drift accumulates across 24 layers (0.03 → 98.36)

### Practical Implications
- At 3x compression: Medium is usable (+9.91%), Small is degraded (+21.36%)
- At 4x compression: Medium is acceptable (+20.71%), Small is broken (+152%)
- At 6x compression: Both are degraded, but Medium is 5x more robust

## Phase C Part 3: GPT-2 Large Validation Results

### Model: GPT-2 Large (774M params, 36 layers, 1280 hidden)
- Baseline PPL: 36.50
- Mean effective rank: 952.3

### Scaling Trend Confirmed

| Ratio | Small (124M) | Medium (355M) | Large (774M) | Trend |
|-------|-------------|---------------|--------------|-------|
| 2x | +5.50% | +2.63% | +2.10% | IMPROVING |
| 3x | +21.36% | +9.91% | +7.45% | IMPROVING |
| 4x | +152.12% | +20.71% | +14.40% | IMPROVING |
| 6x | +261.81% | +48.60% | +33.71% | IMPROVING |

### Key Finding: It's a Scaling Issue, NOT a Fundamental Problem

The "broken" behavior in Small models is because they lack redundancy:
- Small: eff_rank ~354 (low redundancy)
- Medium: eff_rank ~737 (medium redundancy)
- Large: eff_rank ~952 (high redundancy)

At 3x compression, GPT-2 Large gives +7.45% PPL — very usable.
Production models (7B+, 70B+) would be even more robust.

### Practical Compression Limits by Model Size

| Model Size | Max Usable Compression | PPL Degradation |
|------------|----------------------|-----------------|
| 124M (Small) | ~3x | +21% |
| 355M (Medium) | ~4x | +21% |
| 774M (Large) | ~6x | +34% |
| 7B+ (Production) | ~6-8x (predicted) | <20% (predicted) |

---

## Drift-Aware Weighted Low-Rank Fit (2026-08-09)

### Hypothesis
The 3x full-stack failure (+21.36%, c_proj rank 256 all 12 layers) is driven
by fitting layers on teacher inputs while inference receives drifted student
inputs (QEP idea, arXiv 2504.09629, adapted to low-rank replacement).

### Method
`src/drift_aware_svd.py`. Objective min ||W X - W_hat X_s||^2 + beta n ||W - W_hat||^2
solved closed-form (M* = C G^-1) + G-weighted rank-r truncation (EoRA-style
eigenspace restriction). Sequential layer 0->11 with student-input recapture.
Matched control reproduced in-run: plain SVD +21.34% (prior +21.36%).

### Run contract
- Command: `rtk .venv-cuda\Scripts\python.exe src\drift_aware_svd.py`
- Model: gpt2 (HF default revision); dataset: wikitext-2-raw-v1
  (calibration = TRAIN split 64x512 tokens; eval = TEST split 50 texts)
- Targets: attn.c_proj all 12 layers, rank 256 (one-sided 3x)
- Baseline PPL 56.47; matched control plain SVD +21.34%
- Artifacts: results/drift_aware_svd.json (sweep 1),
  results/drift_aware_svd_sweep2.json (sweep 2)

### Results (PPL delta vs 56.47)

| variant | sweep 1 | sweep 2 |
|---|---|---|
| plain_svd (control) | +21.34% | +21.34% |
| alpha=0 beta=0.01 | +4.32% | - |
| **alpha=0 beta=0.1** | **+3.58%** | - |
| alpha=0 beta=0.3 | - | +3.65% |
| alpha=0 beta=1.0 | +5.03%* | +5.03% |
| alpha=0 beta=3.0 | - | +7.23% |
| alpha=1 beta=0.01 | +5.59% | - |
| alpha=1 beta=0.1 | +4.85% | - |
| alpha=1 beta=0.3 | - | +4.66% |
| alpha=1 beta=1.0 | +6.32% | +6.32% |
| alpha=1 beta=3.0 | - | +8.56% |

(*beta=1 alpha=0 also run in pre-bugfix sweep with catastrophic results,
superseded.)

### Verdict
- 3x c_proj full-stack: **+21.34% -> +3.58%** (best: alpha=0, beta=0.1),
  well inside the pre-registered <=10% success gate.
- Gain came from (a) fixing the Conv1D bias double-counting bug and
  (b) activation-weighted fit with weight-space anchor -- NOT from
  drift-awareness: alpha=0 (teacher inputs) beats alpha=1 by ~1pp at all
  matched betas. Sequential student recapture currently adds no value.
- Hypothesis update: "teacher-vs-student input mismatch drives the 3x
  failure" -> WEAKENED. "Unweighted weight-space SVD wastes rank on
  input-rare directions + bias bug" -> SUPPORTED.

### Debugging lessons (runs 1-4, recorded in script docstring)
1. Naive ALS has scale degeneracy (singular Gram).
2. Activation-weighted fit without anchor is ill-posed: unobserved input
   directions are free -> +8000-11000% PPL.
3. Conv1D bias: captured Y = WX + b; fitting Y while keeping b in the module
   double-counts the bias (capture sanity ||WX-Y||/||Y|| ~ 0.22).
4. Alternating subspace iteration diverges (||M-W||2 grows 9.9 -> 23.6 over
   4 iters) because the Z = M P G^{1/2} update ignores C. Closed-form +
   weighted truncation is stable.

### Next
- Generality: MLP family (c_fc/c_proj) + Gemma 3 1B
  (`src/drift_aware_extended.py`, results/drift_aware_extended.json).
- Reproduce winner across seeds (deterministic given fixed data order; check).

### Generality results (same session)

All-matrices-simultaneous at 3x (`src/drift_aware_extended.py`,
results/drift_aware_extended.json) -- both methods break, weighted still
relatively better:

| target set | plain SVD | weighted a0 b0.1 |
|---|---|---|
| GPT-2 MLP (c_fc + c_proj, 24 matrices) | +13885.87% | +9882.60% |
| Gemma3-1B (o_proj + gate/up/down, 104 matrices) | +622329056% | +2739.14% |

One-family-at-a-time on GPT-2 Small (`src/weighted_one_family.py`,
results/weighted_one_family.json) -- weighted beats SVD in EVERY family
(5-25x lower delta):

| family | plain SVD | weighted |
|---|---|---|
| mlp.c_proj only | +546.12% | +76.65% |
| mlp.c_fc only | +7544.82% | +2149.19% |
| attn.c_attn only | +807.70% | +37.17% |
| attn full (c_attn + c_proj) | +1072.37% | +49.11% |

Cross-architecture, o_proj-only Gemma 3 1B (`src/weighted_gemma_oproj.py`,
results/weighted_gemma_oproj.json):

| method | PPL delta |
|---|---|
| plain SVD r384 | +71.86% (reproduces phase D +71.95%) |
| **weighted r384** | **-4.45% (PPL 70.40 -> 67.27)** |

The weighted fit BETTERS the uncompressed Gemma baseline on 50 eval texts
(verification on 200 texts: results/weighted_gemma_oproj_200.json).

### Interpretation
- Attention output projections (attn.c_proj, o_proj) are the compressible
  family at 3x; MLP matrices carry too little redundancy at this size.
- The weighted fit's gain is consistent (5-25x relative) across every
  family and architecture tested -> method generalizes; the absolute
  outcome is family-dependent (redundancy budget).
- Central-thesis update: full-stack replacement CAN work when (a) the
  matrix family has redundancy and (b) the fit is activation-weighted +
  anchored. Drift is manageable; the prior +21.36% "fundamental" figure
  was dominated by the unweighted-fit + bias bugs.

### Robustness + ratio frontier

Robustness (`src/drift_aware_robustness.py`,
results/drift_aware_svd_robustness.json): rebuild with DISJOINT calibration
(chunks 64-127) gives **+4.94%** vs +3.58% -> calibration-stable.

Ratio frontier on GPT-2 c_proj (`src/weighted_ratio_frontier.py`,
results/weighted_ratio_frontier.json). SVD controls reproduce phase C
exactly (+5.50/+21.36/+152.12/+261.81):

| ratio | rank | plain SVD | weighted | gain |
|---|---|---|---|---|
| 2x | 384 | +5.50% | **+0.10%** | ~lossless |
| 3x | 256 | +21.34% | **+3.58%** | 6.0x lower |
| 4x | 192 | +152.13% | **+17.94%** | 8.5x lower |
| 6x | 128 | +261.86% | **+53.38%** | 4.9x lower |
| 8x | 96 | +374.49% | **+94.18%** | 4.0x lower |

The weighted fit recovers a FULL compression tier: weighted 4x beats SVD 3x
(+17.94% < +21.34%), weighted 3x beats SVD 2x (+3.58% < +5.50%).

Gemma verification on 200 eval texts (`src/weighted_gemma_oproj_200.py`,
results/weighted_gemma_oproj_200.json): baseline 59.75 -> weighted 55.37,
**delta -7.34%** (improvement over baseline confirmed, not eval noise).
Interpretation: the weighted fit is a data-aware denoising refit -- it
removes weight components that never fire on the data manifold (o_proj has
large near-null singular subspaces) while preserving in-manifold behavior.

### Promotion-gate status (candidate weighted-lowrank-fit-20260809-0001)
- [x] beats matched baseline at same budget (held-out PPL): yes, 2 archs
- [x] reproduces across independent calibrations (proxy for seeds; method
      is deterministic): +3.58% / +4.94%
- [x] improves on more than one layer/group: all 12 layers, 5 families
- [ ] serialized size / peak memory / warmed-up latency accounting
- [ ] drift metric pre-registered alongside PPL

### Gemma output comparison (`src/compare_gemma_output.py`,
results/compare_gemma_output.json)

Logit divergence on 30 WikiText-2 test texts (~5.9k tokens), teacher forcing:

| metric | value |
|---|---|
| KL(orig\|\|sub) / KL(sub\|\|orig) | 0.268 / 0.292 |
| logit cosine similarity | 0.9925 |
| top-1 agreement | 74.4% |
| top-5 agreement | 96.3% |
| softmax entropy (orig / sub) | 2.19 / 2.47 |

Greedy generation (5 prompts, 64 tokens): both models coherent, on-topic,
fluent; no repetition loops or collapse. Contents diverge (parallel
continuations), quality comparable.

Reading: the student is NOT a near-copy of the teacher (26% argmax flips,
KL ~0.27) yet PPL improves. Entropy is 12% HIGHER while NLL is LOWER ->
the student is less overconfident and better calibrated: removing
never-firing weight components flattens spurious peaks and moves mass onto
correct tokens. Consistent with the data-aware denoising interpretation.

### 7B validation: Qwen2.5-7B-Instruct o_proj 3x (`src/weighted_qwen7b_oproj.py`,
results/weighted_qwen7b_oproj.json)

Run contract: Qwen/Qwen2.5-7B-Instruct rev a09a35458c702b33eeacc393d103063234e8bc28,
fp16, device_map="auto" (17 layers GPU / 11 CPU-offloaded, RTX 4070 Ti 12GB),
WikiText-2 TEST 50 texts max_length 256, calib TRAIN 16x512, targets all 28
o_proj (3584x3584), rank 1194 (one-sided 3x), alpha=0 beta=0.1 closed-form.
Weights read from safetensors shards + written via set_module_tensor_to_device
(accelerate leaves offloaded params on the meta device -- direct .data access
crashes; RUN 1/2 note in script docstring).

| variant | PPL | delta |
|---|---|---|
| baseline (fp16) | 12.29 | -- |
| plain SVD | 15.53 | +26.31% |
| weighted b=0.1 | 12.77 | **+3.84%** |

Pre-registered decision rule: <=10% supports / >=20% falsifies / else
inconclusive -> **verdict: SUPPORTS**. The redundancy hypothesis holds at 7B:
weighted 3x o_proj compression costs <4% PPL, plain SVD control fails badly
(+26%), and the weighted-vs-SVD gap (6.9x) matches the pattern from GPT-2
Small (+3.58%) and Gemma-1B (-4.45%). The central 7B validation of the
redundancy-vs-drift analysis is complete; the scaling trend is NOT falsified.

### Storage accounting vs quantization (`src/storage_vs_quantization.py`,
results/storage_vs_quantization.json)

Analysis only (no GPU). Factored low-rank storage U[m,r]+V[r,n]; one-sided
kx rank convention gives k/2 x per square matrix. o_proj = 5.09% of
Qwen2.5-7B params.

| method | storage ratio | PPL cost | scope |
|---|---|---|---|
| low-rank 3x-rank o_proj (measured) | 1.5x/matrix -> 1.7% whole model | +3.84% (Qwen7B) | o_proj only |
| low-rank 4x-rank o_proj (measured) | 2.0x/matrix -> 2.5% whole model | +17.94% (GPT-2) | o_proj only |
| int8 quantization | 2.0x whole model | ~0-0.1% | ALL weights |
| int4 (GPTQ/AWQ) | ~3.6x whole model | ~0.3-1% | ALL weights |

Verdict: as pure storage compression the low-rank method does NOT compete
with quantization -- even pushing o_proj to 8x-rank (4x/matrix) saves only
3.8% of the model. Quantization wins on both axes (ratio AND quality).
The candidate's value is quality-at-rank and the redundancy/denoising
insight; the practical path is combination: quantize the low-rank factors,
or SVDQuant-style low-rank outlier branch + int4 residual (~3.5x with
quality above plain int4, arXiv 2411.05007).

### Factor quantization hybrid (`src/weighted_qwen7b_factor_quant.py`,
results/weighted_qwen7b_factor_quant.json)

Symmetric per-row quantization of the fitted U,V factors (rank 1194, 3x
convention, all 28 o_proj, same run contract as the 7B validation).

| variant | PPL | delta | per-matrix storage |
|---|---|---|---|
| baseline | 12.29 | -- | 1x |
| plain SVD (control) | 15.53 | +26.31% | -- |
| weighted, fp16 factors | 12.77 | +3.86% | 1.5x |
| weighted, **int8 factors** | **12.76** | **+3.80%** | **3.0x** |
| weighted, int4 factors | 13.30 | +8.22% | 6.0x |

Findings:
- int8 factors are FREE: +3.80% vs +3.86% fp16 (within noise) -> the fitted
  low-rank subspace is robust to 8-bit factor noise. Per-matrix storage
  doubles for zero quality cost.
- int4 factors cost +4.4pp extra (+8.22% total) -- still inside the <=10%
  gate but no longer negligible.
- Whole-model savings remain small (o_proj = 5.09% of params): ~3.4% with
  int8 factors, ~4.2% with int4 factors. The hybrid proves factor
  quantization is lossless at int8; the path to meaningful whole-model
  storage is extending the fit to more matrix families and/or combining
  with whole-model quantization (SVDQuant-style residual split).

### Hybrid residual split, SVDQuant-style (`src/hybrid_residual_split.py`,
results/hybrid_residual_split.json)

Per-matrix: W_hat = U@V + Q_int4(W - U@V); branch rank 256 (fp16 factors),
residual symmetric int4 group=128; all 28 o_proj, same run contract.

| variant | PPL | delta | per-matrix storage |
|---|---|---|---|
| baseline | 12.29 | -- | 25.69 MB |
| pure int4 (reference) | 12.39 | +0.78% | 6.62 MB (3.88x) |
| hybrid, SVD branch | 12.37 | +0.60% | 10.29 MB (2.50x) |
| hybrid, weighted branch | 12.36 | **+0.56%** | 10.29 MB (2.50x) |

Both pre-registered checks pass: weighted branch beats SVD branch at
matched storage (+0.56 vs +0.60) and beats the cheaper pure-int4 reference
(+0.56 vs +0.78). BUT margins are small on o_proj -- int4 alone is nearly
free on this family (+0.78%), unlike at aggressive rank truncation (+3.8%).
Reading: o_proj has no severe outliers, so the branch adds little; the
hybrid's payoff should concentrate on outlier-heavy families (MLPs, q/k/v),
which is exactly where plain low-rank fails. Next: route per family --
hybrid on MLPs, low-rank-only where it wins, plain quant where int4 is free.






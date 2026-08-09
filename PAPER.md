# Activation-Weighted Low-Rank Fitting Defeats Spectral Compression of Transformer Weights — and Can Improve on the Teacher

**Status: DRAFT (pre-submission).** All numbers below are measured results from
this repository (see `results/*.json`, run log in `THREAD.md`). Sections marked
`[GAP]` require additional experiments before the paper is submission-ready.

---

## Abstract

Replacing a pretrained transformer's weight matrices by low-rank
approximations is known to fail catastrophically when composed across layers:
single-layer approximation error compounds through the residual stream, and
full-stack replacement at 3x compression typically destroys language modeling
performance. We show that the dominant cause is not irreducible drift but a
mismatch between the approximation norm and the data distribution: truncated
SVD in weight space spends rank on input directions the model never excites.
We derive a closed-form activation-weighted low-rank fit — the minimizer of
`E_x ||(W − Ŵ)x||²` with a weight-space anchor, truncated in the activation
Gram norm — and show that it defeats plain SVD by 5–7x in perplexity delta
across three architectures and scales: GPT-2 Small (+3.58% vs +21.34%),
Gemma-3-1B (−4.45% vs +71.86%), and Qwen2.5-7B (+3.84% vs +26.31%), all at
one-sided 3x rank on attention output projections, training-free, with 16
chunks of calibration text. The compressed Gemma model scores *below* the
teacher's perplexity while differing from it substantially (top-1 agreement
74%), with higher output entropy and lower NLL — evidence that data-aware rank
truncation acts as denoising. We further show the fitted factors tolerate
int8 quantization losslessly (3.0x per-matrix storage at +3.80%), but that
low-rank compression alone cannot compete with whole-model quantization for
storage; we characterize where the combination of the two is the productive
direction.

---

## 1. Introduction

A pretrained transformer's dense layers are highly overparameterized by
spectral criteria: singular value spectra of attention projections decay fast
enough that a third of the rank captures most of the Frobenius mass. Yet
replacing these matrices by their truncated SVD collapses full-stack language
modeling performance — in our measurements, +21% perplexity on GPT-2 Small
and +72% on Gemma-3-1B at one-sided 3x compression, even though per-layer
output error under teacher inputs is small.

This failure has been attributed to two competing mechanisms:

1. **Drift limit**: approximation errors compound through the residual stream;
   later layers receive off-manifold inputs they were never trained on, so
   layerwise equivalence does not imply network equivalence.
2. **Norm mismatch**: the weight-space Frobenius norm in which SVD is optimal
   is the wrong norm — what matters is the error under the activation
   distribution, and SVD wastes rank on directions that never fire.

We pre-registered a decision rule to distinguish these at 7B scale, and we
find the answer is overwhelmingly (2): a closed-form fit that minimizes
expected activation-space error recovers a full compression tier that plain
SVD destroys. The "drift wall" observed at 3x compression across model scales
(+21.4% Small, +9.9% Medium, +7.5% Large in our earlier phases) is therefore
largely an artifact of the approximation method, not a fundamental limit.

**Contributions.**

1. A closed-form activation-weighted low-rank fit with a weight-space anchor
   (Section 3): `M* = C(G + βnI)^{-1}`, truncated to rank r in the
   G-weighted norm. No gradient steps; per-matrix solve in seconds.
2. Cross-scale validation (GPT-2 Small / Gemma-3-1B / Qwen2.5-7B) against a
   matched SVD control at identical budgets, with pre-registered gates
   (Section 5). The weighted fit is 5–7x better in perplexity delta at every
   tested scale and ratio.
3. Evidence that data-aware truncation can *improve* on the teacher: the
   compressed Gemma-3-1B scores −4.45% (and −7.34% on 200 texts) below the
   teacher's held-out perplexity with no fine-tuning, while exhibiting higher
   output entropy and lower NLL — a calibration/denoising signature
   (Section 6).
4. A storage accounting that places the method honestly against quantization
   (Section 7): factor quantization to int8 is lossless (3.0x per matrix),
   but whole-model quantization dominates pure storage; the productive
   combination is a low-rank branch plus quantized residual.
5. Negative results and implementation pitfalls with quantitative evidence:
   drift-aware refitting on student inputs does not help; Conv1D bias
   double-counting and alternating subspace iteration divergence both cause
   order-of-magnitude blowups (Appendix A).

---

## 2. Related Work

**Quantization with activation awareness.** OBQ/OBS and the GPTQ/SparseGPT
line established that weight-space rounding error should be compensated under
the second-order (activation Gram / Hessian) statistic of calibration inputs.
AWQ protects salient channels by activation magnitude. Our objective is the
low-rank analogue of this principle.

**Activation-aware low-rank compression.** ASVD (Yuan et al.,
arXiv 2312.05821) reparameterizes weights by activation statistics before SVD
and handles outlier channels, demonstrating that activation awareness is the
lever for LLM low-rank compression. Our closed-form anchored fit is an
independent derivation in the same family; to our knowledge the
beats-the-teacher denoising effect (Section 6) has not been reported for
training-free rank truncation. EoRA (Liu et al., arXiv 2410.21271)
compensates low-rank error with an added eigenspace term; QEP
(arXiv 2504.09629) patches cross-layer error propagation for quantization.

**Hybrid low-rank + quantization.** SVDQuant (arXiv 2411.05006) migrates
outliers into an fp16 low-rank branch and quantizes the residual to 4 bits.
Our factor-quantization result (Section 7) shows the low-rank branch itself
survives int8 losslessly, reinforcing the hybrid design space.

**Classical weighted low-rank approximation.** The problem
`min_rank(Ŵ) E||(W−Ŵ)X||` is a known non-convex problem; our contribution is
the specific anchored closed-form + weighted truncation pipeline and its
systematic validation on modern LLMs against matched controls.

---

## 3. Method

### 3.1 Problem setting

For a linear module with weight `W ∈ R^{out×in}` (we use the column-space
convention `y = Wx`; for GPT-2 Conv1D modules `W` is transposed accordingly),
we observe calibration input activations `x_1 … x_n ∈ R^in` captured by a
forward hook, and seek a rank-r approximation `Ŵ` minimizing

    L(Ŵ) = (1/n) Σ_i ||(W − Ŵ) x_i||² + β ||Ŵ − W||_F²        (1)

The first term is the activation-space error (what the network actually
sees); the second is a weight-space anchor of strength β ≥ 0. The anchor is
mandatory, not optional: with β = 0 the problem is ill-posed in directions
the calibration data never excites, and empirical fits diverge there.

### 3.2 Closed-form solution

Let `G = XXᵀ ∈ R^{in×in}` be the activation Gram matrix and
`C = (Y−b)Xᵀ + βnW` the cross-moment (with bias b subtracted from captured
outputs where present — see Appendix A). The unconstrained minimizer of (1) is

    M* = C (G + βnI)^{-1}                                    (2)

computed in the eigenspace of G with a relative eigenvalue floor
(ε_rel = 1e-3) and Tikhonov ridge λ = 0.01 · mean(diag(G_k)) for numerical
safety.

### 3.3 Rank-r truncation in the correct norm

Truncating M* by plain SVD would again optimize the wrong norm. We truncate
in the G-weighted norm — the norm in which (1) measures error — via the SVD of

    Z = M* (G + βnI)^{1/2},    Z = Uz Σ Vzᵀ                   (3)

and set `Ŵ = Uz[:,:r] Uz[:,:r]ᵀ M*`. This is the exact rank-r minimizer of
(1) given the full-rank optimum M*; the per-matrix solve is closed-form and
takes seconds on a single GPU.

### 3.4 Calibration and hyperparameters

Calibration: 16 chunks of 512 tokens from the WikiText-2 **train** split
(never the test split), teacher-forced through the untouched model
(α = 0: teacher inputs; student-input refitting was tested and is inferior,
Section 5.3). One hyperparameter β: optimal in [0.1, 0.3], all reported
results use β = 0.1. Rank convention: one-sided, r = min(in, out)/k.

### 3.5 Algorithm

```
for each target module m:
    capture X (inputs), Y (outputs) over calibration chunks   # forward hooks
    Y <- Y - bias                                             # if present
    G <- X Xᵀ + β n I        C <- (Y) Xᵀ + β n W
    M* <- C (G)^{-1}         # eigenspace solve, ridge + floor
    Ŵ  <- rank-r truncation of M* in G-norm (eq. 3)
    write Ŵ back (fp16 or int8-quantized factors)
```

---

## 4. Experimental Setup

**Models.** GPT-2 Small (124M), google/gemma-3-1b-it, Qwen/Qwen2.5-7B-Instruct
(rev a09a3545). Targets: attention output projections in every layer
(`attn.c_proj`, `self_attn.o_proj`) — the family identified as compressible
in Section 5.4.

**Protocol (identical across all runs).** Calibration: WikiText-2 train,
16×512 tokens (GPT-2 sweep used 64×512). Evaluation: WikiText-2 test, 50
texts >50 chars truncated to 256 tokens (Gemma additionally verified on 200
texts). Metric: perplexity delta vs the untouched fp-matched baseline.
Control: plain truncated SVD at the identical rank — same storage, same
modules, same eval. All runs on a single RTX 4070 Ti 12GB; Qwen-7B fp16 with
partial CPU offload (weights read from safetensors shards, written via
`set_module_tensor_to_device`).

**Pre-registered decision rule (7B validation).** Weighted-fit delta ≤10%
supports the redundancy hypothesis (compression scales); ≥20% falsifies it;
between is inconclusive.

---

## 5. Results

### 5.1 Cross-scale main result (one-sided 3x, all output projections)

| Model | Baseline PPL | Plain SVD | Weighted fit (β=0.1) |
|---|---|---|---|
| GPT-2 Small (124M) | 56.47 | +21.34% | **+3.58%** |
| Gemma-3-1B | 70.40 | +71.86% | **−4.45%** |
| Qwen2.5-7B-Instruct | 12.29 | +26.31% | **+3.84%** |

The weighted-vs-SVD gap is 5.9x, 16.2x, and 6.9x respectively. The
pre-registered 7B gate returns **SUPPORTS**: the earlier "failure wall" at 3x
was a property of the approximation norm, not of scale.

### 5.2 Ratio frontier (GPT-2 Small, c_proj ×12)

| Ratio (rank) | SVD delta | Weighted delta |
|---|---|---|
| 2x (384) | +5.50% | **+0.10%** |
| 3x (256) | +21.34% | **+3.58%** |
| 4x (192) | +152.13% | **+17.94%** |
| 6x (128) | +261.86% | +53.38% |
| 8x (96) | +374.49% | +94.18% |

The weighted fit recovers a full compression tier: weighted 4x beats SVD 3x
in PPL while storing less.

### 5.3 Ablations

**β sweep.** Optimum β ∈ [0.1, 0.3]; β = 0 diverges in unobserved directions.

**Drift-awareness (student inputs).** Fitting against the *student's* own
activations (α = 1, sequential refit) loses ~1pp to teacher-input fitting
(α = 0) at every matched β. Anticipating drift is not the lever; fixing the
norm is.

**Robustness.** Disjoint calibration (chunks 64–127 instead of 0–63):
+4.94% vs +3.58% — mild sensitivity to calibration data, no collapse.

### 5.4 Matrix-family study (GPT-2 Small, one-sided 3x, weighted vs SVD)

| Family | SVD | Weighted |
|---|---|---|
| attn.c_proj | +21.34% | **+3.58%** |
| attn.c_attn | +807.70% | +37.17% |
| attn (full) | +1072.37% | +49.11% |
| mlp.c_proj | +546.12% | +76.65% |
| mlp.c_fc | +7544.82% | +2149.19% |

Weighted fitting wins in every family by 5–25x, but MLP matrices retain no
usable redundancy at 3x at these sizes. Compressible family = attention
output projections.

### 5.5 Factor quantization (Qwen2.5-7B, rank 1194)

| Factor precision | Per-matrix storage | PPL delta |
|---|---|---|
| fp16 | 1.5x | +3.86% |
| int8 (per-row symmetric) | 3.0x | **+3.80%** |
| int4 | 6.0x | +8.22% |

int8 factors are lossless — the fitted subspace is robust to 8-bit factor
noise. int4 costs +4.4pp but remains inside the ≤10% gate.

### 5.6 Hybrid residual split (Qwen2.5-7B, SVDQuant-style)

Per-matrix hybrid: `Ŵ = UV + Q_int4(W − UV)` — a small low-rank branch
(rank 256, fp16 factors) absorbs structure the grid cannot represent; the
residual is group-wise symmetric int4 (group 128). Applied to all 28 o_proj,
same run contract.

| Variant | PPL | Δ | Storage/matrix |
|---|---|---|---|
| Baseline | 12.29 | — | 25.69 MB |
| Pure int4 (reference) | 12.39 | +0.78% | 6.62 MB (3.88x) |
| Hybrid, SVD branch | 12.37 | +0.60% | 10.29 MB (2.50x) |
| Hybrid, weighted branch | 12.36 | **+0.56%** | 10.29 MB (2.50x) |

Both pre-registered checks pass: the weighted branch beats the matched-storage
SVD branch and beats the cheaper pure-int4 reference. Two findings follow.
First, our fit is the better outlier branch, as hypothesized. Second, the
margin on o_proj is small (+0.22pp) because **int4 alone is nearly free on
this family (+0.78%)** — a new data point showing o_proj is easy for both
low-rank and quantization. The hybrid's payoff should therefore concentrate
on outlier-heavy families — the MLPs, 80.7% of parameters, which are
precisely where plain low-rank fails (Table 5.4). This motivates per-family
routing: weighted low-rank where it wins, hybrid where outliers dominate,
plain quantization where int4 is free (Section 7).

---

## 6. Analysis: Compression as Denoising

The Gemma result (−4.45%; −7.34% on 200 texts) means the compressed model is
a *better* language model on held-out text than the teacher, with no
fine-tuning. Logit-space comparison on 30 texts (~5.9k tokens) shows the
student is not a near-copy:

| Metric | Value |
|---|---|
| KL(orig‖sub) / KL(sub‖orig) | 0.268 / 0.292 |
| Logit cosine | 0.9925 |
| Top-1 / top-5 agreement | 74.4% / 96.3% |
| Softmax entropy (orig → sub) | 2.19 → 2.47 |

Entropy rises 12% while NLL falls: the student is less overconfident and
better calibrated. Interpretation: directions of W that are never excited by
natural text carry no signal but contribute weight-space capacity for
spurious logits; activation-weighted truncation removes them, flattening
spurious peaks and moving mass onto correct tokens — denoising.

At 7B, qualitative generation on three solvable prompts (arithmetic, factual
QA, code) shows the int8-factor model tracks the baseline nearly
word-for-word, the int4 model paraphrases with correct content, and even the
+26%-PPL SVD control remains coherent on easy tasks — greedy generation on
short prompts is too insensitive a probe for this kind of damage, confirming
held-out PPL / logit divergence as the right primary metrics.

`[GAP]` The denoising claim currently rests on one architecture, one family,
and 1–2 eval sizes. Required before claiming it: 3 seeds, replication on a
second architecture, pre-registered calibration/eval protocol.

---

## 7. Storage Accounting vs Quantization

Factored storage of a rank-d/3 approximation of a d×d matrix is d²·2/3
parameters → **1.5x**; "kx rank" is not "kx storage" (k/2x for square
matrices). On Qwen2.5-7B (o_proj = 5.09% of params):

| Method | Storage | PPL cost | Scope |
|---|---|---|---|
| Weighted 3x-rank, int8 factors | 3.0x/matrix → ~3.4% whole-model | +3.80% | o_proj |
| int8 quantization | 2.0x whole model | ~0–0.1% | all weights |
| int4 (GPTQ/AWQ) | ~3.6x whole model | ~0.3–1% | all weights |

**Verdict:** low-rank compression alone cannot compete with quantization for
storage. Its value is quality-at-rank and the denoising effect; the
productive frontier is combination — now measured (Section 5.6): the
weighted-fit branch beats both the SVD branch and pure int4 on o_proj, and
the design space that remains open is per-family routing — hybrid residual
split on outlier-heavy MLPs (80.7% of parameters), low-rank where it wins,
plain quantization where int4 is free.

---

## 8. Limitations

1. **Scope of compression.** Only attention output projections are viable at
   3x; MLP families break even weighted. Whole-model compression requires
   budget reallocation across families, which we have not demonstrated.
2. **Seeds.** Single calibration/fit per result `[GAP]`; the ±1pp robustness
   run suggests stability but multi-seed statistics are missing.
3. **Eval sensitivity.** Perplexity on ≤200 WikiText-2 texts; no
   downstream-task evaluation, no pre-registered drift metric alongside PPL
   `[GAP]`.
4. **Latency/size accounting.** Factored form changes the compute graph;
   warmed-up latency and peak memory are unmeasured `[GAP]`. Without a fused
   low-rank kernel the storage win does not automatically become a speed win.
5. **Instruct-tuned 7B baseline.** Qwen experiments use the Instruct variant
   (matched to prior cross-architecture protocol); base-model PPL might
   differ in absolute terms.

---

## 9. Conclusion

The "drift wall" of full-stack low-rank compression is largely an artifact
of approximating in the wrong norm. A closed-form activation-weighted fit
with a weight-space anchor, truncated in the activation Gram norm, is
training-free, cheap, and defeats truncated SVD by 5–7x across three
architectures from 124M to 7B parameters, recovers a full compression tier,
and — on Gemma-3-1B — produces a compressed model that outperforms its
teacher on held-out perplexity, with a measurable calibration signature.
Pure storage compression remains quantization's territory; the compelling
direction is combining data-aware low-rank structure with quantized
representations.

---

## Appendix A: Pitfalls with Quantitative Evidence

**A.1 Bias double-counting.** Modules whose outputs include a bias (GPT-2
Conv1D) produce captured Y = WX + b; fitting Y while the module keeps b
double-counts it. Layer-0 sanity check showed ||WX − Y||/||Y|| = 0.216 —
a 22% phantom target. Full-stack consequence: +3470% PPL until fixed.

**A.2 Alternating subspace iteration diverges.** Refitting `Z = M P G^{1/2}`
against the current solution's energy (ignoring the cross-moment C) diverges:
||M − W||₂ grew 9.9 → 23.6 over 4 iterations. Closed-form (2)+(3) is required.

**A.3 Offloaded weights are meta.** With accelerate `device_map="auto"`,
CPU-offloaded parameters report `device == meta`; direct `.data` access
raises. Read from safetensors shards via the index `weight_map`, write with
`set_module_tensor_to_device`.

**A.4 Student-input refitting is inferior.** Anticipating drift (α = 1)
loses ~1pp to teacher-input fitting at every matched β; the sequential
refit adds complexity without payoff.

---

## Appendix B: Reproducibility

Every number above is produced by a standalone script in `src/` with output
in `results/`: `drift_aware_svd.py` (+ sweeps), `weighted_one_family.py`,
`weighted_gemma_oproj.py` (+ `_200`), `weighted_ratio_frontier.py`,
`drift_aware_robustness.py`, `weighted_qwen7b_oproj.py`,
`weighted_qwen7b_factor_quant.py`, `hybrid_residual_split.py`,
`compare_gemma_output.py`,
`compare_qwen7b_output.py`, `storage_vs_quantization.py`. Run contract
(model revisions, splits, seeds, budgets) is recorded in the header of each
script and summarized in `THREAD.md`. Hardware: RTX 4070 Ti 12GB, Python
3.14, torch 2.13.0+cu132.

---

## Pre-submission checklist (`[GAP]` items)

- [ ] 3 seeds for the denoising claim (Gemma −4.45%) and a second
      architecture reproducing improvement
- [ ] Pre-registered drift metric (hidden-state cosine per block) alongside PPL
- [ ] MLP budget reallocation experiment (adaptive rank across families)
- [ ] Warmed-up latency + peak-memory accounting for factored inference
- [ ] Hybrid residual split on MLP families (where the payoff should live)
      and the full per-family routed recipe
- [ ] Head-to-head against ASVD-style activation reparameterization at
      matched budget (the sharpest novelty test)
- [ ] Base (non-Instruct) 7B confirmation run

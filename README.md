# ResearchCompression

Can a pretrained transformer's dense weights be replaced by a more compact
representation **without breaking the global computation**?

This repository is an evidence-driven research loop answering that question.
After a long negative phase (layerwise low-rank replacements compound into
catastrophic full-stack drift), we found the failure was mostly an artifact
of approximating in the wrong norm — and that an **activation-weighted
closed-form low-rank fit** defeats plain SVD by 5–7x in perplexity delta
across three architectures and scales, training-free.

Paper draft: [PAPER.md](docs/PAPER.md) (markdown) / [paper.tex](docs/paper.tex)
(LaTeX). Full experiment log: [THREAD.md](THREAD.md).

## Headline results (all measured, one-sided 3x rank on attention output projections)

| Model | Baseline PPL | Plain SVD | Weighted fit (β=0.1) |
|---|---|---|---|
| GPT-2 Small (124M) | 56.47 | +21.34% | **+3.58%** |
| Gemma-3-1B (`gemma-3-1b-it`) | 70.40 | +71.86% | **−4.45%** (beats the teacher) |
| Qwen2.5-7B-Instruct (fp16) | 12.29 | +26.31% | **+3.84%** |

Protocol: WikiText-2 train for calibration (16×512 tokens), WikiText-2 test
for evaluation (50 texts, 256 tokens; Gemma also verified on 200 texts:
−7.34%). The 7B run cleared its pre-registered decision rule (≤10% supports
/ ≥20% falsifies) → **SUPPORTS**: compression redundancy scales.

## The method (one paragraph)

For a linear layer `y = Wx`, capture calibration activations `X` with a
forward hook, then solve in closed form — no gradient steps:

```
G  = X Xᵀ + β·n·I                      # activation Gram + anchor
C  = (Y − bias) Xᵀ + β·n·W             # cross-moment (bias subtracted!)
M* = C · G⁻¹                           # full-rank optimum (eigenspace solve)
Ŵ  = rank-r truncation of M* in the G-weighted norm
```

Three bugs cost us order-of-magnitude blowups before this worked: Conv1D
bias double-counting (+3470% PPL), an ill-posed β=0 fit, and divergent
alternating subspace iteration. Details in [THREAD.md](THREAD.md).

## Storage: what actually gets converted (byte-level, real numbers)

**"3x compression" is a rank convention, not a storage ratio.** The rank-d/3
approximation of a d×d matrix is stored *factored*, and here is exactly what
changes on disk for one Qwen2.5-7B `o_proj` matrix (3584×3584, rank 1194 —
our measured 7B configuration):

**Before conversion** — the dense matrix:
```
3584 × 3584 = 12,845,056 weights × 2 bytes (fp16) = 25.69 MB
```

**The conversion** — the dense matrix is discarded and replaced by two
factors `U` (3584×1194) and `V` (1194×3584) with `Ŵ = U·V`. Inference
`y = Wx` becomes `y = U(Vx)` — two smaller matmuls.

**After conversion** (per matrix):

| Serialization | What is stored | Total | Ratio | Measured PPL cost |
|---|---|---|---|---|
| fp16 factors | 8,558,592 factor params × 2 B | 17.12 MB | 1.50x | +3.86% |
| **int8 factors** | 8,558,592 codes × 1 B + 4,778 per-row scales (9.6 KB) | **8.57 MB** | **3.00x** | **+3.80% (lossless vs fp16)** |
| int4 factors | 8,558,592 codes × 0.5 B + 9.6 KB scales | 4.29 MB | 5.99x | +8.22% |

So at int8, one `o_proj` matrix shrinks from **25.69 MB to 8.57 MB — one
third of the space — at no extra quality cost**.

**Model-level honesty:** Qwen2.5-7B has 28 `o_proj` matrices = 5.09% of its
7.07B params. Converting all of them saves 28 × 17.12 MB = **479 MB**,
i.e. the model goes 14.14 GB → ~13.66 GB (~3.4% whole-model). That is why
we do not claim to beat quantization on storage — int4 quantization gives
~3.6x the *entire* model at ~0.3–1% PPL. Our value is quality-at-rank and
the denoising effect; the productive direction is combining the two
(quantized factors, or SVDQuant-style low-rank branch + int4 residual).

## Repository map

| Path | What |
|---|---|
| `src/` | Standalone experiment scripts (one hypothesis each) |
| `results/` | JSON artifacts for every run — never overwritten |
| `THREAD.md` | Chronological experiment log with run contracts |
| `docs/MEMORY.md` | Condensed state of knowledge |
| `docs/PAPER.md` / `docs/paper.tex` | Paper draft (markdown / LaTeX) |
| `research-loop/` | Orchestrator, frontier, candidate records |
| `docs/LITERATURE.md` | Related work notes (QEP, EoRA, GPTQ, ...) |

Key scripts for the current findings:

| Script | Result file | Finding |
|---|---|---|
| `src/drift_aware_svd.py` | `results/drift_aware_svd*.json` | Breakthrough: +21.34% → +3.58% (GPT-2) |
| `src/weighted_gemma_oproj.py` | `results/weighted_gemma_oproj*.json` | Gemma beats baseline (−4.45% / −7.34%) |
| `src/weighted_qwen7b_oproj.py` | `results/weighted_qwen7b_oproj.json` | 7B validation: +3.84% (verdict: SUPPORTS) |
| `src/weighted_qwen7b_factor_quant.py` | `results/weighted_qwen7b_factor_quant.json` | int8 factors lossless (3.0x/matrix) |
| `src/hybrid_residual_split.py` | `results/hybrid_residual_split.json` | Hybrid UV + int4 residual: weighted branch +0.56%, beats SVD branch and pure int4 |
| `src/storage_vs_quantization.py` | `results/storage_vs_quantization.json` | Storage accounting vs quantization |
| `src/compare_gemma_output.py` / `compare_qwen7b_output.py` | `results/compare_*.json` | Qualitative generation comparisons |

## Running experiments

```powershell
# Windows PowerShell; every command goes through the rtk wrapper (AGENTS.md)
.venv-cuda\Scripts\python.exe src\<script>.py     # run from repo root
```

Requirements: `requirements.txt` (torch 2.13+cu132, transformers, datasets,
accelerate, safetensors). Hardware used: RTX 4070 Ti 12 GB, 32 GB RAM
(Qwen-7B runs in fp16 with partial CPU offload — offloaded weights sit on
the meta device; the scripts read them from safetensors shards instead).

## Current status and open questions

- ✅ Breakthrough method validated at 3 scales with matched SVD controls
- ✅ 7B decision rule pre-registered and cleared (SUPPORTS)
- ✅ Factor quantization proven lossless at int8
- ✅ Hybrid residual split measured: weighted-fit branch beats SVD branch and pure int4 (o_proj)
- ⬜ `[GAP]` 3-seed replication of the Gemma denoising effect
- ⬜ `[GAP]` Pre-registered drift metric alongside PPL
- ⬜ `[GAP]` MLP budget reallocation (adaptive rank across families)
- ⬜ `[GAP]` Latency / peak-memory accounting for factored inference
- ⬜ `[GAP]` Hybrid residual split on MLP families + per-family routed recipe
- ⬜ `[GAP]` Head-to-head vs ASVD-style reparameterization (novelty test)

# Research plan: activation-weighted compression, strong baselines, and global behavior

Prepared: 2026-10-13. Execution is planned for the main PC. This document does not authorize running experiments on the current machine.

## 1. Goal and agreed limits

Produce a rigorous empirical paper answering:

> **When does activation-weighted low-rank compression preserve a transformer's global behavior, and what benefit remains after comparison with existing activation-aware methods?**

Start with an empirical contribution. Consider a new algorithm only after the baseline comparisons identify a specific unresolved failure.

- Execute later on your **main PC**.
- Allocate **24 GPU-hours to the first study**, excluding implementation and environment preparation.
- Start with Gemma-3-1B and replicate on GPT-2 Small.
- Begin with WikiText-2; add a separate corpus and downstream benchmarks afterward.
- Defer cloud experiments until the GPU and spending cap are specified.
- Preserve all existing code, drafts, and results. Use a new runner and fresh result directories.

The current Gemma evidence—roughly **6.7% higher perplexity than the teacher, versus a much larger SVD degradation**—is the starting observation. “Compression improves the teacher” remains a hypothesis to investigate.

## 2. Prepare a trustworthy experiment system

### Main-PC preflight

Before fitting any model, record:

- GPU, VRAM, RAM, Python, CUDA, and exact package versions.
- Git revision and hashes of modified research files.
- Model and tokenizer snapshot revisions.
- Dataset revision and hashes of calibration, validation, and evaluation manifests.
- Exact target modules, shapes, ranks, factor parameter counts, and precision.
- Calibration token positions, seeds, numerical regularization, and evaluation settings.

Use cached assets initially. Missing assets should produce an explicit preparation requirement; the runner must never silently download or substitute models or datasets.

Use the historical Gemma revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`. Recover and pin the GPT-2 revision on the main PC before its replication stage.

### Add a new runner

Implement an additive `src.paper_benchmark` interface with three operations:

```powershell
rtk python -m src.paper_benchmark preflight --config configs\paper_protocol_v2.json
rtk python -m src.paper_benchmark run --config configs\paper_protocol_v2.json --stage local-core
rtk python -m src.paper_benchmark report --study local-core
```

These are **proposed commands requiring implementation**, not commands to run against the current repository.

The configuration must contain explicit asset paths, methods, numerical settings, seeds, stage limits, and the data protocol. Each run creates a unique directory and saves:

- Resolved configuration and complete run contract.
- stdout/stderr, progress, and terminal status.
- Incremental metrics and per-seed results.
- Factor checkpoints and their hashes.
- A concise report identifying supported, weakened, rejected, or inconclusive hypotheses.

Reject existing output directories and mismatched comparisons. Stop on NaN/Inf, wrong assets, changed splits, or GPU exhaustion; preserve partial results and tracebacks.

### Audit the actual mathematical method

Separate three implementations:

1. **Canonical activation-weighted truncation.**
2. **Exact anchored truncation**, with \(H=XX^\top+\beta nI\).
3. **The current solver**, which adds further ridge shrinkage before truncation.

For exact teacher outputs, the anchored full-rank optimum is \(W\). The current extra ridge instead produces a shrinkage operation. The paper must describe these differences explicitly.

Use the existing tests and add targeted checks for:

- Full-rank recovery without extra shrinkage.
- Predicted shrinkage when ridge is present.
- Bias subtraction and GPT-2/Gemma matrix orientation.
- Correct handling of rank-deficient calibration.
- Teacher immutability.
- Split separation, metric masking, and accumulation across multiple batches.
- Packed checkpoint reload and output parity.
- FP16 factorized execution; the current packed layer's FP32 factor conversion needs correction.

Keep mathematical repairs separate from research experiments, and document their rationale before running the repaired implementation.

### Lock evaluation protocol v2

- **Calibration:** WikiText-2 train; 16 non-overlapping chunks of 512 tokens.
- **Seeds:** 11, 23, 37, paired across calibration-dependent methods.
- **Selection:** native validation split only.
- **Evaluation:** entire native test split, including the final partial window.
- **Context/stride:** 512/256.
- **Text processing:** retain dataset order, join rows with two newlines, disable added special tokens.
- **PPL:** exponentiate total NLL divided by the actual number of valid shifted targets; score each eligible target once.

Recompute every baseline under this protocol. Keep old subset and disjoint-window measurements labelled as historical results. Context and stride can materially change PPL, as explained in the [Hugging Face evaluation documentation](https://huggingface.co/docs/transformers/en/perplexity).

WikiText-2 has already influenced this research. Treat its test results as regression evidence; reserve the later second corpus for a fresh, locked confirmation.

## 3. First study: a bounded 24 GPU-hour sequence

These are spending caps, not runtime predictions. Finish complete comparisons before advancing. If a stage exceeds its allocation, preserve it as incomplete rather than changing the evaluation or seed budget.

| Stage | GPU-hour cap | Main question |
|---|---:|---|
| A. Readiness and identity controls | 1 | Are substitutions and metrics correct? |
| B. Regularization controls | 4 | Is the benefit weighting, anchoring, or shrinkage? |
| C. Strong baseline comparison | 7 | Does the method improve on activation-aware alternatives? |
| D. Composition and drift | 4 | Where does local error become global damage? |
| E. GPT-2 replication | 4 | Does the finding transfer across architectures? |
| F. Calibration and packed execution | 4 | Is apparent improvement smoothing, and is storage real? |

### A. Readiness and identity controls

Run an untouched teacher, a copied teacher, and a full-rank factorized replacement.

Require matching logits within a declared dtype tolerance, matching token counts, finite metrics, and no mutation of the teacher. Diagnose failures before proceeding.

### B. Separate regularization from truncation

On Gemma attention output projections, use rank 384 and calibration seed 11. Evaluate these controls on validation:

- Plain SVD.
- Unanchored full-covariance weighted truncation.
- Exact anchored truncation at \(\beta=0.1\).
- Current anchored-plus-ridge solver at its recorded settings.
- Full-rank ridge shrinkage without rank reduction.

Then change one variable at a time:

- Anchor sweep: \(\beta\in\{0,0.03,0.1,0.3,1\}\), without extra shrinkage.
- Extra-ridge sweep: fractions \(\{0,0.001,0.01\}\), holding \(\beta=0.1\).

Use a declared pseudoinverse convention for the unanchored singular case. Select settings by validation NLL, with ties resolved by less regularization. Freeze settings before test evaluation.

**Decision:** if full-rank shrinkage reproduces the improvement, rank truncation alone cannot explain it.

### C. Compare strong baselines fairly

Evaluate all attention output projections at the same rank, calibration tokens, factor precision, and parameter budget:

| Method | Purpose |
|---|---|
| Untouched dense teacher | Reference behavior |
| Plain SVD | Weight-space control |
| Fixed-rank diagonal ASVD | Channel-wise activation weighting |
| SVD-LLM whitening stage | Full-covariance activation weighting |
| Exact anchored truncation | Isolate the anchor |
| Current solver | Evaluate the recorded implementation |

Use ASVD's absolute-mean channel statistic and \(\alpha=0.5\) as its published default. Label fixed-rank ASVD and whitening-only SVD-LLM as **component comparisons**, not reproductions of their complete pipelines. Current SVD-LLM also includes recovery fine-tuning, which has a different training budget. [ASVD methods](https://arxiv.org/html/2312.05821v5), [SVD-LLM methods](https://arxiv.org/html/2403.07378v5).

Run calibration-dependent methods across all three seeds. Fit seed-independent SVD once.

Record:

- NLL/PPL and teacher-relative change.
- Teacher-to-student logit KL at temperature 1.
- Local activation-output error on teacher inputs.
- Residual hidden-state error after every block.
- Factor parameters and actual serialized bytes.
- Fit time and peak fitting memory.

Use normalized residual squared error, averaged across blocks, as the primary drift measure. Cosine distance, norm ratios, and linear CKA are secondary diagnostics.

### D. Test composition directly

Compare the selected solver with the strongest activation-aware baseline, chosen by validation NLL.

At fixed per-matrix rank, replace:

- One projection.
- Two consecutive projections.
- Four consecutive projections.
- The entire attention-output stack.

For each smaller group, use early, middle, and late positions: starts \(0\), \(\lfloor(L-g)/2\rfloor\), and \(L-g\), removing duplicates.

Measure drift immediately after the group and at every downstream block. Accumulate results over fixed held-out windows; do not retain only the last prompt.

**Decision:** improvement in local output error without improvement in downstream drift or NLL is a negative composition result. It does not justify declaring the drift problem solved.

If cumulative drift remains the main failure, the next comparison is a fixed-rank SAES-SVD reproduction. It explicitly addresses accumulated and local errors, making it more informative than another unrestricted objective sweep. [SAES-SVD](https://arxiv.org/abs/2602.03051).

### E. Replicate the decisive result on GPT-2

Repeat the frozen protocol on GPT-2 Small attention output projections at rank 256:

- Dense teacher.
- Plain SVD.
- Strongest activation-aware baseline.
- Selected solver.

Use three calibration seeds and the same metrics. Transfer Gemma-selected settings without retuning for this confirmation.

Add a validation-only rank frontier at divisors 2, 3, and 4 for the selected solver and strongest baseline. Report **rank reduction, target-matrix storage reduction, and whole-model savings separately**.

### F. Test calibration and actual factor storage

**Calibration control:** fit one logit temperature per model on validation, then freeze it. Compare raw teacher, temperature-scaled teacher, raw compressed model, and temperature-scaled compressed model.

Report NLL/PPL, next-token accuracy, entropy, and token confidence calibration using 15 fixed equal-width bins. Temperature scaling is a control adapted from [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html), not proof of an LLM denoising mechanism.

**Packed execution:** compare FP16 and INT8 factors for the selected solver. Store factors and quantization metadata; verify the replaced modules contain no dense reconstructed weights.

Measure actual checkpoint bytes and end-to-end inference:

- Batch size 1.
- Prefill lengths 128 and 512.
- Decode 128 tokens with identical cache settings.
- 10 warm-up iterations and 30 measured iterations.
- Synchronized GPU timing, median/p95 latency, peak allocated and reserved memory.

A slower packed model can still be a storage result. Replace “lossless INT8” with “no measurable additional quality loss under this protocol” unless stronger evidence supports exact equivalence.

## 4. Expansion, decision gates, and paper outputs

### Second wave: fresh evaluation and broader scope

After freezing the first study:

1. Prepare a pinned English C4 validation shard.
2. Take its first 1,024 unique non-empty documents, sort by content hash, and assign alternating documents to 512 development and 512 confirmation examples.
3. Cap each document at 2,048 tokens and record exact token manifests.
4. Evaluate frozen models on confirmation data without adjusting compression settings.
5. Add zero-shot HellaSwag and ARC-Challenge using a pinned lm-evaluation-harness revision, task definitions, and prompt format.

These are explicitly sampled benchmarks, not full-C4 measurements.

Keep further work in this order:

- SAES-SVD comparison if composition remains unresolved.
- MLP down-projection study, with separate conclusions.
- Adaptive rank allocation against SVD-LLM V2 or Zero Sum SVD at the same total budget.
- Quantized residual plus low-rank compensation, accounting for the complete backbone and branch storage.
- Qwen2.5-7B replication on a sufficiently large GPU.

Do not use CPU-offloaded 7B timings to claim GPU inference speedups.

### Evidence and stopping rules

Report each seed and mean/standard deviation. For NLL differences, use 2,000 paired bootstrap resamples of fixed contiguous 4,096-target blocks; do not bootstrap individual dependent tokens.

Advance a candidate only when it:

- Improves NLL or logit KL against the matched activation-aware baseline.
- Improves the predeclared drift measure.
- Reproduces across three seeds.
- Meets its declared storage and hardware limits.

Use proposed hardware limits of no more than 5% latency or peak-memory regression against the matched factorized baseline.

If differences remain within uncertainty, report no established advantage. If local metrics improve repeatedly while global behavior does not, stop tuning that hypothesis and retain the negative result.

A new-method claim requires a demonstrable mathematical distinction and evidence against the relevant strong comparator. Ridge-whitened truncation itself is already represented in prior work.

### Manuscript deliverables

Prepare a versioned revised manuscript with:

- One consistent abstract, contribution statement, and conclusion.
- Separate historical exploratory and protocol-v2 tables.
- Correct equations for weighting, anchoring, and extra shrinkage.
- Main baseline, composition, replication, and practical-cost tables.
- Drift-versus-depth and quality-versus-storage figures.
- Complete reproducibility appendix and all failed/incomplete runs.
- The bibliography below, with every reference used substantively in the text.

## 5. Expanded bibliography: 22 relevant papers

Use primary-source metadata to create the final bibliography; distinguish publication years from original preprint dates.

| # | Paper | Role |
|---:|---|---|
| 1 | Chen et al., 2021 — [DRONE: Data-aware Low-rank Compression for Large NLP Models](https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html) | Earlier data-aware low-rank theory |
| 2 | Hsu et al., 2022 — [Language Model Compression with Weighted Low-Rank Factorization](https://openreview.net/pdf?id=uPv9Y3gmAI5) | Task-sensitive weighted factorization |
| 3 | Yuan et al., 2023 — [ASVD](https://arxiv.org/abs/2312.05821) | Activation-aware decomposition baseline |
| 4 | Wang et al., 2024 — [SVD-LLM](https://arxiv.org/abs/2403.07378) | Covariance whitening and recovery |
| 5 | Wang et al., 2025 — [SVD-LLM V2](https://aclanthology.org/2025.naacl-long.217/) | Truncation and rank allocation |
| 6 | Liu et al., 2024 — [EoRA](https://arxiv.org/abs/2410.21271) | Activation-eigenspace residual compensation |
| 7 | Hu et al., 2026 — [SAES-SVD](https://arxiv.org/abs/2602.03051) | Local and accumulated error suppression |
| 8 | Liu et al., 2026 — [LACE-SVD](https://arxiv.org/abs/2607.03057) | Loss-aware allocation and cumulative correction |
| 9 | Abbasi et al., 2026 — [Zero Sum SVD](https://arxiv.org/abs/2602.02848) | Regularized whitening and global component selection |
| 10 | Abbasi et al., 2026 — [IO-SVD](https://arxiv.org/abs/2605.15626) | Input/output geometry and predictive sensitivity |
| 11 | Arai and Ichikawa, 2025 — [Quantization Error Propagation](https://arxiv.org/abs/2504.09629) | Related cross-layer error problem |
| 12 | Frantar et al., 2022 — [GPTQ](https://arxiv.org/abs/2210.17323) | Second-order quantization comparison |
| 13 | Lin et al., 2023 — [AWQ](https://arxiv.org/abs/2306.00978) | Activation importance and practical compression |
| 14 | Frantar and Alistarh, 2023 — [SparseGPT](https://arxiv.org/abs/2301.00774) | Reconstruction-aware pruning context |
| 15 | Li et al., 2024 — [SVDQuant](https://arxiv.org/abs/2411.05007) | Hybrid branch and runtime lessons; diffusion scope |
| 16 | Kornblith et al., 2019 — [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html) | CKA diagnostic |
| 17 | Guo et al., 2017 — [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html) | Temperature-scaling control |
| 18 | Biderman et al., 2024 — [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) | Evaluation protocol and reproducibility |
| 19 | Merity et al., 2016 — [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) | WikiText provenance |
| 20 | Raffel et al., 2019 — [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | C4 provenance |
| 21 | Zellers et al., 2019 — [HellaSwag](https://aclanthology.org/P19-1472/) | Downstream commonsense evaluation |
| 22 | Clark et al., 2018 — [Think You Have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457) | Downstream reasoning evaluation |

**Expected outcome:** a paper that establishes a reproducible advantage, identifies equivalence to existing methods, or explains a bounded failure. Each outcome is useful evidence; none requires endless tuning.

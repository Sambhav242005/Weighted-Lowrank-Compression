# Gemma 3 270M: completed FP32 CPU/GPU experiment

## Experiment status — 20261016-recovery-mvapk6bo

- State: completed. All 2,283 evaluation windows and 292,281 next-token targets are covered for each condition.
- Hypothesis: a small ReLU bottleneck can replace attention-output projections with a better quality/cost tradeoff than activation-weighted linear low-rank factors.
- Configuration and matched baseline: pinned `google/gemma-3-270m`, FP32, all 18 attention-output projections, common 70% target-parameter ceiling, shared train calibration and validation windows, two ReLU optimization seeds (11/23). Deterministic whitening is reused, not two independent baseline fits. Both widths are 275; ReLU has 458,515 parameters per projection versus whitening's 457,600 because of biases. This is a common ceiling, not exact parameter equality.
- Observed evidence: ReLU worsens full-test perplexity, logit divergence and hidden-state drift versus whitening in both optimization seeds. Whole-checkpoint savings are about 1.32%; no consistent runtime advantage is established.
- Decision: the hypothesis is weakened under this bounded protocol. These fitted ReLU replacements do not preserve global computation. This does not reject every nonlinear architecture or longer fitting budget.
- Next smallest experiment: if separately authorized, predeclare a calibration/validation-only learning-curve comparison changing fitting steps alone. ReLU validation loss was still decreasing at update 200. Do not tune against this test set or automatically expand the completed compute budget.
- Risks or missing evidence: one architecture, shared calibration, two optimization seeds, slight parameter-count mismatch, previously used WikiText-2, fixed benchmark order with substantial CPU timing variation, no full CPU test-split likelihood evaluation, no hand-written fused CUDA kernel, and memory measurements that retain reference/replacement modules.

## Quality

Native WikiText-2 raw test at dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, context/stride 256/128, final tail included. Lower values are better. Quality was measured on GPU.

| Method | Test perplexity | Teacher-to-student KL | Final normalized-state relative RMSE |
|---|---:|---:|---:|
| Dense | 47.2231 | 0 | 0 |
| Whitening (same result under both seed labels) | 58.3059 | 0.256396 | 0.324377 |
| ReLU, seed 11 | 191.9389 | 1.673883 | 0.635147 |
| ReLU, seed 23 | 214.2870 | 1.741339 | 0.644645 |

All 18 individual layers have higher local validation error for ReLU than whitening under this fitting budget. Local degradation therefore precedes the full-stack failure; it cannot be attributed solely to composition. Median local validation NMSE across layers is 0.01767 for whitening and 0.11577/0.11781 for ReLU. Method-specific full-rank whitening checks passed before accepting truncated factors.

CPU/GPU parity was checked on one fixed 256-token validation window for every condition. All choose the same next token; relative final-logit differences are below 0.000006. This is a numerical check, not full CPU test-set evaluation.

## Runtime and storage

GTX 1050 4 GB and Intel i5-8300H (four CPU threads); PyTorch 2.7.1+cu118, Transformers 4.55.4. Batch one, FP32, five warmups and twenty synchronized samples per workload. Decode workload includes a 128-token prompt plus 32 cached decode steps with no EOS stopping; it is not decode-only throughput. Prefill computes hidden states and final-position logits.

| Method | GPU prefill 128 (ms) | GPU prompt + decode (s) | CPU prefill 128 (ms) | CPU prompt + decode (s) |
|---|---:|---:|---:|---:|
| Dense | 47.17 | 1.797 | 228.42 | 2.289 |
| Whitening, first measurement | 77.01 | 1.821 | 203.50 | 2.239 |
| ReLU, seed 11 | 50.59 | 1.822 | 217.46 | 2.891 |
| Whitening, second measurement | 47.13 | 1.806 | 277.99 | 2.738 |
| ReLU, seed 23 | 49.40 | 1.842 | 266.36 | 2.669 |

These are medians. Raw samples, p95, 256-token prefill and memory readings are in `metrics.json`. Identical whitening checkpoints differ substantially across CPU timing passes; the second ReLU CPU p95 for prompt/decode is 10.02 seconds. Small apparent speed differences are not reliable evidence of acceleration. No extra timing experiment was launched to chase a favorable result.

| Representation | Target parameters (all 18) | Serialized targets (bytes) | Complete FP32 checkpoint (bytes) |
|---|---:|---:|---:|
| Dense | 11,796,480 | 47,191,981 | 1,072,489,021 |
| Whitening | 8,236,800 | 32,959,863 | 1,058,261,499 |
| ReLU | 8,253,270 | 33,036,235 | 1,058,338,019 |

ReLU reduces target parameters by about 30% and complete checkpoint bytes by about 1.32%. Complete files preserve tied embeddings and use PyTorch serialization; they must not be compared directly with the downloaded BF16 checkpoint. GPU benchmark peak allocated memory is about 1.30–1.31 GB, with retained teacher/replacement modules, so this is not isolated deployment memory. The approximately 2.74 GB process peak RSS is a cumulative high-water mark shared across conditions; it cannot establish per-method CPU memory savings. During the quality pass, device-wide observed use was about 2.5 GB.

## Provenance and recovery

Model revision: `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. Download SHA256 matches upstream: `abb58eb73aece5163624090d5383c59cbcdb9e9539ec84cb864f74386c4b21d5`.

Original immutable run: [20261016-mvanbe94](../results/gemma270m-relu/20261016-mvanbe94/). It stopped unexpectedly after the 1,881-window checkpoint without a traceback; its running status is stale and an interruption observation is retained. The cause remains unknown.

Recovery: [20261016-recovery-mvapk6bo](../results/gemma270m-relu/20261016-recovery-mvapk6bo/). It verifies package/source/checkpoint/test-token hashes, reuses completed fitting/timing/storage/parity and the quality prefix, and evaluates the remaining 402 windows using the same metric definitions. Uncheckpointed tail work is not double-counted. A CPU equivalence test confirms recovered quality equals uninterrupted evaluation. Fourteen offline checks passed before recovery launch.

Conservative original charge: 3,593 seconds, from directory creation through the first interruption observation. Recovery receives the remaining 3,607 seconds of the original 7,200-second cap. Total recorded charge at completion: 4,145.109 seconds (about 69.1 minutes), including the uncertain original tail and recovery verification. No new fitting or benchmark budget was added.

The recovery folder contains the combined metrics/report, recovery configuration, incremental logs, immutable numbered quality checkpoints and final audit. Large model/replacement checkpoints remain in the original run and are referenced rather than copied.

Final audit passed: 236 teacher tensors match the pinned upstream checkpoint, full-checkpoint hashes and replacement reloads match, non-target tensors and tied embeddings are preserved, numerical parity checks pass, and all methods cover the complete 292,281-target test sequence.

- [Combined metrics](../results/gemma270m-relu/20261016-recovery-mvapk6bo/metrics.json)
- [Generated report](../results/gemma270m-relu/20261016-recovery-mvapk6bo/report.md)
- [Recovery contract](../results/gemma270m-relu/20261016-recovery-mvapk6bo/resolved-config.json)
- [Audit](../results/gemma270m-relu/20261016-recovery-mvapk6bo/audit.json)

The earlier paper cohorts and manuscript claims are unchanged by this separate exploratory study.

The subsequent [forward-only diagnosis](GEMMA270M_DIAGNOSIS_20261016.md) isolates initialization and sign-clipping problems, incomplete fitting, compressed-input drift and the small CPU runtime share of the chosen targets. It does not replace these results or establish an improved trained candidate.

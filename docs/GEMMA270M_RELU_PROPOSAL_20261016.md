# Gemma 3 270M ReLU replacement: proposed exploratory study

State: completed, including interrupted-prefix recovery and final audit. This file preserves the planned protocol; see [the measured outcome](GEMMA270M_RESULTS_20261016.md).

## Authorization and preflight

The user authorized downloading Gemma 3 270M and evaluating a small ReLU replacement on CPU and GPU on 16 October 2026. This is a new study, separate from the closed 1B/Qwen/LFM cohorts and their exhausted allocations.

- Model: `google/gemma-3-270m` (pretrained variant).
- Resolved Hub revision: `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`.
- Model access was initially HTTP 401; the user configured access and the pinned model subsequently downloaded successfully. Do not record credentials in research artifacts.
- GPU observed: NVIDIA GeForce GTX 1050, 4096 MiB, driver 582.78. Native BF16 is unavailable on this GPU.
- CPU observed: Intel Core i5-8300H, four cores/eight logical processors.
- Existing research CUDA environment is absent in this checkout. Default Python 3.15 and bundled Python 3.12.14 have no torch/transformers/datasets installation.
- Precision: FP32 on both devices, explicitly approved by the user on 16 October 2026 for this new study, superseding the earlier BF16-only restriction within this study. Retain at most two calibration seeds, 11 and 23; results cannot meet the three-seed promotion gate.

## Proposed additive implementation

Add a separate experiment runner and focused offline tests; preserve historical code and results. First implement the replacement using standard PyTorch device operations: `B(ReLU(Ax + a)) + b`. This executes on GPU when tensors are on CUDA, but is not a hand-written fused CUDA kernel. Custom fusion is a later experiment after quality is established.

Compare the original dense attention-output projections, linear low-rank replacements, and small ReLU bottleneck replacements. Read actual matrix dimensions from the pinned model before fixing ranks. Set a common parameter ceiling at approximately 70% of dense target weights, accounting for every bias and factor. Choose the largest valid bottleneck/rank within that ceiling and report exact count differences. Use the existing whitening baseline where its numerical checks pass; include plain SVD as a diagnostic reference rather than silently replacing whitening.

Fit against teacher activations, freeze the teacher, select checkpoints using validation only, and apply replacements to all attention-output projections for the main comparison. Retain single-layer checks as implementation diagnostics only. Proposed bounded fitting: 200 minibatch updates per target and seed, batch 256 activation rows, Adam learning rate 0.001, no hyperparameter sweep. Proposed active-compute cap: two hours total, excluding environment installation/download. Freeze these values in the run contract before execution; any unfinished stage remains incomplete rather than extending the cap.

## Data and evaluation

Reuse the existing WikiText-2 raw protocol and dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`. No dataset is present in this checkout, so explicitly announce its required download before preparation. Calibration uses train only, checkpoint selection validation only, and final evaluation test only. Record exact document/token positions and hashes before fitting. Proposed calibration: 16 disjoint 256-token training windows. Validation: 8 fixed 256-token windows. Final quality: full native test, context/stride 256/128, including the final partial window. These lengths differ from the historical study and must not be treated as its matched continuation.

Measure NLL/perplexity, teacher-to-student logit KL, and normalized residual drift after every block. Stream/chunk vocabulary logits to stay within the 4 GB GPU budget; do not retain full-test teacher logits on GPU. Evaluate prediction quality on GPU, then check fixed-window CPU/GPU numerical agreement for each saved candidate. If full CPU quality evaluation cannot fit the frozen time cap, mark it incomplete; do not describe CPU parity as a full CPU test-set evaluation.

For each device and method, report target-only serialized bytes and full checkpoint bytes, batch-one 128/256-token prefill, and greedy 32-token cached decode after a 128-token prompt. Use five warmups and twenty synchronized measurements; record median and p95, CPU thread count (four), peak allocated/reserved CUDA memory, and peak process RSS with measurement scope stated. Compare methods within each device in the same precision and workload. Save raw timing samples and execution order; identify unrelated GPU workloads and offloading. A shorter exploratory timing workload must not be compared directly with historical timings.

## Run contract and outputs

Before launch create a unique `results/gemma270m-relu/<timestamp>-<id>/` directory, refusing reuse. Save the exact command, git revision and dirty diff/source hashes, Python/package/CUDA versions and GPU capability check, model/tokenizer/data pins, seeds, split manifests, target names/shapes, representation sizes, fitting budget, baseline definition, and timing protocol in `resolved-config.json`. Preserve `stdout.log`, `stderr.log`, per-stage status, fitted replacements, `metrics.json`, raw latency samples, and `report.md`. Refuse unsupported CUDA binaries, nonfinite losses, split changes, and mismatched budgets.

Decision: exploratory and inconclusive until measurements exist. A local-error improvement alone is insufficient. A ReLU candidate supports further investigation only if held-out global quality and a declared drift metric improve over the matched linear baseline, with measured storage/memory/latency tradeoffs. No claim of general preservation or practical acceleration follows from this one architecture and two seeds.

## Execution clarification

Authentication and precision are resolved. The pinned model and WikiText-2 are downloaded. PyTorch 2.7.1+cu118 is installed in `.venv-gemma270m`, with Transformers 4.55.4 and datasets 3.6.0; CUDA GEMM and offline checks must pass before launch.

Seeds 11 and 23 control ReLU initialization and minibatch ordering. Calibration windows are shared across these optimization seeds, so deterministic whitening rows are reused comparisons, not independent calibration repeats. The common 70% parameter ceiling gives rank/width 275 for both representations; the linear baseline has 457,600 weights and ReLU has 458,515 parameters per target. This small mismatch is disclosed rather than called an exact parameter match. CPU performance uses the full model; CPU quality is a fixed-window numerical parity check, not a full-test likelihood evaluation.

Official model/access reference: https://huggingface.co/google/gemma-3-270m
Official download reference: https://huggingface.co/docs/huggingface_hub/guides/download

# GPU-only custom U(Vx): execution result

## Experiment status — 20261016-relaunch-mvatd77f

- State: completed, 249.968 seconds including the conservatively charged failed startup, within the 900-second cap.
- Hypothesis: one custom FP32 CUDA kernel can execute the existing Gemma low-rank factors more efficiently without changing their mathematical function.
- Configuration and matched baseline: pinned Gemma 3 270M, GTX 1050 (compute capability 6.1), FP32, unchanged saved whitening seed-11 factors, rank 275, all 18 attention-output projections. Compare dense, GPU PyTorch low-rank, fused kernel for every row count, and fused single-row decode with GPU PyTorch prefill. No ReLU, refitting, dense-weight reconstruction, CPU inference or model/data download.
- Observed evidence: numerical checks pass, but this implementation establishes no speed advantage. Pure fusion is substantially slower for prefill; hybrid dispatch has similar prompt/decode latency to PyTorch.
- Decision: functionally validated in the checked cases; performance hypothesis weakened for this implementation/device. Retain the existing PyTorch factors for practical use.
- Next smallest experiment: if further work is separately authorized, profile GPU kernel and host launch time, then change one implementation variable. A two-stage optimized GPU execution path or CUDA Graph may outperform recomputing the intermediate per output tile; neither has been established here.
- Risks or missing evidence: one device/workload family, small implementation grid, approximate FP32 equivalence rather than bitwise identity, no full test-perplexity rerun, and no isolated deployment-memory measurement.

## Numerical verification

Seven GPU tests pass, including irregular dimensions, Gemma dimensions, a non-default stream, empty inputs, dtype/shape guards and loading the kernel in a fresh process before any CUDA tensor exists.

Eight original 256-token validation windows show fused final-logit relative error at most 2.47e-6 against the same PyTorch factors, with matching next-token choices. A 16-position greedy decode check has matching token choices at every checked position; maximum relative logit error is 5.42e-6 for pure fusion and below 8.8e-7 for hybrid dispatch. These checks validate the execution path on these inputs. They do not establish equality for all inputs or recover the linear approximation's existing quality loss against the dense teacher.

The original dense and low-rank perplexity results remain separate. The kernel executes the same saved approximation; it introduces no new compression objective or parameters.

## Full-model GPU latency

All values below are warmed host-wall medians in milliseconds, using five warmups and twenty samples per condition. Method order rotates within each measurement round. CUDA intervals, p95 and all raw samples are retained in the metrics file. Prompt transfer, allocations, Python/driver wrappers and cache handling are included identically. All model inference executes on GPU.

| Method | Prefill 128 | Prefill 256 | Prompt 128 + 32 cached steps |
|---|---:|---:|---:|
| Dense reference | 45.762 | 65.793 | 1,730.622 |
| Existing GPU PyTorch low-rank | 47.313 | 65.924 | 1,777.013 |
| Fused custom kernel, all row counts | 151.193 | 286.964 | 1,881.689 |
| Custom single-row decode, GPU PyTorch prefill | 47.230 | 66.642 | 1,785.707 |

The hybrid prompt/decode median is about 0.5% slower than PyTorch low-rank, a small difference relative to observed variation. Pure fusion is about 3.20x slower for 128-token prefill and 4.35x slower for 256-token prefill. No speedup should be claimed.

## Kernel mechanism and limits

`src/gpu_lowrank.py` compiles the hand-written CUDA C kernel through NVRTC 11.8 bundled with PyTorch. It loads PTX through the NVIDIA driver and launches on PyTorch's current CUDA stream. There is no extra toolkit/package installation. Output is FP32 and the module is inference-only.

Each block calculates the rank-sized `Vx` vector into shared memory, synchronizes, and calculates an output tile with `U`. Blocks for different output tiles independently recompute `Vx`; rows do not cooperate to reuse factor tiles as a GEMM implementation would. The fixed output-tile grid is 64/128/640 and input-row grid is 1/4/16/128/256. Tile 64 was the fastest custom one-row configuration in this measured grid, but still slower than the PyTorch microbaseline (approximately 0.125 versus 0.103 ms per operation including the host wrapper).

At tile 64, the 640-dimensional output requires ten blocks per row, hence ten independently computed latent vectors. This is a concrete arithmetic/parallelism tradeoff, not evidence that all fused kernels must be slow. The large prefill regression is consistent with this repeated work and the lack of matrix-tile reuse. Separating CUDA compute from host-wrapper costs needs further profiling; CUDA event intervals in the current script may include host launch gaps.

The first attempt failed before collecting timings because runtime initialization did not establish a current CUDA context before `cuModuleLoadDataEx`. The tests had previously allocated tensors and masked that startup path. A fresh-process regression test reproduced the error; explicitly materializing a PyTorch CUDA allocation fixed it, after which all seven checks passed. The failed traceback and pre-fix sources remain in the original attempt directory. The relaunch charges its setup time against the same cap.

## Artifacts

- [Final generated report](../results/gpu-lowrank/20261016-relaunch-mvatd77f/report.md)
- [Metrics and samples](../results/gpu-lowrank/20261016-relaunch-mvatd77f/metrics.json)
- [Resolved contract](../results/gpu-lowrank/20261016-relaunch-mvatd77f/resolved-config.json)
- [Audit](../results/gpu-lowrank/20261016-relaunch-mvatd77f/audit.json)
- [CUDA source snapshot](../results/gpu-lowrank/20261016-relaunch-mvatd77f/kernel.cu)
- [Preserved failed startup](../results/gpu-lowrank/20261016-mvascc7n/)

The audit verifies current source hashes, timing sample counts and cumulative cap compliance. Raw memory readings include retained reference/candidate state; they must not be interpreted as per-method deployment memory savings. Historical models, experiment files and results are preserved.

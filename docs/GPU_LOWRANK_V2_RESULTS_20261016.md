# GPU-only U(Vx): reuse and graph replay

## Experiment status — 20261016-mvb5e5qd

- State: completed in 290.984 seconds within the separate 600-second cap. Audit passed.
- Hypothesis: computing Vx once, then reusing it for U(Vx), with optional CUDA Graph replay, reduces overhead versus the first custom kernel and existing PyTorch factors.
- Configuration and matched baseline: same immutable Gemma 3 270M whitening seed-11 factors, rank 275, FP32, all 18 attention-output projections, GTX 1050. No ReLU, refitting, changed model/data revisions, CPU inference, dense reconstruction or new precision. All methods use the existing GPU cuBLAS prefill.
- Observed evidence: every new execution path passes numerical checks, but no measured speed advantage appears. Existing PyTorch has the lowest median for the one-row microbenchmark and full prompt/decode workload.
- Decision: performance hypothesis weakened for these implementations/device. Keep PyTorch as the execution baseline. This result does not establish that whole-block/model CUDA Graphs or other optimized kernels cannot help.
- Next smallest experiment: if further engineering work is separately authorized, profile the GPU/host timeline before changing another kernel. Compare exact work/dispatch/copy costs; do not combine a new precision, cache strategy and kernel in one comparison.
- Risks or missing evidence: twenty samples with variable tail latency, one architecture/device, small numerical checks rather than full-test reevaluation, retained graph/reference state in memory measurements, no per-operation timeline isolating each overhead.

## What changed

The v1 tiled fused kernel calculates Vx independently for each output tile. The v2 kernel instead writes Vx once to a rank-sized GPU buffer and reads it for the second projection. The default decode path launches these two stages separately.

Two additional decode variants capture either those custom stages or the existing cuBLAS matrix operations in a CUDA Graph. They copy the new input into a static GPU buffer, replay the graph and clone the output. The clone preserves ordinary tensor semantics: later replays cannot overwrite a caller-held output. Graph buffers are separate per CUDA stream. All variants dispatch multi-row prefill to the original GPU cuBLAS path.

This fixes repeated computation as an implementation issue. It does not automatically make the entire operation cheaper: staged calls retain host/driver dispatch costs, while graph replay adds input-copy, output-clone, graph-management and wrapper work. Their individual contributions are not isolated by this benchmark. The official [PyTorch CUDA Graph documentation](https://docs.pytorch.org/docs/2.7/notes/cuda.html#cuda-graphs) explains that replay benefits depend on static pointers and reduced launch overhead; it does not promise an improvement for a graph containing only two operations.

## Measurements

Five warmups and twenty samples per condition; method order rotates in each round. Values are GPU-inference host-wall medians, including the wrappers, allocations, prompt transfer and cache handling. Raw CUDA intervals and p95 are retained. Compilation and graph setup are excluded from warmed timings, but included in total run elapsed time.

| Implementation | One-row projection microbenchmark (ms) | Prefill 128 (ms) | Prefill 256 (ms) | Prompt128 + 32 cached steps (s) |
|---|---:|---:|---:|---:|
| Existing PyTorch factors | 0.0995 | 46.273 | 66.184 | 1.867 |
| v1 custom decode / cuBLAS prefill | 0.1311 | 46.586 | 66.085 | 1.874 |
| Two custom stages / cuBLAS prefill | 0.1663 | 46.994 | 65.906 | 1.921 |
| Graph of custom stages / cuBLAS prefill | 0.1258 | 47.286 | 65.813 | 1.941 |
| Graph of cuBLAS factors / cuBLAS prefill | 0.1231 | 46.944 | 66.099 | 1.939 |

Microbenchmarks use ten calls per timing sample; they are per-operation averages within those groups. Full workload measurements use one invocation per sample. The prompt/decode PyTorch p95 is 2.489 seconds; graph/custom p95 values range from approximately 2.437 to 2.899 seconds. Therefore small median differences should not be described as statistically established improvements or regressions. The new paths have higher observed medians and fail to establish acceleration.

Prefill is now around 46–47 ms for 128 tokens and 66 ms for 256 tokens, avoiding the first pure-fusion prefill result of 151/287 ms by retaining GPU cuBLAS. This does not outperform the existing PyTorch baseline; it restores the efficient prefill execution strategy.

## Numerical and implementation checks

- Fourteen GPU checks pass across v1/v2, including tails, stream handling, fresh-process startup, graph output independence and prefill dispatch.
- Eight 256-token original validation windows produce zero measured prefill discrepancy because these paths share the same cuBLAS prefill implementation.
- In the 16-position greedy decode checks, all checked token choices match. Custom stages and their graph show relative logit error below 8.8e-7; the cuBLAS graph matches the checked reference logits exactly.
- Final audit verifies source hashes, twenty timing samples per condition, parity checks and elapsed time below the cap.

These checks support numerical execution agreement on the measured inputs. They do not establish bitwise equality for all inputs or improve the existing compression quality. The original dense/whitening test perplexities remain 47.22/58.31; no full-test PPL rerun was performed for this execution-only change.

## Artifacts

- [Generated report](../results/gpu-lowrank-v2/20261016-mvb5e5qd/report.md)
- [Raw metrics](../results/gpu-lowrank-v2/20261016-mvb5e5qd/metrics.json)
- [Run contract](../results/gpu-lowrank-v2/20261016-mvb5e5qd/resolved-config.json)
- [Audit](../results/gpu-lowrank-v2/20261016-mvb5e5qd/audit.json)
- [v2 CUDA source](../results/gpu-lowrank-v2/20261016-mvb5e5qd/kernel.cu)
- [First GPU kernel result](GPU_LOWRANK_RESULTS_20261016.md)

Earlier source files and experiment results remain preserved. No benchmark is left running by this completed run.

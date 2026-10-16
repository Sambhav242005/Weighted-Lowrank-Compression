# GPU-only reuse and CUDA Graph execution ablation

Same FP32 factors, no ReLU, refitting or CPU inference. All methods use GPU cuBLAS prefill.
Staged computes Vx once and U(Vx) once. Graphs capture those stages or existing cuBLAS calls.
Graph timings include input copy and output clone; no mutable output aliases escape.

| Workload | Method | Wall median ms | Wall p95 ms | CUDA interval median ms |
|---|---|---:|---:|---:|
| prefill128 | pytorch | 46.273 | 50.529 | 46.240 |
| prefill128 | v1_fused_decode | 46.586 | 53.123 | 46.553 |
| prefill128 | staged | 46.994 | 54.450 | 46.958 |
| prefill128 | graph_staged | 47.286 | 58.919 | 47.254 |
| prefill128 | graph_cublas | 46.944 | 52.295 | 46.913 |
| prefill256 | pytorch | 66.184 | 68.666 | 66.130 |
| prefill256 | v1_fused_decode | 66.085 | 70.222 | 66.035 |
| prefill256 | staged | 65.906 | 70.213 | 65.853 |
| prefill256 | graph_staged | 65.813 | 67.816 | 65.768 |
| prefill256 | graph_cublas | 66.099 | 67.806 | 66.049 |
| prompt128_decode32 | pytorch | 1866.640 | 2489.380 | 1866.650 |
| prompt128_decode32 | v1_fused_decode | 1873.530 | 2591.352 | 1873.549 |
| prompt128_decode32 | staged | 1921.132 | 2436.596 | 1921.149 |
| prompt128_decode32 | graph_staged | 1940.790 | 2898.510 | 1940.800 |
| prompt128_decode32 | graph_cublas | 1939.413 | 2657.544 | 1939.434 |

Five warmups, twenty rotated-order samples; compilation/capture excluded from warmed timings.
Prompt transfer, host wrappers, allocations and cache handling included identically. CUDA intervals may include host gaps.
Buffers retained per stream; factor buffers must not be replaced after graph capture.
Checked parity on eight validation windows plus16 generated positions; no full-test PPL rerun.
No quality improvement or broad hardware speedup is implied. Original low-rank quality loss remains.

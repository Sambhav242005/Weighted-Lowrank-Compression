# GPU-only U(Vx) execution benchmark

No ReLU, refitting, CPU inference or factor changes.
The custom kernel uses shared-memory latent storage. Hybrid uses it for one-row decode and GPU PyTorch for prefill.
This is a numerical execution comparison, not a new compression quality result.

| Workload | Method | Wall median ms | Wall p95 ms | CUDA interval median ms |
|---|---|---:|---:|---:|
| prefill128 | dense | 45.762 | 58.774 | 45.721 |
| prefill128 | pytorch_lowrank | 47.313 | 56.386 | 47.274 |
| prefill128 | fused_all_rows | 151.193 | 152.729 | 151.152 |
| prefill128 | fused_decode_pytorch_prefill | 47.230 | 55.675 | 47.184 |
| prefill256 | dense | 65.793 | 69.252 | 65.723 |
| prefill256 | pytorch_lowrank | 65.924 | 68.187 | 65.877 |
| prefill256 | fused_all_rows | 286.964 | 289.092 | 286.923 |
| prefill256 | fused_decode_pytorch_prefill | 66.642 | 72.436 | 66.593 |
| prompt128_decode32 | dense | 1730.622 | 1796.224 | 1730.631 |
| prompt128_decode32 | pytorch_lowrank | 1777.013 | 1862.932 | 1777.001 |
| prompt128_decode32 | fused_all_rows | 1881.689 | 1941.558 | 1881.690 |
| prompt128_decode32 | fused_decode_pytorch_prefill | 1785.707 | 1818.529 | 1785.714 |

Times include Python/driver wrappers, allocations, and identical prompt transfer. CUDA intervals can include host launch gaps.
Compile/setup excluded from warmed timings; full run elapsed time includes setup. Methods interleave to reduce order bias.
Pure fusion can recompute Vx per output tile; this trades extra arithmetic for parallelism and one launch.
Validation checks cover eight windows and 16 generated positions; complete test PPL was not rerun.
The saved linear approximation still has its previously measured quality loss versus dense; a kernel does not remove it.

# GPU-only WikiText-2 test perplexity

FP32, context/stride256/128, complete native test tail.
Same saved rank275 factors; no ReLU, fitting or precision changes.

| Metric | Dense | Old PyTorch low-rank | New staged | New graph staged | New graph cuBLAS |
|---|---:|---:|---:|---:|---:|
| Full test perplexity | 47.22310574 | 58.30585813 | 58.30585813 | 58.30585813 | 58.30585813 |
| Full test targets | 292281 | 292281 | 292281 | 292281 | 292281 |
| Decode-subset perplexity (128 targets only) | 42.89579617 | 58.92170507 | 58.92170262 | 58.92170262 | 58.92170507 |

Standard full-sequence PPL exercises the shared cuBLAS prefill path for new modules.
The separate teacher-forced decode subset exercises custom/graph paths; it is not full-test PPL.
These are repeated measurements on previously seen data, not independent generalization evidence.

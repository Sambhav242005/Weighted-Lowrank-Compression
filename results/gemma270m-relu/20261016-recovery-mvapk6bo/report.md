# Gemma 3 270M exploratory ReLU study

FP32, GTX 1050 and i5-8300H; two seeds.
Standard PyTorch GPU operations; no custom fused CUDA kernel.

| Method | Test PPL | Teacher KL | Final drift RMSE | Full bytes |
|---|---:|---:|---:|---:|
| dense | 47.2231 | 0.000000 | 0.000000 | 1072489021 |
| whitening_seed11 | 58.3059 | 0.256396 | 0.324377 | 1058261499 |
| relu_seed11 | 191.9389 | 1.673883 | 0.635147 | 1058338019 |
| whitening_seed23 | 58.3059 | 0.256396 | 0.324377 | 1058261499 |
| relu_seed23 | 214.2870 | 1.741339 | 0.644645 | 1058338019 |

| Device | Method | Prefill 128 ms | Prefill 256 ms | Prefill128 + decode32 ms |
|---|---|---:|---:|---:|
| cuda | dense | 47.17 | 67.44 | 1796.76 |
| cuda | whitening_seed11 | 77.01 | 65.95 | 1821.02 |
| cuda | relu_seed11 | 50.59 | 67.11 | 1821.99 |
| cuda | whitening_seed23 | 47.13 | 67.79 | 1805.83 |
| cuda | relu_seed23 | 49.40 | 66.18 | 1841.64 |
| cpu | dense | 228.42 | 396.83 | 2288.67 |
| cpu | whitening_seed11 | 203.50 | 364.06 | 2239.19 |
| cpu | relu_seed11 | 217.46 | 369.11 | 2891.09 |
| cpu | whitening_seed23 | 277.99 | 450.82 | 2737.58 |
| cpu | relu_seed23 | 266.36 | 497.41 | 2668.54 |

Prediction quality is evaluated on GPU; CPU uses fixed-window numerical parity checks.
Two seeds and one architecture cannot meet the project's promotion gate. Prior WikiText use limits generalization.
Process peak RSS is cumulative process high-water mark, not an isolated per-method memory delta.
Full checkpoint size uses PyTorch serialization with tied weights retained, not the downloaded BF16 file size.

Recovery reused the interrupted run's fitted weights, CPU/GPU timings and first 1881 quality windows. Checkpoints remain at `C:\Users\sambh\Desktop\Weighted-Lowrank-Compression\results\gemma270m-relu\20261016-mvanbe94`.

# BF16/FP8 numerical readiness

Single-seed full-rank controls; no compression efficacy or hardware-performance claim.

FP8 factors/weights use row-scaled storage decoded to BF16 for GEMM. Native FP8 support is a separate tiny probe.

State: running.

| Variant | Test NLL | PPL | Probe max error | Probe pass | NLL pass |
|---|---:|---:|---:|---|---|
| teacher_bf16 | 3.451549498 | 31.549240 | 0 | True | True |
| copy_bf16 | 3.451549498 | 31.549240 | 0 | True | True |
| identity_bf16 | 3.451549498 | 31.549240 | 0 | True | True |
| svd_bf16 | 3.452658125 | 31.584236 | 1.0625 | False | False |
| svd_fp32_compute | 3.451583723 | 31.550320 | 1.15625 | False | True |
| svd_fp8_e4m3 | 3.455429644 | 31.671893 | 9.73046875 | False | False |
| svd_fp8_e5m2 | 3.470559478 | 32.154727 | 7.8671875 | False | False |

Reference: untouched BF16 dense teacher. Probe: first 512 validation tokens; atol=0.1, rtol=0.01. Test: every native WikiText-2 shifted target, context 512 / stride 256; NLL difference tolerance 0.001. A failed numerical gate is retained and does not prevent the other predeclared controls from being measured.

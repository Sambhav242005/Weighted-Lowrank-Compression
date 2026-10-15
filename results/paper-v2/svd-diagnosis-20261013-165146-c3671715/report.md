# Full-rank SVD rounding diagnosis

Passing declared identity gates: ['dense_svd_fp32', 'float32_execution'].

| Variant | NLL | PPL | KL | Max probe logit error | Probe gate | NLL gate |
|---|---:|---:|---:|---:|---|---|
| dense_svd_fp32 | 3.451660701 | 31.552748 | 5.58944116e-05 | 0.117188 | True | True |
| dense_svd_fp16 | 3.451704808 | 31.554140 | 6.26759572e-05 | 0.140625 | False | True |
| unbalanced_fp16 | 3.451860161 | 31.559043 | 6.08428028e-05 | 0.152344 | False | True |
| balanced_fp16 | 3.451848345 | 31.558670 | 8.98418231e-05 | 0.181641 | False | True |
| float32_execution | 3.451771295 | 31.556238 | 7.56557404e-05 | 0.117188 | True | True |

This diagnoses full-rank numerical behavior; it is not compression evidence.

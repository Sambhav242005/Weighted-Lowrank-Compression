# GPU execution manuscript revision — 16 October 2026

[Updated nine-page IEEE PDF](../paper-future-scope-20261015/ieee-conference-gpu-execution-20261016.pdf)
was compiled with Tectonic 0.17.0. Rendered pages were visually checked, the
numeric table values were verified in the export, and no text lies outside
page boundaries. Export hashes and checks are in [validation.json](validation.json).

The primary [IEEE source](../paper-future-scope-20261015/ieee-conference.tex)
now includes the separate Gemma 3 270M FP32 GPU execution study: protocol,
full-test perplexity, teacher-forced decode diagnostic, warmed latency table,
storage context and limitations. It compares unchanged rank-275 whitening
factors executed through PyTorch, custom CUDA stages and per-projection graphs.
No ReLU is added.

The measured execution paths preserve the compressed reference's likelihood;
they do not recover dense quality or establish acceleration. Full-test PPL
uses shared cuBLAS prefill; the short decode diagnostic tests the custom paths.
Dense latency is omitted from the new table because dense was absent from the
same rotating-order timing cohort. This single-fit, single-device cohort is
kept separate from the paper's earlier FP16/BF16 experiments.

No experiments were launched for this edit. Frozen input hashes and extracted
numbers are in [evidence-extraction.json](evidence-extraction.json).
[The pre-edit source](ieee-conference-before-gpu.tex) is preserved alongside
its hash in [before.json](before.json). The reference-style manuscript and
Markdown remain historical versions; this update targets the primary IEEE
source. Earlier PDF exports remain preserved.

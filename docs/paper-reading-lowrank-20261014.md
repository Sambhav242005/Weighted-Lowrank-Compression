# Primary-paper reading notes: low-rank methods

Read on 14 October 2026 for the two-column manuscript revision. No models were run, and no benchmarks were repeated. Coverage below records complete primary full-text reading, including methods, equations, experiment descriptions, table text, figure captions, references, and the available appendices. Raster plots were not individually inspected. Synthesis paragraphs are kept below 110 words per paper.

Primary files and extracted text are cached outside the repository at `C:/Users/NPC/Downloads/paper-reading-sources-20261014/`.

## DRONE — Chen et al., NeurIPS 2021

- Main source: [Conference PDF](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf).
- Supplement: [Conference supplement](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Supplemental.pdf).
- Read: all 14 main-document pages; Sections 1–6, algorithms, references/checklist; all seven supplementary pages, Appendices A–H, including the theorem proof and rank-search pseudocode.
- Key locators: main Section 3.1, Theorem 1; Section 3.3; Section 4.1; supplementary Appendices A, B, D, E.

DRONE solves a rank-constrained local output approximation, expressed as minimizing `||WX-WMX||_F²`, with an explicit closed form. Its optimal local loss overlaps unanchored covariance-weighted truncation; this does not establish global transformer equivalence. The complete procedure already compresses modules in topological order to handle shifted representations and selects ranks using loss budgets and observed inference costs. Experiments cover BERT/GLUE, distilled models, LSTM language modeling, and translation; some reported results include retraining. Related work should acknowledge both the local closed-form precedent and sequential treatment of representation shifts. Its speedups cannot be transferred to this project's GPU workloads.

## Fisher-weighted factorization — Hsu et al., ICLR 2022

- Source: [Full arXiv HTML, v1](https://arxiv.org/html/2207.00112v1).
- Read: complete HTML document, Sections 1–7, references, and Appendix A with supplementary captions/tables.
- Key locators: Section 2.2, Equation 3; Section 4, Equations 5–6; Sections 5.2.2 and 6.

FWSVD estimates parameter importance from squared gradients of the target-task loss, then simplifies elementwise Fisher weighting into shared row importance so standard SVD can solve the transformed objective. This is task-aware weighting, rather than input-covariance whitening. Evaluations concern BERT-family task-specific models on GLUE and named-entity recognition, with both factorization-only and subsequent fine-tuning results. Its strongest compact-model results should not be described as uniformly training-free. The authors explicitly require a task objective and training data and identify the rowwise approximation as a limitation. Related work should distinguish Fisher weighting from the calibration-input geometry studied here.

## ASVD — Yuan et al., arXiv v5, 28 August 2025

- Sources: [Full HTML](https://arxiv.org/html/2312.05821v5), [PDF](https://arxiv.org/pdf/2312.05821v5).
- Read: complete 17-page PDF, Sections 1–5, references, Appendices A.1–A.8.
- Key locators: Section 3.3, Equations 8–9; Sections 3.4–3.5 and 4.1–4.2; Appendices A.3–A.4 and A.8.

ASVD combines activation-magnitude scaling with sensitivity-based per-layer rank search; v5 also evaluates ASVD+, using a Cholesky factor of the activation Gram matrix. For invertible covariance in exact arithmetic, this covariance alternative shares the unanchored output-error geometry of our whitening baseline, although spectral cutoffs and implementations differ. Our diagonal fixed-rank row isolates only magnitude scaling. The complete paper also studies KV-cache compression and factor quantization. Its limitations include MLP sensitivity, calibration/search cost, and worsening low-bit quality. Related work should name ASVD+ explicitly and avoid treating diagonal ASVD as the strongest complete ASVD pipeline.

## SVD-LLM — Wang et al., arXiv v5, ICLR 2025

- Sources: [Full HTML](https://arxiv.org/html/2403.07378v5), [PDF](https://arxiv.org/pdf/2403.07378v5).
- Read: complete 21-page v5 PDF, Sections 1–6, references, Appendices A.1–A.11.
- Key locators: Section 3.1, Theorem 3.2/Corollary 3.3; Section 3.2; experimental setup; Appendices A.5, A.8, A.11.

SVD-LLM whitens using the activation Gram matrix and links discarded singular values to local output loss; Appendix A.5 explicitly relates its optimum to DRONE. The v5 full method additionally performs sequential factor-wise LoRA fine-tuning using Alpaca data, while SVD-LLM(W) denotes whitening alone. The project's fixed-rank eigenvalue-pseudoinverse baseline isolates that local objective, with different conditioning and factor conventions; it does not reproduce the full method. Experiments span several decoder families and hardware workloads. Declared limitations include high-compression quality, KV reconstruction latency, and poor generation despite favorable perplexity. Related work must identify the version and whitening-only comparison.

## EoRA — Liu et al., arXiv v6, 17 March 2026

- Sources: [Full HTML](https://arxiv.org/html/2410.21271v6), [PDF](https://arxiv.org/pdf/2410.21271v6).
- Read: complete 25-page PDF, Sections 1–6, references, Appendices A.1–A.14.
- Key locators: Sections 2–3, Equations 3–9/Theorem 1; Sections 4.1 and 4.5; Appendices A.6, A.12–A.14.

EoRA fits an activation-eigenspace-weighted approximation of the residual error `W-W_compressed`, adding low-rank compensation while retaining a shared compressed backbone. This differs in representation and total budget from replacing W outright. Core compensation is gradient-free; separate experiments use EoRA initialization for fine-tuning. Its inverse spectral factors require a convention for zero eigenvalues, despite discussion emphasizing symmetry. Task-specific calibration, added branch storage, and runtime overhead remain relevant. Reported speedups use specialized fused quantized execution; they are not generic low-rank speedups. Related work should preserve the compensation/replacement distinction and avoid transferring its accuracy or kernel results to this study.

## Manuscript follow-through

The revised Related Work now acknowledges sequential representation shifts in DRONE, ASVD+ covariance overlap, SVD-LLM(W) versus the v5 factor-wise LoRA update, and EoRA's additive compensation setting. For covariance-overlap wording, explicitly restricting equivalence to invertible covariance in exact arithmetic avoids conflating Cholesky whitening with this implementation's truncated spectral pseudoinverse. No full-pipeline reproduction or novel general closed-form algorithm is established by the saved project comparisons.

---
bibliography: paper_protocol_v2.bib
---

# Activation-weighted compression and global behavior: revised evidence manuscript

Version: 2026-10-14T06:15:26.638218+00:00

## Abstract

This report examines whether matched activation-weighted low-rank approximations preserve global transformer behavior. It retains weighting/anchoring controls, full-stack and composition measurements, separate BF16/FP8 numerical controls, and the user-amended hybrid-architecture panel. Tables below are generated from retained artifacts; failed/incomplete attempts remain visible. The current scientific conclusion is inconclusive, promotion is disabled, and no universal or practical compression advantage is established. A terminal dispatcher does not imply every scientific cohort completed.

## Study status and scope

Dispatcher: **cancelled**. Report mode: terminal orchestration report.

The amended panel uses Gemma, Qwen3.5-0.8B and LFM2.5-1.2B-Instruct; GPT-2 is excluded. Only full-attention output projections are targeted in the hybrids. Native-tokenizer PPL is compared within each model. Validation frontier ranks and primary test ranks remain separate; no selection is made using confirmation outcomes.

## Mathematical methods and frozen constants

Let $W\in\mathbb{R}^{o\times i}$ and $X\in\mathbb{R}^{i\times n}$. Write $A=UV$ with rank at most $r$. The anchored objective is $\|(W-A)X\|_F^2+\beta n\|W-A\|_F^2$.

Set $H=XX^T+\beta nI$ and $C=WH$ for exact teacher outputs. For measured outputs subtract the teacher bias first: $Y_b=Y-b\mathbf{1}^T$, then $C=Y_bX^T+\beta nW$. The extra-ridge solver adds $\lambda I$ to the geometry, with $\lambda=\rho\,\operatorname{mean}(\operatorname{eig}(H))$. For $G=H+\lambda I$, form $A_\mathrm{full}=CG^\dagger$ and truncate $A_\mathrm{full}G^{1/2}$ to rank $r$; multiply back by $G^{\dagger/2}$. Thus ridge-free full-rank anchored recovery gives $W$ on retained spectral directions, while positive extra ridge generally shrinks it.

The implementation computes weighted covariance, eigendecomposition and weighted truncation in FP64, then returns FP32 factors for the declared storage cast. It clamps negative numerical eigenvalues to zero and retains regularized eigenvalues strictly above $\mathrm{rcond}\,\lambda_\max(G)$. Plain SVD uses FP32. Fixed-rank diagonal ASVD uses $s_j=\max(\operatorname{mean}_t|X_{jt}|,10^{-8})^{0.5}$, truncates $W\operatorname{diag}(s)$, and scales back. Factors use the recorded unbalanced convention unless a run contract explicitly declares a separate convention.

The predeclared validation anchor sweep is beta={0,0.03,0.1,0.3,1}; the extra-ridge fractions are rho={0,0.001,0.01} at beta=0.1. Validation NLL ties within 1e-6 prefer less regularization. Native evaluation uses context/stride 512/256, full shifted targets including the tail, and 16 disjoint 512-token calibration chunks for seeds 11/23/37. The primary Gemma attention-output rank is 384. The amended hybrid rank is floor((17/24)*out*in/(out+in)); frontier ranks are floor(min(out,in)/divisor) for validation-only divisors2/3/4. Exact observed run constants and factor precisions follow below; missing fields remain unrecorded.

| Run | Precision | Main rank | Pseudoinverse rcond | Config beta | Config ridge fraction | Selected solver / baseline |
|---|---|---:|---:|---:|---:|---|
| baselines-20261013-175255-6a00d4e8 | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": "SVD and fixed-rank ASVD / whitening-only components, same FP16/rank", "solver": null} |
| composition-20261013-185512-2bffd477 | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0}, "solver": null} |
| composition-20261014-032120-486d799f | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0}, "solver": null} |
| composition-20261014-054423-dbd821ff | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0}, "solver": null} |
| precision-readiness-20261014-060840-6db37daf | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": "untouched dense BF16 teacher and copied BF16 control", "solver": null} |
| regularization-20261013-171941-804f4c2e | float16 | 384 | 1e-07 | 0.1 | 0.01 | {"baseline": "SVD and fixed-rank ASVD / whitening-only components, same FP16/rank", "solver": null} |
## Related work and scope of the planned evidence

Data-aware low-rank compression predates this study. [DRONE](https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html)
derives a closed-form approximation informed by layer representations in BERT
[@chen2021drone]. [Fisher-weighted factorization](https://arxiv.org/abs/2207.00112)
uses parameter importance to address disagreement between matrix reconstruction
and task accuracy [@hsu2022weighted]. These are context for the present question:
low local weight error alone cannot establish preserved global computation.

[ASVD](https://arxiv.org/abs/2312.05821) adapts decomposition to activation
distributions and layer sensitivity [@yuan2023asvd].
[SVD-LLM](https://arxiv.org/abs/2403.07378) combines truncation-aware whitening
with sequential updates after compression [@wang2024svdllm]. Our fixed-rank
diagonal ASVD and whitening-only SVD-LLM isolate their weighting components
under a matched factor budget. They are not full-pipeline reproductions.
[SVD-LLM V2](https://aclanthology.org/2025.naacl-long.217/) changes rank allocation
and singular-value truncation [@wang2025svdllmv2], so it belongs in a later
adaptive-budget study rather than a claim that fixed per-matrix ranks reproduce
its complete procedure.

Global composition is already an explicit target of related work.
[SAES-SVD](https://arxiv.org/abs/2602.03051) addresses accumulated and local
errors [@hu2026saessvd]; [LACE-SVD](https://arxiv.org/abs/2607.03057) combines
loss-aware allocation with residual-stream correction [@liu2026lacesvd].
Our smaller-group and downstream-drift comparisons test the location and extent
of composition damage, without claiming to resolve it. The next comparator is
SAES-SVD if cumulative drift remains the decisive failure.
[Quantization Error Propagation](https://arxiv.org/abs/2504.09629) demonstrates
why accumulated errors also matter in layer-wise quantization [@arai2025qep].
That connection motivates measuring downstream behavior, but does not equate
low-rank truncation with quantization or transfer an existing method's results.

Loss-aware global allocation provides further distinctions from this study's
fixed-rank setup. [Zero Sum SVD](https://arxiv.org/abs/2602.02848) uses whitened
component sensitivities under a global pruning rule [@abbasi2026zerosum].
[IO-SVD](https://arxiv.org/abs/2605.15626) adds an output-side predictive metric
and heterogeneous rank selection [@abbasi2026iosvd]. These motivate future
equal-total-budget comparisons. They do not establish novelty for an anchored
input-covariance truncation or for the current extra-ridge solver.

The practical comparison distinguishes storage, accuracy, and execution cost.
[GPTQ](https://arxiv.org/abs/2210.17323) uses approximate second-order information
for post-training quantization [@frantar2022gptq].
[AWQ](https://arxiv.org/abs/2306.00978) uses activation-informed channel protection
and an inference implementation [@lin2023awq].
[SparseGPT](https://arxiv.org/abs/2301.00774) studies one-shot pruning
[@frantar2023sparsegpt]. These are different representations, not matched
fixed-rank baselines in the first wave. Their practical emphasis motivates
recording actual checkpoint bytes and warmed end-to-end timings.

[EoRA](https://arxiv.org/abs/2410.21271) compensates compressed models with
eigenspace-informed low-rank branches [@liu2024eora].
[SVDQuant](https://arxiv.org/abs/2411.05007) absorbs outliers in a high-precision
low-rank branch while quantizing a residual branch, and identifies the runtime
overhead of naive separate branch execution [@li2024svdquant]. SVDQuant's
published scope is diffusion models. It supplies a system-design warning for
future hybrid experiments here, not evidence of LLM quality or speedups.

[CKA](https://proceedings.mlr.press/v97/kornblith19a.html) measures alignment
between representation similarities [@kornblith2019cka]. We use it as a
secondary diagnostic beside the predeclared normalized residual squared error;
neither representation similarity nor norm preservation substitutes for held-out
NLL or logit KL. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)
is a validation-fitted calibration control [@guo2017calibration], adapted here
to next-token probabilities. Improvement after temperature fitting would not
by itself identify an LLM denoising mechanism.

Evaluation conventions can change apparent model rankings; the reproducibility
guidance in [Biderman et al.](https://arxiv.org/abs/2405.14782) motivates pinned
assets, task formats and declared target masking [@biderman2024evaluation].
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) introduced
WikiText [@merity2016wikitext]. Because WikiText has already informed this
project, its present test evaluation is regression evidence. The later fresh
confirmation samples a locked C4 shard, whose provenance is the
[T5 study](https://jmlr.org/papers/v21/20-074.html) [@raffel2020t5]. Neither its
sampled confirmation nor the initial WikiText wave represents all text domains.
[HellaSwag](https://aclanthology.org/P19-1472/) supplies a commonsense completion
benchmark [@zellers2019hellaswag], and [ARC](https://arxiv.org/abs/1803.05457)
supplies science-question reasoning with an explicit Challenge subset
[@clark2018arc]. Those downstream evaluations remain pending; no downstream
accuracy claim follows from the readiness controls.

## Verified bibliography and dates

The additive [BibTeX file](paper_protocol_v2.bib) contains the 22 plan references,
verified from primary metadata on 2026-10-13. Preprint entries use the first
submission year, record the checked revision, and separately note a venue when
the primary page identifies one. In particular SVD-LLM is a 2024 preprint with
ICLR 2025 metadata, GPTQ is a 2022 preprint with ICLR 2023 metadata, AWQ is a
2023 preprint with MLSys 2024 metadata, and SVDQuant is a 2024 preprint with ICLR
2025 metadata. EoRA's current primary metadata identifies **ICLR 2026 workshops**,
not main-conference publication. T5 is cited as its 2020 JMLR article, with its
2019 preprint date recorded separately. Unverified venues are omitted.

Author lists reflect the verified source version. The ASVD v5 author list and
the EoRA v6 list differ from some earlier bibliographies; this file does not
silently mix versions. Hsu et al.'s author spelling follows the conference PDF
(Sung-En Chang); the arXiv metadata spells that name Sungen Chang. The direct
OpenReview page returned a browser challenge, so metadata was cross-checked
with the primary paper's indexed PDF and its arXiv record.

## Recorded numerical controls

FP16 SVD readiness remains failed. Later BF16/FP8 variants are separate numerical cohorts, not repaired FP16 evidence. FP8 codes are decoded before BF16 multiplication; native tiny-GEMM support does not establish native model speed.

| Run / state | Variant | NLL | PPL | Maximum probe error | Probe gate | NLL gate |
|---|---|---:|---:|---:|---|---|
| precision-readiness-20261014-060840-6db37daf / cancelled | teacher_bf16 | 3.4515495 | 31.5492399 | 0 | True | True |
| precision-readiness-20261014-060840-6db37daf / cancelled | copy_bf16 | 3.4515495 | 31.5492399 | 0 | True | True |
| precision-readiness-20261014-060840-6db37daf / cancelled | identity_bf16 | 3.4515495 | 31.5492399 | 0 | True | True |
| precision-readiness-20261014-060840-6db37daf / cancelled | svd_bf16 | 3.45265812 | 31.5842357 | 1.0625 | False | False |
| precision-readiness-20261014-060840-6db37daf / cancelled | svd_fp32_compute | 3.45158372 | 31.5503197 | 1.15625 | False | True |
| precision-readiness-20261014-060840-6db37daf / cancelled | svd_fp8_e4m3 | 3.45542964 | 31.6718934 | 9.73046875 | False | False |
| precision-readiness-20261014-060840-6db37daf / cancelled | svd_fp8_e5m2 | 3.47055948 | 32.1547273 | 7.8671875 | False | False |
| readiness-20261013-164122-cdc919e9 / failed | teacher | 3.45156254 | 31.5496515 | 0 | True | unrecorded |
| readiness-20261013-164122-cdc919e9 / failed | copied_teacher | 3.45156254 | 31.5496515 | 0 | True | unrecorded |
| readiness-20261013-164407-bda5b4d9 / completed | teacher | 3.45156254 | 31.5496515 | 0 | True | unrecorded |
| readiness-20261013-164407-bda5b4d9 / completed | copied_teacher | 3.45156254 | 31.5496515 | 0 | True | unrecorded |
| readiness-20261013-164407-bda5b4d9 / completed | full_rank_factorized | 3.45156254 | 31.5496515 | 0 | True | unrecorded |
| svd-diagnosis-20261013-165146-c3671715 / completed | teacher | 3.45156254 | 31.5496515 | unmeasured | unrecorded | unrecorded |
| svd-diagnosis-20261013-165146-c3671715 / completed | dense_svd_fp32 | 3.4516607 | 31.5527485 | 0.1171875 | True | True |
| svd-diagnosis-20261013-165146-c3671715 / completed | dense_svd_fp16 | 3.45170481 | 31.5541402 | 0.140625 | False | True |
| svd-diagnosis-20261013-165146-c3671715 / completed | unbalanced_fp16 | 3.45186016 | 31.5590426 | 0.15234375 | False | True |
| svd-diagnosis-20261013-165146-c3671715 / completed | balanced_fp16 | 3.45184835 | 31.5586697 | 0.181640625 | False | True |
| svd-diagnosis-20261013-165146-c3671715 / completed | float32_execution | 3.4517713 | 31.5562382 | 0.1171875 | True | True |

## Validation regularization controls

Every row identifies its attempt and scope. Mean ± sample standard deviation is across available calibration seeds; a single seed has no estimated seed standard deviation. Incomplete attempts and partial seed coverage do not constitute a full cohort.

| Run / state | Specification | Split / group / rank scope | Seeds | NLL mean | NLL seed SD | Drift mean | Drift seed SD |
|---|---|---|---|---:|---:|---:|---:|
| regularization-20261013-171941-804f4c2e / completed | {"method": "svd"} | {} | [11] | 5.14486078 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.1, "method": "fullrank_ridge", "ridge_fraction": 0.01} | {} | [11] | 3.49923448 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {} | [11] | 3.60078899 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {} | [11] | 3.59662239 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.1, "method": "anchored", "ridge_fraction": 0.0} | {} | [11] | 3.59675176 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.3, "method": "anchored", "ridge_fraction": 0.0} | {} | [11] | 3.607465 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 1.0, "method": "anchored", "ridge_fraction": 0.0} | {} | [11] | 3.61579886 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.0} | {} | [11] | 3.59685438 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.001} | {} | [11] | 3.59727845 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.01} | {} | [11] | 3.60682971 | unmeasured | unmeasured | unmeasured |
| regularization-20261013-171941-804f4c2e / completed | {"method": "asvd"} | {} | [11] | 3.73607467 | unmeasured | unmeasured | unmeasured |

## Gemma full-stack component comparisons

Every row identifies its attempt and scope. Mean ± sample standard deviation is across available calibration seeds; a single seed has no estimated seed standard deviation. Incomplete attempts and partial seed coverage do not constitute a full cohort.

| Run / state | Specification | Split / group / rank scope | Seeds | NLL mean | NLL seed SD | Drift mean | Drift seed SD |
|---|---|---|---|---:|---:|---:|---:|
| baselines-20261013-175255-6a00d4e8 / completed | {"method": "svd"} | {} | [11] | 5.08868112 | unmeasured | 0.0742475831 | unmeasured |
| baselines-20261013-175255-6a00d4e8 / completed | {"method": "asvd"} | {} | [11, 23, 37] | 3.69446606 | 0.00241956806 | 0.0309717453 | 0.00019731531 |
| baselines-20261013-175255-6a00d4e8 / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {} | [11, 23, 37] | 3.57281383 | 0.0118191598 | 0.0106375166 | 0.000382136395 |
| baselines-20261013-175255-6a00d4e8 / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {} | [11, 23, 37] | 3.56592113 | 0.00957592299 | 0.0105470011 | 0.000492296849 |
| baselines-20261013-175255-6a00d4e8 / completed | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.01} | {} | [11, 23, 37] | 3.57903427 | 0.00852077408 | 0.0118359709 | 0.000138552104 |

## Composition and group heterogeneity

Every row identifies its attempt and scope. Mean ± sample standard deviation is across available calibration seeds; a single seed has no estimated seed standard deviation. Incomplete attempts and partial seed coverage do not constitute a full cohort.

| Run / state | Specification | Split / group / rank scope | Seeds | NLL mean | NLL seed SD | Drift mean | Drift seed SD |
|---|---|---|---|---:|---:|---:|---:|
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45206785 | 0.000410778252 | 0.00207792164 | 0.00017472334 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45201942 | 0.000409256997 | 0.00207452887 | 0.000178507413 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45825421 | 0.00132397251 | 8.76745161e-05 | 2.4463997e-06 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4579111 | 0.00143757277 | 8.91802058e-05 | 2.24916162e-06 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45142891 | 0.000329325004 | 1.69831209e-06 | 9.05406025e-08 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45122641 | 0.000131137763 | 2.023157e-06 | 9.42299447e-08 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45442908 | 0.00205462111 | 0.00365644585 | 0.000137627131 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45438131 | 0.00214728679 | 0.00356760921 | 0.000238304358 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353003 | 0.00115146509 | 0.000141608504 | 4.06719991e-06 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353582 | 0.00129669463 | 0.000144554107 | 4.09979657e-06 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45212529 | 0.000451146203 | 1.23266207e-05 | 3.68456941e-07 |
| composition-20261013-185512-2bffd477 / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11] | 3.45195513 | unmeasured | 1.21042136e-05 | unmeasured |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45206785 | 0.000410778252 | 0.00207792164 | 0.00017472334 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45201942 | 0.000409256997 | 0.00207452887 | 0.000178507413 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45825421 | 0.00132397251 | 8.76745161e-05 | 2.4463997e-06 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4579111 | 0.00143757277 | 8.91802058e-05 | 2.24916162e-06 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45142891 | 0.000329325004 | 1.69831209e-06 | 9.05406025e-08 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45122641 | 0.000131137763 | 2.023157e-06 | 9.42299447e-08 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45442908 | 0.00205462111 | 0.00365644585 | 0.000137627131 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45438131 | 0.00214728679 | 0.00356760921 | 0.000238304358 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353003 | 0.00115146509 | 0.000141608504 | 4.06719991e-06 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353582 | 0.00129669463 | 0.000144554107 | 4.09979657e-06 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45212529 | 0.000451146203 | 1.23266207e-05 | 3.68456941e-07 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45194198 | 0.000101102007 | 1.22970593e-05 | 3.91576035e-07 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46224947 | 0.000734496385 | 0.0061389891 | 0.000142663396 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4619553 | 0.00102918121 | 0.00600512161 | 0.000161076046 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | [11, 23, 37] | 3.47310172 | 0.000470679639 | 0.00038058521 | 1.72440575e-05 |
| composition-20261014-032120-486d799f / interrupted | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | [11, 23] | 3.4765793 | 1.58332576e-05 | 0.000438405635 | 3.31525237e-07 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45206785 | 0.000410778252 | 0.00207792164 | 0.00017472334 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45201942 | 0.000409256997 | 0.00207452887 | 0.000178507413 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45825421 | 0.00132397251 | 8.76745161e-05 | 2.4463997e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4579111 | 0.00143757277 | 8.91802058e-05 | 2.24916162e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45142891 | 0.000329325004 | 1.69831209e-06 | 9.05406025e-08 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45122641 | 0.000131137763 | 2.023157e-06 | 9.42299447e-08 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45442908 | 0.00205462111 | 0.00365644585 | 0.000137627131 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45438131 | 0.00214728679 | 0.00356760921 | 0.000238304358 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353003 | 0.00115146509 | 0.000141608504 | 4.06719991e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46353582 | 0.00129669463 | 0.000144554107 | 4.09979657e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45212529 | 0.000451146203 | 1.23266207e-05 | 3.68456941e-07 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45194198 | 0.000101102007 | 1.22970593e-05 | 3.91576035e-07 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | [11, 23, 37] | 3.46224947 | 0.000734496385 | 0.0061389891 | 0.000142663396 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4619553 | 0.00102918121 | 0.00600512161 | 0.000161076046 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | [11, 23, 37] | 3.47310172 | 0.000470679639 | 0.00038058521 | 1.72440575e-05 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | [11, 23, 37] | 3.47582544 | 0.00130576482 | 0.000449140822 | 1.85953677e-05 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [22, 23, 24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.45396348 | 0.00136122732 | 0.000123680895 | 1.21816693e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [22, 23, 24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.4529696 | 0.000385430754 | 0.00011315281 | 2.34008283e-06 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.0, "method": "whitening", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.57281383 | 0.0118191598 | 0.0106375166 | 0.000382136395 |
| composition-20261014-054423-dbd821ff / completed | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "rank": 384, "split": "test"} | [11, 23, 37] | 3.56592113 | 0.00957592299 | 0.0105470011 | 0.000492296849 |

Group-dependent outcomes must not be pooled into a universal-improvement claim. A lower full-stack NLL with unchanged/worse residual drift does not satisfy the predeclared joint gate. Three paired seeds with both higher NLL and higher drift weaken the candidate only at that measured scope.

## Amended hybrid architecture and rank-frontier comparisons

Every row identifies its attempt and scope. Mean ± sample standard deviation is across available calibration seeds; a single seed has no estimated seed standard deviation. Incomplete attempts and partial seed coverage do not constitute a full cohort.

| Run / state | Specification | Split / group / rank scope | Seeds | NLL mean | NLL seed SD | Drift mean | Drift seed SD |
|---|---|---|---|---:|---:|---:|---:|
| Unmeasured | pending | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured |

## Practical costs and calibration

Whole-model checkpoint bytes include archive overhead; factor payload size and whole-model savings are separate quantities. Placements and precision must agree before comparing timings. Below are measured records only; missing conditions remain unmeasured.

| Run / state | Label | Seed | Bits | Whole bytes | Factor bytes | Raw NLL | Scaled NLL | Temperature | Mode | Prefill | Median s | p95 s | Allocated bytes | Reserved bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Unmeasured | pending | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured |

## Statistical interpretation

NLL differences use 2,000 paired bootstrap resamples of fixed contiguous 4,096-target blocks. The final partial block remains in full-test NLL but is excluded from fixed-block resampling. Individual dependent tokens are not bootstrapped. Per-seed confidence intervals, exact excluded counts, matched target checks and all scoped candidate-minus-baseline differences are in report.json and report.md. These are evaluation-block intervals, not seed intervals.

## Reproducibility and incomplete-run appendix

The accompanying report.json embeds each retained contract, source hashes, package/device versions, asset revisions, calibration/split hashes, target recipes, run-specific command, raw per-seed metrics, integrity checks, artifact paths and all failed/interrupted attempts. Read errors are retained explicitly; promotion is disabled.

| Stage | Recorded charged wall seconds | Declared cap seconds | Unknown-cost attempts |
|---|---:|---:|---|
| readiness | 1381.392 | 3600 | [] |
| regularization | 1589.148 | 14400 | [] |
| baselines | 3736.553 | 25200 | [] |
| composition | 12825.574 | 14400 | [] |
| architecture | 0.000 | 14400 | [] |
| practical | 0.000 | 14400 | [] |

Elapsed-wall-time charges include failed/retried work and may include reboot downtime. The first reboot has an unbounded unlogged tail; no exact GPU-hour compliance claim.

## Conclusion

No established practical or universal compression advantage; retained within-model findings remain exploratory.

Complete negative or mixed comparisons are retained as bounded evidence. Unmeasured downstream tasks, fresh-corpus confirmation, absent seeds or hardware conditions are not implied by dispatcher completion.

## Measured scientific figures

![Gemma Stage C residual drift: individual seeds (dotted), mean (solid), sample seed SD (shaded). Single retained cohort; strict FP16 readiness remains failed.](figures/baselines-20261013-175255-6a00d4e8-drift.png)


Figure inputs, scope, source hashes and unavailable measurements are recorded in figures/provenance.json.

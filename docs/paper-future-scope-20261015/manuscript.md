---
bibliography: paper_protocol_v2.bib
---

# Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost

Editorial revision: 15 October 2026. Exploratory empirical study; measurements frozen on 14 October 2026.

## Abstract

Large language models use large matrices, or tables of numbers, to turn text into predictions. Replacing some of these matrices with two smaller ones can save storage, but the replacements may change how information passes through the model. We study whether a compression method guided by the model's observed inputs can preserve its predictions and the internal signals passed between layers. We compare this method with a version that also penalizes changes to the original weights, using the same number of stored values. Tests on three pretrained language models show that the penalty provides no consistent improvement in both prediction quality and internal similarity. In one model, prediction quality improves while average error in these internal signals grows to 3.86 times that of the method without the penalty. Tests on consecutive layers also find a group where both prediction quality and internal similarity worsen for all three choices of fitting data. Separate storage and timing tests remove 29.17% of the weights in the selected matrices, but reduce the complete model file by only 0.894%; execution time varies with the workload. These findings show that prediction quality alone cannot establish whether compression preserves the original computation. Failed numerical accuracy checks, incomplete additional tests for one model, and evaluation on a previously used text dataset make the findings exploratory. A practical compression claim therefore needs separate evidence for prediction quality, internal behavior, storage, and speed.

Keywords: low-rank compression; activation weighting; representation drift; transformer evaluation; inference cost.

## 1. Introduction

Language models use weight matrices to transform text into predictions. Low-rank compression replaces a large matrix with two smaller factors. The replacement saves weights, but its effect depends on the rest of the model. A factorized layer may closely match the original on calibration inputs and still receive different inputs after earlier layers have been compressed. A compressed model may also predict text more accurately while its internal representations move farther from those of the original model, which we call the teacher. We measure these outcomes separately.

We test whether a weight-space anchor improves prediction quality and internal similarity when the number of stored factor weights is fixed. Activation weighting gives more importance to input directions observed during calibration. The anchor adds a penalty for changing the original weights. It changes how the factors are fitted but adds no branch at inference. We compare it with covariance weighting without the anchor, using the same factor budget, and measure storage and execution cost.

The study separates the effects of activation weighting, anchoring and extra ridge shrinkage. We test attention-output replacements across the full stack and in consecutive groups, measuring prediction quality alongside the changes that propagate through the residual stream. We also measure complete checkpoint size and runtime. The results show where an average joint improvement fails to hold across calibration seeds or layer groups, and where better prediction quality accompanies greater internal drift.

The results remain exploratory. Some strict numerical accuracy checks failed, and Qwen's primary comparison is available without a complete validation rank frontier. The later practical cohort uses bfloat16 (BF16), has only two calibration seeds, and adds no BF16 drift measurements. We do not claim a new whitening algorithm or reliable preservation of the complete model's computation.

## 2. Related work and comparison scope

Data-aware low-rank approximation predates this study. DRONE and weighted language-model factorization establish earlier approaches that use data or task information when selecting a low-rank representation [@chen2021drone] [@hsu2022weighted]. ASVD combines activation-magnitude scaling with sensitivity-based rank selection; its v5 covariance variant, ASVD+, uses a Cholesky factor of the activation Gram matrix [@yuan2023asvd]. SVD-LLM v5 combines covariance whitening with sequential factor-wise low-rank adaptation fine-tuning, while SVD-LLM(W) denotes its whitening-only component [@wang2024svdllm]. Our controls use fixed ranks and omit rank search and recovery fine-tuning. They isolate the diagonal-scaling and covariance-weighting components of those methods. For invertible covariance in exact arithmetic, unanchored covariance weighting shares the whitening geometry; numerical cutoffs and regularization still require separate qualification.

Accumulated error and rank allocation motivate distinct comparisons. SVD-LLM V2 studies truncation and rank allocation [@wang2025svdllmv2]. The SAES-SVD and LACE-SVD preprints address accumulated/local error and cumulative correction, respectively [@hu2026saessvd] [@liu2026lacesvd]. Zero Sum SVD considers loss-sensitive global component selection, and IO-SVD introduces input/output whitening with heterogeneous ranks [@abbasi2026zerosum] [@abbasi2026iosvd]. These methods remain candidates for future comparisons at matched budgets; we did not run them here. Related work on quantization-error propagation reinforces the need to evaluate downstream effects, without making quantization and low-rank truncation equivalent [@arai2025qep].

Compression also depends on the deployed representation. GPTQ, AWQ and SparseGPT study quantization or pruning rather than our fixed-rank factors [@frantar2022gptq] [@lin2023awq] [@frantar2023sparsegpt]. EoRA uses eigenspace-informed low-rank compensation [@liu2024eora]. SVDQuant illustrates the execution cost of separate branches in diffusion models, a different application domain [@li2024svdquant]. These methods provide context for the practical measurements; we did not benchmark them or establish faster language-model inference. Centered kernel alignment and temperature scaling provide representation and calibration diagnostics, respectively; neither establishes teacher preservation [@kornblith2019cka] [@guo2017calibration].

## 3. Methods

### 3.1 Representation and parameter budget

For a projection $W\in\mathbb{R}^{o\times i}$, replace $Wx+b$ with $U(Vx)+b$, where $U\in\mathbb{R}^{o\times r}$ and $V\in\mathbb{R}^{r\times i}$. The original bias is applied once. The factors contain $r(o+i)$ weights, compared with $oi$ in the dense matrix. Parameter savings therefore require $r<oi/(o+i)$. Rank reduction by itself does not specify the storage saving.

Only the output projections of full-attention blocks are replaced. Attention heads are not pruned. Embeddings, feed-forward layers and other untargeted weights remain dense; Qwen's DeltaNet and LFM's convolutional mixers also remain dense. For Gemma, each of 26 targets has shape 1152 by 1024. Rank 384 retains 835,584 of 1,179,648 weights per target: a 29.17% targeted reduction, or 8,945,664 weights across the stack. This fraction is not the whole-model saving.

The hybrid primary ranks follow $r=\lfloor(17/24)oi/(o+i)\rfloor$, giving rank 483 for Qwen and 725 for LFM. This approximately matches Gemma's retained target-weight fraction, rather than imposing the same integer rank on different shapes. Validation-only frontier ranks use $\lfloor\min(o,i)/d\rfloor$, with $d\in\{2,3,4\}$. Within each comparison, the anchor and whitening methods use identical target shapes, ranks and factor precision.

### 3.2 Weighting, anchoring and extra ridge

Let $X\in\mathbb{R}^{i\times n}$ contain calibration activations and write $A=UV$. The anchored objective is $J_\beta(A)=\|(W-A)X\|_F^2+\beta n\|W-A\|_F^2$, subject to rank at most $r$. The first term weights directions by observed activation energy; the second penalizes weight-space deviation even in weakly observed directions. The anchor changes the fitting geometry while keeping the factor budget and inference graph fixed.

With exact teacher outputs, define $H=XX^T+\beta nI$ and $C=WH$. The BF16 practical implementation instead fits measured outputs: it first removes the bias, $Y_b=Y-b\mathbf{1}^T$, and minimizes $\|Y_b-AX\|_F^2+\beta n\|W-A\|_F^2$, giving $C=Y_bX^T+\beta nW$. This equals the exact-output objective only when $Y_b=WX$; finite-precision measured outputs need not satisfy that identity. The historical extra-ridge solver uses $G=H+\lambda I$, with $\lambda=\rho\operatorname{mean}(\operatorname{eig}(H))$. It forms $A_{\mathrm{full}}=CG^\dagger$, truncates $A_{\mathrm{full}}G^{1/2}$ to rank $r$, and maps back using $G^{\dagger/2}$.

For nonsingular $H$, exact outputs and no extra ridge, the solution is $A_\beta=[WH^{1/2}]_rH^{-1/2}$, where brackets denote the best rank-$r$ singular-value truncation. Without rank reduction, this recovers $W$. For singular or numerically truncated geometry, recovery is restricted to retained spectral directions. In contrast, positive extra ridge generally shrinks the unrestricted solution even before truncation. Consequently, an effect of the historical solver cannot automatically be attributed to low rank or to anchoring alone.

The unanchored whitening component sets $\beta=\rho=0$. The selected anchor uses $\beta=0.03$ and $\rho=0$. Plain singular value decomposition (SVD) is a weight-space control. The fixed-rank diagonal ASVD component uses $s_j=\max(\operatorname{mean}_t|X_{jt}|,10^{-8})^{0.5}$, truncates $W\operatorname{diag}(s)$ and scales back. Covariance fitting and weighted decomposition use 64-bit floating-point computation (FP64), then 32-bit floating-point factors (FP32) before the declared storage cast; plain SVD uses FP32. The weighted implementation retains regularized eigenvalues strictly above $10^{-7}\lambda_{\max}(G)$ and clamps negative numerical eigenvalues to zero. These finite-precision choices limit exact-arithmetic equivalence claims.

### 3.3 Local error and propagated residual drift

Local error compares the original and replacement projection on identical teacher activations. Propagated drift instead compares the outputs of every complete block in independent teacher and compressed-model forward passes on the same token sequence. The student therefore receives its own upstream hidden states. Both models use the observed token context; this is not free-running generated-text evaluation.

Let $h_{\ell,t}$ and $\widehat h_{\ell,t}$ be teacher and student block-output states at scored positions $t\in\mathcal{T}$. We use $D_\ell=\sum_{t\in\mathcal{T}}\|\widehat h_{\ell,t}-h_{\ell,t}\|_2^2/\max(\sum_{t\in\mathcal{T}}\|h_{\ell,t}\|_2^2,10^{-30})$ and $D=L^{-1}\sum_{\ell=1}^{L}D_\ell$ over all model blocks. Errors and reference energy accumulate across all scored windows before normalization. The metric $D$ averages the energy ratio for each block. It differs from both a last-layer error and an average of per-token ratios. For subgroup replacements, it includes unchanged upstream blocks as well as downstream blocks, so group location affects its interpretation.

Cosine distance, residual norm ratios, window-based linear centered kernel alignment (CKA), and teacher-to-student logit Kullback-Leibler (KL) divergence are secondary diagnostics. Low drift indicates similarity in the measured representation coordinates; it is not a theorem of functional equivalence. A representation change can preserve or improve an output metric. Conversely, a favorable average drift score need not guarantee good final logits or likelihood.

## 4. Experimental protocol and evidence boundaries

### 4.1 Data, selection and uncertainty

All reported quality comparisons use the native WikiText-2 splits [@merity2016wikitext]. Calibration draws 16 disjoint chunks of 512 tokens from training data, paired across methods for seeds 11, 23 and 37 in the historical quality cohorts. Native validation selects regularization; native test evaluates the frozen setting. Rows retain dataset order and are joined with two newlines without added special tokens. Evaluation uses context 512, stride 256, exactly-once shifted-target masking and the final partial window. Negative log-likelihood (NLL) is total token loss divided by scored targets; perplexity (PPL) is its exponential. Native-tokenizer metrics are compared within models, never used to rank different tokenizers.

The original anchor sweep is $\beta\in\{0,0.03,0.1,0.3,1\}$. Extra-ridge fractions are $\rho\in\{0,0.001,0.01\}$ at $\beta=0.1$. Validation NLL ties within $10^{-6}$ prefer less regularization. The selected ridge-free anchor is frozen before the primary comparisons; hybrid validation frontiers are development evidence, not additional confirmation.

Tables report calibration-seed means and sample standard deviations, with seed-independent SVD fitted once. Seeds describe calibration variation, not independently pretrained models. Paired 95% bootstrap intervals use 2,000 resamples of contiguous 4,096-target evaluation blocks. The incomplete tail remains in full-test NLL but is excluded from fixed-block resampling. These intervals therefore describe evaluation-block variation conditional on the fitted model; it does not replace seed uncertainty. Comparisons across many groups and ranks remain descriptive rather than multiplicity-corrected discoveries [@biderman2024evaluation].

### 4.2 Numerical controls, coverage and practical amendment

The original strict numerical check for 16-bit floating-point (FP16) factorization failed. Copied-teacher and factor-split identity controls can match exactly, but do not establish that SVD-based factor preparation has equivalent numerical behavior. Later BF16 controls likewise do not retroactively validate the FP16 cohort: the BF16 full-rank SVD control has NLL 3.452658 versus 3.451550 for its teacher and fails its recorded probe and NLL gates. Higher-precision execution improves NLL agreement while the strict probe still fails. Because these checks failed, the results describe the recorded implementations. They cannot isolate the ideal mathematical objectives from numerical error.

The Qwen continuation contains all eight primary-test records but only 10 of 18 declared validation-frontier records. The continuation reuses observations from an interrupted attempt; they do not count as independent replications. LFM's eight primary-test and 18 frontier records are complete. Completion of either dispatcher or saved rows does not remove the readiness limitation.

A later practical amendment evaluates only BF16 Gemma: one dense teacher and whitening/anchor factors for seeds 11 and 23, five conditions in total. It is separate from the stopped FP16/INT8 practical cohort and does not supply new drift evidence. Its two seeds cannot satisfy the original three-seed promotion gate. No missing frontier point, seed, corpus or task is inferred from execution completion.

## 5. Results

### 5.1 Activation-aware fitting improves on plain SVD, but anchoring has a small mixed Gemma effect

Table 1 compares all 26 Gemma attention-output projections at rank 384 using the historical FP16 cohort. The dense reference NLL is 3.451563. Activation-aware components have substantially lower NLL and drift than plain SVD at the same factor budget. The anchor's mean NLL is slightly lower than whitening's, but both remain worse than the dense reference on this corpus.

Table 1. Gemma full-stack native-test comparison. Values are mean ± sample seed standard deviation; a dash indicates a single seed-independent fit. Drift is the dimensionless primary residual error $D$.

| Method | NLL | Residual drift | Seeds |
|---|---|---|---|
| Plain SVD | 5.088681 (-) | 0.074248 (-) | 1 |
| Diagonal ASVD component | 3.694466 ± 0.002420 | 0.030972 ± 0.000197 | 3 |
| Covariance whitening | 3.572814 ± 0.011819 | 0.010638 ± 0.000382 | 3 |
| Anchor (beta=0.03) | 3.565921 ± 0.009576 | 0.010547 ± 0.000492 | 3 |
| Extra ridge (beta=0.1, rho=0.01) | 3.579034 ± 0.008521 | 0.011836 ± 0.000139 | 3 |

Anchoring changes mean NLL by -0.006893 relative to whitening. The mean drift also decreases slightly, from 0.010638 to 0.010547, but drift worsens in two of the three paired seeds. Therefore the aggregate does not establish a reproducible joint improvement. The historical extra-ridge setting is worse than whitening on both metrics in every paired seed. The validation full-rank ridge control further demonstrates why shrinkage must be distinguished from rank reduction; an unrestricted-rank control is not a matched compressed baseline.

![Figure 1. Historical Gemma residual drift after each complete block, using saved measurements only. Dotted traces show individual calibration seeds; solid traces show means and shading shows sample seed standard deviation. The strict FP16 readiness failure still applies.](figures/gemma-drift.png)

### 5.2 A full-stack average does not predict every consecutive group

The completed composition cohort evaluates single projections, groups of two, groups of four and all 26 projections. Smaller groups occur at early, middle and late positions, with fixed per-target rank. Table 2 reports paired mean differences, anchor minus whitening; negative values favor the anchor. It uses only the unique completed cohort. Interrupted predecessors remain in the evidence ledger and are not counted as new observations.

Table 2. Composition heterogeneity across three calibration seeds. Block indices are zero-based. Jointly better counts seeds with strictly lower NLL and strictly lower $D$; it does not assert statistical significance.

| Blocks | Mean NLL difference | Mean drift difference | Jointly better |
|---|---|---|---|
| 0 | -0.0000484 | -3.392768e-06 | 2/3 |
| 12 | -0.0003431 | +1.50569e-06 | 0/3 |
| 25 | -0.0002025 | +3.248449e-07 | 0/3 |
| 0-1 | -0.0000478 | -8.883664e-05 | 1/3 |
| 12-13 | +0.0000058 | +2.945603e-06 | 0/3 |
| 24-25 | -0.0001833 | -2.956141e-08 | 0/3 |
| 0-3 | -0.0002942 | -0.0001338675 | 2/3 |
| 11-14 | +0.0027237 | +6.855561e-05 | 0/3 |
| 22-25 | -0.0009939 | -1.052808e-05 | 2/3 |
| All 26 | -0.0068927 | -9.051551e-05 | 1/3 |

For middle blocks 11-14, anchor NLL is 3.475825 versus 3.473102 for whitening, and drift is 0.00044914 versus 0.00038059. Both metrics worsen in all three seeds. Other groups have mixed signs or much smaller differences. The full-stack average gain therefore does not hold for every group. Drift also need not grow monotonically with group size: layer position, downstream sensitivity and the unchanged blocks included in the average all affect the result.

### 5.3 Hybrid-model results separate likelihood from teacher-state fidelity

Table 3 shows primary-test means for the two hybrid models. Each method targets six full-attention output projections, with matching ranks within model. Qwen's rank is 483 and LFM's is 725. Their other mixer families remain dense. Three calibration seeds are present for each activation-aware method; the dense teacher and plain-SVD controls are recorded once.

Table 3. Within-model hybrid primary-test results. NLL and drift entries show whitening → anchor. Cross-model differences are not architecture-only causal effects.

| Model / primary rank | NLL: whitening → anchor | Drift: whitening → anchor | Record coverage |
|---|---|---|---|
| Qwen, rank 483 | 2.934439 → 2.943324 | 0.006155 → 0.012762 | 8/8 test; 10/18 validation |
| LFM, rank 725 | 4.056405 → 3.835916 | 0.035880 → 0.138346 | 8/8 test; 18/18 validation |

Qwen's anchor has higher NLL and higher drift than whitening in every seed. This weakens the anchor hypothesis at its measured primary rank, even though its incomplete validation frontier prevents a complete frontier claim.

LFM shows the clearest disagreement between the metrics: anchor NLL is lower in every seed, while drift increases in every seed and is approximately 3.86 times the whitening mean. Its anchor NLL is also below the dense teacher's 3.998736, whereas its drift remains nonzero and much larger than whitening's. Better observed next-token likelihood therefore does not certify reproduction of the teacher's hidden-state computation. The result alone also cannot identify denoising, beneficial regularization or a particular causal mechanism. The three fully measured LFM validation ranks show the same qualitative likelihood/drift tradeoff, but remain development observations.

### 5.4 Targeted compression yields modest whole-model storage savings and mixed timing

The BF16 practical cohort's dense checkpoint contains 1,999,883,711 bytes; each factorized checkpoint contains 1,982,003,507 bytes, including archive overhead. The reduction is 17,880,204 bytes, or 0.894%. Identical anchor and whitening factor budgets yield identical checkpoint sizes. The 29.17% reduction in targeted weights therefore translates to less than 1% of this whole-model serialization, because most weights remain untouched.

Table 4. Separate BF16 practical cohort. Temperature is selected on validation from a bounded grid, then frozen; all five saved selections are approximately 1.090508. GiB uses $2^{30}$ bytes. These are descriptive full-test aggregates without paired block intervals.

| BF16 condition | Seed | Raw NLL | Scaled NLL | Whole GiB |
|---|---|---|---|---|
| Dense teacher | - | 3.451549 | 3.422601 | 1.862537 |
| Whitening | 11 | 3.556086 | 3.541726 | 1.845885 |
| Anchor | 11 | 3.556826 | 3.542255 | 1.845885 |
| Whitening | 23 | 3.568431 | 3.553267 | 1.845885 |
| Anchor | 23 | 3.562949 | 3.545341 | 1.845885 |

Temperature scaling reduces NLL for both the teacher and compressed conditions, without closing the teacher-relative NLL gap. This supports treating calibration as a control rather than interpreting raw likelihood differences as denoising. The practical BF16 rows are not pooled with historical FP16 quality or drift measurements.

Timing uses a resident NVIDIA GeForce RTX 4070 Ti, batch size one, 10 warmups and 30 synchronized samples. Prefill is measured at 128 and 512 tokens. Cached 128-token decode is timed separately and excludes the initial prompt pass. Table 5 compares anchor with same-seed BF16 whitening under identical workloads. Ratios below one indicate lower median latency; they are descriptive workload comparisons, not a new acceptance gate.

| Workload | Seed 11 median ratio | Seed 23 median ratio |
|---|---|---|
| Prefill 128 | 1.1876 | 0.9370 |
| Decode 128 after prompt 128 | 1.1073 | 0.9679 |
| Prefill 512 | 1.0108 | 0.9473 |
| Decode 128 after prompt 512 | 0.9876 | 0.9580 |

Median ratios vary by seed and workload: the anchor is slower for seed 11 at the 128-token conditions, while the observed seed-23 ratios favor it. Peak allocated/reserved memory and serialized bytes match between the paired factorized methods. These observations establish neither a consistent anchoring speed advantage nor general acceleration over the dense teacher. CUDA allocator peaks are not total device memory or system RAM. Whole-model reload probes cover a 128-token prompt and one cached next-token forward, not arbitrary generation or the entire test corpus.

## 6. Discussion

LFM predicts text better under the anchored fit while its residual representations differ more from the teacher's. This result limits what likelihood alone can tell us about preservation. Residual similarity also has limits: $D$ compares states in fixed coordinates, and different states can implement similar functions. A stronger preservation claim needs agreement across final logits, task performance, drift measurements and generation tests. Smaller storage or a better score on one measure does not supply that evidence.

Matched covariance weighting is the relevant baseline for the anchor. Gains over plain SVD show the value of data-aware fitting in this implementation, but cannot establish what the anchor adds or whether it is novel. Gemma's small average advantage, Qwen's joint worsening and LFM's tradeoff provide no consistent basis for adding the same anchor across models. The models also differ in pretrained weights, tokenizers, mixers and historical inference precision, so we cannot attribute those outcomes solely to architecture.

The numerical controls also constrain causal interpretation. The measured operator includes factor casting, multiplication order, spectral cutoffs and finite-precision execution. Without passing the strict readiness checks, we cannot assign every downstream difference to the mathematical objective alone. Future comparisons need to resolve these failures. An exact identity split does not validate a different decomposition path.

Model-level storage and speed depend on which layers are compressed and how the factors execute. A targeted parameter saving can be diluted by embeddings and other dense components. Two dependent matrix multiplications can incur launch and intermediate-memory costs despite using fewer weights. Faster covariance preparation, fused BF16 factor execution and runtime-aware rank allocation are reasonable future hypotheses, but require new frozen protocols and matched quality/resource budgets. They are not measured benefits of the present study.

## 7. Limitations and reproducibility

WikiText-2 has already influenced this project; native-test results are regression evidence, not a fresh-domain confirmation. No downstream task or free-running generation equivalence is established. A locked sample from C4, whose provenance is described in the T5 study, could provide fresh confirmation [@raffel2020t5]; HellaSwag and ARC-Challenge could evaluate additional capabilities [@zellers2019hellaswag] [@clark2018arc]. None was executed here.

The scope is attention-output replacement at recorded budgets. Multilayer perceptrons (MLPs), hybrid non-attention mixers, larger models and other rank allocations require separate conclusions. Three calibration seeds provide limited variation, and the later BF16 practical amendment has only two. Within-model matching does not make different model families or FP16/BF16 cohorts interchangeable. The Qwen frontier's eight absent validation measurements remain missing. The original strict numerical-readiness failure, incomplete coverage and unfulfilled promotion gates prevent promotion of a general preservation claim.

Every main table is reconstructed from the frozen evidence. The [frozen evidence report](../../results/paper-v2/revised-report-20261014-151432-07c08d61/report.md) identifies the source runs, and the revision manifest records the exact extraction fields. Selected paired intervals remain in the repository's [evidence file](report.json). The full historical report preserves commands, package/device versions, asset revisions, token manifests, factor recipes, source/checkpoint hashes, failed attempts and recovery accounting. Reused records are not independent replications. Recorded charged wall time includes failed/retried work and some reboot downtime; the first reboot has an unbounded unlogged tail, so no exact 24 GPU-hour compliance claim is made. This revision uses the saved measurements without changing the fitting budget, model or dataset.

## 8. Future Scope

### 8.1 Compression with a small neural network

A possible extension is to replace a selected dense projection with a small bottleneck neural network. Its input would pass through a narrower hidden layer before returning to the original output dimension, allowing a compact mapping to be learned from the original model's activations. Transformer knowledge distillation provides a related precedent for training smaller students from larger teachers [@jiao2020tinybert]. The proposed projection-level replacement would still require evaluation in the models and compression setting studied here.

Another option is to keep the low-rank factors and train a small network to correct their output error. The factor rank would need to leave enough room for this correction network within the matched storage budget. Both options should count all network weights and biases, together with any retained factors. A reduction in local error would need to improve held-out predictions and control residual drift when the replacements are composed across layers. Neither option has been implemented or tested in this study.

### 8.2 Numerical checks and broader evaluation

Further work should first resolve the failed numerical controls and complete the missing Qwen validation measurements. Controlled comparisons could then test whether matching the output of a group of layers reduces the downstream drift seen after composition. Fresh evaluation data, downstream tasks and generated-text tests would help assess whether any improvement extends beyond the text already used in this project. The model, data splits, fitting budget and decision criteria should be fixed before each comparison, with replication across calibration seeds.

### 8.3 Storage and execution cost

The small-network proposals also need measurements of complete checkpoint size, peak memory and warmed-up inference latency. Fused factor execution and rank allocation informed by runtime cost are further candidates for study. Each must be compared with matched low-rank baselines under the same workloads, since fewer stored weights alone do not establish a practical speed improvement.

## 9. Conclusion

Activation-aware low-rank fitting improves on plain SVD at the tested attention-output budgets. Adding the anchor provides no consistent joint improvement in likelihood and residual fidelity over matched whitening. Gemma's small average gain does not hold across all layer groups. Qwen worsens on both metrics, while LFM improves likelihood and increases drift. The separate BF16 practical cohort saves 0.894% of the complete checkpoint size and has mixed timing results. Compression remains unproven as a way to preserve the original computation across the tested models.

## Acknowledgment

OpenAI Codex [@openaicodex2026] assisted with drafting and editing the Abstract, Introduction, Methods, Discussion, Future Scope and Conclusion, and with preparing formatting, table-assembly and plotting scripts. Numerical results in this revision come from the saved experiment records; this editorial work did not run a new compression experiment.

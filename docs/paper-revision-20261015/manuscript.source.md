---
bibliography: paper_protocol_v2.bib
---

# Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost

Editorial revision: 15 October 2026. Exploratory empirical study; measurements frozen on 14 October 2026.

## Abstract

Low-rank replacement reduces a transformer's weight count, but local reconstruction quality does not establish preservation of its computation. We study this distinction using attention-output projections in Gemma-3-1B, Qwen3.5-0.8B and LFM2.5-1.2B-Instruct, comparing covariance-weighted truncation with an anchored objective at matched factor budgets. Three calibration seeds reveal different outcomes: Gemma's full-stack mean negative log-likelihood changes from 3.572814 to 3.565921, without a consistent per-seed drift improvement; Qwen's anchoring worsens both likelihood and drift at the primary test rank; and LFM's anchoring lowers negative log-likelihood from 4.056405 to 3.835916 while increasing residual drift by approximately 3.86 times. Gemma composition tests also expose a four-block group in which both metrics worsen in every seed. A separate two-seed BF16 practical cohort reduces the whole-model checkpoint by 0.894%, despite removing 29.17% of targeted projection weights, and shows workload-dependent latency. These observations support a scoped dissociation between likelihood, representation similarity and execution cost. They do not establish reliable global preservation, a general anchoring advantage or practical acceleration. Failed strict numerical-readiness controls, incomplete Qwen frontier coverage and evaluation on a previously used corpus limit the conclusions.

Keywords: low-rank compression; activation weighting; representation drift; transformer evaluation; inference cost.

## 1. Introduction

Replacing a dense projection with two smaller matrices offers a direct way to reduce stored weights. The difficult question is whether those replacements preserve the behavior of the complete model. A projection can reconstruct its teacher outputs on calibration inputs yet encounter different activations after earlier projections have also been replaced. Conversely, an altered model can improve held-out next-token likelihood while moving farther from the teacher's internal representations. These outcomes test different properties and should not be summarized by a single compression score.

We ask: **at a fixed factor budget, does adding a weight-space anchor to activation-weighted low-rank fitting consistently improve both held-out likelihood and residual-state fidelity, and what storage and execution benefit remains?** The anchor changes the fitting objective without adding an inference branch. This makes it a controlled comparison against unanchored covariance weighting, rather than a comparison between differently sized representations.

Our contribution is a bounded empirical analysis with three parts. First, we distinguish activation weighting, anchoring and extra ridge shrinkage mathematically and compare their recorded controls. Second, we examine complete attention-output stacks and consecutive groups, keeping likelihood and propagated residual drift separate. Third, we distinguish targeted parameter savings from serialized whole-model size and measured runtime. The results expose useful failures: average improvements need not reproduce across seeds or groups, and better likelihood can accompany substantially greater drift.

This is an exploratory study, rather than a claim of a new whitening algorithm or a solved global-preservation problem. The strict numerical-readiness checks were not uniformly satisfied. The Qwen primary comparison is available, but its validation frontier is incomplete. The later BF16 practical cohort has only two calibration seeds and adds no BF16 drift measurement. We retain those boundaries throughout the analysis.

## 2. Related work and comparison scope

Data-aware low-rank approximation predates this study. DRONE and weighted language-model factorization establish earlier approaches that use data or task information when selecting a low-rank representation [@chen2021drone] [@hsu2022weighted]. ASVD combines activation-magnitude scaling with sensitivity-based rank selection; its v5 covariance variant, ASVD+, uses a Cholesky factor of the activation Gram matrix [@yuan2023asvd]. SVD-LLM v5 combines covariance whitening with sequential factor-wise low-rank adaptation fine-tuning, while SVD-LLM(W) denotes its whitening-only component [@wang2024svdllm]. Our fixed-rank diagonal-scaling and covariance-weighted controls omit rank search and recovery fine-tuning. They are component comparisons, not reproductions of either complete pipeline. For invertible covariance in exact arithmetic, unanchored covariance weighting shares the whitening geometry; numerical cutoffs and regularization still require separate qualification.

Accumulated error and rank allocation motivate distinct comparisons. SVD-LLM V2 studies truncation and rank allocation [@wang2025svdllmv2]. The SAES-SVD and LACE-SVD preprints address accumulated/local error and cumulative correction, respectively [@hu2026saessvd] [@liu2026lacesvd]. Zero Sum SVD considers loss-sensitive global component selection, and IO-SVD introduces input/output whitening with heterogeneous ranks [@abbasi2026zerosum] [@abbasi2026iosvd]. These approaches motivate future matched-budget studies; none was executed as a comparator here. Related work on quantization-error propagation reinforces the need to evaluate downstream effects, without making quantization and low-rank truncation equivalent [@arai2025qep].

Compression also depends on the deployed representation. GPTQ, AWQ and SparseGPT study quantization or pruning rather than our fixed-rank factors [@frantar2022gptq] [@lin2023awq] [@frantar2023sparsegpt]. EoRA uses eigenspace-informed low-rank compensation [@liu2024eora]. SVDQuant illustrates the execution cost of separate branches in diffusion models, a different application domain [@li2024svdquant]. We cite these methods as context, not as measured baselines or evidence of an LLM speedup. Centered kernel alignment and temperature scaling supply diagnostic and calibration tools, respectively; neither certifies teacher preservation [@kornblith2019cka] [@guo2017calibration].

## 3. Methods

### 3.1 Representation and parameter budget

For a projection $W\in\mathbb{R}^{o\times i}$, replace $Wx+b$ with $U(Vx)+b$, where $U\in\mathbb{R}^{o\times r}$ and $V\in\mathbb{R}^{r\times i}$. The original bias is applied once. Factors contain $r(o+i)$ weights rather than $oi$; a parameter reduction therefore requires $r<oi/(o+i)$. A smaller rank alone is not a sufficient storage claim.

Only the output projections of full-attention blocks are replaced. Attention heads are not pruned. Embeddings, feed-forward layers and other untargeted weights remain dense; Qwen's DeltaNet and LFM's convolutional mixers also remain dense. For Gemma, each of 26 targets has shape 1152 by 1024. Rank 384 retains 835,584 of 1,179,648 weights per target: a 29.17% targeted reduction, or 8,945,664 weights across the stack. This fraction is not the whole-model saving.

The hybrid primary ranks follow $r=\lfloor(17/24)oi/(o+i)\rfloor$, giving rank 483 for Qwen and 725 for LFM. This approximately matches Gemma's retained target-weight fraction, rather than imposing the same integer rank on different shapes. Validation-only frontier ranks use $\lfloor\min(o,i)/d\rfloor$, with $d\in\{2,3,4\}$. Within each comparison, the anchor and whitening methods use identical target shapes, ranks and factor precision.

### 3.2 Weighting, anchoring and extra ridge

Let $X\in\mathbb{R}^{i\times n}$ contain calibration activations and write $A=UV$. The anchored objective is $J_\beta(A)=\|(W-A)X\|_F^2+\beta n\|W-A\|_F^2$, subject to rank at most $r$. The first term weights directions by observed activation energy; the second penalizes weight-space deviation even in weakly observed directions. Anchoring changes fitting geometry, not the factor budget or inference graph.

With exact teacher outputs, define $H=XX^T+\beta nI$ and $C=WH$. The BF16 practical implementation instead fits measured outputs: it first removes the bias, $Y_b=Y-b\mathbf{1}^T$, and minimizes $\|Y_b-AX\|_F^2+\beta n\|W-A\|_F^2$, giving $C=Y_bX^T+\beta nW$. This equals the exact-output objective only when $Y_b=WX$; finite-precision measured outputs need not satisfy that identity. The historical extra-ridge solver uses $G=H+\lambda I$, with $\lambda=\rho\operatorname{mean}(\operatorname{eig}(H))$. It forms $A_{\mathrm{full}}=CG^\dagger$, truncates $A_{\mathrm{full}}G^{1/2}$ to rank $r$, and maps back using $G^{\dagger/2}$.

For nonsingular $H$, exact outputs and no extra ridge, the solution is $A_\beta=[WH^{1/2}]_rH^{-1/2}$, where brackets denote the best rank-$r$ singular-value truncation. Without rank reduction, this recovers $W$. For singular or numerically truncated geometry, recovery is restricted to retained spectral directions. In contrast, positive extra ridge generally shrinks the unrestricted solution even before truncation. Consequently, an effect of the historical solver cannot automatically be attributed to low rank or to anchoring alone.

The unanchored whitening component sets $\beta=\rho=0$. The selected anchor uses $\beta=0.03$ and $\rho=0$. Plain singular value decomposition (SVD) is a weight-space control. The fixed-rank diagonal ASVD component uses $s_j=\max(\operatorname{mean}_t|X_{jt}|,10^{-8})^{0.5}$, truncates $W\operatorname{diag}(s)$ and scales back. Covariance fitting and weighted decomposition use FP64 computation, then FP32 factors before the declared storage cast; plain SVD uses FP32. The weighted implementation retains regularized eigenvalues strictly above $10^{-7}\lambda_{\max}(G)$ and clamps negative numerical eigenvalues to zero. These finite-precision choices limit exact-arithmetic equivalence claims.

### 3.3 Local error and propagated residual drift

Local error compares the original and replacement projection on identical teacher activations. Propagated drift instead compares the outputs of every complete block in independent teacher and compressed-model forward passes on the same token sequence. The student therefore receives its own upstream hidden states. Both models use the observed token context; this is not free-running generated-text evaluation.

Let $h_{\ell,t}$ and $\widehat h_{\ell,t}$ be teacher and student block-output states at scored positions $t\in\mathcal{T}$. We use $D_\ell=\sum_{t\in\mathcal{T}}\|\widehat h_{\ell,t}-h_{\ell,t}\|_2^2/\max(\sum_{t\in\mathcal{T}}\|h_{\ell,t}\|_2^2,10^{-30})$ and $D=L^{-1}\sum_{\ell=1}^{L}D_\ell$ over all model blocks. Errors and reference energy accumulate across all scored windows before normalization. Thus $D$ is an average of block-level energy ratios, not a last-layer error or an average of per-token ratios. For subgroup replacements, it includes unchanged upstream blocks as well as downstream blocks, so group location affects its interpretation.

Cosine distance, residual norm ratios, window-based linear centered kernel alignment (CKA), and teacher-to-student logit Kullback-Leibler (KL) divergence are secondary diagnostics. Low drift indicates similarity in the measured representation coordinates; it is not a theorem of functional equivalence. A representation change can preserve or improve an output metric. Conversely, a favorable average drift score need not guarantee good final logits or likelihood.

## 4. Experimental protocol and evidence boundaries

### 4.1 Data, selection and uncertainty

All reported quality comparisons use the native WikiText-2 splits [@merity2016wikitext]. Calibration draws 16 disjoint chunks of 512 tokens from training data, paired across methods for seeds 11, 23 and 37 in the historical quality cohorts. Native validation selects regularization; native test evaluates the frozen setting. Rows retain dataset order and are joined with two newlines without added special tokens. Evaluation uses context 512, stride 256, exactly-once shifted-target masking and the final partial window. Negative log-likelihood (NLL) is total token loss divided by scored targets; perplexity (PPL) is its exponential. Native-tokenizer metrics are compared within models, never used to rank different tokenizers.

The original anchor sweep is $\beta\in\{0,0.03,0.1,0.3,1\}$. Extra-ridge fractions are $\rho\in\{0,0.001,0.01\}$ at $\beta=0.1$. Validation NLL ties within $10^{-6}$ prefer less regularization. The selected ridge-free anchor is frozen before the primary comparisons; hybrid validation frontiers are development evidence, not additional confirmation.

Tables report calibration-seed means and sample standard deviations, with seed-independent SVD fitted once. Seeds describe calibration variation, not independently pretrained models. Paired 95% bootstrap intervals use 2,000 resamples of contiguous 4,096-target evaluation blocks. The incomplete tail remains in full-test NLL but is excluded from fixed-block resampling. Appendix uncertainty therefore describes evaluation-block variation conditional on the fitted model; it does not replace seed uncertainty. Comparisons across many groups and ranks remain descriptive rather than multiplicity-corrected discoveries [@biderman2024evaluation].

### 4.2 Numerical controls, coverage and practical amendment

The original strict FP16 factorization-readiness gate failed. Copied-teacher and factor-split identity controls can match exactly, but do not establish that SVD-based factor preparation has equivalent numerical behavior. Later BF16 controls likewise do not retroactively validate the FP16 cohort: the BF16 full-rank SVD control has NLL 3.452658 versus 3.451550 for its teacher and fails its recorded probe and NLL gates. Higher-precision execution improves NLL agreement while the strict probe still fails. These controls make the following results exploratory observations under recorded implementations, rather than a numerically certified comparison of ideal objectives.

The Qwen continuation contains all eight primary-test records but only 10 of 18 declared validation-frontier records. Its inherited observations are reused records, not independent replications. LFM's eight primary-test and 18 frontier records are complete. Completion of either dispatcher or saved rows does not remove the readiness limitation.

A later practical amendment evaluates only BF16 Gemma: one dense teacher and whitening/anchor factors for seeds 11 and 23, five conditions in total. It is separate from the stopped FP16/INT8 practical cohort and does not supply new drift evidence. Its two seeds cannot satisfy the original three-seed promotion gate. No missing frontier point, seed, corpus or task is inferred from execution completion.

## 5. Results

### 5.1 Activation-aware fitting improves on plain SVD, but anchoring has a small mixed Gemma effect

Table 1 compares all 26 Gemma attention-output projections at rank 384 using the historical FP16 cohort. The dense reference NLL is 3.451563. Activation-aware components have substantially lower NLL and drift than plain SVD at the same factor budget. The anchor's mean NLL is slightly lower than whitening's, but both remain worse than the dense reference on this corpus.

Table 1. Gemma full-stack native-test comparison. Values are mean ± sample seed standard deviation; a dash indicates a single seed-independent fit. Drift is the dimensionless primary residual error $D$.

{{GEMMA_TABLE}}

Anchoring changes mean NLL by -0.006893 relative to whitening. The mean drift also decreases slightly, from 0.010638 to 0.010547, but drift worsens in two of the three paired seeds. Therefore the aggregate does not establish a reproducible joint improvement. The historical extra-ridge setting is worse than whitening on both metrics in every paired seed. The validation full-rank ridge control further demonstrates why shrinkage must be distinguished from rank reduction; an unrestricted-rank control is not a matched compressed baseline.

![Figure 1. Historical Gemma residual drift after each complete block, using saved measurements only. Dotted traces show individual calibration seeds; solid traces show means and shading shows sample seed standard deviation. The strict FP16 readiness failure still applies.](figures/gemma-drift.png)

### 5.2 A full-stack average does not predict every consecutive group

The completed composition cohort evaluates single projections, groups of two, groups of four and all 26 projections. Smaller groups occur at early, middle and late positions, with fixed per-target rank. Table 2 reports paired mean differences, anchor minus whitening; negative values favor the anchor. It uses only the unique completed cohort. Interrupted predecessors remain in the evidence ledger and are not counted as new observations.

Table 2. Composition heterogeneity across three calibration seeds. Block indices are zero-based. Jointly better counts seeds with strictly lower NLL and strictly lower $D$; it does not assert statistical significance.

{{COMPOSITION_TABLE}}

For middle blocks 11-14, anchor NLL is 3.475825 versus 3.473102 for whitening, and drift is 0.00044914 versus 0.00038059. Both metrics worsen in all three seeds. Other groups have mixed signs or much smaller differences. This counterexample rules out an interpretation that the modest full-stack average gain implies uniform group-level benefit. It does not prove that drift must grow monotonically with group size: layer position, downstream sensitivity and the inclusion of unchanged blocks all affect the aggregate.

### 5.3 Hybrid-model results separate likelihood from teacher-state fidelity

Table 3 shows primary-test means for the two hybrid models. Each method targets six full-attention output projections, with matching ranks within model. Qwen's rank is 483 and LFM's is 725. Their other mixer families remain dense. Three calibration seeds are present for each activation-aware method; the dense teacher and plain-SVD controls are recorded once.

Table 3. Within-model hybrid primary-test results. NLL and drift entries show whitening → anchor. Cross-model differences are not architecture-only causal effects.

{{HYBRID_TABLE}}

Qwen's anchor has higher NLL and higher drift than whitening in every seed. This weakens the anchor hypothesis at its measured primary rank, even though its incomplete validation frontier prevents a complete frontier claim.

LFM exhibits the central metric disagreement: anchor NLL is lower in every seed, while drift increases in every seed and is approximately 3.86 times the whitening mean. Its anchor NLL is also below the dense teacher's 3.998736, whereas its drift remains nonzero and much larger than whitening's. Better observed next-token likelihood therefore does not certify reproduction of the teacher's hidden-state computation. The result alone also cannot identify denoising, beneficial regularization or a particular causal mechanism. The three fully measured LFM validation ranks show the same qualitative likelihood/drift tradeoff, but remain development observations.

### 5.4 Targeted compression yields modest whole-model storage savings and mixed timing

The BF16 practical cohort's dense checkpoint contains {{DENSE_BYTES}} bytes; each factorized checkpoint contains {{FACTOR_BYTES}} bytes, including archive overhead. The reduction is {{SAVED_BYTES}} bytes, or 0.894%. Identical anchor and whitening factor budgets yield identical checkpoint sizes. The 29.17% reduction in targeted weights therefore translates to less than 1% of this whole-model serialization, because most weights remain untouched.

Table 4. Separate BF16 practical cohort. Temperature is selected on validation from a bounded grid, then frozen; all five saved selections are approximately 1.090508. GiB uses $2^{30}$ bytes. These are descriptive full-test aggregates without paired block intervals.

{{PRACTICAL_TABLE}}

Temperature scaling reduces NLL for both the teacher and compressed conditions, without closing the teacher-relative NLL gap. This supports treating calibration as a control rather than interpreting raw likelihood differences as denoising. The practical BF16 rows are not pooled with historical FP16 quality or drift measurements.

Timing uses a resident NVIDIA GeForce RTX 4070 Ti, batch size one, 10 warmups and 30 synchronized samples. Prefill is measured at 128 and 512 tokens. Cached 128-token decode is timed separately and excludes the initial prompt pass. Table 5 compares anchor with same-seed BF16 whitening under identical workloads. Ratios below one indicate lower median latency; they are descriptive workload comparisons, not a new acceptance gate.

{{TIMING_TABLE}}

Median ratios vary by seed and workload: the anchor is slower for seed 11 at the 128-token conditions, while the observed seed-23 ratios favor it. Peak allocated/reserved memory and serialized bytes match between the paired factorized methods. These observations establish neither a consistent anchoring speed advantage nor general acceleration over the dense teacher. CUDA allocator peaks are not total device memory or system RAM. Whole-model reload probes cover a 128-token prompt and one cached next-token forward, not arbitrary generation or the entire test corpus.

## 6. Discussion

The results distinguish three questions. A representation can be smaller than its dense source, fit a data distribution better than a comparator, and still diverge from the teacher internally. None of these outcomes logically supplies the others. In particular, the LFM observation is evidence against using likelihood improvement alone as a preservation certificate. It is not evidence that residual similarity is always the correct optimization target: $D$ measures similarity in fixed coordinates, while different internal states can implement similar functions. A future claim about global preservation needs converging evidence from final logits, tasks, drift and generation rather than privileging one proxy.

The strongest comparison for anchoring is matched covariance weighting. Large gains over plain SVD primarily establish the importance of data-aware fitting within this implementation; they do not establish the incremental value or novelty of the anchor. Gemma's small average advantage, Qwen's joint worsening and LFM's tradeoff do not support a model-independent recommendation to add the same anchor. Because model weights, tokenizers, mixer structures and historical inference precision differ, this study cannot attribute the different outcomes solely to architecture.

The numerical controls also constrain causal interpretation. The measured operator includes factor casting, multiplication order, spectral cutoffs and finite-precision execution. Without passing the strict readiness checks, we cannot assign every downstream difference to the mathematical objective alone. Retaining these failures is useful: it identifies a prerequisite for stronger future comparisons instead of allowing a successful identity split to validate a different decomposition path.

Finally, model-level storage and speed depend on coverage and implementation. A targeted parameter saving can be diluted by embeddings and other dense components. Two dependent matrix multiplications can incur launch and intermediate-memory costs despite using fewer weights. Faster covariance preparation, fused BF16 factor execution and runtime-aware rank allocation are reasonable future hypotheses, but require new frozen protocols and matched quality/resource budgets. They are not measured benefits of the present study.

## 7. Limitations and reproducibility

WikiText-2 has already influenced this project; native-test results are regression evidence, not a fresh-domain confirmation. No downstream task or free-running generation equivalence is established. A locked sample from C4, whose provenance is described in the T5 study, could provide fresh confirmation [@raffel2020t5]; HellaSwag and ARC-Challenge could evaluate additional capabilities [@zellers2019hellaswag] [@clark2018arc]. None was executed here.

The scope is attention-output replacement at recorded budgets. MLPs, hybrid non-attention mixers, larger models and other rank allocations require separate conclusions. Three calibration seeds provide limited variation, and the later BF16 practical amendment has only two. Within-model matching does not make different model families or FP16/BF16 cohorts interchangeable. The Qwen frontier's eight absent validation measurements remain missing. The original strict numerical-readiness failure, incomplete coverage and unfulfilled promotion gates prevent promotion of a general preservation claim.

Every main table is reconstructed from the frozen evidence, with its source run identified in Appendix A and exact extraction fields retained in the revision manifest. The full historical report preserves commands, package/device versions, asset revisions, token manifests, factor recipes, source/checkpoint hashes, failed attempts and recovery accounting. Reused records are not independent replications. Recorded charged wall time includes failed/retried work and some reboot downtime; the first reboot has an unbounded unlogged tail, so no exact 24 GPU-hour compliance claim is made. The editorial revision launches no research experiment and changes no saved metric, fitting budget, model or dataset.

## 8. Conclusion

At the measured attention-output budgets, activation-aware low-rank fitting improves on plain SVD, but anchoring does not consistently improve both likelihood and propagated residual fidelity over matched whitening. Gemma's small full-stack average change does not transfer uniformly to layer groups; Qwen worsens on both metrics; LFM improves likelihood while increasing drift. The separate BF16 practical cohort achieves a measured 0.894% whole-checkpoint saving with mixed timing. These findings support reporting likelihood, teacher-state fidelity and practical cost as distinct outcomes. Reliable preservation of global computation remains unresolved.

## Appendix A. Evidence map and completion status

{{EVIDENCE_TABLE}}

The complete predecessor/continuation history and all unabridged tables remain in the [frozen evidence report](../../results/paper-v2/revised-report-20261014-151432-07c08d61/report.md). The [original manuscript](../../results/paper-v2/revised-report-20261014-151432-07c08d61/manuscript.md) is unchanged. The [revision evidence file](report.json) retains selected already-recorded paired intervals for the primary comparisons and the middle-four-block counterexample, also reproduced in the PDF and LaTeX appendices. The complete comparison family remains in the frozen report. These intervals are not adjusted for the many tested groups or configurations.

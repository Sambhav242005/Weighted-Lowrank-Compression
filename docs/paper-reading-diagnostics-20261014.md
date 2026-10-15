# Diagnostic and adjacent-paper reading record — 14 October 2026

Read primary full texts for the manuscript citation review; no benchmarks, datasets, or model evaluations were run. Coverage below distinguishes main-paper reading from selected appendix reading. Reading does not certify every proof. Summaries are intentionally short (under 110 words per paper).

## GPTQ

- Metadata: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323 (2022), ICLR 2023; reviewed v2 (22 March 2023).
- Full text: https://arxiv.org/html/2210.17323v2
- Read: main sections 1–8; appendix A.2 evaluation/timing details; inspected A.1/A.3/A.4 comparisons and tables.

GPTQ uses approximate second-order information to compensate remaining weights during sequential weight quantization, with batching and a Cholesky formulation for scale and stability. Later blocks receive inputs produced by already quantized blocks. Its low-batch generative speedups depend on specialized kernels and reduced memory traffic, rather than fewer multiplications. Cite as adjacent activation-sensitive compression. Do not call it whitening, pure low-rank replacement, or a method that ignores changed upstream inputs. Reported OPT/BLOOM perplexities are not matched baselines for this manuscript’s Gemma panel.

## AWQ

- Metadata: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. arXiv:2306.00978 (2023), MLSys 2024; reviewed v6 (25 April 2026).
- Full text: https://arxiv.org/html/2306.00978v6
- Read: main sections 1–6, including channel-scaling derivation, scale search, TinyChat, calibration sensitivity and experiments.

AWQ uses activation magnitudes to identify salient channels and searches diagonal channel scales before low-bit weight quantization. Its calibration search minimizes layer output discrepancy without gradient-based weight reconstruction; clipping also contributes. TinyChat translates the representation into hardware-specific acceleration. Distinguish diagonal scaling from full covariance weighting and rank truncation. The paper’s calibration-distribution robustness is empirical, not a universal non-overfitting guarantee. Describe GPTQ/AWQ as adjacent quantization methods rather than competitive baselines measured in this study.

## CKA

- Metadata: Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. *Similarity of Neural Network Representations Revisited*. ICML 2019, PMLR 97:3519–3529.
- Full text: https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf ; additionally read primary arXiv HTML https://arxiv.org/html/1905.00414
- Read: main sections 1–7; appendix A invariance theorem proof. Other appendix experiments were not comprehensively reviewed.

CKA compares centered representational similarity structures. Linear CKA is invariant to orthogonal transformations and isotropic scaling, and emphasizes high-variance directions. These desirable comparison properties also make it insufficient to certify faithful computation under fixed downstream weights. State centering, sample alignment and the actual estimator; distinguish CKA from uncentered cosine or covariance statistics. Use it as a complementary representation diagnostic rather than an invariant whose preservation guarantees logits or perplexity. The original experiments include CNNs and transformer encoders, not compressed contemporary decoder models.

## Temperature calibration

- Metadata: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks*. ICML 2017, PMLR 70:1321–1330.
- Full text: https://proceedings.mlr.press/v70/guo17a/guo17a.pdf
- Read: complete main paper sections 1–6 and experimental tables; separately referenced supplementary S1–S4 were not read.

Temperature scaling fits one positive scalar to validation NLL while leaving model weights and class ordering unchanged. The paper studies image and document classification with same-distribution splits. It supports the calibration procedure, not a claim that temperature restores compressed language-model representations. Keep native and temperature-adjusted perplexities separate; identify the validation data used to choose temperature. Lower adjusted perplexity alone does not establish improved ECE, hidden-state fidelity, or repaired computation. No hidden states change when only final logits are rescaled.

## WikiText / Pointer Sentinel

- Metadata: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher. *Pointer Sentinel Mixture Models*. arXiv:1609.07843v1 (26 September 2016).
- Full text: https://arxiv.org/html/1609.07843v1
- Read: main sections 1–7; supplementary pointer-use examples and WikiText-103 Zipfian plot captions.

The paper introduces a gated pointer/LSTM vocabulary mixture and the WikiText corpora. WikiText preserves article ordering, punctuation, casing and a larger vocabulary than commonly processed PTB. Cite it for dataset origin, then specify the actual raw dataset configuration, revision, tokenization and split used here. Original word-level perplexities are not directly comparable with this manuscript’s model-tokenizer perplexities. WikiText-2 and WikiText-103 have distinct training sets but share validation/test articles; using both does not create independent held-out corpora.

## SparseGPT

- Metadata: Elias Frantar and Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. arXiv:2301.00774v3 (22 March 2023); 2023.
- Full text: https://arxiv.org/html/2301.00774v3
- Read: main sections 1–6; appendices A–E, including reconstruction approximations, seed sensitivity, evaluation and acceleration details.

SparseGPT combines approximate Hessian-based reconstruction and adaptive pruning masks, with extensions to semi-structured sparsity and joint quantization. Strong results at very large OPT/BLOOM scales do not establish equal behavior for smaller models or low-rank factorization. Its success also prevents a universal claim that local reconstruction cannot compose. Dense zero-valued weights do not automatically yield smaller serialization or faster kernels. Mention as adjacent sparsification, preserving its different representation and hardware assumptions; never label manuscript magnitude pruning as SparseGPT without the corresponding algorithm and artifacts.

## Quantization Error Propagation (QEP)

- Metadata: Yamato Arai and Yuma Ichikawa. *Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization*. arXiv:2504.09629 (2025); accepted NeurIPS 2025 according to primary record; reviewed v3 (13 January 2026).
- Full text: https://arxiv.org/html/2504.09629v3
- Read: main sections 1–7, including limitations; appendix A; selected appendix B.1/B.2/B.3/B.4 statements and derivations. Detailed appendix D tables were not comprehensively reviewed.

QEP explicitly matches teacher outputs on teacher inputs against compressed outputs on altered upstream inputs. A continuous weight correction precedes an existing quantizer; propagation strength controls calibration sensitivity and cost. Restore this directly relevant prior art when motivating cross-layer drift. Its quantization representation differs from pure low-rank substitution. Avoid universal global-error guarantees: the appendix’s formal result concerns bounds under restrictive assumptions, while the main statement is stronger. This study may contribute a diagnostic comparison in its scoped protocol, not discovery of error accumulation itself.

## SVDQuant

- Metadata: Muyang Li, Yujun Lin, Zhekai Zhang, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, Song Han. *SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models*. arXiv:2411.05007 (2024); ICLR 2025 Spotlight; reviewed v4 (8 November 2025).
- Full text: https://arxiv.org/html/2411.05007v4
- Read: main sections 1–6; appendices A–F, including proof derivations, implementation, ablations, rank/latency tradeoffs and limitations at extreme bitwidths.

SVDQuant transfers activation outliers to weights, absorbs dominant components in a high-precision low-rank branch and retains a quantized residual branch. Nunchaku fuses kernels to reduce extra activation movement. Its primary experiments evaluate diffusion models and image quality; it does not establish successful pure low-rank replacement in language models. Cite only as a related hybrid low-rank/quantization design and practical-cost example. Preserve the residual branch distinction and hardware/workload context whenever discussing its speedups.

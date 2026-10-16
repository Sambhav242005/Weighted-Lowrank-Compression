# Gemma 3 270M: alternatives after the ReLU replacement result

Date: 2026-10-16. Research only; no experiment, training, model download, or source modification was performed for this note. Run facts below were supplied by the coordinating investigation and should be checked against its audit. Primary sources were consulted directly. Published successes on larger LLMs or CNNs are not evidence of success on this Gemma checkpoint.

## Evidence and interpretation

The completed exploratory run reports perplexity 47.223 dense, 58.306 activation-whitened linear low-rank, and 191.939 / 214.287 for two ReLU optimization seeds. It replaces all 18 attention output projections (1024 inputs, 640 outputs) with width-275 factors under the same 70%-of-dense parameter ceiling, with 200 local Adam updates and frozen teacher inputs. It uses ordinary PyTorch operations on a GTX 1050 and Core i5. Two optimization seeds are insufficient for the repository's three-seed promotion gate.

The result supports a narrow conclusion: this initialization/objective/budget for nonlinear replacements is worse than its matched linear baseline. It does not establish that custom GPU kernels, nonlinear compression generally, or all matrix families fail.

### Why ReLU is a substantial change

This is mathematical reasoning, not a newly measured cause. If the linear approximation is BAx, inserting ReLU gives B max(0, Ax). Negative bottleneck coordinates disappear unless the representation explicitly encodes their sign or shifts them. ReLU output coordinates being nonnegative does not imply final outputs cannot be negative: B can contain negative coefficients. The issue is loss of signed intermediate information and a different function class under a narrow budget.

Sign-split identity: z = ReLU(z) - ReLU(-z). Thus [B,-B] ReLU([A;-A]x) exactly equals BAx. But a conventional untied implementation doubles bottleneck width and factor parameter count. At width 275, at most rank 137 can be represented using complete pairs, compared with rank 275 for the linear baseline. Tying both signs can preserve storage, but two branches collapse algebraically to the original linear map; it supplies no extra expressive power or automatic speed advantage. CReLU studies reconstruction and phase preservation in CNNs; that motivates this diagnostic, not a transferable Gemma quality guarantee. [Shang et al., ICML 2016](https://proceedings.mlr.press/v48/shang16.pdf).

A shifted variant B ReLU(Ax+a)-Ba equals BAx wherever Ax+a >= 0 coordinatewise. Selecting a from calibration minima can give a function-preserving initialization on calibration inputs, but not on arbitrary held-out or compressed-model inputs. Bias parameters count against the ceiling. Report the fraction/magnitude of violated coordinates on held-out data. This is our proposed algebraic diagnostic, not a cited established LLM method.

Local teacher-input fitting also misses inference-time inputs produced by upstream compressed layers. Small per-layer errors can therefore compound. The two bad seeds do not identify whether insufficient optimization, sign loss, output biases, or input distribution mismatch dominates; saved per-layer local losses and downstream drift must decide that.

## Alternatives and what each actually addresses

| Route | Primary-source result or mechanism | Relevance and limits here |
|---|---|---|
| Keep a linear replacement; correct sequentially | SVD-LLM combines truncation-aware whitening with a layer-wise closed-form update. [Wang et al.](https://arxiv.org/abs/2403.07378) | Targets compression-induced mismatch without adding ReLU. Compare the existing implementation against the actual method before calling it SVD-LLM. |
| Allocate ranks by sensitivity | SVD-LLM V2 uses layer/matrix-specific truncation-loss allocation and loss-optimized truncation. [Wang et al.](https://arxiv.org/abs/2503.12340) | Equal width across 18 projections need not be optimal. Preserve total parameter budget while reallocating. V2 is not itself a group-end/logit distillation method. |
| Reconstruct short groups on student inputs | Official QAD documentation fits consecutive decoder blocks jointly and optionally propagates preceding compressed outputs. [LLM Compressor QAD](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/qad/) | Mechanistic support for changing teacher-only inputs to student inputs, then testing groups of 2 and 4. QAD is a quantization workflow; transplanting its objective to low rank is a proposal and requires custom validation. |
| Logit distillation | AQLM Appendix A uses end-to-end teacher/student KL recovery after block fitting. [Egiazarian et al.](https://arxiv.org/pdf/2401.06118) | Optimizes a global behavior signal beyond local MSE. Its published recovery uses millions of tokens; do not silently import that budget into the 200-update study or promise comparable recovery. |
| Low-rank error compensation | EoRA derives activation-eigenspace residual factors to compensate compression errors without gradient fitting. [Liu et al.](https://arxiv.org/abs/2410.21271), [official implementation](https://github.com/NVlabs/EoRA) | Useful especially around a quantized/pruned base. Count base, residuals, and quantization metadata. Keeping dense W and adding factors does not compress W. Two unconstrained linear low-rank paths at the same total rank combine into one rank-bounded map, so splitting existing factors alone need not beat a correctly solved weighted low-rank optimum. |
| Quantization, then optional residual recovery | GPTQ uses second-order information for weight quantization. [Frantar et al.](https://arxiv.org/abs/2210.17323) | Preserves matrix dimensions rather than removing rank. Test as a separate precision/storage protocol, with a declared byte budget and the original FP32 reference. Published A100/A6000 speedups do not predict GTX 1050 speed. |

Do not confuse LoRA training adapters with compression: an adapter normally augments retained weights. To reduce storage, the deployment representation must remove or quantize the original dense storage and count everything that remains.

## Hardware and runtime eligibility

Official bitsandbytes requirements distinguish features: NF4/FP4 requires compute capability 6.0+, while LLM.int8() requires 7.5+. Therefore Pascal-generation support for one feature cannot justify claiming all int8 kernels work on the GTX 1050. Exact installed wheel, CUDA toolkit, OS, driver, compute dtype and kernel support still need checking. [bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation).

llama.cpp offers quantized representations and separate CPU/CUDA implementations; it is a plausible runtime to investigate, not a verified Gemma-270M deployment here. Conversion correctness, architecture support at a pinned revision, tokenizer parity, and this GPU's build target must be established before timing. Compare unquantized and quantized models inside that same runtime to separate runtime changes from compression changes. [Official llama.cpp repository](https://github.com/ggml-org/llama.cpp).

The existing ordinary PyTorch two-matmul-plus-ReLU implementation is not a fused custom kernel. Fewer stored parameters do not guarantee less latency: additional launches, intermediate tensors, small matrix shapes, embedding/output costs and untargeted layers remain. These are engineering hypotheses; a profiler must establish which dominates. Optimize kernels after a quality-preserving candidate exists. Record prefill and decode separately, same tokens/batch/cache/threads, warm-up, synchronized GPU timings, peak allocated/reserved memory, serialized bytes and variation over repeats.

## Smallest next experiments, in order

These are proposals only. Keep the existing model revision, data/splits, FP32 and budget unless separately authorizing a changed protocol. One hypothesis-driving variable per comparison; retain existing artifacts.

1. **Saved-artifact diagnosis first:** compare initialization/final local error, held-out local error, signed bottleneck statistics, output mean/norm and block-by-block drift for the existing candidates. The point is to distinguish fitting failure from composition failure before spending another training budget.
2. **Activation ablation:** same initial factors, width 275, optimizer/updates/data/seed, replace ReLU by identity. Compare both local fit and held-out global behavior. Improvement toward the whitened baseline implicates the nonlinear parameterization; it does not alone separate initialization from optimization.
3. **Initialization-only test, if ReLU remains relevant:** shifted linear-preserving initialization with explicitly budgeted biases; identical subsequent updates and split. Also perform the sign-split identity as an algebra/unit diagnostic; do not present doubled-width sign split as budget matched.
4. **Input-source ablation on the better linear candidate:** fit the same factors against teacher targets using inputs propagated through the compressed prefix. Keep updates and rank fixed. Reduced downstream drift plus improved held-out KL/PPL supports the input-mismatch hypothesis.
5. **Composition scale test:** same selected candidate on one layer, then two consecutive layers, then four. Change only replacement scope, and compare a scope-matched baseline. Record every block's relative hidden error/cosine plus final KL/PPL. This localizes where compounding starts; it is not a whole-model promotion result.
6. **Only after these:** choose either group-end reconstruction, global KL recovery, or fixed-total-budget rank allocation as one next change. Quantization/hybrid experiments are a separate branch requiring an explicit precision/byte-budget protocol.

Decision rule: local MSE alone cannot advance a candidate. Require held-out PPL or logit KL improvement over the scope- and budget-matched linear baseline and a predeclared drift improvement; then repeat three seeds and evaluate storage, peak memory and warmed latency. A method with smaller files but unchanged/slower inference is a storage result. Current ReLU hypothesis: weakened for this precise configuration; broader nonlinear and fused-kernel hypotheses remain inconclusive.

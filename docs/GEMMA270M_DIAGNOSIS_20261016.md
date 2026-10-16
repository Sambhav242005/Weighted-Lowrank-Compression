# Why the Gemma 270M ReLU replacement failed

## Experiment status — forward diagnosis 20261016-mvaq5xb4

- State: completed. Forward-only diagnostics took 29.937 seconds under a separately declared 600-second diagnostic cap. No optimizer updates, new model/data download, test-set probing or historical artifact replacement occurred.
- Hypothesis: initialization, activation clipping, limited fitting and input-distribution drift explain the existing quality failure; the selected target also accounts for little CPU runtime.
- Configuration and matched baseline: identical pinned Gemma/data revisions, FP32, saved seed-11/23 checkpoints, all 18 attention-output projections, original 16 training windows and eight validation windows, rank/width 275. Whitening is the saved linear reference. All controls are diagnostics, not newly trained candidates. The doubled-width sign control is explicitly outside the compression budget.
- Observed evidence: numerical replay matches recorded trained local losses within approximately 1.5e-8; direct matrix orientation checks pass. ReLU changes the copied linear function dramatically, original initialization predicts almost zero, both seeds continue improving at the final fitting step, compressed-prefix inputs diverge, and target projections occupy about 4% of the instrumented dense CPU workload.
- Decision: the initial nonlinear parameterization and fitting protocol are inadequate in this setting. No loading/transposition defect was found. The completed experiment does not isolate the effect of ReLU from initialization and optimization. Nonlinear compression in general remains inconclusive.
- Next smallest experiment: a predeclared initialization-only comparison described below. Do not combine initialization, step budget, rank allocation and objective changes in one run.
- Risks or missing evidence: local forward controls do not establish better full-model perplexity; shifted coordinates can leave their calibrated region after upstream compression; three CPU profiling samples are not a GPU profile; longer-training benefits and causal contributions remain untested.

## Reproduction and controls

The frozen symptom was reproduced by loading the audited full-test metrics and asserting `relu_seed11.perplexity <= whitening_seed11.perplexity`. It deterministically fails: 191.939 versus 58.306. All 18 local errors are worse as well. This artifact replay establishes the symptom, not its cause.

The minimal mathematical example `x=[-2,-1], A=B=I` reproduces the sign-clipping change in the actual ReLU operation. Two offline tests verify (a) positive/negative paired units recover the original linear function and (b) the shifted construction is exact on its calibrated region but can fail outside it:

```powershell
rtk .venv-gemma270m/Scripts/python.exe -m pytest tests/unit/test_gemma270m_diagnosis.py -q
```

Result: 2 passed. The real-model forward probes then replay both saved candidates on exactly the original validation activations. No tensor-orientation or checkpoint discrepancy explains the loss: replayed seed-11 losses agree with recorded ones to within 1.5e-8. The previous audit already checked all non-target tensors, model provenance, replacement reloads and full-test coverage.

## 1. The initialization was a much worse function than the linear baseline

In `src/gemma270m_relu_study.py`, `fit_one` copies `right` into `candidate.down.weight`, zeros biases, and leaves `candidate.up.weight` at its random default. It does **not** start from the existing low-rank approximation. This was an experimental-design choice in the first implementation, not evidence against every nonlinear replacement.

Whitening stores very differently scaled factors. Across layers, the median norm of the saved output factor is 1,553.6, whereas the default random output factor has norm 14.61. These are actual norms, not a proof that Adam cannot learn the transformation: low-rank factors can be rescaled while preserving their product. They do show why copying only one factor is an inadequate warm start.

The median initial validation NMSE is **0.99976**, compared with **0.017668** for whitening. Thus the network spends its short fit learning a transformation that the baseline already supplies analytically. The first experiment changes initialization, activation and optimization method together, so its final PPL gap cannot be assigned uniquely to ReLU.

## 2. Simply inserting ReLU breaks the signed latent representation

For a linear map `y = B(Ax)`, the latent coordinate `Ax` is signed. Replacing it with `ReLU(Ax)` changes the function. On the actual validation activations, approximately 50.17% of coordinates are negative and they account for approximately 50.37% of latent squared energy (medians across layers). These percentages alone do not prove irrecoverable information loss: the output factor can contain negative weights and a trained representation can encode signs differently. The controlled output errors provide stronger evidence.

| Forward control | Median validation NMSE across 18 layers | Interpretation |
|---|---:|---|
| Saved linear whitening | 0.0176683 | Reference |
| Original random-output ReLU initialization | 0.999760 | Starts far from the reference |
| Copy both factors, insert ReLU | 0.732352 | Copying the missing output factor alone is insufficient |
| Train-derived shift plus copied factors | 0.0177780 | Nearly recovers the reference locally, without fitting |
| Paired signs at budget-compatible width 274 / effective rank 137 | 0.0579396 | Preserving signs consumes capacity at this parameter ceiling |

For paired signs, `z = ReLU(z) - ReLU(-z)`, so `[B,-B] ReLU([A;-A]x)` recovers `BAx`. At rank 275 this needs 550 conventional units and 915,200 untied factor weights, exceeding the original dense projection's 655,360 weights. Its measured discrepancy versus the linear reference is about 1.5e-13 median NMSE, validating the algebra. It is not a compression success. At fixed total width, complete sign pairs reduce effective rank.

For the shifted control, derive `a = -min_train(Ax) + 1e-4` independently per coordinate, and use `B ReLU(Ax+a) - Ba`. It equals `BAx` wherever all shifted coordinates are nonnegative. Its calibration discrepancy is numerical roundoff. On held-out validation, the median clipped-coordinate fraction is about **0.125%**, and median local error is nearly the whitening reference. Its 458,515 parameters fit the existing ReLU ceiling, including both biases.

This is a useful initialization control, not a verified solution. It restores approximately the linear behavior rather than demonstrating a nonlinear advantage, and the input distribution can change after compression. No full-stack shifted model was evaluated.

## 3. The optimizer was still improving when the fixed budget ended

For both optimization seeds, all **18/18** best checkpoints occur at update **200**, the final allowed step. Across layers, median validation loss falls another **6.10%** (seed 11) / **5.64%** (seed 23) between steps 180 and 200. This supports incomplete convergence under the chosen budget. It does not establish that more updates will reach whitening or improve global behavior.

Keep this distinct from a numerical failure: outputs/losses are finite, full-rank checks passed, and saved weights replay. The result is consistent with underfitting and a poor starting parameterization, rather than NaNs or a broken checkpoint.

Every fitted layer also has units inactive on all 2,048 validation activation rows. Median counts are 49.5/275 for seed 11 and 50/275 for seed 23; the worst individual layer has 138 inactive units. These are validation-inactive units, not proof of globally dead neurons or zero training gradients. They nevertheless show that part of the nominal bottleneck contributes nothing on the measured held-out activations.

## 4. Later replacements receive inputs unlike their fitting data

The original fits use dense-teacher inputs for every projection. On validation inputs propagated through the compressed model, the relative RMSE of the input to the final target is **0.4452** for whitening and **0.7816 / 0.7854** for the ReLU seeds. The first target input is identical by construction; discrepancy grows after preceding replacements, though it is not monotonic at every layer.

This directly measures an input-distribution mismatch. It supports accumulated drift as a further mechanism, while the already-worse local fits show composition is not the sole cause. The full test confirms final normalized-state drift 0.3244 for whitening versus 0.6351 / 0.6446 for ReLU.

Sequential updates based on compressed inputs are a relevant research direction: [SVD-LLM](https://arxiv.org/abs/2403.07378) includes a sequential parameter update after truncation. Our current whitening component is not a reproduction of that complete method.

## 5. We compressed a small part of the CPU workload

Three instrumented dense CPU repeats, each comprising a 128-token prefill plus 32 cached steps after two warmups, attribute **3.95–4.07%** of wall time to all 18 target projections, versus **32.84–34.34%** to the vocabulary output head. The remaining time is other layers, attention, cache handling and framework work. This profile has hooks and only three repeats; it cannot establish the GPU bottleneck or isolate every overhead.

Using the median target fraction of about 4.01% in Amdahl's formula, removing target work completely would yield only about **1.042x** speed for this instrumented CPU workload. A 30% improvement in target time corresponds to roughly **1.012x** total speed under the same assumptions, before extra overhead. Parameter reduction is not automatically a 30% target-time reduction either.

The deployed prototype uses two ordinary matrix operations and a ReLU, not one fused custom kernel. Additional operations and intermediate tensors can erase small savings. The existing benchmark's fixed ordering and substantial repeated-baseline variation also prevent interpreting small timing differences as method improvements. PyTorch describes fusion as a way to reduce launches and memory traffic; that is an optimization hypothesis requiring measurement here, not a promise. [Official performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Proposed next experiment: change initialization only

This configuration is a proposal, not a training run or authorization to extend the completed study:

```json
{
  "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
  "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
  "dtype": "float32",
  "targets": "same18_attention_outputs",
  "width": 275,
  "seeds": [11, 23],
  "calibration_and_validation": "exact original token manifests",
  "optimizer": "Adam",
  "learning_rate": 0.001,
  "updates_per_target": 200,
  "batch_size": 256,
  "changed_variable": "initialization policy only",
  "control_initialization": "original copied input factor, random output factor, zero biases",
  "candidate_initialization": "copy both whitening factors, train-derived positive shift, compensating output bias",
  "first_gate": "finite gradients and near-whitening pre-fit local error",
  "selection": "validation-only, predeclared mean-NMSE and block-drift comparison; no test-driven tuning",
  "compute_budget": "must be separately frozen before any new training",
  "test_evaluation": "only after freezing the candidate; this previously seen test is not fresh confirmation"
}
```

Potential factor-balancing is a separate initialization/conditioning ablation; do not quietly combine it with the shift comparison. If locally competitive initialization still fails after composition, change teacher-input fitting to compressed-prefix inputs as the next separate variable, then test scope-matched groups of 2 and 4. Only after isolating these should group-end or logit-distillation objectives be compared.

For practical deployment, a separate quantization study could preserve matrix dimensions and reduce bytes in larger storage/runtime contributors, with a declared precision and byte budget. EoRA is a relevant residual-compensation method around compressed bases, not a guarantee for this device. Keeping the dense matrix plus an adapter does not constitute compression. [EoRA](https://arxiv.org/abs/2410.21271).

## Artifacts

- [Raw diagnostic metrics](../results/gemma270m-diagnosis/20261016-mvaq5xb4/metrics.json)
- [Diagnostic contract](../results/gemma270m-diagnosis/20261016-mvaq5xb4/resolved-config.json)
- [Detailed alternatives and primary sources](GEMMA270M_ALTERNATIVES_RESEARCH_20261016.md)
- [Original full-test result](GEMMA270M_RESULTS_20261016.md)

The original research runner, checkpoints and full-test measurements are unchanged. The diagnostic controls explain concrete defects and limitations in the first setup; they do not yet provide a replacement meeting the project's promotion gates.

Final focused verification: 16 tests passed across the original replacement, recovery-equivalence and new algebraic-diagnostic tests. Original runner/math source hashes still match the frozen experiment contract.

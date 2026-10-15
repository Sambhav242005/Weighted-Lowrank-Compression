# Diagnosis snapshot — 2026-10-14

This additive audit preserves the historical drafts and results. It separates
observed numerical/runtime failures from older interpretations that are no
longer supported by the current matched experiment. It does not authorize a new
experiment, change source code, or change any model, split, tolerance, or budget.

## Main finding

The evidence does **not** identify insufficient GPU capacity as the cause of the
readiness failure. The measured problem is full-rank FP16 factorization failing
the declared elementwise logit tolerance. A separate unexpected system reboot
interrupted composition; its cause is unknown. The research problem remains
global behavior after composition: activation weighting substantially improves
on plain SVD, but the selected anchor has not established a reproducible drift
advantage over the stronger whitening comparator.

## Observed failures and their scope

| Issue | Observable evidence | Interpretation and current action |
|---|---|---|
| Initial Python environment incompatibility | [Implementation log](PAPER_V2_IMPLEMENTATION.md), lines 70–82, records system Transformers 4.53.3 with incompatible Hub 2.2.0. The initial attempt produced no metrics. | An environment failure, separate from model fitting. Experiments use the recorded `.venv-cuda` environment; do not invoke the system research stack. |
| Strict FP16 identity failure | [Diagnosis metrics](../results/paper-v2/svd-diagnosis-20261013-165146-c3671715/metrics.json), [completed status](../results/paper-v2/svd-diagnosis-20261013-165146-c3671715/status.json), and [v2 manuscript](paper_protocol_v2.md), lines 45–67. | Numerical sensitivity is supported by the precision controls. Readiness remains false; downstream comparisons are exploratory. This is not evidence of an out-of-memory failure. |
| Unexpected reboot | [Retained Windows-event evidence](../results/paper-v2/composition-20261013-185512-2bffd477/reboot-evidence.json) records events 41 and 6008. The interrupted run retained 34 completed records. | These events establish an unclean shutdown, not whether power loss, a driver, hardware, or another system condition caused it. Preserve the interruption and resume only verified complete records. |
| Interrupted-run accounting uncertainty | [Continuation accounting caveat](../results/paper-v2/composition-20261014-032120-486d799f/accounting-caveat.json) charges 8,972.373 seconds from the last log plus 60 seconds. | The allowance is an estimate, not a proven upper bound on unlogged work. Do not claim exact 24 GPU-hour compliance from it. |
| Secondary CKA numerical overflow, previously repaired | [Precision repair patch](geometry_precision_repair.patch), and the normalized/FP64 reductions in `src/paper_comparisons.py`, lines 204–215. | The previous FP32 product of large Gram squared sums could overflow. Completed Stage C uses the repaired metric. This secondary diagnostic defect does not explain the FP16 logit identity failure or the reboot. |
| Stale documentation test harness, repaired during this audit | The supervising agent's explicit suite initially produced five failures, all in `docs/test_composition_cache_repair.py`: its harness tried to reapply a patch already installed in source. | The docs-only harness now tests the installed helpers when present; its five targeted checks pass. This was a test-harness failure, not a new GPU experiment failure. `TESTING.md` also no longer incorrectly claims there are no tests. |

No new GPU probe or experiment was run for this audit. Composition continuation
`composition-20261014-032120-486d799f` had state `running` and 37 complete records
when inspected. This is a snapshot, not a terminal conclusion.

The supervising agent's subsequent snapshot reached 38 of 60 composition
records and verified all 74 recorded source hashes unchanged. Separate
GPT-2/practical-stage code auditing is pending at this audit's writing.

## What the precision controls actually say

The untouched FP16 teacher has full-test NLL 3.451562542535798 and PPL
31.549651486382494 on 292,281 shifted targets. Original tolerances remain
unchanged. The probe uses combined absolute and relative tolerance, so maximum
absolute error by itself is not the decision rule.

| Full-rank control | Maximum absolute probe logit difference | Test NLL | Probe gate | NLL gate |
|---|---:|---:|---|---|
| FP32 SVD reconstruction, copied into dense teacher-dtype weights | 0.1171875 | 3.451660701 | Pass | Pass |
| Reconstruction from FP16 factors, copied into dense teacher-dtype weights | 0.140625 | 3.451704808 | Fail | Pass |
| Unbalanced FP16 factor execution | 0.15234375 | 3.451860161 | Fail | Pass |
| Balanced FP16 factor execution | 0.181640625 | 3.451848345 | Fail | Pass |
| FP32 factor execution, output returned to the input dtype | 0.1171875 | 3.451771295 | Pass | Pass |

The exact execution definitions matter: `src/paper_svd_diagnosis.py`, lines
19–30 and 87–104, shows that “FP32 reconstruction” is not an all-FP32 language
model. It reconstructs weights in FP32 and copies them into the original
FP16 dense module. The FP32 factor module casts its output back to input dtype.
These controls support a rounding/execution-precision diagnosis within the
tested setting. They do not establish a universal FP16 error bound or make the
failed readiness gate pass. The exact identity-factor control passed separately;
that does not replace the SVD-factor readiness test.

The supervising agent also checked the official
[PyTorch 2.13 numerical-accuracy documentation](https://docs.pytorch.org/docs/2.13/notes/numerical_accuracy.html)
through Context7. It explains that mathematically equivalent floating-point
operations need not give identical results and that operation order matters.
This supports the finite-precision interpretation; it does not substitute for
the project's predeclared numerical gate or attribute the reboot to PyTorch.

## What the completed comparisons establish

Stage B selected exact anchoring beta 0.03, ridge zero using validation only.
The strongest activation-aware baseline was whitening. The selected anchor and
selected solver are the same candidate, not independent methods. Source:
[frozen selection](../results/paper-v2/regularization-20261013-171941-804f4c2e/selection.json)
and [Stage-B metrics](../results/paper-v2/regularization-20261013-171941-804f4c2e/metrics.json).

Stage C is completed, with 13 records in
[the measured metrics](../results/paper-v2/baselines-20261013-175255-6a00d4e8/metrics.json).
All methods replace the same 26 Gemma attention output projections, each
1152 by 1024, rank 384, FP16 factors. All evaluate the entire native test split.

| Method | PPL, mean across calibration seeds | Primary drift, mean across calibration seeds |
|---|---:|---:|
| Dense teacher | 31.54965149 | Reference |
| Plain SVD, one deterministic fit | 162.17583073 | 0.074247583 |
| ASVD component | 40.22416796 | 0.030971745 |
| Whitening component | 35.61833233 | 0.010637517 |
| Exact anchor beta 0.03 | 35.37310360 | 0.010547001 |
| Recorded legacy ridge, FP64 algebraic reproduction | 35.83978035 | 0.011835971 |

Anchoring improves NLL against whitening in seeds 11, 23, and 37, respectively,
by 0.003119252, 0.008670166, and 0.008888696. Primary drift improves only in
seed 23; it slightly worsens in seeds 11 and 37. Mean drift alone hides that
failure of the three-seed gate. All compressed methods have worse PPL than the
dense teacher. See [the v2 manuscript](paper_protocol_v2.md), lines 94–145, for
every seed and the limits of the paired block uncertainty estimates.

This supports activation awareness over weight-only SVD **in this setting**.
It does not establish that anchoring solves composition, improves on the
teacher, generalizes to another architecture, or meets practical cost gates.
Those are distinct hypotheses.

## Historical documentation requiring careful interpretation

The following are claims in preserved historical drafts, not current
protocol-v2 conclusions. This audit does not erase their measurements.

| Source and lines | Historical statement | Why it must not drive current decisions |
|---|---|---|
| [RESEARCH_SUMMARY.md](archive/historical-drafts/RESEARCH_SUMMARY.md), 60–70; [MEMORY.md](MEMORY.md), 19 and 162–186 | Weight-space SVD is “the best method”; later passages say activation weighting beats SVD. | The documents combine successive research stages without retiring earlier generalizations. Current matched C measurements favor weighting, especially full covariance, over plain SVD. |
| [RESEARCH_SUMMARY.md](archive/historical-drafts/RESEARCH_SUMMARY.md), 54–58; [MEMORY.md](MEMORY.md), 15 | Shared compression is “ALWAYS worse” and shared structure is false/a myth. | Failure of the tested shared-input-subspace implementation cannot reject every shared representation, family, budget, or architecture. Scope the negative result to its actual experiment. |
| [RESEARCH_SUMMARY.md](archive/historical-drafts/RESEARCH_SUMMARY.md), 9–27 and 114–121; [MEMORY.md](MEMORY.md), 133–147 | Universal compression limits and size-based causal explanations or 7B+ predictions. | These are historical protocol-dependent observations and extrapolations. A rank divisor is not a factor-storage or whole-model compression ratio; architecture and protocol also change. No causal isolation establishes redundancy or depth as the reason. |
| [CONTEXT.md](CONTEXT.md), 24 and 57–60 | Drift is *the* reason compression fails, with architecture independence. | Accumulated deviation is measurable, but it is not by itself a unique causal explanation. Weight-space error, activation distribution, precision, and matrix family also matter. Current replication remains pending at this snapshot. |
| [Research.md](archive/historical-drafts/Research.md), 82–88 and 98–100 | Meeting any one permissive criterion suffices for “promising”; breakthrough partially achieved from local/older results. | Current AGENTS promotion gates and [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), 238–253, require matched global and drift evidence, three seeds, and practical limits. Old local success does not satisfy them. |
| [PAPER.md](archive/historical-drafts/PAPER.md), 135–163; [MEMORY.md](MEMORY.md), 166–171 | The recorded ridge/floor solver is the exact anchored rank-constrained optimum and an anchor is mandatory. | For exact teacher outputs and no extra ridge, the full-rank anchored optimum is W. Additional ridge changes it to shrinkage; finite-precision floors further change the metric. A specified pseudoinverse handles an unanchored singular metric. Current beta-zero whitening is stable and competitive, so the universal anchor requirement is unsupported. |
| [PAPER.md](archive/historical-drafts/PAPER.md), 26–31, 194–201 and 300–341; [MEMORY.md](MEMORY.md), 235–251 | Compression beats the teacher and removes “dead directions,” establishing denoising. | The positive historical measurement used up to 200 selected/truncated texts, with distinct baseline values. The current full native-test protocol gives worse PPL than the teacher. Higher entropy plus lower subset NLL does not identify a causal mechanism; validation-frozen temperature controls and fresh confirmation are needed. A direction literally null on an input cannot change Wx on that same input; calibration-null and universally irrelevant directions must not be conflated. |
| [PAPER.md](archive/historical-drafts/PAPER.md), 33–34 and 264–271; [MEMORY.md](MEMORY.md), 203–219 | INT8 factors are lossless. | Similar aggregate PPL does not imply identical factors, logits, or outputs. Report no measurable additional quality loss only when supported under the declared protocol; packed byte/reload/timing checks remain separate. |
| [PAPER.md](archive/historical-drafts/PAPER.md), 354–369; [MEMORY.md](MEMORY.md), 204–213 | Later passages correctly distinguish rank and factor storage. | Preserve that correction, but do not transfer a Qwen matrix fraction to Gemma or call the Stage-C factor archive whole-model storage. Gemma projections are rectangular. |
| [MEMORY.md](MEMORY.md), 259–263 | Ready for submission preparation; all key findings documented. | This historical status predates the current readiness failure and incomplete composition/replication/cost evidence. The additive v2 manuscript is the current working account. |

## Next decisions within the authorized study

1. Continue the existing composition continuation with its frozen checkpoints,
   splits, seeds, and remaining cap; retain every completed and interrupted
   record with provenance. Do not infer a full composition conclusion from a
   partial group/seed table.
2. Compare the selected anchor with whitening, not only plain SVD. Report where
   group-end drift, downstream drift, and NLL agree or disagree. A local error
   improvement alone is not success.
3. Run the already-authorized frozen GPT-2 transfer and practical/calibration
   stages only through the study's existing sequencing and budgets. Record
   uncertainty and incomplete stages rather than reducing evaluation or seeds.
4. Maintain exploratory classification while strict FP16 readiness is false.
   Do not silently change precision or relax tolerance to obtain promotion.
5. Update the additive v2 account after terminal evidence arrives. Keep older
   drafts as historical provenance; cite the v2 account for current conclusions.
6. If composition remains unresolved, the plan identifies a fixed-rank
   SAES-SVD comparison as the next proposal, before an unrestricted objective
   sweep. It is not part of a silent budget expansion.

## Audit limits

The completed [runner audit](RUNNER_AUDIT_20261014.md) additionally checks the
actual GPT-2 adapter and FP16/INT8 whole-model checkpoint reloads on a tiny CPU
model, including cached next-token output and tied embeddings. Those checks
pass. The report also records a conservative limitation: paired uncertainty is
computed for NLL, so automatic KL-only promotion cannot currently be established.
This does not invalidate saved KL measurements or change the failed readiness
gate. The five repaired composition-harness checks pass; the other 69 checks
and 74 subtests passed in the broader invocation before that harness repair.

This is a documentation/evidence audit, not a new model, driver, power, or
hardware stress test. It does not independently identify the reboot cause or
establish a universal numerical tolerance. Historical numbers were inspected
in their source documents; they were not rerun or relabeled as comparable
protocol-v2 evidence. Primary completed JSON results were read directly for the
current numerical and B/C conclusions. No research source/configuration or
historical result was modified by this audit.

The current interim uncertainty analysis covers paired NLL blocks. A matched
paired uncertainty analysis for teacher KL has not yet been established here;
an aggregate KL improvement alone must not be described as statistically
established. This is a reporting gap rather than evidence of corrupted KL
measurements, and it cannot override the already-failed readiness/drift gates.

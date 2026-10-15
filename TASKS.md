# Remaining paper tasks

**Authorized bounded work complete: BF16 only, maximum two seeds.**
The earlier stop remains preserved in
[user stop evidence](results/paper-v2/user-stop-1791984635/stop.json).
Only the new BF16 cohort is authorized: dense teacher plus whitening/anchor at
seeds 11 and 23, five conditions. Do not restart historical other-format tests,
completed quality/composition cohorts or the exhausted Qwen frontier. See the
[BF16 amendment](docs/BF16_TWO_SEED_20261014.md) and
[active ledger](results/paper-v2/revised-orchestration-20261014-5-bf16/status.json).

Updated 14 October 2026. The user has resumed only the remaining **BF16 practical
tests with two seeds**, under ledger 5. Earlier continuations are historical;
see the [continuation history](docs/CONTINUATION_20261014.md).
This list does not authorize expanding budgets or restarting an active process.

The paper can report measured weight-count reduction and bounded quality
tradeoffs. Establishing reliable preservation of global computation can remain
future work; it must not be implied by the current title, abstract or conclusion.

## Already completed

- [x] Additive pinned experiment system, split checks, cumulative budgets,
  checkpoints, source hashes, drift metrics and report generation.
- [x] Gemma regularization selection on validation; frozen anchor beta 0.03 with
  no extra ridge; matched whitening baseline.
- [x] Gemma full-stack baseline comparison across seeds 11/23/37.
- [x] Gemma composition: 60 declared records, with reuse identified explicitly.
- [x] Seven of nine BF16/FP8 numerical controls; failures retained.
- [x] Pinned Qwen/LFM downloads and CPU checkpoint/adapter preflight.
- [x] Draft manuscript, 22-reference bibliography, measured Gemma plots and
  an educational architecture comparison.
- [x] Report/PDF/standalone-LaTeX export path prepared and tested on actual
  interim evidence. The 24-page interim PDF has no off-page text, its first,
  figure and final pages were inspected, and its LaTeX source compiled. Final
  post-experiment manuscript proof is now complete; see the final outcome below.

## Finish the bounded first study, in order

Status checked on 14 October 2026: Qwen has
**8/8 primary test records and 10/18 validation-frontier records** saved in
[its closed recovery cohort](results/paper-v2/architecture-replication-20261014-093845-97378b5e/metrics.json).
The three-seed primary comparison is complete: anchor mean PPL 18.979 versus
whitening 18.811, with higher drift for anchor in every seed. These results
weaken the anchor hypothesis on this architecture. The full cohort remains
incomplete. The third reboot is recorded in the
[interruption observation](docs/archive/study-interruption-observed-20261014.json).
Recovery validated and reused all ten saved records in a new run, with the
3,647.159-second launch-to-next-boot interval charged. About 3,552.841 seconds
remained in the original Qwen cap at recovery start. Ledger 4 is historical;
original artifacts remain unchanged. Qwen has now closed at its cumulative cap:
8/8 primary test and 10/18 validation records are saved. Eight validation scopes
remain incomplete: divisor 3, seed 37, both methods; divisor 4, all three seeds,
both methods. LFM completed all 26 records before the new seed limit. The old
Gemma FP16/INT8 practical stage stopped with seven complete records; a separate
five-condition BF16/two-seed practical cohort completed within the remaining
original practical budget. Historical seed-37 data remains preserved.

- [x] **1. Close the numerical investigation and review the continuation.**
  Failed FP16/BF16 readiness and all seven completed controls remain preserved.
  The two unfinished dense-FP8 storage controls are explicitly omitted in the
  continuation decision, rather than silently dropped or reported as complete.
  No partial precision attempt is repeated; its recorded cost remains charged.
  The new dispatcher does not reapply the one-time accounting edit.
- [x] **2. Close the bounded Qwen3.5-0.8B attempt — budget-limited.** Six full-attention output targets;
  primary rank 483; eight native-test records and eighteen validation-frontier
  records, with seeds 11/23/37 and the frozen methods. DeltaNet mixers remain
  dense. Keep the existing two-hour model cap.
  The full scientific cohort remains **incomplete**, with 8/8 test and 10/18
  validation records saved. Recovery elapsed 3,553.015 seconds; with the reboot
  charge, cumulative time is 7,200.173 seconds (0.173-second cooperative boundary
  overshoot). This checkbox closes authorized execution, not the missing science.
  No further Qwen execution is authorized under this exhausted allocation.
- [x] **3. Run LFM2.5-1.2B confirmation — 26/26 records completed.** Six full-attention output targets;
  primary rank 725; the same eight test/eighteen validation scopes. Convolution
  mixers remain dense. Keep its two-hour cap and the shared four-hour E cap.
- [x] **4. Complete Gemma temperature-calibration controls.** Fit temperatures
  on validation, freeze before test, and report raw/scaled NLL, perplexity,
  accuracy, entropy and 15-bin confidence calibration. This tests a possible
  explanation for apparent quality gains; it does not establish denoising.
  Remaining execution is BF16 only, seeds 11/23; earlier other-format rows remain
  a separate cancelled cohort. Do not evaluate a third seed.
- [x] **5. Measure practical storage and execution.** Compare matched whitening
  and anchored BF16 models for seeds 11/23 only. Verify
  factor and whole-model reload parity; measure actual factor/whole-model bytes,
  target and whole-model savings, synchronized median/p95 latency, and allocated/
  reserved GPU memory. Use the declared 128/512-token prefill and 128-token decode,
  ten warmups and thirty measurements. Calibration and these measurements share
  the original four-hour F allocation. Record offload placement separately.
- [x] **6. Freeze and audit the final evidence.** Check exact pins, target counts,
  seed coverage, split-specific rank scopes, full native-test tails, immutable
  teachers and checkpoint hashes. Generate 2,000 paired resamples of fixed
  4,096-target blocks; report per-seed values and mean/sample SD. Do not rank raw
  PPL across different tokenizers or count reused rows as independent repeats.
  Preserve budget-limited cohorts as incomplete. The first reboot prevents an
  exact 24 GPU-hour compliance claim; do not borrow unused stage budgets.
  The completed Gemma C/D audit has passed checkpoint SHA, full-test coverage,
  matched recipe and reuse checks; see
  [the audit](results/paper-v2/evidence-audit-20261014-095952-f9c97619/report.md).
  The [Qwen partial-cohort audit](results/paper-v2/qwen-evidence-audit-20261014-104445-4f5593f2/report.md)
  and [completed LFM audit](results/paper-v2/lfm-evidence-audit-20261014-140916-9e1e9fcf/report.md)
  also passed. The final BF16 practical audit passed; final report and export checks are complete.

## Finish the manuscript and review package

- [x] **7. Finalize the claim and mathematical description.** Explain that
  low-rank factorization supplies the smaller representation; anchoring changes
  the fitting objective. Distinguish canonical weighting, exact anchoring and
  extra ridge. Compare the anchor with the strong matched whitening component,
  not only plain SVD. A bounded negative/mixed empirical paper is a valid outcome;
  no task requires tuning until our method wins.
- [x] **8. Refresh every table and scientific figure from the frozen evidence.**
  Include baseline, composition, three-model confirmation, validation rank
  frontier, calibration and practical-cost tables; drift-versus-depth and
  quality-versus-storage plots. Clearly separate parameter counts, serialized
  bytes and measured GPU resources. Final measured storage and runtime tables and storage-quality figures are included.
- [x] **9. Finish the prose and citations.** Align abstract, contribution,
  methods, results, limitations and conclusion; label historical experiments;
  verify that the 22 references are used accurately. Keep the unresolved global
  preservation question explicit rather than presenting it as solved.
- [x] **10. Finish reproducibility and release review.** Include exact commands,
  packages/CUDA/device, model/data revisions, calibration positions, seed/split
  manifests, artifacts/hashes, failed/incomplete attempts and reboot accounting.
  Verify setup instructions and the required CPU tests. Review source/config
  changes before committing; keep model weights and raw cached datasets out of
  Git while retaining small evidence/provenance files.
- [x] **11. Prepare and proof the final submission files.** Produce the revised
  Markdown/LaTeX/PDF, consistent figures/captions, reference list and appendix.
  The rendered PDF was inspected; venue selection and venue-specific formatting remain human review. The historical
  LaTeX/Word drafts are not a finished revised submission.

## Conditional follow-up requiring a separate frozen protocol and budget

- [ ] **Fresh-data confirmation** for stronger generalization claims: pin the
  proposed sampled C4 development/confirmation protocol before examining results.
  Existing WikiText-2 evidence has influenced the research.
- [ ] **Downstream task confirmation** if claiming maintained task capability:
  pin HellaSwag, ARC-Challenge, harness revision and prompt definitions.
- [ ] **Broader compression research**, only if needed: accumulated-error-aware
  comparison such as SAES-SVD, MLP targets, adaptive ranks, residual compensation,
  or larger-model replication. These are not required to finish a paper scoped
  to the current attention-output experiments and do not have an automatic
  extension of the first-wave budget.

See [the current review handoff](docs/REVIEW_20261014.md),
[the original plan](docs/EXPERIMENT_PLAN.md) and
[the model/precision amendment](docs/EXPERIMENT_PLAN_20261014_AMENDMENT.md).

## Repository cleanup

- [x] Replace the front page with current evidence and clear limitations;
  preserve the earlier README and Word draft in the archive.
- [x] Group five historical paper drafts/plans in `docs/archive/historical-drafts/`;
  verify unchanged contents in the
  [relocation manifest](docs/archive/cleanup-historical-drafts-20261014.json).
- [x] Update documentation links and add a [source index](src/README.md).
- [x] Keep model weights, environments, caches and local PDF proof images ignored;
  retain experiment JSON and provenance as reviewable evidence.
- [x] Refresh front-page/result links after final experiments and manuscript proof.
- [ ] Human review of the final changes and publication format. Commit, push and
  publication remain separate human decisions.

## Final bounded-study outcome

All authorized execution and artifact preparation are complete. BF16 practical: 5/5 conditions, seeds 11/23, 4,688.669 seconds; cumulative original practical charge 11,524.515 / 14,400 seconds. No GPU research process remains. Qwen retains eight missing validation scopes under its exhausted cap. No experiment was restarted to fill them.

- [Final manuscript PDF](results/paper-v2/revised-report-20261014-151432-07c08d61/manuscript.pdf), [Markdown](results/paper-v2/revised-report-20261014-151432-07c08d61/manuscript.md), [evidence report](results/paper-v2/revised-report-20261014-151432-07c08d61/report.md).
- [BF16 checkpoint audit](results/paper-v2/bf16-practical-audit-20261014-150957-37e12636/report.md): passed.
- [Final compilation/PDF proof](results/paper-v2/revised-report-20261014-151432-07c08d61/final-validation.json): LaTeX success, 45-page separate ReportLab PDF, no off-page text, representative pages inspected.
- Reporting/export verification: 27 tests and 26 subtests passed.

Remaining: human scientific/code review, venue choice and formatting, and optional separately budgeted future studies. No automatic commit, push or publication.

- [x] Add explicit future scope for offline fitting and BF16 runtime optimization; regenerate and compile the review manuscript. These are proposals, not measured speedups.

# Continuation after review — 14 October 2026

The user's instruction to continue resumes the bounded paper work. Preserve the
cancelled-for-review dispatcher and numerical run unchanged; use a new dispatcher
ledger, `results/paper-v2/revised-orchestration-20261014-3`.

## Third-reboot recovery

Ledger 3 was interrupted by an unexpected shutdown confirmed by Windows events
41/6008. Its files remain unchanged. The active replacement ledger is
`results/paper-v2/revised-orchestration-20261014-4/status.json`, dispatched by
`docs/continue_after_reboot3_20261014.py`.

The additive `src/paper_architecture_recovery.py` validates and reuses the ten
saved Qwen scopes (eight test, two validation), including original checkpoint,
selection, token and source hashes. It evaluates only missing scopes into a new
directory. The frozen original architecture runner and v3 configuration are
unchanged. Reused rows are continuation evidence, not independent replications.

`results/paper-v2/interruption-accounting-20261014-reboot3.json` binds the original
artifacts to a conservative 3,647.1586377620697-second launch-to-next-boot charge,
leaving 3,552.8413622379303 seconds in Qwen's original two-hour cap. The report
validates this sidecar without rewriting the old running status. LFM retains
its own two-hour limit, both models share four hours, and Gemma practical work
retains its independent four-hour cap. Missing scopes at a cap remain incomplete.

## Numerical-control closure

Retain all seven completed BF16/FP8 controls, including failed gates. Explicitly
omit the two unfinished dense-FP8 storage controls (`dense_fp8_e4m3` and
`dense_fp8_e5m2`) from this continuation. The controls were a side investigation;
the paired FP8 factor-storage results already fail the numerical gates, and
native FP8 model inference is outside the implemented protocol. More dense-FP8
controls do not answer the central matched low-rank comparison. This is a
declared scope decision, not successful completion of the original nine controls.

The stopped run remains cancelled with its partial artifacts. Its charged
382.99257254600525 seconds remain in shared readiness accounting, for a total
1381.3921549460051 seconds. No precision attempt is repeated and the original
failed FP16 readiness is not changed to a pass.

## Execution sequence

1. Run the existing frozen Qwen/LFM BF16 architecture panel: eight primary-test
   plus eighteen validation-frontier records per model. Keep the exact pins,
   seeds, native splits and target scopes. Two hours/model, four shared hours;
   prior Stage-E spending is zero at this continuation boundary.
2. Run the existing matched Gemma practical/calibration stage under its original
   four-hour cap; prior Stage-F spending is zero. This stage uses the original
   FP16 Gemma cohort and is not a BF16 cross-model speed comparison.
3. Generate the final artifact-based report, manuscript, bibliography and measured
   figures. Keep missing/budget-limited cohorts explicit. Review the resulting
   evidence before deciding on any larger claim or further study.

The new dispatcher applies no source patch, changes no scientific hyperparameter,
and imports no CUDA model itself. GPU stages run sequentially in subprocesses.
Unexpected failures stop for diagnosis; budget-limited independent cohorts do not
borrow budgets or cancel unrelated authorized work. Fresh C4/downstream/7B and
new-method studies remain outside this continuation. Exact 24 GPU-hour compliance
remains unproven because of the first reboot's unlogged tail.

## Observed Qwen runtime notes

Optional fast DeltaNet/convolution kernels are absent; the installed PyTorch
fallback is used without installing another backend. Tokenization warns that the
whole test corpus has 297,193 tokens, above the model's advertised context, but
the evaluator feeds only fixed 512-token windows. The untouched teacher completed
all 297,192 shifted test targets without an indexing error. Readiness copy and
identity probes match; the full-rank BF16 SVD probe fails the unchanged logit
tolerance, so the architecture evidence remains exploratory.

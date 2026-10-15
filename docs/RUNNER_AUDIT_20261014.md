# Runner audit, 14 October 2026

Read-only audit performed while composition remained active. No experimental
source or active configuration was changed; no audit GPU work or downloads ran.

## Confirmed behavior

- The GPT-2 adapter targets attention `attn.c_proj` only, transposes Conv1D's
  stored matrix into the canonical `[out,in]` orientation, and retains bias.
- Replication transfers the frozen Gemma settings at rank 256. Its divisor
  frontier remains on validation; divisor 3 shares fitted factors with the
  rank-256 test condition and still evaluates the separate validation split.
- The test protocol and calibration metrics include the final partial window,
  score every eligible shifted target once, and accumulate across windows.
- Temperature fitting executes one model forward per validation window. The
  33 temperatures cause 33 loss evaluations on those logits, rather than 33
  model forwards. Each frozen temperature file is written before test metrics.
- Whole-model restore creates FP16 factor buffers through the packed layer's
  constructor. INT8 codes and FP32 scale/minimum metadata retain their saved
  types. No reconstructed dense projection weight is registered.

A tiny random GPT-2 CPU check validates actual Conv1D orientation and bias,
then saves/restores both FP16 and INT8 packed models. Prompt logits and one
cached next-token forward match bitwise after restore, and tied token-embedding
and output-head storage remains tied. This is a software correctness check,
not research evidence about GPT-2 compression quality or GPU performance.

Reproduction: `rtk .venv-cuda/Scripts/python.exe -m docs.audit_gpt2_cpu_20261014`.
Every invocation creates a unique `results/paper-v2/cpu-adapter-audit-*`
directory and preserves prior artifacts.

## Reporting limitation requiring later work

`paper_reporting.assess_promotion` accepts a separately established KL advantage,
but `build_run_report` currently computes only NLL block-bootstrap uncertainty.
Consequently the automatic report cannot establish a KL-only global advantage,
even if mean KL improves for all three seeds. It conservatively leaves such a
case inconclusive. It does not silently promote candidates and does not change
any saved NLL, KL, drift, or hardware measurement.

No source repair is needed during the active run. A later protocol amendment
would need to predeclare a suitable KL uncertainty analysis and save paired
KL blocks before enabling that promotion path; a synthetic flag alone would
not establish scientific uncertainty. Current promotion is already disabled
because the strict FP16 readiness gate failed.

## Limits

The full practical stage has not completed. Its wall time must still be checked
against the unchanged four-hour cumulative cap. The audit does not establish
why Windows rebooted, and found no evidence that an out-of-memory condition
caused the numerical readiness failure.

Read local `docs/EXPERIMENT_PLAN.md`, `src/paper_followups.py`,
`src/paper_protocol.py`, `src/paper_reporting.py`, `src/paper_comparisons.py`,
`src/paper_study.py`, `src/compression_core.py` and existing follow-up tests.
Context7's official Transformers material confirms GPT-2 uses Conv1D and
documents cached generation: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/gpt2.md.
The installed Transformers 5.14.1 implementation was exercised by the CPU check.

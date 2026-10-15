# Research evidence index

Experiment artifacts are evidence, including failures. Keep JSON contracts,
resolved configs, metrics, manifests, source hashes and reports. Do not delete or
overwrite a run to make a successful-looking history.

## Current review snapshot

The bounded study's final review report is
[revised-report-20261014-151432-07c08d61](paper-v2/revised-report-20261014-151432-07c08d61/report.md),
with a [PDF review draft](paper-v2/revised-report-20261014-151432-07c08d61/manuscript.pdf).
It includes completed LFM and BF16 practical evidence, with Qwen explicitly incomplete.
The [active BF16-only ledger](paper-v2/revised-orchestration-20261014-5-bf16/status.json)
is the current execution authority: seeds 11/23 only, five practical conditions.
The old FP16/INT8 practical run is stopped and preserved; see the
[amendment](../docs/BF16_TWO_SEED_20261014.md). The following snapshots stay preserved.

| Evidence | Location |
|---|---|
| Report, manuscript, bibliography and measured figures | [revised-report-20261014-061526-3ee7c395](paper-v2/revised-report-20261014-061526-3ee7c395/report.md) |
| Validation selection | [regularization-20261013-171941-804f4c2e](paper-v2/regularization-20261013-171941-804f4c2e/report.md) |
| Full-stack baseline comparison | [baselines-20261013-175255-6a00d4e8](paper-v2/baselines-20261013-175255-6a00d4e8/report.md) |
| Completed 60-record composition | [composition-20261014-054423-dbd821ff](paper-v2/composition-20261014-054423-dbd821ff/report.md) |
| Independent Gemma C/D checkpoint, coverage, budget and reuse audit | [evidence-audit-20261014-095952-f9c97619](paper-v2/evidence-audit-20261014-095952-f9c97619/report.md) |
| Terminal Qwen partial-cohort audit | [qwen-evidence-audit-20261014-104445-4f5593f2](paper-v2/qwen-evidence-audit-20261014-104445-4f5593f2/report.md) |
| Completed LFM BF16 cohort audit | [lfm-evidence-audit-20261014-140916-9e1e9fcf](paper-v2/lfm-evidence-audit-20261014-140916-9e1e9fcf/report.md) |
| Seven completed BF16/FP8 controls; run stopped for review | [precision-readiness-20261014-060840-6db37daf](paper-v2/precision-readiness-20261014-060840-6db37daf/status.json) |
| Review pause and preserved process/accounting evidence | [review-pause-20261014-1](paper-v2/review-pause-20261014-1/pause.json) |
| Perplexity comparison illustration from saved measurements | [review-graphs-20261014-8c7e4533](paper-v2/review-graphs-20261014-8c7e4533/perplexity.png) |

`paper-v2/` contains unique dated run directories, incremental artifacts,
interrupted attempts, recovery ledgers and generated report versions. A later
report does not erase an earlier run. Read terminal `status.json` with metrics:
some intermediate per-run reports were written while work was running.

The files and directories outside `paper-v2/` are earlier exploratory evidence
with different protocols; do not pool their numbers into the revised tables.
Prior GPT-2/7B artifacts remain historical even though the current panel excludes
them. Recorded reuse is not independent replication.

Large checkpoints, model caches, datasets and environment files are excluded by
the existing `.gitignore`. Small research JSON and provenance are intentionally
eligible for Git. Do not blanket-ignore `results/` to hide legitimate evidence.

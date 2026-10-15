# Documentation index

Start with the root [project overview](../README.md) and
[remaining paper tasks](../TASKS.md). Architecture confirmation was resumed under
the [post-review continuation](CONTINUATION_20261014.md); earlier review snapshots
remain preserved and are not live status pages. Recovery ran under ledger
4 with saved rows reused and the reboot charged; the later user stop and
[BF16-only two-seed amendment](BF16_TWO_SEED_20261014.md) now supersede that
execution. Ledger 5's five BF16 practical conditions are complete; see
[remaining tasks](../TASKS.md).

## Current evidence and review

| Document | Purpose |
|---|---|
| [15 October revised paper](paper-revision-20261015/manuscript.md) / [PDF](paper-revision-20261015/manuscript.pdf) | Compact scientific narrative and tables; original frozen evidence preserved |
| [Editorial revision notes](paper-revision-20261015/README.md) | Changes, evidence checks, build instructions and export limitations |
| [Review handoff](REVIEW_20261014.md) | Completed measurements, code changes, unfinished work |
| [Final bounded-study evidence report](../results/paper-v2/revised-report-20261014-151432-07c08d61/report.md) | Saved continuation results; architecture cohort still partial |
| [Review manuscript PDF](../results/paper-v2/revised-report-20261014-151432-07c08d61/manuscript.pdf) | Artifact-generated review draft, with unavailable measurements explicit |
| [Architecture illustration](figures/gemma-1b-anchored-architecture-20261014.png) | Dense versus anchored low-rank Gemma attention output |
| [Diagnosis](DIAGNOSIS_2026-10-14.md) | Limits of historical claims and numerical failures |

## Protocol and implementation

| Document | Purpose |
|---|---|
| [Pending commit organization](COMMIT_PLAN_20261015.md) | File groups, cleanup decisions and archived-draft verification |
| [Original experiment plan](EXPERIMENT_PLAN.md) | Original bounded study and manuscript deliverables |
| [Current amendment](EXPERIMENT_PLAN_20261014_AMENDMENT.md) | Gemma/Qwen/LFM, GPT-2 exclusion, precision/frontier budgets |
| [Implementation guide](PAPER_V2_IMPLEMENTATION.md) | Additive runner and its limitations |
| [Architecture adapter audit](ARCHITECTURE_ADAPTER_AUDIT_20261014.md) | Exact Qwen/LFM targets and verification |
| [Runner audit](RUNNER_AUDIT_20261014.md) | Provenance, reload and execution checks |
| [Practical measurement audit](PRACTICAL_AUDIT_20261014.md) | Reload, timing, memory and precision-comparison scope |
| [Finalization guide](REVISED_STUDY_FINALIZATION_20261014.md) | Reports, manuscript and scientific plots |
| [LaTeX/PDF export](LATEX_EXPORT_20261014.md) | Standalone source, PDF rendering and proof checks |
| [Testing guide](TESTING.md) | Supported CPU verification commands |
| [Verified bibliography](paper_protocol_v2.bib) | Bibliography source for the revised paper |
| [Configuration index](../configs/README.md) | Current versus preserved configurations |
| [Results index](../results/README.md) | Immutable evidence and current run identifiers |

The dated Python scripts and patch files in this directory are audit/recovery
utilities. Their locations are preserved because recorded contracts and source
hashes can refer to them. Their presence is not permission to rerun an experiment.

## Historical material

[PAPER.md](archive/historical-drafts/PAPER.md),
[paper.tex](archive/historical-drafts/paper.tex), [MEMORY.md](MEMORY.md),
[LITERATURE.md](LITERATURE.md) and the root [THREAD.md](../THREAD.md) retain earlier
research history. Their metrics and success claims are not substitutes for the
current matched protocol. The [archive index](archive/README.md) identifies the
preserved drafts, former front page and Word draft. Historical drafts are grouped
under `archive/historical-drafts/`; their relocation manifest verifies unchanged bytes.

The graph in `graphify-out/` predates the new paper runner; use it as a historical
navigation aid, with current source and retained contracts taking precedence.

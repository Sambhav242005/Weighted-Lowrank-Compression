# Experiment configurations

The study has resumed under the [post-review continuation](../docs/CONTINUATION_20261014.md).
These files document experiments; do not launch a competing process merely
because a configuration is listed here. Every actual attempt needs a fresh
output directory and cumulative accounting of earlier work.

| Configuration | Role |
|---|---|
| `paper_bf16_practical_two_seeds_20261014.json` | Active BF16-only practical amendment, seeds 11/23 only, five conditions, remaining original practical budget |
| `paper_architecture_recovery_v4.json` | Closed third-reboot panel: Qwen budget-limited, LFM complete; do not restart |
| `paper_architecture_panel_v3.json` | Frozen original pinned Qwen3.5-0.8B/LFM2.5-1.2B BF16 panel; 8 test + 18 validation records/model |
| `paper_practical_after_panel_v3.json` | Stopped historical FP16/INT8 stage; do not resume |
| `paper_exploratory_resume_v2.json` | Retained Gemma continuation and input to the separate precision controls |
| `paper_exploratory_composition_resume_v2.json` | Preserved composition-recovery configuration; superseded by completed recovery output |
| `paper_exploratory_v2.json` | Earlier exploratory execution settings |
| `paper_identity_control_v2.json` | Earlier identity/readiness control settings |
| `paper_protocol_v2.json` | Original protocol configuration; not the current model-panel dispatcher |

Some preserved v2 configs contain GPT-2 settings. GPT-2 is excluded from the
current amendment; do not restart a historical `local-core` dispatcher that
would reach its old replication branch. The previous revised dispatcher already
applied a one-time accounting edit, so a review-approved continuation must inspect
its ledger before restarting. All selected assets are pinned; no model/data
substitutions or automatic downloads are implied.

# Source index

The current paper study uses the modules below. Start with the root
[overview](../README.md), [task list](../TASKS.md), and
[protocol guide](../docs/PAPER_V2_IMPLEMENTATION.md).

| Module | Responsibility |
|---|---|
| `paper_bf16_practical.py` | Currently authorized BF16-only practical/calibration tests, seeds 11/23 and five conditions |
| `compression_core.py` | Shared packed low-rank representation and checkpoint utilities |
| `paper_benchmark.py` | Offline preflight and experiment entry point |
| `paper_study.py` | Sequential stages and cumulative budget accounting |
| `paper_comparisons.py` | Matched fitting and evaluation comparisons |
| `paper_architecture_panel.py` | Pinned Qwen/LFM attention-output confirmation |
| `paper_followups.py` | Composition, replication and practical/calibration protocols |
| `paper_protocol.py` | Calibration positions, evaluation windows and metrics |
| `paper_math.py` | Factor-fitting mathematics |
| `paper_reporting.py` | Evidence summaries and reporting helpers |
| `paper_precision.py`, `paper_svd_diagnosis.py` | Retained numerical-control investigations |
| `prepare_paper_assets.py` | Explicit asset preparation |

The dated continuation, export and recovery helpers live in `docs/` at their
recorded paths. See the [documentation index](../docs/README.md). Existing run
contracts and source hashes depend on stable locations.

Other modules preserve earlier experiments. They are not additional steps in
the active study and should not be run merely because they appear here. Keep
active study code and configurations frozen while GPU work is running.

The separately authorized Gemma 3 270M study uses `gemma270m_relu_study.py` and
the additive `gemma270m_resume.py` recovery wrapper. Its completed FP32 CPU/GPU
results and limits are in [the dated report](../docs/GEMMA270M_RESULTS_20261016.md).
These entry points do not extend or restart the earlier paper cohorts.

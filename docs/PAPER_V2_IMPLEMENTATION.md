# Protocol-v2 implementation and execution state

Updated 2026-10-14: the user authorized execution on this PC and downloads of the plan's assets.

Current scope amendment: [the three-model architecture panel and precision plan](EXPERIMENT_PLAN_20261014_AMENDMENT.md)
replaces GPT-2 with pinned Qwen3.5-0.8B and LFM2.5-1.2B-Instruct. Gemma remains
the existing reference; at most four models are permitted, and three are selected.
BF16/FP8 readiness and RAM offload are authorized. Historical GPT-2 descriptions
below record the original protocol, not authorization to start it now.

The two requested local configuration changes were discarded. Existing research
results and legacy experiment scripts remain preserved.

## Implemented and verified

- Separate spectral weighting, exact anchoring, legacy extra-ridge, and fixed-rank ASVD primitives in `src/paper_math.py`.
- Spectral pseudoinverse convention: retain eigenvalues above `1e-7` times the largest eigenvalue; null-space directions are zero.
- Offline preflight and Stage-A identity controls in `src/paper_benchmark.py`.
- Native split token hashing, deterministic calibration positions, sliding 512/256 windows, exactly-once shifted targets, and final partial-window inclusion.
- Total NLL accumulation and contiguous 4,096-target sums for later paired bootstrap analysis.
- Packed execution follows input dtype; FP16 factors stay FP16 until an explicit input-dtype conversion is needed.
- The historical solver is preserved. Additional ridge is labelled shrinkage rather than silently presented as exact anchoring.

The new math accepts teacher-exact outputs implicitly as `W @ X`; noisy-output fitting is outside this implementation.
ASVD and whitening are component implementations, not reproductions of complete published pipelines.

Readiness requires one-window logits within the declared FP16 tolerance (`atol=0.1`, `rtol=0.01`),
full-test NLL agreement within 0.001, complete target counts, and an unchanged teacher hash.
These are readiness checks, not compression promotion criteria.

## Exact assets

| Asset | Revision | State |
|---|---|---|
| google/gemma-3-1b-it | dcc83ea841ab6100d6b47a070329e1ba4cf78752 | Prepared after authorized authentication |
| openai-community/gpt2 | 607a30d783dfa663caf39e06633721c8d4cfcd7e | Prepared locally |
| Salesforce/wikitext, wikitext-2-raw-v1 | b08601e04326c79dfdd32d625aee71d232d685c3 | Prepared locally; train/validation/test |

The GPT-2 revision was resolved and pinned during authorized preparation. No old
GPT-2 revision was recoverable from the recorded default cache.

## Historical preparation snapshot — superseded by execution below

- State at initial preparation: waiting for a human; authentication was subsequently resolved.
- Hypothesis: protocol-v2 can distinguish weighting, anchoring, shrinkage, and accumulated representation drift.
- Configuration and matched baseline: `configs/paper_protocol_v2.json`; three seeds; planned Gemma rank 384; strong baselines remain pending.
- Observed evidence: RTX 4070 Ti with 12 GB VRAM and CUDA available; 25 repository tests passed. Gemma preparation failed with gated-repository 401; a subsequent check reported no saved Hugging Face token.
- Decision: do not fit or substitute a different model. No compression hypothesis has been evaluated under protocol v2.
- Next smallest experiment: prepare the exact Gemma snapshot after authenticating; execute Stage A.
- Historical limitations: identity controls and Stages B–F were pending at this snapshot. All are now implemented, including `--stage local-core`; empirical completion is reported below.

## Historical preparation commands (use the existing CUDA venv for execution)

Authenticate in a human terminal using an account with access to Gemma; do not put tokens in chat:

```powershell
rtk hf auth login
rtk python -u -m src.prepare_paper_assets --asset gemma
rtk python -m src.paper_benchmark preflight --config configs/paper_protocol_v2.json
rtk python -u -m src.paper_benchmark run --config configs/paper_protocol_v2.json --stage readiness
rtk python -m src.paper_benchmark report --study local-core
```

Readiness has a one-hour elapsed-time cap. Later stages must enforce the remaining
4/7/4/4/4 hour caps and preserve incomplete comparisons; the full 24-hour study
was not started at this historical snapshot. It is now active; cumulative charges apply. Preflight evidence is retained under `results/paper-v2`.

API behavior was checked against official Hugging Face Transformers, Hub, and
Datasets documentation through Context7. Research package versions were preserved; test-package installation is recorded below.

## Authentication resolved and readiness resumed

Gemma access was verified after the user's login. The exact pinned snapshot was
downloaded successfully; preflight `preflight-20261013-163923-3ee24162` passed.

The first readiness attempt (`readiness-20261013-163926-aa1c82c9`) stopped during
initialization without metrics. A separate import probe found the system Python
stack incompatible: Transformers 4.53.3 requires Hub below 1.0, while Hub 2.2.0
is now installed. No package downgrade or upgrade was performed by this agent.

The existing `.venv-cuda` was checked instead: Python 3.14.6, Torch
2.13.0+cu132, Transformers 5.14.1, Datasets 5.0.1, and Hub 1.27.0.
Model/data imports and CUDA mathematical/FP16 sanity checks passed. Pytest 9.0.2
and its test dependencies were subsequently installed in this existing venv.
Current verification is 69 tests passed, plus 74 subtests; the earlier 25-test
result belongs to the original system environment.

Fresh readiness run: `readiness-20261013-164122-cdc919e9`. Its contract records
the actual environment. The teacher loaded, its identity control matched
exactly, and full-test evaluation began. Consult this run's `status.json`,
`stdout.log`, and incremental `metrics.json` for subsequent progress.
Stages B–F and the local-core dispatcher are now implemented. Implementation
does not imply empirical completion; see the execution snapshot below.

## Completed identity diagnosis

Run `readiness-20261013-164122-cdc919e9` failed the declared logit tolerance
for FP16 SVD factors: maximum absolute difference 0.15234375. Dense teacher
and copied teacher matched on all 292,281 test targets (NLL
3.451562542535798; PPL 31.549651486382494). The failed run, configuration,
partial metrics, and traceback were retained.

A separate config `configs/paper_identity_control_v2.json` changed only the
full-rank factorization basis to weight times identity (or identity times weight
for wide matrices), preserving precision, factor count, model, splits, and
tolerance. Run `readiness-20261013-164407-bda5b4d9` completed in 173.10 seconds.
Teacher, copy, and full-rank identity factors had identical NLL/PPL on all test
targets, zero maximum logit difference in the identity probe, and the final
teacher hash was unchanged.

This establishes correctness of this packed identity execution path, not
equivalence of FP16 SVD factors or successful compression. The SVD-specific
numerical discrepancy remains an open readiness issue for the study's real
factorization methods; its tolerance has not been weakened. Downstream
comparisons therefore remain explicitly exploratory.

## Completed SVD diagnosis and exploratory scope

Diagnosis `svd-diagnosis-20261013-165146-c3671715` completed in 686.97 seconds.
FP32 reconstruction and FP32 factor execution pass the combined probe gate;
balanced and unbalanced FP16 SVD factors fail, although all NLL differences
are below 0.001. This isolates a numerical floor, not compression success.
**Strict FP16 readiness remains false, and promotion is disabled.** Conservative
readiness charge is 998.40 seconds. No tolerance, asset, or budget was relaxed.

## Completed Stage B: validation selection

Run `regularization-20261013-171941-804f4c2e` completed in 1,589.15 seconds.
Eleven controls score all 257,289 validation targets, seed 11. All 26 Gemma
attention output projections have shape 1152×1024; matched rank 384 uses
835,584 factor parameters per target, stored in FP16.

Validation selected exact anchoring beta 0.03, additional ridge zero, as both
anchor and solver (NLL 3.59662239; PPL 36.47482843). Whitening beta zero is
the frozen strongest baseline (NLL 3.60078899; PPL 36.62712134). Plain SVD PPL
is 171.54760067; ASVD PPL is 41.93306556. Full-rank ridge PPL 33.090111 is
an unmatched shrinkage control, not a compressed baseline. Recorded legacy
beta 0.1/ridge 0.01 gives PPL 36.84904531. Frozen selection, split hashes,
calibration hashes/positions, and factors remain in the run artifacts.

The algebraic legacy comparison fits recorded teacher outputs with bias removed
in FP64; it is not a bitwise reproduction of the historical FP32 solver.
ASVD and whitening are component comparisons, not full published pipelines.

## Completed Stage C: native-test comparisons

Run `baselines-20261013-175255-6a00d4e8` completed in 3,736.55 seconds with
13 records. Every record scores all 292,281 test targets. Four data-aware
methods each use seeds 11/23/37; deterministic plain SVD is evaluated once.
Each factor archive is 43,465,927 bytes: target-factor storage, not whole-model
checkpoint size. Test data did not select beta.

| Method | Test PPL, mean ± sample SD | Teacher KL, mean | Primary drift, mean |
|---|---:|---:|---:|
| SVD (one deterministic record) | 162.17583 | 1.558672 | 0.074248 |
| ASVD component | 40.22417 ± 0.09736 | 0.374854 | 0.030972 |
| Whitening component | 35.61833 ± 0.42185 | 0.138217 | 0.010638 |
| Exact anchor beta 0.03 | 35.37310 ± 0.33963 | 0.136755 | 0.010547 |
| Recorded legacy beta 0.1/ridge 0.01 | 35.83978 ± 0.30517 | 0.157781 | 0.011836 |

Anchoring lowers NLL against whitening in all three seeds, but improves primary
drift only in seed 23; seeds 11 and 37 regress slightly. Every compressed
method has worse PPL than the dense teacher (31.54965). These observations do
not establish reproducible drift improvement or practical compression success.
Uncertainty intervals and measured costs are required for stronger conclusions.

Actual depth curves and source-hash provenance are in
`docs/figures/20261014-032234-41dc5749/`. Shading is sample seed standard
deviation, not confidence intervals. The generated PNG was visually checked.

## Experiment status — composition-20261014-032120-486d799f

- State: running (observed 2026-10-14; consult live status artifacts).
- Hypothesis: group size and location distinguish local fit from downstream composition damage.
- Configuration and matched baseline: frozen Stage-B anchor versus whitening, rank 384, seeds 11/23/37; full native test and groups of 1/2/4/all projections.
- Observed evidence: earlier `composition-20261013-185512-2bffd477` was interrupted by an unexpected reboot (Windows events 41/6008), with 34 completed records and no remaining process. Its estimated charge is 8,972.373 seconds (last log plus 60 seconds); continuation allocates 5,427.627 seconds under the original four-hour cap. The estimate is not a proven upper bound on pre-reboot elapsed time; `accounting-caveat.json` records this uncertainty.
- Decision: continue the authorized bounded stage and preserve both directories. Proven identical full-stack C reuse is reuse, not independent replication.
- Next smallest experiment: complete remaining composition comparisons, then follow the frozen first-wave dispatcher.
- Risks or missing evidence: composition remains incomplete at this snapshot; GPT-2 and practical measurements remain pending. Strict FP16 readiness is false; no candidate is promoted.

Stages A–F, provenance checks, cumulative caps, paired bootstrap reporting,
calibration, reload parity, and practical measurement are implemented. Failures
and interrupted time count against the original 1/4/7/4/4/4-hour stage limits;
time limits do not authorize fewer windows/seeds or a fresh budget. Failed and
cancelled runs and controlled boundary handoffs remain under `results/paper-v2`.

The CPU interim analysis `interim-analysis-d2080d7f/report.json` reports anchored
minus whitening NLL differences and paired-block intervals: seed 11
−0.00311925 [−0.00559843, −0.000577106], seed 23 −0.00867017
[−0.0123616, −0.00487939], seed 37 −0.00888870 [−0.0115643, −0.00619303].
Each uses 2,000 paired resamples of 71 complete 4,096-target blocks. The 1,465
tail targets are excluded from intervals but included in full-test NLL.
These intervals describe evaluation-block variation, not calibration-seed
variation, and do not override readiness, drift, or pending hardware gates.

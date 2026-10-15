# Revised study evidence report

Mode: terminal orchestration report. Orchestration: cancelled.

No established practical or universal compression advantage; retained within-model findings remain exploratory.

Promotion disabled. Research hypothesis: inconclusive.

| Attempt | Terminal/observed state | Model | Precision |
|---|---|---|---|
| architecture-asset-preflight-de585fb3 | completed | unrecorded | unrecorded |
| asset-preparation-20261013-163641-aa339227 | completed | unrecorded | unrecorded |
| baselines-20261013-175255-6a00d4e8 | completed | google/gemma-3-1b-it | float16 |
| composition-20261013-185512-2bffd477 | interrupted | google/gemma-3-1b-it | float16 |
| composition-20261014-032120-486d799f | interrupted | google/gemma-3-1b-it | float16 |
| composition-20261014-054423-dbd821ff | completed | google/gemma-3-1b-it | float16 |
| precision-readiness-20261014-060840-6db37daf | cancelled | google/gemma-3-1b-it | float16 |
| preflight-20261013-162201-5deb6cfb | waiting for a human | unrecorded | unrecorded |
| preflight-20261013-162440-e537056e | waiting for a human | unrecorded | unrecorded |
| preflight-20261013-163923-3ee24162 | completed | unrecorded | unrecorded |
| preflight-20261013-170927-db2a9f91 | completed | unrecorded | unrecorded |
| preflight-20261013-175153-4f9eda7f | completed | unrecorded | unrecorded |
| readiness-20261013-163926-aa1c82c9 | failed | unrecorded | unrecorded |
| readiness-20261013-164122-cdc919e9 | failed | google/gemma-3-1b-it | float16 |
| readiness-20261013-164407-bda5b4d9 | completed | google/gemma-3-1b-it | float16 |
| reboot-recovery-20261014-2 | completed | unrecorded | unrecorded |
| regularization-20261013-171941-804f4c2e | completed | google/gemma-3-1b-it | float16 |
| revised-orchestration-20261014-1 | interrupted | unrecorded | unrecorded |
| revised-orchestration-20261014-2 | cancelled | unrecorded | unrecorded |
| study-20261013-171941-f6a7f400 | cancelled | google/gemma-3-1b-it | float16 |
| study-20261013-175254-ba317f9c | interrupted | google/gemma-3-1b-it | float16 |
| study-20261014-032120-72463b92 | interrupted | google/gemma-3-1b-it | float16 |
| svd-diagnosis-20261013-165146-c3671715 | completed | google/gemma-3-1b-it | float16 |

## Accounted costs

| Stage | Recorded wall seconds | Cap seconds | Unknown-cost attempts |
|---|---:|---:|---|
| readiness | 1381.392 | 3600 | none |
| regularization | 1589.148 | 14400 | none |
| baselines | 3736.553 | 25200 | none |
| composition | 12825.574 | 14400 | none |
| architecture | 0.000 | 14400 | none |
| practical | 0.000 | 14400 | none |

Elapsed-wall-time charges include failed/retried work and may include reboot downtime. The first reboot has an unbounded unlogged tail; no exact GPU-hour compliance claim.

## Interpretation limits

- Original FP16 SVD readiness failed; BF16/FP8 controls are separate cohorts.
- FP8 storage is decoded for BF16 multiplication; a tiny native GEMM probe is not model acceleration.
- Architecture panel targets only six full-attention outputs per new model; hybrid mixers remain dense.
- Raw PPL cannot rank tokenizers/models. Compare within-model differences only.
- No pooling across precision, placements, reboot attempts, or resumed/reused cohorts.
- Downstream tasks and fresh sampled-C4 confirmation remain unmeasured unless separately evidenced.

Full retained contracts, source/asset provenance, per-seed summaries, paired block uncertainty, integrity checks and read failures are in report.json. Each run remains a separate cohort.
Report-generation errors do not change experiment terminal states or erase successful measurements.

## Within-model, scope-specific comparisons

Each delta is candidate minus baseline. Group outcomes remain separate; a full-stack NLL gain does not imply improved block drift or improvement at every layer group.

| Run | Candidate | Scope | Seed | NLL delta | Drift delta | Scoped hypothesis |
|---|---|---|---:|---:|---:|---|
| baselines-20261013-175255-6a00d4e8 | {"method": "svd"} | {} | 11 | +1.52592704 | +0.0639228729 | inconclusive |
| baselines-20261013-175255-6a00d4e8 | {"method": "asvd"} | {} | 11 | +0.13435745 | +0.0208573326 | weakened |
| baselines-20261013-175255-6a00d4e8 | {"method": "asvd"} | {} | 23 | +0.124064776 | +0.020418124 | weakened |
| baselines-20261013-175255-6a00d4e8 | {"method": "asvd"} | {} | 37 | +0.106534451 | +0.0197272296 | weakened |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {} | 11 | -0.00311925166 | +3.02265165e-06 | inconclusive |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {} | 23 | -0.00867016564 | -0.000321961736 | inconclusive |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {} | 37 | -0.0088886965 | +4.73925457e-05 | inconclusive |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.01} | {} | 11 | +0.0073417433 | +0.00135895433 | weakened |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.01} | {} | 23 | +0.0100857299 | +0.00134531503 | weakened |
| baselines-20261013-175255-6a00d4e8 | {"beta": 0.1, "method": "legacy_ridge", "ridge_fraction": 0.01} | {} | 37 | +0.00123384365 | +0.000891093688 | weakened |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 11 | -4.60899341e-05 | -6.43828241e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 23 | -3.27137553e-05 | +1.55730395e-05 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 37 | -6.64797922e-05 | -1.93130612e-05 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 11 | -0.0008545763 | +1.4748007e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 23 | -9.84808365e-05 | +1.8589235e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 37 | -7.62656914e-05 | +1.18334494e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 11 | +0.000172930499 | +3.12111105e-07 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 23 | -0.000707979189 | +3.33152092e-07 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 37 | -7.24395315e-05 | +3.29271524e-07 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 11 | -0.000276457771 | -8.40125732e-05 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 23 | +0.000274831323 | -0.000205128507 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 37 | -0.000141676135 | +2.26311751e-05 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 11 | -0.000598313739 | +3.0037094e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 23 | +0.000449291597 | +2.81076527e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 37 | +0.000166412477 | +3.02233395e-06 | inconclusive |
| composition-20261013-185512-2bffd477 | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 11 | +8.0425554e-05 | -1.1740639e-07 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 11 | -4.60899341e-05 | -6.43828241e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 23 | -3.27137553e-05 | +1.55730395e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 37 | -6.64797922e-05 | -1.93130612e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 11 | -0.0008545763 | +1.4748007e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 23 | -9.84808365e-05 | +1.8589235e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 37 | -7.62656914e-05 | +1.18334494e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 11 | +0.000172930499 | +3.12111105e-07 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 23 | -0.000707979189 | +3.33152092e-07 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 37 | -7.24395315e-05 | +3.29271524e-07 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 11 | -0.000276457771 | -8.40125732e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 23 | +0.000274831323 | -0.000205128507 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 37 | -0.000141676135 | +2.26311751e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 11 | -0.000598313739 | +3.0037094e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 23 | +0.000449291597 | +2.81076527e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 37 | +0.000166412477 | +3.02233395e-06 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 11 | +8.0425554e-05 | -1.1740639e-07 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 23 | -0.000811169275 | +1.7246151e-08 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 37 | +0.000180791641 | +1.14760223e-08 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 11 | -0.000657116931 | -0.000190415906 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 23 | +7.18358325e-05 | -0.000187657446 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 37 | -0.000297241087 | -2.35291143e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | 11 | +0.00388530813 | +6.89941006e-05 | inconclusive |
| composition-20261014-032120-486d799f | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | 23 | +0.00358999649 | +6.64840569e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 11 | -4.60899341e-05 | -6.43828241e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 23 | -3.27137553e-05 | +1.55730395e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0], "rank": 384, "split": "test"} | 37 | -6.64797922e-05 | -1.93130612e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 11 | -0.0008545763 | +1.4748007e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 23 | -9.84808365e-05 | +1.8589235e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12], "rank": 384, "split": "test"} | 37 | -7.62656914e-05 | +1.18334494e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 11 | +0.000172930499 | +3.12111105e-07 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 23 | -0.000707979189 | +3.33152092e-07 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [25], "rank": 384, "split": "test"} | 37 | -7.24395315e-05 | +3.29271524e-07 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 11 | -0.000276457771 | -8.40125732e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 23 | +0.000274831323 | -0.000205128507 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1], "rank": 384, "split": "test"} | 37 | -0.000141676135 | +2.26311751e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 11 | -0.000598313739 | +3.0037094e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 23 | +0.000449291597 | +2.81076527e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [12, 13], "rank": 384, "split": "test"} | 37 | +0.000166412477 | +3.02233395e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 11 | +8.0425554e-05 | -1.1740639e-07 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 23 | -0.000811169275 | +1.7246151e-08 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [24, 25], "rank": 384, "split": "test"} | 37 | +0.000180791641 | +1.14760223e-08 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 11 | -0.000657116931 | -0.000190415906 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 23 | +7.18358325e-05 | -0.000187657446 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3], "rank": 384, "split": "test"} | 37 | -0.000297241087 | -2.35291143e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | 11 | +0.00388530813 | +6.89941006e-05 | weakened |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | 23 | +0.00358999649 | +6.64840569e-05 | weakened |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [11, 12, 13, 14], "rank": 384, "split": "test"} | 37 | +0.000695857577 | +7.01886799e-05 | weakened |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [22, 23, 24, 25], "rank": 384, "split": "test"} | 11 | -0.00042231288 | -1.086791e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [22, 23, 24, 25], "rank": 384, "split": "test"} | 23 | -0.00290206343 | -1.14651198e-05 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [22, 23, 24, 25], "rank": 384, "split": "test"} | 37 | +0.000342722158 | -9.25122474e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "rank": 384, "split": "test"} | 11 | -0.00311925166 | +3.02265165e-06 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "rank": 384, "split": "test"} | 23 | -0.00867016564 | -0.000321961736 | inconclusive |
| composition-20261014-054423-dbd821ff | {"beta": 0.03, "method": "anchored", "ridge_fraction": 0.0} | {"group": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], "rank": 384, "split": "test"} | 37 | -0.0088886965 | +4.73925457e-05 | inconclusive |

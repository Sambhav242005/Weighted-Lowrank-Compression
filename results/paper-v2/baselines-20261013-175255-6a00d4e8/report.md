# baselines

Completed 13 comparisons on test. Hypothesis status: inconclusive pending paired uncertainty, three-seed global comparisons and hardware gates. Component baselines are not complete ASVD/SVD-LLM pipelines.

| Method | Seed | NLL | PPL | KL | Primary drift |
|---|---:|---:|---:|---:|---:|
| seed-11-00-svd | 11 | 5.088681 | 162.1758 | 1.558672 | 0.07424758312303882 |
| seed-11-01-asvd | 11 | 3.697112 | 40.3306 | 0.376601 | 0.03118204278984501 |
| seed-11-02-whitening | 11 | 3.562754 | 35.2602 | 0.132628 | 0.010324710225531702 |
| seed-11-03-anchored | 11 | 3.559635 | 35.1504 | 0.132312 | 0.010327732877186257 |
| seed-11-04-legacy_ridge | 11 | 3.570096 | 35.5200 | 0.152995 | 0.01168366455975469 |
| seed-23-01-asvd | 23 | 3.693921 | 40.2022 | 0.377352 | 0.030942521331881596 |
| seed-23-02-whitening | 23 | 3.569857 | 35.5115 | 0.135988 | 0.010524397309095334 |
| seed-23-03-anchored | 23 | 3.561186 | 35.2049 | 0.133343 | 0.010202435573502321 |
| seed-23-04-legacy_ridge | 23 | 3.579942 | 35.8715 | 0.155793 | 0.011869712339292703 |
| seed-37-01-asvd | 37 | 3.692365 | 40.1397 | 0.370609 | 0.030790671865169903 |
| seed-37-02-whitening | 37 | 3.585831 | 36.0833 | 0.146035 | 0.011063442235805195 |
| seed-37-03-anchored | 37 | 3.576942 | 35.7640 | 0.144609 | 0.011110834781549542 |
| seed-37-04-legacy_ridge | 37 | 3.587065 | 36.1279 | 0.164555 | 0.011954535923705998 |

No promotion claim: hardware limits and second architecture remain separate gates.

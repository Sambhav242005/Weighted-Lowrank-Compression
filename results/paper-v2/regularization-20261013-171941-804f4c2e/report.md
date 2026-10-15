# regularization

Completed 11 comparisons on validation. Hypothesis status: inconclusive pending paired uncertainty, three-seed global comparisons and hardware gates. Component baselines are not complete ASVD/SVD-LLM pipelines.

| Method | Seed | NLL | PPL | KL | Primary drift |
|---|---:|---:|---:|---:|---:|
| seed-11-00-svd | 11 | 5.144861 | 171.5476 | 1.545474 | not measured |
| seed-11-01-fullrank_ridge | 11 | 3.499234 | 33.0901 | 0.009088 | not measured |
| seed-11-02-anchored | 11 | 3.600789 | 36.6271 | 0.124118 | not measured |
| seed-11-03-anchored | 11 | 3.596622 | 36.4748 | 0.123975 | not measured |
| seed-11-04-anchored | 11 | 3.596752 | 36.4795 | 0.135886 | not measured |
| seed-11-05-anchored | 11 | 3.607465 | 36.8725 | 0.159728 | not measured |
| seed-11-06-anchored | 11 | 3.615799 | 37.1810 | 0.195374 | not measured |
| seed-11-07-legacy_ridge | 11 | 3.596854 | 36.4833 | 0.135920 | not measured |
| seed-11-08-legacy_ridge | 11 | 3.597278 | 36.4988 | 0.136581 | not measured |
| seed-11-09-legacy_ridge | 11 | 3.606830 | 36.8490 | 0.143638 | not measured |
| seed-11-10-asvd | 11 | 3.736075 | 41.9331 | 0.362992 | not measured |

Full-rank shrinkage validation NLL change versus teacher: +0.001855. Selected solver change: +0.099243. A matching sign alone does not establish mechanistic equivalence.

No promotion claim: hardware limits and second architecture remain separate gates.

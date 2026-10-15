# BF16 practical evidence audit

State: passed

Five BF16 conditions, seeds 11/23 only. Full native targets, frozen calibration, checkpoint dtypes/hashes, reload probes and timing workloads checked.

Cumulative practical seconds: 11524.515054043139 / 14400.

- CPU checkpoint inspection performs no model forward or new seed test.
- Exact reload covers the recorded prompt/cached-token probes, not arbitrary generation.
- Practical NLL has aggregate calibration metrics, not paired block loss intervals.
- Two seeds do not meet the historical three-seed promotion criterion.

Errors: []

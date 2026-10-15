# Practical measurement audit — 14 October 2026

Read-only preflight of `src/paper_followups.py`, `src/paper_study.py`,
`src/paper_comparisons.py`, `src/compression_core.py`, the existing follow-up
tests, and `configs/paper_practical_after_panel_v3.json`. No GPU experiment,
download, configuration change, or research-source edit was performed.

## Declared comparison

Stage F measures the pinned Gemma FP16 teacher, then the frozen Stage C
whitening and anchored checkpoints for seeds 11, 23, and 37, each stored as
FP16 and row-quantized INT8 factors: 13 complete records in total. The frozen
selection file supplies the method settings; stale top-level beta/ridge values
do not replace the selected specifications. GPT-2 is not used by this stage.

The source checkpoints are checked against the fresh teacher digest, all native
split token hashes, calibration positions and token hashes, selection hash,
target names/shapes/ranks, factor convention, dtype, tensor shapes, finiteness,
and saved checkpoint hashes before use.

## What the measurements establish

- Temperature selection uses validation only, over the declared 33-point grid.
  Its chosen temperature and boundary-hit flag are saved before test scoring.
  Raw and temperature-scaled test metrics count every shifted target once,
  including the partial final window; ECE uses 15 fixed bins.
- Factor files are reloaded and required to reproduce an all-ones local probe
  exactly. Whole-model files include all remaining model weights and explicit
  packed recipes; shared tied tensor storage is preserved when saving. File
  sizes are actual serialized bytes, with the same serializer used for the
  dense teacher and compressed models.
- A compressed whole-model checkpoint is reconstructed from its saved config
  without downloading weights. Exact reload parity covers a 128-token prompt's
  logits and one cached next-token forward. It does **not** establish exactness
  over the entire test set or arbitrary generation. The dense teacher file is
  measured but does not undergo this whole-model reload-probe check.
- Timing uses batch size 1, prompt lengths 128 and 512, CUDA synchronization,
  10 warmups and 30 measured samples per condition. Prefill is timed separately
  from 128 greedy cached decode steps; decode excludes the initial prompt pass.
  Median and p95 are recorded. Peak allocated/reserved memory includes the
  resident model and cache; these are CUDA allocator statistics, not total
  system RAM or whole-device driver memory.
- INT8 factors are unpacked/dequantized into floating-point factors during
  forward execution. These measurements cannot establish native INT8 kernel
  acceleration. No FP8 control is part of this stage.

## Reporting constraints

The existing `hardware-gates.json` compares both solver storage formats against
the same seed's **FP16 whitening** baseline, using median latency and peak
allocated memory with a 5% threshold. This reference must be named explicitly.
It is not a same-bit INT8 comparison, and its gate does not test p95 latency,
reserved memory, serialized size, or quality. Same-bit comparisons can be
derived from the already measured whitening/solver rows without rerunning work.
Dense-relative cost comparisons must likewise name the dense reference.

The four-hour Stage F cap is cooperative: progress callbacks check elapsed time
before each calibration window and timing iteration. Loading, serialization,
hashing, checkpoint validation, one window's 33-temperature calculation, and
one decode iteration may run between checks. Record actual terminal elapsed
time rather than claiming a hard real-time cutoff. The study controller deducts
previous retained practical attempts before launching a continuation.

Only complete, persisted records support paired comparisons. A cap can interrupt
a cohort before its metric row is saved; retain its partial files and state the
missing rows. Do not borrow another stage's allowance to complete the grid.
Readiness remains failed and promotion disabled regardless of cost outcomes.

## Disposition

No definite crash requiring a source repair was found in this audit. Existing
tests cover partial-window target accounting, validation-only temperature fit,
freeze-before-test ordering, exact FP16/INT8 factor reload, whole-model packed
reconstruction with a synthetic factory, and source-checkpoint provenance
rejection. They do not replace the actual Gemma reload and CUDA measurements.
The measurement scopes above must accompany the final paper's hardware claims.

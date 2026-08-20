# Testing Guide

## Purpose

Tests must establish that a compression result is both technically correct and scientifically credible. A small reconstruction error, a passing single-layer experiment, or a visually plausible generation is not sufficient evidence that a compressed transformer preserves global behavior.

This guide covers four layers of verification:

1. deterministic unit tests for mathematical and metric correctness;
2. offline integration tests for extraction, replacement, and experiment orchestration;
3. controlled GPU/model tests for real inference behavior; and
4. scientific regression tests for reproducibility and matched-baseline comparisons.

## Current status

The repository currently has no dedicated `tests/` directory or test configuration. Files named `quick_test.py`, `gpu_test.py`, and `causality_test.py` are executable research scripts, not isolated tests. Pytest attempts to collect them and fails because their imports assume execution from `src`.

Until the test suite is introduced, do not use a bare `pytest` run as evidence that the project is healthy. Keep experimental scripts out of test discovery and avoid model downloads or GPU execution at import time.

## Required test layout

```text
tests/
  unit/
  integration/
  gpu/
  regression/
```

- `unit/` contains fast, deterministic tests with synthetic tensors only.
- `integration/` uses a tiny locally constructed transformer and never downloads a model or dataset.
- `gpu/` contains opt-in real-model checks and is marked `gpu`.
- `regression/` validates result schemas and compares reproducible benchmark summaries against approved baselines.

Research scripts belong outside `tests/` and must use explicit CLI entry points.

## Unit tests

Every representation family must test:

- output tensor shape and dtype;
- parameter-count and compression-ratio calculations;
- deterministic output when the seed and configuration are fixed;
- exact reconstruction of a known rank-`r` synthetic matrix when mathematically expected;
- finite loss/parameters for valid inputs;
- validation errors for unsupported rank, invalid budget, NaN/Inf input, and incompatible tensor shape; and
- serialized representation size rather than only nominal factor count.

Evaluation tests must verify:

- identical teacher and student logits yield zero KL divergence and full top-1 agreement;
- perplexity counts only valid shifted, non-padding target tokens;
- hidden-state drift metrics are zero for identical tensors and have defined behavior for zero norms;
- the evaluator does not mutate the teacher model;
- every metric record identifies its model, dataset split, seed, target group, budget, and method; and
- a comparison is rejected when the candidate and baseline do not share the same experimental contract.

Model-adapter tests must verify, using a tiny local model, that extraction and replacement preserve matrix orientation for each supported architecture and target type. They must also prove that substitution changes only the copied student model, never the original teacher.

## Integration tests

Build a tiny transformer from configuration in the test itself. Do not call `from_pretrained` or `load_dataset`.

The minimum end-to-end integration test should:

1. create deterministic synthetic token batches and a tiny teacher model;
2. select one projection and one two-layer group;
3. fit an independent and a joint low-rank replacement at a matched budget;
4. run evaluation and drift instrumentation;
5. write results to a temporary directory; and
6. verify the result schema, run metadata, and no mutation of the teacher.

The test asserts structural behavior rather than a research-quality PPL target: the run completes, metrics are finite, outputs have the expected keys, and matched-budget comparisons are enforced.

## GPU and network tests

Real GPT-2/Gemma and WikiText checks are opt-in because they are slow, hardware-dependent, and may require cached assets or network access.

- Mark GPU tests with `@pytest.mark.gpu`.
- Mark model/dataset-download tests with `@pytest.mark.network`.
- Skip them with a clear reason when CUDA, required memory, model cache, or dataset cache is unavailable.
- Pin the model and dataset revisions. Log CUDA version, PyTorch version, GPU name, dtype, and warm-up policy.

GPU tests must use a fixed number of warm-up and measurement iterations before asserting latency or peak-memory behavior. They should never make performance claims from one cold run.

## Scientific regression tests

Scientific regression checks guard the research claim rather than numerical equality alone.

For each approved benchmark configuration, store a compact immutable reference record containing:

- complete run configuration and package/device metadata;
- dense baseline metrics;
- matched independent-baseline metrics;
- candidate metrics for PPL/NLL, logit KL, drift profile, serialized bytes, peak memory, and warmed-up latency; and
- seed-level summaries.

A new candidate is comparable only if its run contract matches the reference. Flag—not silently overwrite—a result when it changes the model revision, tokenizer, data split, parameter budget, metric definition, or measurement hardware.

## Commands after the test suite is implemented

```powershell
# Fast, offline checks for every code change
py -3 -m pytest -q -m "not gpu and not network"

# Local integration tests only
py -3 -m pytest -q tests/integration

# Opt-in GPU/model verification on a prepared machine
py -3 -m pytest -q -m gpu

# Full scientific regression on cached models/data
py -3 -m pytest -q tests/regression -m "gpu and network"
```

The default command must finish without GPU, internet access, a Hugging Face cache, or writes outside its temporary test directory.

## Test gates for an experiment

Before trusting or promoting an experiment result, confirm:

1. unit and offline integration tests pass;
2. the experiment used a recorded, matched baseline;
3. calibration, validation, and final test data are disjoint;
4. all three seeds completed or failures are reported explicitly;
5. the result contains all required behavior, drift, compression, and hardware metrics; and
6. the conclusion matches the measured evidence and does not generalize beyond the tested layer family or architecture.

## Failure triage

| Failure | Action |
|---|---|
| Unit test fails | Fix the mathematical or metric defect before rerunning an expensive experiment. |
| Integration test fails | Fix orchestration, tensor orientation, result schema, or accidental teacher mutation before using a real model. |
| GPU test fails only | Capture the environment and distinguish a device/caching issue from a model-behavior regression. |
| Candidate improves local metrics but fails PPL/KL | Record a valid negative result; inspect drift rather than tuning the same objective indefinitely. |
| Candidate fails reproducibility across seeds | Mark it inconclusive and do not promote it. |

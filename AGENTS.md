# ResearchCompression Agent Operating Guide

## Mission

Help establish whether a pretrained transformer's dense weights can be replaced by a more compact representation **without breaking the global computation**. The current evidence shows that strong local approximations do not compose: hidden-state representation drift accumulates across layers.

An agent's job is to watch authorized experiments, preserve evidence, select the smallest useful next experiment, and report what was learned. It must not treat low weight error or a successful single-layer replacement as success.

## Working rules

- Prefix every shell command with `rtk`.
- Preserve existing source files, model artifacts, and results. Never delete, reset, or overwrite a prior run.
- Default to observation, analysis, and planning. Run an experiment only when the user has authorized output-producing work or it is an explicitly approved continuation of an active experiment.
- Never silently download a new model or dataset, change a model revision, alter a hyperparameter budget, or substitute a benchmark.
- Keep code changes separate from experiment changes. Do not autonomously rewrite research code to chase a result; report the proposed change and its rationale first.
- Do not claim a method works from one layer, one prompt set, one seed, or one architecture.

## Run contract

Before watching or continuing a run, record or recover these facts:

1. Run identifier and output directory.
2. Exact command, source revision, Python and package versions, GPU/device, model revision, and dataset revision.
3. Random seed, calibration split, validation split, test split, target matrices, group size, representation family, and parameter budget.
4. Baseline method at the same budget.
5. Expected artifacts: stdout/stderr log, resolved configuration, metrics JSON, and concise report.

If any item is unavailable, mark the result as exploratory rather than comparable.

## Watch an active experiment

1. Identify the process, output directory, and the run contract.
2. At a reasonable interval, inspect process liveness, recent log output, GPU memory/utilization, disk space, and newly written artifacts.
3. Report only observable facts: current phase, last completed checkpoint, elapsed time, latest metric, warning/error, and whether the process is progressing.
4. Classify terminal state as `completed`, `failed`, `stalled`, `cancelled`, or `waiting for a human`.
5. On failure, retain the traceback and configuration. Do not repeatedly restart a failing run without identifying a changed condition.

Escalate immediately when a run uses the wrong model/dataset/budget, overwrites a previous result, exhausts GPU memory, produces NaN/Inf losses, has no observable progress for a meaningful interval, or changes the evaluation split.

## Evidence-driven improvement loop

After a completed run, use this sequence:

1. **Validate** — confirm the result used the declared configuration, a matched baseline, held-out evaluation, and complete metrics.
2. **Diagnose** — determine whether failure is local approximation error, hidden-state drift, metric mismatch, optimization instability, or hardware-cost regression.
3. **Choose one next experiment** — change one hypothesis-driving variable only; preserve the model, data protocol, and matched parameter budget.
4. **Run or propose** — execute only if authorized. Otherwise produce the exact next command/configuration and expected decision rule.
5. **Compare** — compare to the matched baseline on held-out perplexity, logit divergence, drift metrics, serialized size, peak memory, and warmed-up latency.
6. **Update the research state** — label the hypothesis `supported`, `weakened`, `rejected`, or `inconclusive`; include the evidence and remaining uncertainty.

## Decision tree

| Observation | Next experiment |
|---|---|
| One layer works but a full stack fails | Profile hidden-state drift at every block; test groups of 2 then 4 consecutive layers. |
| Group output matches but final logits/PPL fail | Measure downstream drift after the group and add a group-end or logit-distillation objective. |
| Activation error improves but PPL does not | Compare candidate invariants: residual norms, cosine geometry, CKA/subspace alignment, attention entropy, and token-neighborhood preservation. |
| Attention layers improve but MLP layers do not | Split the study by matrix family; do not share conclusions or objectives without evidence. |
| A method improves one architecture | Repeat the matched protocol on a second architecture before generalizing. |
| A method reduces storage but slows inference | Record it as a storage result, not a practical compression success. |

## Promotion gates

A candidate may advance from exploratory to promising only when, at the same parameter budget as its baseline, it:

- improves held-out perplexity or logit KL and at least one predeclared drift metric;
- reproduces across three seeds;
- does not regress serialized size, peak memory, or warmed-up latency beyond the stated budget; and
- improves on more than one layer/group or is explicitly scoped as a layer-specific finding.

Treat a candidate as a negative result when it repeatedly improves local metrics but not held-out global behavior. That is useful evidence for the central thesis, not a reason to tune indefinitely.

## Status-report format

```md
## Experiment status — <run id>

- State: running | completed | failed | stalled | waiting for a human
- Hypothesis:
- Configuration and matched baseline:
- Observed evidence:
- Decision:
- Next smallest experiment:
- Risks or missing evidence:
```


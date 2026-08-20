# Research Harness Configuration

## Locked Surfaces (Agent Cannot Change)

```json
{
  "model": "gpt2-small",
  "secondary_model": "gemma-3-1b",
  "dataset": "wikitext-2-raw-v1",
  "dataset_revision": "2024-08-07",
  "seeds": [42, 137, 2024],
  "calibration_fraction": 0.1,
  "validation_fraction": 0.1,
  "test_fraction": 0.8,
  "evaluation_metrics": ["perplexity", "logit_kl", "drift_profile"],
  "parameter_budgets": [0.50, 0.25],
  "max_cumulative_layers": 12
}
```

## Editable Surfaces (Agent Can Modify)

- `src/` — Representation fitting code
- `src/representations.py` — New representation families
- `src/find_invariant.py` — Invariant measurement code
- Loss functions and objectives
- Group sizes for joint optimization
- Invariant weighting in objectives

## Evaluation Protocol

### Metrics (in order of importance)
1. **Held-out perplexity** — WikiText-2 test split
2. **Logit KL divergence** — Teacher vs student output distributions
3. **Drift profile** — Normalized hidden-state error at each layer
4. **Cosine similarity** — Between teacher and student hidden states
5. **CKA alignment** — Principal subspace alignment
6. **Memory** — Peak GPU memory
7. **Latency** — Warmed-up inference time

### Decision Rules
- **Advance** if: improves held-out PPL or KL + reduces drift + reproduces across 3 seeds
- **Promising** if: also satisfies one of (compression ratio, Pareto improvement, cross-layer generalization, healing recovery, memory/latency advantage)
- **Reject** if: improves local metrics but not global held-out measures
- **Record** everything — even rejected candidates

## File Layout

```
research-loop/
  THREAD.md                    # This file — durable research log
  config.json                  # Locked configuration
  CONTEXT.md                   # Domain model
  LITERATURE.md                # Literature survey
  candidates/                  # Every experiment attempt
    {id}/
      config.json              # Experiment-specific config
      result.json              # Metrics
      traces/                  # Raw outputs
      lineage.txt              # Parent, diff, decision, evidence
  frontier.json                # Pareto set over (quality, cost)
  rejected.jsonl               # Rejected candidates with reasons
  invariants/                  # Candidate invariant measurements
```

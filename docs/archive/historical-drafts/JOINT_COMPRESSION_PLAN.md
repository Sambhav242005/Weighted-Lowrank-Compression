# Joint Compression Research Plan

## Research objective

Determine whether post-training compression can preserve a transformer's **global computation** when approximations are fitted jointly, rather than independently per layer. The plan directly tests the working explanation in `results/RESEARCH_SUMMARY.md`: small local deviations accumulate into hidden-state representation drift.

The study does not define success as low Frobenius error, singular-subspace overlap, or an isolated layer result. It defines success as improved held-out model behavior at a matched parameter and hardware budget.

## Current evidence and hypothesis

- Individual GPT-2 layers can be compressed with little isolated perplexity loss.
- Cumulative replacement causes severe perplexity degradation.
- Activation-preserving fitting is more behaviorally relevant than weight fitting, but still does not solve composition.
- Attention and MLP matrices behave differently across GPT-2 and Gemma.

**Primary hypothesis:** jointly optimizing consecutive replacement layers against the original model's group-level computation will reduce representation drift and improve cumulative behavior versus independently fitted replacements at the same budget.

**Alternative hypothesis:** no compact post-training representation at the tested budgets can preserve the cross-layer invariants required for global behavior. This is a valid and publishable negative result.

## Experimental protocol

### Fixed starting protocol

- Start with GPT-2 Small and attention output projections (`W_O`) only, because they have the clearest existing evidence.
- Use WikiText-2 Raw V1 with disjoint calibration, validation, and final test subsets. Record the exact dataset revision, selection seed, text filtering, tokenization, and token counts.
- Run three fixed seeds for every candidate and baseline.
- Match the compressed parameter count exactly across methods. Compare actual serialized bytes as well as nominal factor counts.
- Use the original, frozen model as the teacher. Evaluate only on held-out text that was not used to fit factors.

### Baselines

For every group size and budget, compare:

1. Original dense weights.
2. Truncated SVD fitted independently per layer.
3. Existing activation/function-preserving fitting independently per layer.
4. Jointly fitted factors with the same total parameter budget.

No candidate should be compared with a baseline that has a different rank, target layers, evaluation texts, or model revision.

## Phased work

### Phase A — Establish a trustworthy drift profile

Instrument the original and compressed models on the same held-out inputs. After every transformer block, record:

- normalized hidden-state error;
- cosine similarity and residual-stream norm distribution;
- CKA or principal-subspace alignment;
- attention entropy/pattern similarity where applicable; and
- final-logit KL divergence and top-1 agreement.

Run this for independent replacements at group sizes 1, 2, 4, 8, and 12. Identify the earliest block at which drift becomes predictive of held-out perplexity degradation. This phase selects metrics; it does not yet assume which invariant is causal.

### Phase B — Joint group compression

Fit replacements for consecutive groups of 2 and 4 `W_O` layers jointly. The teacher provides the input activation to the first layer of each group; the student runs the entire compressed group. Optimize the student against the teacher at the **group output**, not each intermediate weight independently.

Begin with normalized group-end hidden-state loss. Then evaluate additions only when Phase A shows they predict behavior: cosine/geometry preservation, distributional statistics, selected intermediate checkpoints, and final-logit KL. Do not combine arbitrary losses merely because they are available.

Compare joint and independent fitting at the same total factor budget. If two- and four-layer groups help, extend to 8 and then all attention-output layers.

### Phase C — Invariant ablation

For the best joint group setting, run controlled ablations. Add one candidate invariant at a time to the group-end objective and measure whether it improves held-out PPL, KL, and the drift profile across all three seeds.

The candidate invariants are:

- residual-stream norm and covariance;
- token-pair cosine geometry;
- principal subspace/CKA alignment;
- attention entropy or attention-map similarity; and
- output-logit distribution similarity.

Retain only invariants that improve global behavior over the group-end baseline. Report non-helpful invariants as negative evidence.

### Phase D — Matrix-family and architecture validation

After an attention result is reproducible, repeat the protocol for MLP down projections. Keep attention and MLP conclusions separate until results justify a shared objective.

Validate the winning and strongest negative result on Gemma 3 1B using an architecture adapter and exactly the same data, budget, and metric protocol. A result observed only on GPT-2 remains architecture-specific.

### Phase E — Practicality and healing

For only the strongest candidates, measure warmed-up end-to-end latency, peak GPU memory, FLOP estimate, and actual checkpoint size. Then test a fixed, small LoRA healing budget to determine whether the compressed solution lies in a recoverable optimization basin.

Do not frame a method as practical if its computational overhead removes its memory-bandwidth benefit.

## Decision rules

Advance a method when it improves over the matched independent baseline on held-out PPL or logit KL, reduces at least one drift metric selected in Phase A, and reproduces across three seeds.

Call a method **promising** only if it also satisfies one of the project criteria: meaningful compression with negligible PPL change, Pareto improvement over classical baselines, cross-layer generalization without per-layer retuning, recoverability with a fixed healing budget, or a real memory/latency advantage.

Stop extending a representation family when it repeatedly improves local reconstruction or activation error but fails the global held-out measures. Record that as support for the layerwise-noncompositionality thesis.

## Test and verification plan

Before any expensive run, verify:

1. Representation unit tests on known low-rank tensors, including shape, exact reconstruction where expected, parameter count, serialization, and deterministic seeds.
2. Model-adapter tests using a tiny local transformer, verifying extraction/substitution orientation and that the original model remains unchanged.
3. Evaluator tests proving padded tokens are excluded correctly, identical teacher/student outputs give zero divergence, and metrics use the declared split.
4. Integration tests that run a tiny complete group experiment offline and emit a schema-valid result record.
5. GPU/nightly tests that run the real-model protocol, preserve the run contract, and compare the result to the matched baseline.

## Expected research outputs

- A reproducible drift-versus-depth plot for independently compressed groups.
- Matched joint-versus-independent Pareto curves for PPL/KL, compression, memory, and latency.
- An invariant-ablation table identifying which measurements predict and improve global behavior.
- A cross-architecture replication or a clearly bounded architecture-specific result.
- A defensible conclusion: joint optimization preserves a measurable invariant, or post-training compact replacements remain non-compositional at the tested budgets.

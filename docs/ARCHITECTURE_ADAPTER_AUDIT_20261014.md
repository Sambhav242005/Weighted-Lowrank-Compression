# Attention adapter audit for the amended three-model study

The selected cohort is the existing Gemma-3-1B-IT reference plus the two pinned
models below. GPT-2 is excluded. No fourth model is selected. The new panel
runner uses BF16; the preserved Gemma component study used FP16. Comparisons
across architectures must use each model's teacher-relative changes, with that
precision difference visible. Tokenizer-specific raw PPL values must not be
ranked or pooled.

| Model | Decoder blocks | Full-attention block indices, zero-based | Attention output | Weight shape [out,in] | Rank |
|---|---:|---|---|---|---:|
| Qwen/Qwen3.5-0.8B | 24 | 3, 7, 11, 15, 19, 23 | `model.layers.i.self_attn.o_proj` | [1024, 2048] | 483 |
| LiquidAI/LFM2.5-1.2B-Instruct | 16 | 2, 5, 8, 10, 12, 14 | `model.layers.i.self_attn.out_proj` | [2048, 2048] | 725 |

Qwen revision: `2fc06364715b967f1860aea9cf38778875588b17`.
LFM revision: `0f604ada3f766f9f257460c4c9f0b5d6f69d431b`.
Both records and their original configs are preserved in
`paper-assets/architecture-selection-20261014.json`.

Each model has six attention output targets. Qwen's 18 DeltaNet output
projections and LFM's ten convolution mixer outputs remain unchanged. Including
them would define a separate mixer-compression experiment, with a different
scope and budget; it is not an adapter convenience. Drift is measured at all
24 or 16 decoder blocks, including unchanged mixers.

## What required a new adapter

The existing Gemma/GPT-2 target selector only recognizes `self_attn.o_proj` and
`attn.c_proj`, so it rejects LFM's attention output `self_attn.out_proj`. The
existing Stage-E dispatcher also hardcodes GPT-2 and rank 256. Those active-study
files remain unchanged. `src/paper_architecture_panel.py` is an additive runner
with explicit model pins, native class loading, and attention-only target
validation.

The existing composition helper assumes the target count equals decoder depth
when assigning group drift indices. That assumption does not hold for these
hybrids. The new confirmation runner keeps target block indices explicit and
collects residual drift across every actual decoder block; it does not reuse
the Gemma composition helper for hybrid groups.

## Loader evidence

Installed Transformers 5.14.1 provides `Qwen3_5ForCausalLM` and
`Qwen3_5TextConfig`. The latter's `base_config_key="text_config"` extracts the
text configuration from the multimodal snapshot. Its `qwen3_5_text`
`PrefixChange` conversion removes `language_model` under the `model` prefix,
mapping checkpoint `model.language_model.layers.*` onto text-model
`model.layers.*`. The text class explicitly ignores vision/MTP weights.
Loading checks reject remaining missing, mismatched, unexpected, or error keys.

A tiny genuinely multimodal random Qwen checkpoint was saved locally, loaded
through the installed text class, and checked without network or GPU access.
All text weights matched exactly, full text logits were bitwise equal, and
the loader reported no remaining missing, mismatched, or unexpected keys.
Meta-device instantiation of both real pinned configs confirmed the table's
shapes, indices and counts without allocating full model weights.

Reproduction: `rtk .venv-cuda/Scripts/python.exe -m docs.audit_hybrid_adapters_cpu_20261014`.
Evidence: `results/paper-v2/cpu-hybrid-adapter-audit-20261014-e1c741dc/audit.json`.
These are software checks; actual pinned-weight quality and performance remain
measurements for the authorized panel.

Primary documentation explicitly recommends the text-only Qwen class and config:
[Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/v5.13.0/en/model_doc/qwen3_5).
LFM's hybrid attention/convolution architecture and causal model class are
documented in the [official LFM2 model documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/lfm2.md).
Installed implementation evidence is in:

- `transformers/models/qwen3_5/configuration_qwen3_5.py`, lines 56 and 107;
- `transformers/models/qwen3_5/modeling_qwen3_5.py`, lines 670, 760 and 1602;
- `transformers/conversion_mapping.py`, line 974;
- `transformers/configuration_utils.py`, lines 680–682;
- `transformers/models/lfm2/modeling_lfm2.py`, lines 241, 422 and 495.

## Rank and evidence protocol

Gemma's rank-384 factors use 835584 parameters for a 1179648-parameter
attention-output matrix: fraction 17/24. The panel applies
`floor((17/24) * out * in / (out+in))` to each native matrix. This gives 483 for
Qwen and 725 for LFM, rather than transferring a shape-dependent rank 256.
The resulting factor/dense fractions are 0.70751953125 and 0.7080078125.
Within each model, every paired method has identical target shapes, ranks,
BF16 factor precision, calibration tokens and parameter budget.

Each model produces eight full native WikiText-2 test records: dense teacher
once, SVD once, whitening for seeds 11/23/37, and the frozen selected exact
anchor (beta .03, no extra ridge) for seeds 11/23/37. No settings are retuned.
The original plan's additional validation-only rank frontier is retained:
divisors 2, 3 and 4 use `min(out,in)//divisor`, with whitening and the frozen
anchor across all three seeds, giving eighteen further records per model.
Qwen frontier ranks are 512, 341 and 256; LFM ranks are 1024, 682 and 512.
These are native rank divisors, distinct from the primary confirmation's
factor-parameter-fraction policy. They do not select new test settings or
constitute a new independent confirmation cohort.

Each frontier row records its validation split, divisor, actual rank, complete
target recipes and checkpoint provenance. A separate untouched validation
teacher is evaluated once and saved in `validation-teacher.json`; validation
teacher-relative NLL and PPL use this reference, never the test teacher. The
completed cohort contains 26 explicitly scoped records plus that separate
validation teacher artifact.

Calibration uses native BF16 activations from 16 nonoverlapping 512-token train
chunks; weighted fitting uses FP64 covariance/SVD. Evaluation scores every
shifted test target once, including the tail, at context 512 / stride 256.
Records include teacher KL, local target output error, every block's drift,
complete NLL blocks, actual factor-checkpoint bytes, fitting cost, all model
and dataset pins, source hashes, selection hash and per-tokenizer manifests.
Whole-model parameter savings are separate from target savings and serialized
factor bytes; no unmeasured whole-model byte or inference-speed claim is made.

Copied, full-rank W-times-I and full-rank SVD numerical probes use the first
512 validation tokens. These probes do not establish full-test readiness.
All panel results remain exploratory; no old FP16 failure is relabelled as a
BF16 pass, and no three-architecture promotion is inferred automatically.

## Bounds and failure handling

The two model allocations are independent, two hours each, within the existing
four-hour Stage-E cap. Old cancelled/failed GPT-2 Stage-E work and all new panel
attempts count against the shared cap. Repeated attempts also count against
their model's two-hour allocation. No downloads occur inside the runner.

A per-model budget limit preserves that model as incomplete and permits the
other independent allocation to run under the remaining shared cap. The final
panel still returns incomplete/nonzero, preventing automatic advancement to
Stage F. NaN/Inf, wrong assets, OOM and other unexpected failures stop the panel.
A complete model cohort can be recovered after interruption only after its
artifact hashes, implementation hashes, selection/data protocol, factors and
all eight test and eighteen validation scopes verify. Recovery checks both
splits' actual target counts, the frozen frontier protocol, teacher-reference
split provenance, and each factor checkpoint's stored specification, BF16
precision, target recipes and actual tensor shapes. A cohort containing only
the eight primary rows cannot be recovered as complete. Reuse is labelled and
never called replication. Frontier evaluation and its validation teacher count
against the same unchanged two-hour model/four-hour shared Stage-E caps.

Placement defaults to explicitly paired GPU models. Optional `placement="cpu"`
means CPU execution and remains budget-bound; it is not CUDA CPU-weight offload
and provides no GPU timing evidence. There is no automatic precision or device
fallback. The installed Qwen DeltaNet path emits a warning that optional fast
kernels are absent and uses the Torch implementation; the runner neither
downloads kernels nor claims optimized runtime.

Verification: `rtk .venv-cuda/Scripts/python.exe -m pytest -q tests/unit/test_architecture_panel.py`
passed twelve meaningful CPU tests covering native Qwen/LFM adapters, sparse
attention indices, unchanged other mixers, BF16 factors, all-block drift and
partial-window accumulation, copied/identity probes, pins, rank budgets,
cumulative cancelled costs, independent budget limits and failure stopping.
The frontier tests verify all 26 scopes, original divisor ranks, three paired
seeds, no recipe mutation, recovery rejection of missing frontier rows, wrong
split references/counts, changed rank recipes and wrong actual factor shapes.

# User-authorized architecture and precision amendment — 14 October 2026

This amendment supersedes GPT-2 replication in the 13 October plan. Preserve
the original plan and all completed Gemma evidence. The user requested skipping
GPT-2, using at most four models, testing BF16/FP8, and allowing RAM offload.

## Selected panel: three models, maximum four

| Model | Pinned revision | Architecture and role |
|---|---|---|
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` | Existing transformer reference; preserve the completed FP16 comparisons. |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Dense hybrid language backbone: 18 gated DeltaNet blocks and 6 full-attention blocks. Evaluate text only; do not claim vision results. |
| `LiquidAI/LFM2.5-1.2B-Instruct` | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` | 16-block hybrid: 10 short-convolution blocks and 6 full-attention blocks. The published checkpoint is 1.2B, not exactly 1B. |

No fourth model is selected. GPT-2 is excluded from the revised execution and
its previously prepared assets remain preserved. Exact primary configuration
metadata is saved in `paper-assets/architecture-selection-20261014.json`.
The new snapshots are explicitly prepared at their commit revisions; no new
dataset or benchmark is substituted.

Primary sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-0.8B),
[LFM model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), and each
snapshot's pinned `config.json`. Qwen's family-wide MoE marketing description
does not mean this 0.8B checkpoint is an MoE model; inspect its actual config.

## Execution boundary and budgets

1. Let the active Gemma composition stage finish. A verified local controller
   stops the old dispatcher at that boundary before its hard-coded GPT-2 stage.
   Preserve and charge any initialization caught in the boundary race.
2. Run the separate Gemma BF16/FP8 numerical controls in the unused part of the
   original one-hour readiness allocation. Extend shared accounting to include
   the new `precision-readiness-` directories before launching them. Do not
   reset the readiness budget or reinterpret failed FP16 readiness as passed.
3. Replace the original four-hour GPT-2 allocation with a shared four-hour
   architecture allocation: at most two hours each for Qwen and LFM, including
   fitting, evaluation, retries, and partial work. Any old GPT-2 initialization
   charge also reduces the shared allocation. Do not borrow other stage caps.
4. Preserve the original four-hour practical/calibration allocation for the
   existing matched Gemma checkpoints. BF16/FP8 numerical controls are separate
   from original FP16 hardware measurements and cannot be pooled with them.

The pre-reboot composition charge remains an estimate, as documented in the
retained accounting caveat. This amendment does not establish a precise upper
bound on unlogged work or justify an exact 24 GPU-hour compliance claim.

## Precision and RAM placement

The pinned Gemma checkpoint declares BF16. The additive readiness runner uses
a dense BF16 teacher, copied teacher, identity factors, full-rank BF16 SVD
factors, FP32 factor computation returned to BF16, and E4M3/E5M2 FP8 storage
controls for factors and dense target weights. FP8 storage is row-scaled and
decoded to BF16 for matrix multiplication. A separate tiny native FP8 GEMM
probe records actual kernel support; it does not prove native FP8 model speed.

Keep the original probe tolerances and full native test evaluation. Retain a
pristine reference and decompositions in RAM. Active weights may use CPU/RAM
offload with CUDA execution if GPU placement does not fit. Record placement,
failures and cost; do not silently change dtype or report offloaded latency as
equivalent to fully resident GPU latency. No new packages are required.

## Matched architecture comparison

Use native WikiText-2 train/validation/test, the pinned dataset revision, 16
calibration chunks of 512 tokens, seeds 11/23/37, and context/stride 512/256.
Keep the frozen Gemma selection: whitening versus exact anchor beta 0.03 and
zero extra ridge. Fit both on identical model-native calibration tokens for
each seed, alongside a seed-independent SVD control and dense reference.

The primary target family is **full-attention output projections only**.
Qwen and LFM each have six such targets. Do not substitute DeltaNet or
convolution output matrices or describe this as compressing every hybrid mixer.
Measure residual drift at every block, including untargeted hybrid blocks.

Match the Gemma target factor-parameter fraction, 17/24, rather than blindly
copy a rank across different shapes. Use
`rank = floor((17/24) * out * in / (out + in))`, with the same realized rank
for every method on that matrix. Qwen targets are [1024,2048], rank 483;
LFM targets are [2048,2048], rank 725. Record exact realized parameters, bias,
factor bytes, and whole-model accounting separately.

Evaluate the entire native test split including its tail; save NLL/PPL, KL,
local output error, all-block residual drift, paired uncertainty blocks, source
hashes, asset revisions and model-native token/calibration hashes. Different
tokenizers preclude direct raw PPL ranking across models: compare each method
to its own teacher and matched baseline. Precision and architecture differences
remain explicit. These exploratory comparisons cannot remove the original
FP16 readiness failure or alone establish a universal compression method.

Retain the original validation-only rank frontier after the frozen primary
comparison: divisors 2, 3 and 4 mean `min(out,in)//divisor`, with whitening and
the frozen anchor at all three seeds. Qwen frontier ranks are 512/341/256;
LFM ranks are 1024/682/512. This adds 18 validation records to each model's
eight primary test records. Measure a separate native-validation dense reference;
never compare a validation NLL to a test-split teacher or choose a setting on
test outcomes. Fitting and frontier evaluation share the same existing caps.

If a model exhausts its cap or an adapter is unsupported, retain an explicit
incomplete result. Do not silently replace it with GPT-2 or another model,
reduce evaluation coverage/seeds, install a new backend, or expand the budget.

Budget-limited cohorts do not cancel independent authorized cohorts. The second
architecture retains its own allocation, and Gemma practical measurements retain
their original allocation even if precision controls or an architecture cohort
run out of time. A hardware error, unexpected exception, or invalid provenance
still stops the dispatcher for diagnosis. Completion of the bounded work must
identify every incomplete cohort; it is not a claim that every measurement or
promotion gate passed.

# Paper layout and citation audit — 14 October 2026

Open document edited in place: C:/Users/NPC/Downloads/paper_twocolumn_revised_20261014.tex.

## Screenshot diagnosis and repair

Long monospaced run identifiers and revision hashes exceeded the two-column text width. Compilation success had not detected this visual defect.

- Replaced url with xurl, enabling line breaks through long identifiers and hashes.
- Made inline code paths breakable rather than wrapping them in an unbreakable texttt command.
- Placed appendix labels and identifiers on separate lines, with smaller identifier text.
- Split the objective, bias/cross-moment, NLL/PPL, drift, and quadratic-expansion equations into aligned lines.
- Bounded every table to its containing column or full-width float using adjustbox; tightened the paired-interval table spacing.
- Kept the existing editor, document path, two-column format, measured values, and diagram.

These changes address the source mechanisms shown in the screenshots. The final source is checked with the desktop compiler. This audit does not claim that every rendered page has been visually inspected.

## Skill discovery

Found and read K-Dense's scientific-writing skill and relevant writing/display/formatting guidance.

- Skill: https://skills.sh/k-dense-ai/scientific-agent-skills/scientific-writing
- Source: https://github.com/K-Dense-AI/scientific-agent-skills/tree/main/skills/scientific-writing
- Discovery snapshot: approximately 2.3K skill installs; approximately 49K stars on the source repository.
- Purpose: evidence-bound scientific drafting and citation/consistency review. It does not itself provide a LaTeX layout template.
- Not installed globally by this task.
- Optional installation command: rtk npx skills add k-dense-ai/scientific-agent-skills --skill scientific-writing

The recommendation is based on inspected source guidance, not popularity alone. Finding or reading a skill is not evidence of submission readiness.

## Full-text reading coverage

Read the ten works cited in the previous revision and the three additional works cited in the originally supplied paper: 13 total. Reading records identify full-text sources and distinguish main-paper versus appendix coverage.

- Low-rank methods: [reading record](paper-reading-lowrank-20261014.md) — DRONE, Fisher-weighted factorization, ASVD, SVD-LLM, EoRA.
- Diagnostics and adjacent representations: [reading record](paper-reading-diagnostics-20261014.md) — GPTQ, AWQ, SparseGPT, QEP, SVDQuant, CKA, temperature calibration, Pointer Sentinel/WikiText.

This is an AI-assisted source reading and editorial check, not accountable human verification of every proof.

## Changes from reading

- Added four focused Related Work subsections with more precise descriptions.
- Acknowledged DRONE's sequential treatment of shifted representations.
- Named ASVD+ and bounded its objective overlap to invertible covariance in exact arithmetic.
- Distinguished SVD-LLM(W) from the cited v5's sequential factor-wise LoRA fine-tuning.
- Restored QEP as relevant accumulated-error prior art without endorsing an unconditional theorem guarantee.
- Distinguished quantization, pruning, additive compensation, and pure projection replacement.
- Restored SparseGPT and SVDQuant with representation and diffusion-domain caveats.
- Corrected AWQ's current title and ordered the bibliography by first citation.
- All 13 bibliography entries are cited, and every citation resolves to an entry.

## Experiment boundary

No benchmarks, training, calibration fitting, model evaluation, or dataset/model downloads were run. Reading research PDFs and editing text used only existing experimental measurements. No research-code or experiment configuration was changed, and no commit or publication was performed.

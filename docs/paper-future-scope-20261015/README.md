# Paper with Future Scope and no appendices

[Updated PDF](manuscript.pdf) · [IEEE conference source](ieee-conference.tex) ·
[Reference-style LaTeX](manuscript.tex) · [Authored text](manuscript.source.md)

The open **IEEE source** now clearly labels the measured anchored configuration
as **Ours (anchored)** in orange. Gemma 3-1B, Qwen3.5-0.8B and LFM2.5-1.2B use
horizontal bars starting at zero for prediction loss and residual drift. Qwen
and LFM also include their saved dense-reference and plain-SVD controls; their
three-seed sample standard deviations are derived from the same saved test rows.
No new model, experiment or small-network result was added. The existing
per-block drift curve for the anchored method is also identified in orange.
See [model-chart-data.json](model-chart-data.json) for exact source values and
hashes, and [model-chart-validation.json](model-chart-validation.json) for checks.

The open IEEE source was previously edited in place to replace three dense
results tables with self-contained vector charts. Gemma uses paired metric
panels with sample seed standard-deviation whiskers. Composition uses signed
differences around zero, with explicit scaled units and joint-improvement
counts. Hybrid results use separate within-model panels and retain record
coverage. The BF16 practical results and latency ratios remain numerical tables.
All plotted values match the original tables; see
[chart-validation.json](chart-validation.json). The eight equations, abstract,
bibliography and Future Scope remain unchanged by this chart edit.

Compile `ieee-conference.tex` in the current editor to see these charts. The
review PDF, Markdown and reference-style LaTeX retain the earlier table layout.
The original export scripts describe that earlier layout; preserve the edited
IEEE source when making later builds. No separate paper PDF was generated for
the in-place chart edit.

Appendices A and B have been removed from this edition's paper, including their
cross-references. The five main results tables, original equations, abstract,
drift figure and numerical conclusions are preserved. The source experiment
records and selected uncertainty intervals remain in the repository and
[report.json](report.json); their locations are identified in Section 7.
Earlier paper editions remain intact.

Section 8, **Future Scope**, begins on page 6 of the seven-page review PDF.
Section 8.1 proposes a small bottleneck neural network as a learned replacement
for a selected dense projection. It also describes a small correction network
alongside lower-rank factors. Any comparison must include the network's own
weights and biases in the storage budget and evaluate composed model behavior,
memory and inference latency. Neither network has been implemented or evaluated.

[TinyBERT](https://aclanthology.org/2020.findings-emnlp.372/) supplies a related
transformer-distillation precedent. It is not evidence that our proposed
projection replacement will preserve computation. Its verified publication
metadata is appended to the bibliography; all previous entries remain intact.
Sections 8.2 and 8.3 cover numerical controls, missing evaluation, broader tests
and practical execution costs. The Conclusion is now Section 9, and the
acknowledgment includes assistance with Future Scope.

All eight displayed equations retain the previous 11.5-point formatting. All
seven PDF pages have been visually reviewed. Both LaTeX sources compile with the
desktop editor's compiler. The review PDF is an independent ReportLab rendering;
the IEEE source uses the provisional conference class described in the
[publication research](../paper-ieee-preparation-20261015/IEEE_PUBLICATION_RESEARCH.md).

The [preparation script](prepare_manuscript.py) records the exact editorial edit
in [editorial.patch](editorial.patch) and checks table and math preservation.
The [builder](build_edition.py) exports a fresh edition using the local
[renderer](build_paper.py). Run preparation and building only before generated
outputs exist; both scripts refuse existing authored or rendered editions.
[edition-audit.json](edition-audit.json) records preserved source hashes, and
[validation.json](validation.json) records final artifact checks. No research
experiment or training run was performed.

# Review draft exports — 14 October 2026

`export_paper_latex_20261014.export_report(report_directory)` creates a new
`manuscript.tex`; `export_paper_pdf_20261014.export_pdf(report_directory)`
creates a new `manuscript.pdf`. Both refuse overwrite. Call after report.json,
manuscript.md, the checked bibliography, and figures/provenance.json exist.
No experiment source, assets, configuration, or run artifact is changed.

The LaTeX exporter uses the Python standard library and embeds all 22 references
and measured figure coordinates. It needs no external images/BibTeX files.
Native picture graphs retain separate attempts, model identities and primary-test
versus validation-frontier roles; missing measurements stay unavailable. Tables
with many columns are split into linked panels without dropping fields. The
paired-block appendix includes actual intervals, resample counts and excluded
tail targets. Neither exporter promotes exploratory evidence or ranks models
using raw tokenizer-dependent perplexity.

The independent PDF renderer uses existing ReportLab in `C:/Python314/python.exe`;
no package was installed. ReportLab Platypus documentation was resolved and
queried through Context7 before implementation. It preserves manuscript content,
wraps/paginates table cells, embeds actual generated PNGs, adds all references
and the paired-uncertainty appendix, and prints equations in readable ASCII
notation. The companion TeX retains exact typeset mathematical notation. A
ReportLab PDF does not imply successful TeX compilation.

## Verification

- Six LaTeX exporter tests and three PDF tests pass. They cover escaping/math,
  bibliography nesting/count, complete table-column retention, scope separation,
  nonfinite plot rejection, stable filenames and refusal to overwrite.
- Built-in LaTeX compilation used three calls and one repair. First failed on the
  generated manuscript's `\lambda_\max`; the exporter normalizes this to
  `\lambda_{\max}`. Second call succeeded on
  `docs/proofs/latex-review-8ce135e2/manuscript.tex`. After the editorial row-ID
  improvement, a third call succeeded on
  `docs/proofs/rowid-review-e0e9af7c/manuscript.tex`; no further repair was needed.
- Proof inputs are copied from the **old cancelled-for-review report**
  `results/paper-v2/revised-report-20261014-061526-3ee7c395`; they do not represent
  completion of the currently active continuation. Original outputs remain intact.
- Final PDF proof: `docs/proofs/pdf-review-8d3c61fb/manuscript.pdf`.
  Its `pdf-smoke.json` records hashes, page/text counts, embedded-image count and
  geometry checks. First page, crowded results page, figure page and final page
  are rendered beside it for visual inspection. These proof files are documents,
  not synthetic experiment runs.

Actual terminal-study exports must be generated from the new final report, then
compiled/rendered and inspected again. The built-in compiler validates standalone
TeX but does not itself export a filesystem PDF, hence the independent renderer.

## Wide-table record identity repair

Every source table wider than five columns now assigns stable visible R001,
R002, … IDs to original rows. Every continuation panel repeats that ID and the
first original column, followed by three additional columns. A visible note
explains that matching IDs refer to the same original record and restart per
source table. Thus repeated run names remain distinguishable across variant,
method/scope, gates, and timing-condition panels. Every original column remains.

New proof `docs/proofs/rowid-review-e0e9af7c/manuscript.pdf` has 23 pages and
zero off-page text blocks; rendered pages 4 and 5 show the original numerical
variant panel and its gate continuation with matching IDs. Both were visually
checked. Earlier proofs are preserved. `rowid-proof.json` records source hashes.

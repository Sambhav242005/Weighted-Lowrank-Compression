# Single-author reference-style edition

[Paper PDF](manuscript.pdf) · [LaTeX source](manuscript.tex) ·
[Scientific text](manuscript.md)

This edition matches the visual system of the user-provided
`weighted_lowrank_compression_solo_author.pdf`: A4 pages, a centered full-width
title and single-author block, blue project links, italic full-width abstract,
Times body text in two columns, numbered display equations, compact ruled tables
and numbered citations. The PDF is seven pages. Author name Sambhav Surana and
project URLs come from the supplied reference; the repository URL also matches
this checkout's Git origin. No affiliation, ORCID, funding or contribution role
was inferred.

The scientific text is copied byte for byte from the verified 15 October
editorial revision, not from the older reference's superseded results. Its model
scope, numerical-readiness failures, partial Qwen frontier, two-seed BF16 cohort,
measured savings and limits remain unchanged. All 22 references and all six
source tables (five main results tables plus the evidence map) are retained.
The selected paired intervals from the prior revision's supplement are also
typeset as Appendix B.

## Output and verification

- The PDF is generated with ReportLab using Times New Roman. Mathematical
  expressions use 360-dpi STIX rendering rather than ASCII approximations.
- The standalone LaTeX uses a matching two-column Times-style layout and
  compiles successfully with the Codex desktop compiler. It is a separate
  rendering source; the supplied PDF is not represented as compiled from TeX.
- The PDF drift figure is replotted only from saved means, individual-seed
  observations and sample standard deviations, with readable column-sized
  labels. No model evaluation or new measurement is involved.
- The standalone TeX embeds saved mean-curve coordinates without external
  assets. Its caption explicitly points out that seed variation appears in the
  companion PDF and retained original evidence.
- All seven PDF pages were rendered and visually checked. Automated checks
  cover text bounds, original-source hashes, table/citation preservation and
  local links; [validation.json](validation.json) records the exact output hashes.
- The reference's scientific claims and embedded checklists were treated as
  document content, not instructions. No historical denoising, lossless-INT8 or
  generalized compression-success claim was restored.

This remains an exploratory paper and a reference-matched layout, not a claim
of compliance with a particular venue. The original reference PDF, previous
editorial edition, source code, experiment outputs and budgets are unchanged.
AI-assisted formatting was used; no new scientific result was generated.

## Rebuilding

[build_paper.py](build_paper.py) reads the previous editorial manuscript and
retained figure/interval data, checks source hashes, and prepares both exports.
It uses already-installed ReportLab, Matplotlib, Pillow and the existing export
helper. Formula-image caches are local build products excluded from Git; the PDF
embeds them and the TeX does not require them.

```powershell
rtk C:/Python314/python.exe docs/paper-reference-style-20261015/build_paper.py
```

The builder refuses existing generated outputs. Create a new sibling edition
for another build; preserve existing papers and experiment evidence. It also
records the supplied local reference PDF's hash, so that original file must
remain available when rebuilding. [format-manifest.json](format-manifest.json)
records the layout dimensions, source hashes, math expressions and format scope.

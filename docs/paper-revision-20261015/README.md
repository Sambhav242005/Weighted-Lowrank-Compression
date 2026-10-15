# Paper revision - 15 October 2026

[Read the paper](manuscript.pdf), [Markdown](manuscript.md), or
[standalone LaTeX source](manuscript.tex).

This editorial revision reorganizes the 14 October frozen manuscript into a
research question, comparison scope, mathematical methods, protocol, five compact
result tables, discussion, limitations and evidence appendix. The abstract is
rewritten to state quantitative outcomes and their limits. The independent
ReportLab PDF is nine pages; it is not a PDF compiled from the LaTeX source.
The exact final LaTeX source also compiled successfully in the desktop editor.
No venue-specific format or submission-readiness certification is claimed.

## What changed

- Centered the argument on disagreement among likelihood, residual fidelity and
  practical cost, rather than execution status or an unsupported method win.
- Defined the residual drift metric, its block/token aggregation, teacher-context
  evaluation and student-propagated states.
- Distinguished exact-output and measured-output objectives, anchoring and extra
  ridge shrinkage, numerical cutoffs and factor-storage versus fitting precision.
- Presented unique completed composition records without counting interrupted
  predecessors again. Kept Qwen's primary-test coverage separate from its partial
  validation frontier and its reused observations.
- Reported both targeted 29.17% weight savings and exact BF16 whole-checkpoint
  bytes (0.894% saving). The older manuscript rounded those byte counts.
- Corrected related work to include ASVD+ covariance weighting and SVD-LLM v5's
  sequential factor-wise adaptation. The baselines remain fixed-rank components,
  not complete reproductions.
- Kept all failed readiness gates, two-seed practical scope, corpus reuse,
  incomplete frontier points, runtime limitations and budget-accounting caveats.
- Moved verbose run records into the linked frozen evidence and selected paired
  interval appendix. PDF citations now use numbered references.

## Evidence and checks

All five main tables are assembled deterministically from saved records.
[revision-manifest.json](revision-manifest.json) records source hashes, extraction
fields, table contents and protected-artifact checks. [report.json](report.json)
is a selected, explicitly derivative interval supplement, not a replacement for
the [complete frozen report](../../results/paper-v2/revised-report-20261014-151432-07c08d61/report.json).

The saved evidence and research source files remain unchanged. No new model,
dataset, experiment, fitting setting or measurement was introduced. Primary
ASVD/SVD-LLM source text and the four 2026 preprint records were rechecked for the
revised comparison wording; the remaining bibliography retains its prior recorded
verification. All 22 citation keys resolve and are used. The 2026 works remain
preprints without an inferred venue.

All nine PDF pages were rendered and visually reviewed in contact sheets. Text
bounds, local links, required quantitative findings and artifact/source hashes
were checked. The exact final TeX source compiled successfully. See
[validation.json](validation.json) for the checked artifact hashes and scope.
The PDF's equations use the existing exporter's ASCII notation; the TeX source
provides exact mathematical typesetting.

The academic-paper skill informed reviewer, writer and citation roles. Its
installed package lacks the advertised patch-application and acronym-check
scripts, so this is a manual structural revision with deterministic table
assembly and hash checks, not a certified automated revision-pipeline round.
No reviewer comments, author identities, funding or venue approval were invented.

## Source and build

[manuscript.source.md](manuscript.source.md) contains the authored text and table
slots. [build_revision.py](build_revision.py) binds those slots to the frozen
records, checks citation keys and original-file hashes, and uses the existing
LaTeX/PDF exporters. It imports no experiment runner or model library.

```powershell
rtk C:/Python314/python.exe docs/paper-revision-20261015/build_revision.py
```

The command refuses existing generated outputs. To create a later revision,
copy the authoring source and builder into a new sibling directory and regenerate
there with the reporting environment already available. Do not overwrite the
frozen 14 October evidence or present a reporting rebuild as another experiment.

Further scientific progress still requires a separately authorized protocol:
passing strict numerical controls, fresh-data/downstream confirmation, additional
practical replication or another compression family. This revision supplies no
new experimental support for those claims.

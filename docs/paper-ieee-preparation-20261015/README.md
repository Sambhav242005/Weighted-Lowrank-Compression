# Readable paper and IEEE preparation

[Review PDF](manuscript.pdf) · [Abstract](abstract.md) ·
[IEEE conference source](ieee-conference.tex) ·
[Publication research](IEEE_PUBLICATION_RESEARCH.md)

The abstract now introduces matrices as tables of numbers and explains the
internal signals passed between layers before describing the comparison. It is
one paragraph of 233 words, with no abbreviations, citations or equations.
The abstract retains the 3.86-fold internal-error finding, the distinction between
29.17% targeted weight reduction and 0.894% whole-file reduction, and the
numerical, coverage and evaluation limitations.

The [Humanizer skill, version 3.1.0](https://github.com/blader/humanizer/blob/main/SKILL.md)
guided a prose edit of the introduction, explanations and discussion. The pass
removes staged questions, decorative bolding, repetitive contrasts and formulaic
contribution lists. It keeps the scientific qualifications. All six original
tables, mathematical expressions and existing link targets are unchanged. All
22 research references remain, with a new OpenAI Codex reference identifying the
assistance disclosed in the acknowledgment. No text detector or acceptance
prediction was used.

## Submission status

The venue is unknown. The user reports a deadline before 17 October 2026 and a
₹1,000 charge afterward, but no notice or portal identifies the event or fee
stage. Neither detail is independently verified. We have not selected a paid
journal, submitted the paper or paid a fee. The publication research records the
official sources and the teacher-review items that remain.

`ieee-conference.tex` uses the standard `IEEEtran` conference class and compiles
successfully. It is a provisional format until the actual event confirms paper
size, page count, anonymity, author metadata and appendix rules. Author name is
retained from the supplied reference; affiliation, email, ORCID and biography
have not been invented. The source includes an acknowledgment of the drafting,
editing and formatting assistance observed in this chat.

`manuscript.pdf` is an independent two-column review rendering. It is not a
camera-ready PDF compiled from `ieee-conference.tex`. Compile/export that exact
IEEE source and check it against the actual venue's rules before upload. The
reference-style `manuscript.tex` is also retained and compile-checked.

The failed numerical checks, missing frontier measurements and limited external
evaluation remain scientific limits. Improving the wording or using an IEEE
class does not supply missing evidence. This preparation runs no new experiment.

## Authoring and verification

- [manuscript.source.md](manuscript.source.md) is the revised authored text.
- [prepare_manuscript.py](prepare_manuscript.py) records the exact prose changes
  from the previous edition and asserts table/math/link preservation.
- [editorial-audit.json](editorial-audit.json) and [editorial.patch](editorial.patch)
  retain the edits for review.
- [build_preparation.py](build_preparation.py) uses the preserved rendering helper
  to export the review files and assembles the generic IEEE conference source.
- `build_paper.py` is an unchanged copy of that renderer for its source manifest;
  invoke `build_preparation.py` when creating this edition.
- [preparation-manifest.json](preparation-manifest.json) identifies sources and
  unresolved submission details; [validation.json](validation.json) records the
  final artifact checks and compiler results.

Both preparation and rendering commands refuse existing outputs. A later edit
needs a fresh edition, preserving the current paper and original experiment
records. Rendered proof pages and regenerable formula images are ignored local
build products.

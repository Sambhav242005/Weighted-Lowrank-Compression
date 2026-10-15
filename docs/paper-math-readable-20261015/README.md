# Paper with readable mathematics

[Updated PDF](manuscript.pdf) · [IEEE conference source](ieee-conference.tex) ·
[Reference-style LaTeX](manuscript.tex) · [Abstract](abstract.md)

The rank-budget formulas now appear on separate numbered lines with stacked
fractions and scalable floor brackets. The storage condition and the two drift
definitions also use displayed fractions. All eight displayed equations in the
review PDF use 11.5-point mathematics without shrinking to fit the column.
Inline symbols use their measured baseline depth, and body lines have more
space to accommodate subscripts and superscripts.

The seven-page PDF preserves the previous manuscript text, abstract,
bibliography, numerical tables, interval report and figure input data exactly.
The LaTeX sources apply the equivalent mathematical presentation changes.
No experiment was run, and the earlier editions remain intact.

Both LaTeX sources compile successfully with the desktop editor's compiler.
The PDF is an independent ReportLab review rendering. The IEEE source uses the
provisional `IEEEtran` conference class; the actual venue requirements remain
unresolved as recorded in the [publication research](../paper-ieee-preparation-20261015/IEEE_PUBLICATION_RESEARCH.md).

Run `rtk proxy python docs/paper-math-readable-20261015/build_math_edition.py`
only when this edition's generated outputs do not already exist. The builder
refuses to overwrite an existing edition. Its local [renderer](build_paper.py)
records the exact presentation mapping and equation sizes in
[format-manifest.json](format-manifest.json).
[math-audit.json](math-audit.json) records unchanged source hashes, and
[validation.json](validation.json) records the artifact and compilation checks.
Formula images and rendered proof pages are ignored build products.

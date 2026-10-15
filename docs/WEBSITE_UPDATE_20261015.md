# Research website update — 15 October 2026

The existing GitHub Pages site is published from
`results/research-dashboard-20260815` by `.github/workflows/deploy.yml`.
Both the overview and plain-language reader guide now describe the current
bounded study rather than the earlier exploratory success narrative.

The update includes the current abstract, Gemma/Qwen/LFM findings, numerical
and coverage limitations, BF16 storage and timing results, and the untested
small-network proposal in Future Scope. Obsolete denoising, universal drift-wall
and compression-success statements have been removed. Links now point to the
current IEEE source and frozen evidence report. The earlier review PDF is
explicitly identified as retaining its earlier table layout.
Both pages also link to the current eight-page IEEE PDF, exported from the
open source with the revised charts; Future Scope is on page 7.

The three dense comparisons use inline vector charts. The five original Gemma
methods include the reported sample seed standard deviations, with no error bar
invented for the single SVD fit. Composition differences retain their signs,
scaled axis units and joint-improvement counts. Hybrid panels use model-specific
ranges and retain Qwen's incomplete validation coverage. Simple BF16 and timing
values remain tables. Each chart has a description and an expandable semantic
table of the exact values. The same public values are available in
`research-data.json`, including the source paper's SHA-256.

The latest chart revision labels the tested anchored fit **Ours (anchored)** in
orange and uses horizontal, zero-based bars for Gemma, Qwen and LFM. Saved dense
references and plain-SVD controls are now included for Qwen3.5-0.8B and
LFM2.5-1.2B. Repeated fits have sample seed standard-deviation whiskers, computed
from their existing test rows; the single recorded controls have no error bars.
The source paper's model-chart data file retains original values and hashes.
The small-network idea remains unmeasured Future Scope.

The layer-group chart now separates the numbers from the plotting area.
Each panel has a right-aligned Value column, with a signed four-decimal
number on the same row as its dot. The existing scaled units, zero lines,
raw data and exact-value table remain unchanged. The website uses tabular
numerals for scanning the columns. PDF checks verified all 20 displayed
numbers against the saved data, matching row heights and aligned right edges.

The existing 3D illustration is retained as an explicitly synthetic concept
visualization. Native controls support pausing, selecting a stage, rotation and
zoom. Reduced-motion preference starts it paused. The chart content remains
available without JavaScript, WebGL or the optional external 3D library.

`update_research_website_20261015.py` derives the numbers from the current IEEE
source and preserves the paper and underlying experiment records. It can be run
with `rtk proxy python docs/update_research_website_20261015.py` when synchronizing
later editorial changes. It does not run a model or an experiment.

Validation used Microsoft Edge at 1440×1000 and 390×844. Both pages had no script
errors or page-level horizontal overflow. Checks covered exact values against
the manuscript, links to local and repository files, unique SVG descriptions,
native disclosures, animation controls, reduced motion, JavaScript disabled,
and optional-CDN failure. Rendered desktop and mobile proofs were inspected.
See [website-validation-20261015.json](website-validation-20261015.json).

The Modern Web Guidance accessibility guide informed semantic table captions,
chart descriptions, native controls and focus treatment. Its instruction
“Provide data tables as alternatives” is implemented by each chart's disclosure.
This is a scoped check, not a comprehensive screen-reader conformance claim.

The existing deployment workflow is unchanged. A push to `main` publishes this
site through GitHub Pages. The open LaTeX editor and its source, existing paper
PDFs, model artifacts and experiment results were not modified by this update.

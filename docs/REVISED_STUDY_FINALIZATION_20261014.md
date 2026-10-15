# Revised study finalization — 14 October 2026

The additive finalizer reads existing artifacts. It imports no research runner,
model library, or CUDA package and changes no source, configuration, checkpoint,
or prior experiment status. Run it in a new process after the serial driver
saves a terminal status, including a terminal incomplete/stopped outcome:

```powershell
rtk C:/Python314/python.exe -u docs/report_revised_study_20261014.py --root results/paper-v2 --orchestration results/paper-v2/revised-orchestration-20261014-4
```

The driver can call this command in its finalization block **after** saving its
terminal status. Failure of this reporting subprocess must be recorded as a
report-generation problem; it must not rewrite completed GPU stage statuses.
This PC's existing base interpreter has Matplotlib 3.11.0; the CUDA virtual
environment is reserved for experiment execution and lacks Matplotlib. The
reporting interpreter and plot source hashes are recorded with the output.
The finalizer refuses nonterminal orchestration unless `--snapshot` is supplied.
An explicitly requested snapshot is labeled interim and cannot establish study
completion. Every invocation creates a fresh `revised-report-<UTC>-<UUID>`
directory containing `report.md`, `report.json`, a versioned `manuscript.md`,
and a copied verified `paper_protocol_v2.bib`. The manuscript combines actual
control/cohort/composition/architecture/cost tables with the existing equations,
22 substantive references and an incomplete-run/reproducibility appendix.
Its mathematical section states the implemented covariance/extra-ridge solve,
spectral cutoff, ASVD constants and recorded per-run settings without a future
equations placeholder.

The finalizer invokes `docs/plot_revised_study_20261014.py` after writing the
report. With existing matplotlib it exports PNG and SVG figures for completed
C drift (each seed, mean and seed SD) and completed F raw NLL versus actual
whole-checkpoint bytes. Each cohort is plotted independently; interrupted or
reused C/D attempts are never pooled. `figures/provenance.json` includes source
hashes and exact plotted values. PNGs are embedded in the manuscript. Missing F
evidence or unavailable plotting packages yield explicit unavailable flags,
never invented points. Numerical FP8/full-rank factor controls are excluded from
the whole-model storage figure. Reinvoking the companion on an existing figures
directory refuses overwrite; rerun the finalizer to create a fresh report.

The generator also creates two figures per available architecture attempt:
primary native-test teacher-relative NLL by method, and validation-only
teacher-relative NLL versus the actual retained target parameter fraction.
Both show individual seed measurements and mean/sample seed SD when available.
The denominator is dense targeted weight parameters; the numerator is the sum
of rank × (out+in), checked against exact saved recipes and metadata. This is
not whole-model savings or an ambiguous rank compression factor. Models,
tokenizers, splits, rank recipes, attempts and seeds never pool. Validation
points require `validation-teacher.json` with its matching native split hash
and count. Test points use their same-run native-test teacher.

Architecture plots may display partial saved rows, clearly labeled **PARTIAL**
with the observed attempt state and per-point seed count. They do not imply
missing seeds, unmeasured frontier points or a completed confirmation. Existing
Gemma C/F plots still require completed cohorts. Exact parameters, per-seed
values, source hashes and status remain in the figure provenance JSON.
Validation frontiers are development evidence, never independent confirmation.

The resumed ledger is `revised-orchestration-20261014-4`; ledger 3 and its stale
running status remain preserved after the third reboot. The formal scope closure
in `CONTINUATION_20261014.md` omits two unfinished dense FP8 controls while
preserving all seven saved controls and their charged cancelled attempt. The
report embeds this declaration and does not present nine controls as completed.

After the final manuscript appends and figures, the finalizer calls
`export_paper_latex_20261014.export_report(report_directory)` and
`export_paper_pdf_20261014.export_pdf(report_directory)`. These create separate
`manuscript.tex` and `manuscript.pdf` review artifacts, refusing overwrite. The
PDF uses the existing system interpreter's ReportLab and is **not compiled
from LaTeX**. A successful source export does not establish TeX compilation.
`export-status.json` records the prepared/failed state of each artifact, its
absolute path, bytes, SHA256 and exporter source hash. Compilation verification
remains false until a separate compiler tool succeeds on that exact source.

An absent exporter/library or export exception is an explicit reporting failure:
the finalizer attempts both formats, records the individual errors, preserves
any successfully prepared format and exits nonzero. It does not relabel GPU
experiments, erase the report or silently claim PDF availability. A failed
export can be retried through a new finalizer invocation/new output directory.

## Evidence and interpretation

- Each attempt stays separate, including interrupted/retried composition,
  BF16/FP8 controls, architecture cohorts, and practical measurements. Resumed
  records are not counted again as independent replications across attempts.
- Full recorded contracts retain exact source hashes, environment/package
  versions, model/data revisions, split/calibration manifests, targets/ranks,
  placement, selection hashes and existing runtime charges. Top-level artifact
  hashes and existing integrity manifests are checked. This preserves recorded
  provenance; it does not invent missing contract fields or claim to reconstruct
  an unrecorded historical environment.
- Missing/malformed individual files produce exploratory read-error entries.
  Such attempts also enter the accounting audit as unknown cost. One damaged
  historical JSON file cannot abort every other report.
- Paired NLL intervals use 2,000 resamples of complete contiguous 4,096-target
  blocks. The final partial block remains in native-test NLL but is excluded
  from fixed-size resampling, with the excluded count reported explicitly.
  Block intervals do not replace calibration-seed uncertainty.
- Exact canonical aliases are merged only with matching complete metrics and
  nonempty checkpoint content hashes. Conflicts are preserved and excluded
  from seed aggregates. A method is not compared independently against its
  own canonical alias.
- Group/split/rank outcomes remain separate. Three paired seeds with worse NLL
  and worse drift weaken only that scoped candidate hypothesis. Mixed NLL and
  drift behavior remains inconclusive. Full-stack NLL gains cannot establish
  uniform gains over small groups or solve accumulated drift.
- The amended per-model rank frontier has 18 validation-only measurements
  (three seeds × two methods × divisors 2/3/4), separate from eight primary-test
  measurements. `rank_divisor` and exact `recipes` are part of grouping identity;
  neither different ranks nor validation/test observations enter one aggregate.
- BF16/FP8 controls and hybrid panel results never repair the original FP16
  readiness failure retroactively. FP8 storage/decode controls are distinct
  from native model-level FP8 acceleration. CPU offload and resident-GPU
  measurements remain separate.
- Hybrid targets cover the six full-attention outputs per new model; DeltaNet
  and convolution mixers remain dense. Raw PPL is never used to rank models
  with different tokenizers.
- Promotion is always disabled. The research-wide hypothesis remains
  inconclusive pending all predeclared gates, fresh confirmation and practical
  evidence. An exploratory direction or a successful numerical control cannot
  establish a universal/practical compression advantage.

## Budgets and reboot limitations

Ledger 4 names an external interruption-accounting sidecar. The finalizer binds
each charge to the exact source run, model, stage, original artifact hashes and
finite launch-to-next-boot interval before using it. Original status files remain
unchanged; the report preserves their observed state and separately labels the
validated interrupted state. An invalid sidecar is a visible read error and
leaves the old attempt's cost unknown. Recovery contracts identify reused source
records, which remain separate from the predecessor attempt and must not be
counted as independent replications. The third-reboot Qwen charge is
3,647.1586377620697 wall seconds, including the reboot gap.

The audit separately sums retained terminal elapsed charges for readiness
(including SVD and precision diagnosis), regularization, baselines, composition,
the shared replacement architecture allocation, and practical execution. It
includes failed/retried attempts. Recorded architecture charges are also broken
out by model, with the declared two-hour caps applied only to Qwen and LFM;
any preserved GPT-2 initialization contributes to the shared four-hour allocation.

The first reboot's charge is the last observed logged elapsed time plus a
60-second allowance, which does not bound an unknown unlogged tail. The second
reboot recovery uses launch-to-new-boot wall time, including restart downtime.
For a terminal legacy attempt lacking elapsed status, the finalizer uses the
same documented fallback as the original `spent_seconds`: last minus first ISO
log timestamp plus a 60-second allowance. This is explicitly marked estimated.
Missing/invalid logs leave cost unknown; they are never silently zeroed. Thus
the report explicitly declines an exact 24 GPU-hour
compliance claim even if recorded charges fit nominal caps. Existing accounting
caveats are copied verbatim into structured evidence.

## End-to-end integration findings

The original `report_study(config, "local-core")` accepts precision metric
dictionaries through its non-comparison branch and panel/orchestration ledgers
with no metrics as empty exploratory entries. Architecture-replication lists
can produce within-model method summaries, but its generic adapter cannot
establish the amended architecture readiness/hardware gates. The original
`readiness_passed=false` correctly prevents promotion.

The earlier alias-duplication warning is not a reproduced failure of the current
generic reporter. On 14 October, its CPU-only `report_study` invocation parsed
the actual retained runs and returned zero, creating
`results/paper-v2/report-20261014-101240-f87d273a`. Current source already reduces
canonical aliases. Do not reapply an old repair patch based on historical notes.
An unreadable JSON file can still interrupt the generic report; the independent
finalizer isolates malformed artifacts. Any actual reporting failure must stay
separate from experiment completion and must not trigger another GPU run.

A panel cohort reaching its own budget cap must remain incomplete. That failure
does not invalidate already-frozen independent Gemma practical work. The parent
driver was amended separately to preserve precision/panel budget-limited status
while allowing the authorized independent Stage F under its own unchanged cap.
The finalizer reports both facts and never converts incomplete panel evidence
into a completed cohort.

## Verification

```powershell
rtk C:/Python314/python.exe -m unittest docs.test_report_revised_study_20261014 docs.test_export_paper_latex_20261014 docs.test_export_paper_pdf_20261014
```

CPU tests cover partial-block bootstrap, alias conflict handling, frontier scope separation, terminal-state
guarding, malformed-run isolation, unknown-cost accounting and preservation of
existing statuses, drift aggregation and finite-data checks, missing-figure
handling, and actual PNG/SVG rendering when matplotlib is already available.
They create synthetic fixtures only in a temporary test
directory; no synthetic experiment results are inserted into the project.

The combined reporting/export suite has 27 passing tests after adding validated
interruption-sidecar rejection checks (wrong identity, source hashes and charge)
and a scoped negative-result prose check. Primary-test negative prose requires
the three-seed joint-worsening flag plus matched targets and factor budgets;
validation-only or unmatched observations cannot trigger that statement.

Practical reporting additionally derives same-bit anchor/whitening cost ratios
only within a run, seed, factor precision, compute precision, GPU/device,
placement and exact workload (including prompt/decode lengths, batch, cache,
warmups and measurement count). Missing or ambiguous pairs are explicitly
excluded. This does not change the original hardware gates, whose reference is
same-seed FP16 whitening even for INT8 solver factors. The manuscript records
decode-only timing, allocator-only memory, limited reload-probe coverage and
INT8 dequantization caveats. Pairing tests reject cross-condition and duplicate
measurements instead of making a cost claim from them.

For readable architecture presentation, a unique continuation can replace its
interrupted predecessor in the main manuscript table, findings and figures only
after verifying explicit `independent_replication: false`, model/tokenizer/data
pins, split manifests, targets, baseline, source hashes, the reuse count and an
exact unchanged source-record prefix. Partial successors remain explicitly
partial. Invalid or ambiguous recoveries retain all attempts with warnings.
Original attempts remain separate in report JSON/Markdown and the manuscript
appendix, including interruption state, coverage, charged cost and source
hashes. Figure provenance lists each suppressed duplicate and its validated
continuation; no independent replication or pooled result is created.

# Reporting alias repair: apply only after the active study exits

> Historical repair proposal. The current generic reporter already reduces
> canonical aliases, and a CPU-only invocation against actual retained artifacts
> succeeded on 14 October 2026, producing `report-20261014-101240-f87d273a`.
> Do not apply this patch again merely because this proposal is retained.

Status: proposed reporting repair. **Do not modify or reload frozen `src` files
while the serialized experiment process is active.** Preserve its terminal
status even if the final reporting command fails. Recover reporting afterward
in a fresh, uniquely named output directory, retaining raw stage artifacts.

## Confirmed failure

`canonical_specification` intentionally maps exact anchored beta=0/ridge=0 to
whitening. Stage C can execute both the whitening component and the selected
anchor at beta=0, creating two observations per seed in the same canonical
group. Stage D/E may likewise execute solver and baseline labels that describe
the same method. `summarize_seeds` rejects duplicate seeds, so final reporting
can fail although raw experiments completed. That rejection is appropriate for
independent seed summaries; grouping must handle aliases before calling it.

## Required repair

1. Keep every raw result identifier, label, specification, checkpoint path and
   recorded factor hash in an `alias_evidence` list.
2. Within one canonical specification and identical group/rank/split scope,
   merge repeated seeds **only when** the complete metric dictionaries agree,
   actual target counts and NLL blocks agree, and their recorded checkpoint
   content hashes agree. For direct serialized checkpoint payloads, identical
   SHA256 establishes equal bytes; path names alone are insufficient evidence.
   Require a nonempty recorded checkpoint hash. Unavailable hashes do not pass.
3. When repeated aliases disagree, retain all raw evidence and flag
   `conflicting_alias_evidence`. Exclude that seed from aggregate/promotion
   calculations; do not pick the first observation or average them as extra
   seeds. The affected method and comparisons stay exploratory.
4. A canonical method cannot be its own independent baseline. Report the
   equivalent labels/components once, with an equivalence note, and omit a
   purported solver-versus-baseline advantage for identical canonical methods.
5. Practical timing records are separate repeat measurements. Keep baseline
   and solver labels in their report scope when timing records exist; do not
   require their naturally variable timings to be byte-identical, and do not
   present equal-method labels as different algorithms. Stage-F alias timings
   remain descriptive evidence, never an independent method advantage.
6. Canonicalize Stage C/F method keys consistently in hardware joins. Preserve
   both labels if canonical method and seed coincide. A later hardware gate
   must identify the chosen raw record/label explicitly; dictionary overwrite
   cannot select a measurement silently.

The adjacent proposed patch implements steps 1–5 for quality grouping. It also
records ambiguity instead of allowing silent canonical Stage-F key overwrite.
Run focused regression tests against the patched module in a separate CPU
process before replacing the frozen source after study completion. Then run
the existing reporting tests plus the cases below.

## Acceptance cases

- Two canonical alias records with seed 11, identical complete metrics, equal
  nonempty factor SHA256, and equal parameter counts produce one summarized
  seed and two preserved alias source records.
- A differing NLL, target count, NLL block, factor SHA256 or parameter count
  marks a conflict. Missing SHA256 also marks a conflict.
- Three matching seed pairs remain three seeds, not six; they cannot establish
  an advantage over their own canonical baseline.
- Composition groups, frontier ranks, evaluation splits and factor bit widths
  remain separate. Pairing may occur only at equal scopes.
- Practical baseline/solver timing labels remain separate descriptive rows.
- Stage-F canonical alias collisions are recorded and disable ambiguous joins.
- A failed run stays failed; a report never overwrites any old metrics or status.

## Other report hazards found in read-only review

- `report_study` scans active/failed/incomplete historical runs. Concurrent
  `metrics.json` writes or malformed JSON can raise `JSONDecodeError` and abort
  the entire report. Recovery should run after the active process exits and
  isolate per-run read errors as explicit exploratory entries.
- Stage-F hardware join originally keys on raw specification while quality
  groups use canonical specification. Anchored-beta-zero aliases can miss the
  intended join; canonicalize both sides and preserve collision evidence.
- Composition records lack `metadata.factor_parameters`. Quality pairing can
  still report NLL uncertainty, but the current promotion check cannot establish
  the actual group budget; leave it exploratory unless exact source-factor
  metadata and group accounting are recovered.
- Dense teacher in replication has no rank scope, while compressed rows do.
  This properly prevents promotion against a dense teacher at an unmatched
  factor budget, but teacher-relative quality should still be shown descriptively.
- Aggregate run/study classification and claim are hardcoded exploratory/no
  established advantage. That is conservative but can contradict a future
  valid promoted comparison; derive overall wording only from fully audited
  comparisons and never from partial method summaries.
- The report adapter requires readiness evidence. Existing historical contracts
  lacking it intentionally remain exploratory; do not retrofit their provenance.
- A complete hardware timing dictionary may hide duplicate condition rows due
  to keyed overwrite. Require exactly four unique declared timing rows before
  using a gate. Reject extra/duplicate rows explicitly.

These are reporting limitations, not reasons to rerun completed GPU comparisons
or change frozen research settings.

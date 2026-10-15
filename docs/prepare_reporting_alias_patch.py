"""Prepare a reporting-only patch without touching active research sources.

Run with rtk python docs/prepare_reporting_alias_patch.py after reviewing the
proposal. This generator writes a new patch artifact with exclusive creation.
Applying the patch is a separate action permitted only after the study exits.
"""
from pathlib import Path
import difflib
import hashlib


ALIAS_HELPER = '''def reduce_alias_records(records):
    """Deduplicate proven identical quality aliases, retaining every source."""
    by_seed = {}
    for record in records:
        by_seed.setdefault(record.get("seed"), []).append(record)
    clean, evidence = [], []
    for seed, values in by_seed.items():
        sources = [{"id": v.get("id"), "label": v.get("label"),
                    "specification": v.get("specification"),
                    "checkpoint": v.get("checkpoint", v.get("source_checkpoint")),
                    "checkpoint_sha256": v.get("metadata", {}).get("sha256") or
                        v.get("checkpoint_sha256") or v.get("source_sha256")}
                   for v in values]
        def fingerprint(record):
            factor_hash = record.get("metadata", {}).get("sha256") or record.get("checkpoint_sha256") or record.get("source_sha256")
            quality = record.get("metrics")
            if not factor_hash or not isinstance(quality, dict) or not quality:
                return None
            return {"factor_hash": factor_hash, "metrics": quality,
                    "factor_parameters": record.get("metadata", {}).get("factor_parameters"),
                    "serialized_bytes": record.get("metadata", {}).get("serialized_bytes"),
                    "checkpoint_sha256": record.get("checkpoint_sha256"),
                    "group": record.get("group"), "rank": record.get("rank"), "split": record.get("split")}
        reference = fingerprint(values[0])
        conflict = len(values) > 1 and (reference is None or any(fingerprint(v) != reference for v in values[1:]))
        if not conflict:
            clean.append(values[0])
        evidence.append({"seed": seed, "records": len(values), "sources": sources,
                         "equivalent_aliases": len(values) > 1 and not conflict,
                         "conflicting_alias_evidence": conflict,
                         "raw_conflicting_records": values if conflict else []})
    return clean, evidence


'''


def proposed_source(source):
    changed = source
    marker = 'def build_run_report(directory, *, practical_directory=None):'
    assert marker in changed
    changed = changed.replace(marker, ALIAS_HELPER + marker, 1)
    old = 'scope = {name: record[name] for name in ("group", "rank", "split", "bits") if name in record}'
    new = old + '\n        if "timing" in record and "label" in record:\n            scope["timing_label"] = record["label"]'
    assert old in changed
    changed = changed.replace(old, new, 1)
    old = '    normalized = normalize_contract(contract)\n'
    new = '    alias_evidence = {}\n    for key in list(grouped):\n        grouped[key], alias_evidence[key] = reduce_alias_records(grouped[key])\n' + old
    assert old in changed
    changed = changed.replace(old, new, 1)
    old = '            if key == baseline_key:\n                continue\n'
    new = '            if key == baseline_key or not records or not baseline_records:\n                continue\n'
    assert old in changed
    changed = changed.replace(old, new, 1)
    old = '            candidate_flat, baseline_flat = enriched(records, key), enriched(baseline_records, baseline_key)\n'
    new = '            alias_conflict = any(e["conflicting_alias_evidence"] for k in (key, baseline_key) for e in alias_evidence[k])\n            if alias_conflict:\n                matched["comparable"] = False\n                matched["mismatched"] = [*matched["mismatched"], "conflicting_alias_evidence"]\n' + old
    assert old in changed
    changed = changed.replace(old, new, 1)
    old = '            "comparisons": comparisons, "method_summaries": {\n'
    new = '            "alias_evidence": alias_evidence,\n            "equivalence_note": "Canonical aliases are not independent method comparisons; conflicts excluded from aggregates.",\n' + old
    assert old in changed
    changed = changed.replace(old, new, 1)
    changed = changed.replace('key = (json.dumps(record.get("specification", {}), sort_keys=True), record.get("seed"))',
                              'key = (json.dumps(canonical_specification(record.get("specification", {})), sort_keys=True), record.get("seed"))', 1)
    old = '        source = next((r for r in global_records if r.get("seed") == record.get("seed") and\n                       r.get("specification") == record.get("specification")), None)'
    new = '        source = next((r for r in global_records if r.get("seed") == record.get("seed") and\n                       canonical_specification(r.get("specification", {})) == canonical_specification(record.get("specification", {}))\n                       and r.get("metadata", {}).get("sha256") == record.get("source_sha256")), None)'
    assert old in changed
    changed = changed.replace(old, new, 1)
    changed = changed.replace('timing_complete = set(timings) == expected_conditions and all(',
                            'timing_complete = len(record.get("timing", [])) == 4 and set(timings) == expected_conditions and all(', 1)
    old = '        joined[key] = {"matched": matched["comparable"]'
    new = '        if key in joined:\n            joined[key]["matched"] = False\n            joined[key].setdefault("ambiguous_alias_labels", [joined[key].get("label")]).append(record.get("label"))\n            joined[key].setdefault("ambiguous_alias_records", []).append(record)\n            continue\n        joined[key] = {"label": record.get("label"), "matched": matched["comparable"]'
    assert old in changed
    changed = changed.replace(old, new, 1)
    return changed


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    path = root/"src/paper_reporting.py"
    source = path.read_text(encoding="utf-8")
    proposed = proposed_source(source)
    compile(proposed, str(path), "exec")
    destination = root/"docs/report_alias_repair.patch"
    patch = ''.join(difflib.unified_diff(source.splitlines(keepends=True), proposed.splitlines(keepends=True),
                                       fromfile="a/src/paper_reporting.py", tofile="b/src/paper_reporting.py"))
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(patch)
    print({"patch": str(destination), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "source_modified": False})

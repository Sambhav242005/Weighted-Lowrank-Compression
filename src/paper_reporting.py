"""Conservative protocol-v2 comparisons; no fitting or asset access.

Bootstrap inputs are ordered contiguous blocks produced by paper_protocol.evaluate.
The final partial block contributes to the full NLL difference, but is excluded
from bootstrap resampling so every sampled block has exactly 4096 targets.
Intervals quantify evaluation-block variability, not training-seed variability.
"""
from __future__ import annotations

import math
import random
import statistics
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from collections.abc import Mapping


MATCH_FIELDS = (
    "protocol", "model_revision", "dataset_revision", "split_hashes",
    "calibration_token_hash", "target_modules", "ranks", "factor_precision",
    "factor_parameters", "evaluation",
    "readiness_passed", "readiness_evidence",
    "calibration_positions",
)
METRICS = ("nll", "logit_kl", "primary_drift", "serialized_bytes",
           "latency_median_ms", "peak_memory_bytes")


def validate_matched_contracts(candidate, baseline, *, fields=MATCH_FIELDS):
    """Return evidence flags; missing provenance never silently passes."""
    missing = [key for key in fields if key not in candidate or key not in baseline
               or candidate.get(key) is None or baseline.get(key) is None]
    mismatched = [key for key in fields if key not in missing
                  and candidate[key] != baseline[key]]
    ready = candidate.get("readiness_passed") is True and baseline.get("readiness_passed") is True
    if "readiness_passed" in fields and not ready and "readiness_passed" not in missing:
        mismatched.append("readiness_passed")
    return {"comparable": not missing and not mismatched, "missing": missing,
            "mismatched": mismatched, "readiness_passed": ready,
            "classification": "comparable" if not missing and not mismatched else "exploratory"}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _blocks(values, block_size):
    values = list(values)
    if not values:
        raise ValueError("empty NLL observations")
    if isinstance(values[0], Mapping):
        blocks = [(v["nll_sum"], v["targets"]) for v in values]
    else:
        if not all(_number(v) and v >= 0 for v in values):
            raise ValueError("NLL must be finite and nonnegative")
        blocks = [(math.fsum(values[i:i + block_size]), len(values[i:i + block_size]))
                  for i in range(0, len(values), block_size)]
    for index, (total, count) in enumerate(blocks):
        if not _number(total) or total < 0 or not isinstance(count, int) or isinstance(count, bool):
            raise ValueError("invalid NLL block")
        if count <= 0 or count > block_size or (index < len(blocks) - 1 and count != block_size):
            raise ValueError("blocks must be contiguous fixed-size blocks plus a final partial block")
    return blocks


def _quantile(sorted_values, fraction):
    pos = (len(sorted_values) - 1) * fraction
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (pos - lower)


def paired_block_bootstrap(candidate_nll, baseline_nll, *, block_size=4096,
                           resamples=2000, seed=11, confidence=0.95):
    """Paired candidate-minus-baseline NLL percentile interval.

    Accept per-target losses or evaluate() nll_blocks. Contract/split validation
    is separate: equal counts alone do not establish the same token identities.
    """
    if block_size != 4096 or resamples != 2000:
        raise ValueError("protocol-v2 requires 4096-target blocks and 2000 resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    candidate, baseline = _blocks(candidate_nll, block_size), _blocks(baseline_nll, block_size)
    if [n for _, n in candidate] != [n for _, n in baseline]:
        raise ValueError("paired blocks have different target boundaries")
    counts = [n for _, n in candidate]
    differences = [a[0] - b[0] for a, b in zip(candidate, baseline)]
    estimate = math.fsum(differences) / sum(counts)
    total_targets = sum(counts)
    omitted = counts[-1] if counts[-1] < block_size else 0
    if omitted:
        counts, differences = counts[:-1], differences[:-1]
    if not counts:
        return {"difference": estimate, "confidence_interval": None,
                "confidence": confidence, "block_size": block_size, "blocks": 0,
                "targets": total_targets, "resamples": resamples, "seed": seed,
                "direction": "candidate_minus_baseline", "sufficient_blocks": False,
                "bootstrap_targets": 0, "excluded_partial_targets": omitted,
                "established_nll_advantage": False,
                "interpretation": "no full 4096-target blocks; uncertainty unavailable"}
    generator = random.Random(seed)
    draws = []
    for _ in range(resamples):
        selected = [generator.randrange(len(counts)) for _ in counts]
        draws.append(math.fsum(differences[i] for i in selected) / sum(counts[i] for i in selected))
    draws.sort()
    tail = (1 - confidence) / 2
    interval = [_quantile(draws, tail), _quantile(draws, 1 - tail)]
    enough = len(counts) >= 2
    return {"difference": estimate, "confidence_interval": interval,
            "confidence": confidence, "block_size": block_size, "blocks": len(counts),
            "targets": total_targets, "resamples": resamples, "seed": seed,
            "bootstrap_targets": sum(counts), "excluded_partial_targets": omitted,
            "bootstrap_difference": math.fsum(differences) / sum(counts),
            "direction": "candidate_minus_baseline", "sufficient_blocks": enough,
            "established_nll_advantage": enough and interval[1] < 0,
            "interpretation": "evaluation-block uncertainty; seed variation reported separately"}


def summarize_seeds(records, *, expected_seeds=(11, 23, 37)):
    records = list(records)
    seeds = [r.get("seed") for r in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed records")
    summary = {}
    for metric in METRICS:
        values = [r.get(metric) for r in records]
        finite = [v for v in values if _number(v)]
        summary[metric] = {"mean": statistics.mean(finite) if finite else None,
                           "std": statistics.stdev(finite) if len(finite) > 1 else None,
                           "count": len(finite), "complete": len(finite) == len(expected_seeds)}
    complete = set(seeds) == set(expected_seeds) and len(seeds) == len(expected_seeds)
    return {"seeds": seeds, "expected_seeds": list(expected_seeds), "complete": complete,
            "per_seed": records, "metrics": summary}


def assess_promotion(candidate_records, baseline_records, uncertainty=None, *, contracts=None,
                     hardware_limit=0.05):
    """Require matched evidence, all three seeds, drift and practical-cost gates.

    NLL advantage requires a negative block-bootstrap upper bound for each seed.
    KL can independently satisfy global improvement when an explicitly supplied
    established_kl_advantage flag reports an appropriate uncertainty assessment.
    """
    candidate, baseline = summarize_seeds(candidate_records), summarize_seeds(baseline_records)
    missing = []
    if not candidate["complete"] or not baseline["complete"]:
        missing.append("three paired seeds")
    if contracts is None or not contracts.get("comparable"):
        missing.append("matched run contracts")
    if contracts is None or not contracts.get("readiness_passed"):
        missing.append("passed readiness with recorded evidence")
    c = {r["seed"]: r for r in candidate["per_seed"]}
    b = {r["seed"]: r for r in baseline["per_seed"]}
    seeds = (11, 23, 37)
    paired = all(s in c and s in b for s in seeds)
    checks = {}
    for metric in METRICS:
        present = paired and all(_number(c[s].get(metric)) and _number(b[s].get(metric)) for s in seeds)
        if not present:
            missing.append(metric)
        if metric in ("nll", "logit_kl", "primary_drift"):
            checks[metric] = present and all(c[s][metric] < b[s][metric] for s in seeds)
        else:
            limit = 0.0 if metric == "serialized_bytes" else hardware_limit
            checks[metric] = present and all(b[s][metric] > 0 and c[s][metric] <= b[s][metric] * (1 + limit)
                                             for s in seeds)
    uncertainty = uncertainty or {}
    nll_established = all(uncertainty.get(s, uncertainty.get(str(s), {})).get("established_nll_advantage", False)
                          for s in seeds)
    kl_established = all(uncertainty.get(s, uncertainty.get(str(s), {})).get("established_kl_advantage", False)
                        for s in seeds)
    practical = all(checks[m] for m in ("serialized_bytes", "latency_median_ms", "peak_memory_bytes"))
    global_gate = (checks["nll"] and nll_established) or (checks["logit_kl"] and kl_established)
    if not global_gate:
        missing.append("established global advantage under uncertainty")
    promoted = not missing and global_gate and checks["primary_drift"] and practical
    return {"promoted": promoted, "hypothesis": "supported" if promoted else "inconclusive",
            "classification": "promising" if promoted else "exploratory",
            "checks": checks, "global_gate": global_gate, "hardware_regression_limit": hardware_limit,
            "missing_evidence": missing,
            "decision": "advance" if promoted else "no established advantage within uncertainty"}


def normalize_contract(contract):
    """Adapt the additive runner's nested contract without inventing provenance."""
    config = contract.get("configuration", {})
    splits = contract.get("splits", {})
    targets = contract.get("targets", [])
    return {"protocol": config.get("protocol"), "model_revision": config.get("model_revision"),
            "dataset_revision": config.get("dataset_revision"),
            "split_hashes": {s: splits.get(s, {}).get("hash") for s in ("train", "validation", "test")}
                if all(splits.get(s, {}).get("hash") for s in ("train", "validation", "test")) else None,
            "calibration_token_hash": splits.get("calibration_token_hashes", splits.get("calibration_hashes")),
            "calibration_positions": splits.get("calibration_positions"),
            "target_modules": [{"name": t.get("name"), "shape": t.get("shape")} for t in targets]
                if targets and all(t.get("name") and t.get("shape") for t in targets) else None,
            "ranks": [t["rank"] for t in targets] if targets and all(t.get("rank") is not None for t in targets) else None,
            "factor_parameters": sum(t["factor_parameters"] for t in targets)
                if targets and all(t.get("factor_parameters") is not None for t in targets) else None,
            "factor_precision": config.get("precision"),
            "readiness_passed": config.get("readiness_passed"),
            "readiness_evidence": config.get("readiness_evidence"),
            "evaluation": {k: config[k] for k in ("context", "stride", "evaluation_split", "selection_split")}
                if all(k in config for k in ("context", "stride", "evaluation_split", "selection_split")) else None}


def _flatten_record(record):
    metric, metadata = record.get("metrics", {}), record.get("metadata", {})
    return {"seed": record.get("seed"), **{m: metric.get(m) for m in METRICS[:3]},
            "serialized_bytes": metadata.get("serialized_bytes"),
            "latency_median_ms": metadata.get("latency_median_ms"),
            "peak_memory_bytes": metadata.get("peak_memory_bytes")}


def canonical_specification(specification):
    """Merge the exact unanchored component alias, retaining scope/settings."""
    spec = dict(specification)
    if spec.get("method") == "whitening" or (spec.get("method") == "anchored" and
            spec.get("beta") == 0 and spec.get("ridge_fraction", 0) == 0):
        spec.update(method="whitening", beta=0.0, ridge_fraction=0.0)
    return spec


def link_practical_evidence(global_directory, practical_directory):
    """Join Stage C/F only by pinned contracts, selection and source hashes."""
    def read(directory, name):
        return json.loads((Path(directory)/name).read_text(encoding="utf-8"))
    global_contract = read(global_directory, "contract.json")
    practical_contract = read(practical_directory, "contract.json")
    matched = validate_matched_contracts(normalize_contract(global_contract), normalize_contract(practical_contract))
    selection_match = bool(global_contract.get("selection_hash")) and global_contract.get("selection_hash") == practical_contract.get("selection_hash")
    complete = read(practical_directory, "status.json").get("state") == "completed"
    practical = read(practical_directory, "metrics.json")
    global_records = read(global_directory, "metrics.json")
    joined = {}
    expected_conditions = {(mode, length) for mode in ("prefill", "decode") for length in (128, 512)}
    for record in practical:
        if record.get("bits") != 16 or record.get("label") not in ("baseline", "solver"):
            continue
        key = (json.dumps(canonical_specification(record.get("specification", {})), sort_keys=True), record.get("seed"))
        source = next((r for r in global_records if r.get("seed") == record.get("seed") and
                       canonical_specification(r.get("specification", {})) == canonical_specification(record.get("specification", {}))
                       and r.get("metadata", {}).get("sha256") == record.get("source_sha256")), None)
        source_match = source is not None and bool(record.get("source_sha256")) and record["source_sha256"] == source.get("metadata", {}).get("sha256")
        timings = {(r.get("mode"), r.get("prefill_tokens")): r for r in record.get("timing", [])}
        timing_complete = len(record.get("timing", [])) == 4 and set(timings) == expected_conditions and all(
            t.get("batch_size") == 1 and t.get("warmups") == 10 and t.get("measurements") == 30
            and t.get("use_cache") is True and t.get("decode_tokens") == (128 if mode == "decode" else 0)
            and all(_number(t.get(m)) and t[m] > 0 for m in ("median_seconds", "peak_allocated_bytes", "peak_reserved_bytes"))
            for (mode, _), t in timings.items())
        primary = timings.get(("decode", 512), {})
        if key in joined:
            joined[key]["matched"] = False
            joined[key].setdefault("ambiguous_alias_labels", [joined[key].get("label")]).append(record.get("label"))
            joined[key].setdefault("ambiguous_alias_records", []).append(record)
            continue
        joined[key] = {"label": record.get("label"), "matched": matched["comparable"] and selection_match and complete and source_match and timing_complete,
                       "selection_hash_matches": selection_match, "source_hash_matches": source_match,
                       "timing_complete": timing_complete, "timings": timings,
                       "serialized_bytes": record.get("whole_checkpoint", {}).get("bytes"),
                       "factor_checkpoint_bytes": record.get("factor_checkpoint_bytes"),
                       "latency_median_ms": primary.get("median_seconds", 0)*1000 or None,
                       "peak_memory_bytes": primary.get("peak_allocated_bytes"),
                       "primary_hardware_condition": "decode 128 tokens from 512-token prefill"}
    return joined


def reduce_alias_records(records):
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


def build_run_report(directory, *, practical_directory=None):
    """Read retained artifacts; comparisons never cross unrelated run contracts."""
    directory = Path(directory)
    missing = []
    def read(name, default):
        path = directory / name
        if not path.is_file():
            missing.append(name)
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    status = read("status.json", {})
    contract = read("contract.json", {})
    raw = read("metrics.json", [])
    if not isinstance(raw, list):
        return {"run_id": directory.name, "state": status.get("state", "unknown"),
                "classification": "exploratory", "missing_evidence": missing,
                "note": "non-comparison stage; inspect native metrics", "comparisons": []}
    grouped = {}
    for record in raw:
        specification = canonical_specification(record.get("specification", {}))
        scope = {name: record[name] for name in ("group", "rank", "split", "bits") if name in record}
        if "timing" in record and "label" in record:
            scope["timing_label"] = record["label"]
        if scope:
            specification["_report_scope"] = scope
        key = json.dumps(specification, sort_keys=True)
        grouped.setdefault(key, []).append(record)
    alias_evidence = {}
    for key in list(grouped):
        grouped[key], alias_evidence[key] = reduce_alias_records(grouped[key])
    normalized = normalize_contract(contract)
    provenance = validate_matched_contracts(normalized, normalized)
    hardware = link_practical_evidence(directory, practical_directory) if practical_directory else {}
    baseline_keys = [key for key in grouped if json.loads(key).get("method") in ("asvd", "whitening")]
    selection_path = directory / "frozen-selection.json"
    if selection_path.exists():
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        spec = frozen.get("strongest_baseline", {}).get("specification")
        if spec:
            wanted = canonical_specification(spec)
            baseline_keys = [key for key in grouped if {k: v for k, v in json.loads(key).items()
                            if k != "_report_scope"} == wanted]
    comparisons = []
    for baseline_key in baseline_keys:
        baseline_records = grouped.get(baseline_key, [])
        for key, records in grouped.items():
            if key == baseline_key or not records or not baseline_records:
                continue
            if json.loads(key).get("_report_scope") != json.loads(baseline_key).get("_report_scope"):
                continue
            uncertainty = {}
            b = {r["seed"]: r for r in baseline_records}
            params_match = bool(records and baseline_records)
            for record in records:
                paired = b.get(record.get("seed"))
                if not paired:
                    continue
                candidate_params = record.get("metadata", {}).get("factor_parameters")
                baseline_params = paired.get("metadata", {}).get("factor_parameters")
                params_match = params_match and candidate_params is not None and baseline_params is not None and candidate_params == baseline_params
                blocks = record.get("metrics", {}).get("nll_blocks")
                reference = paired.get("metrics", {}).get("nll_blocks")
                if blocks and reference:
                    uncertainty[record["seed"]] = paired_block_bootstrap(blocks, reference)
            matched = dict(provenance)
            if not params_match:
                matched["comparable"] = False
                matched["mismatched"] = [*matched["mismatched"], "actual factor_parameters"]
            def enriched(source_records, specification_key):
                values = []
                for record in source_records:
                    flat = _flatten_record(record)
                    measured = hardware.get((specification_key, record.get("seed")), {})
                    if measured.get("matched"):
                        flat.update({m: measured.get(m) for m in ("serialized_bytes", "latency_median_ms", "peak_memory_bytes")})
                    values.append(flat)
                return values
            alias_conflict = any(e["conflicting_alias_evidence"] for k in (key, baseline_key) for e in alias_evidence[k])
            if alias_conflict:
                matched["comparable"] = False
                matched["mismatched"] = [*matched["mismatched"], "conflicting_alias_evidence"]
            candidate_flat, baseline_flat = enriched(records, key), enriched(baseline_records, baseline_key)
            gate = assess_promotion(candidate_flat, baseline_flat, uncertainty, contracts=matched)
            all_conditions = bool(practical_directory)
            for seed in (11, 23, 37):
                candidate_hw = hardware.get((key, seed), {})
                baseline_hw = hardware.get((baseline_key, seed), {})
                paired_hw = candidate_hw.get("matched") and baseline_hw.get("matched")
                all_conditions = all_conditions and bool(paired_hw)
                if paired_hw:
                    for condition, measured in candidate_hw["timings"].items():
                        reference = baseline_hw["timings"][condition]
                        all_conditions = all_conditions and all(measured[m] <= reference[m]*1.05
                            for m in ("median_seconds", "peak_allocated_bytes", "peak_reserved_bytes"))
                    candidate_bytes, baseline_bytes = candidate_hw.get("factor_checkpoint_bytes"), baseline_hw.get("factor_checkpoint_bytes")
                    all_conditions = all_conditions and _number(candidate_bytes) and _number(baseline_bytes) and candidate_bytes <= baseline_bytes
            if not all_conditions:
                gate.update(promoted=False, hypothesis="inconclusive", classification="exploratory")
                gate["missing_evidence"].append("matched completed Stage-F storage and all four hardware conditions")
            gate["all_hardware_conditions_passed"] = bool(all_conditions)
            if status.get("state") != "completed" or directory.name.startswith("regularization-"):
                gate.update(promoted=False, hypothesis="inconclusive", classification="exploratory")
                gate["missing_evidence"].append("completed held-out test comparison")
            comparisons.append({"candidate": json.loads(key), "baseline": json.loads(baseline_key),
                                "candidate_summary": summarize_seeds(candidate_flat),
                                "baseline_summary": summarize_seeds(baseline_flat),
                                "uncertainty": uncertainty, "matched_contract": matched, "promotion": gate})
    return {"run_id": directory.name, "state": status.get("state", "unknown"),
            "practical_run_id": Path(practical_directory).name if practical_directory else None,
            "classification": "exploratory", "missing_evidence": missing,
            "alias_evidence": alias_evidence,
            "equivalence_note": "Canonical aliases are not independent method comparisons; conflicts excluded from aggregates.",
            "comparisons": comparisons, "method_summaries": {
                key: summarize_seeds([_flatten_record(r) for r in records]) for key, records in grouped.items()},
            "note": "fixed-rank ASVD and whitening-only SVD-LLM are component comparisons"}


def report_study(config, study="local-core"):
    """Create a fresh, reviewable report; never overwrite research results."""
    root = Path(config["output_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"study output root missing: {root}")
    directories = sorted(p for p in root.iterdir() if p.is_dir() and (p / "status.json").is_file()
                         and (study == "local-core" or p.name.startswith(study + "-")))
    practical_runs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("practical-")
                            and all((p/name).is_file() for name in ("contract.json", "status.json", "metrics.json")))
    run_reports = []
    for directory in directories:
        measured = None
        if directory.name.startswith("baselines-") and (directory/"contract.json").exists():
            own_contract = json.loads((directory/"contract.json").read_text(encoding="utf-8"))
            for candidate in practical_runs:
                other = json.loads((candidate/"contract.json").read_text(encoding="utf-8"))
                state = json.loads((candidate/"status.json").read_text(encoding="utf-8"))
                declared_source = other.get("configuration", {}).get("baseline_run_path")
                if state.get("state") == "completed" and declared_source and Path(declared_source).resolve() == directory.resolve() and own_contract.get("selection_hash") and own_contract["selection_hash"] == other.get("selection_hash"):
                    measured = candidate
                    break
        run_reports.append(build_run_report(directory, practical_directory=measured))
    report = {"study": study, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "runs": run_reports,
              "classification": "exploratory", "claim": "no established advantage within uncertainty"}
    output = root / ("report-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = ["# Study report: " + study, "", report["claim"], "",
            "| Run | State | Paired comparisons | Classification |", "|---|---|---:|---|"]
    rows.extend(f"| {r['run_id']} | {r['state']} | {len(r['comparisons'])} | {r['classification']} |" for r in report["runs"])
    rows.extend(["", "See report.json for per-seed metrics, uncertainty intervals, matching checks and missing evidence.",
                 "A report with incomplete drift, three-seed or hardware evidence cannot promote a candidate."])
    (output / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(str(output.resolve()))
    return 0

"""Read-only, stdlib finalization of retained original/revised study artifacts.

Run AFTER the serial dispatcher terminates. No source/config/run is rewritten.
Use --snapshot explicitly for an interim report; promotion is always disabled.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import uuid

TERMINAL = {"completed", "completed with incomplete cohorts", "failed", "stopped", "incomplete", "cancelled", "interrupted"}
PREFIXES = {"readiness": ("readiness-", "svd-diagnosis-", "precision-readiness-"),
            "regularization": ("regularization-",), "baselines": ("baselines-",),
            "composition": ("composition-",),
            "architecture": ("replication-", "architecture-replication-"), "practical": ("practical-",)}
CAPS = dict(readiness=3600, regularization=14400, baselines=25200, composition=14400,
            architecture=14400, practical=14400)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: "+value)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def apply_interruption_accounting(root, reports, path):
    """Bind immutable interrupted evidence to a separately recorded reboot charge."""
    root, path = Path(root).resolve(), Path(path).resolve()
    if path.parent != root:
        raise ValueError("interruption sidecar must be inside the results root")
    sidecar = read(path)
    if sidecar.get("schema") != "researchcompression-interruption-accounting-v1":
        raise ValueError("unsupported interruption sidecar schema")
    by_id = {r["run_id"]:r for r in reports}
    checked, seen = [], set()
    for entry in sidecar.get("runs", []):
        run_id = entry.get("run_id")
        record = by_id.get(run_id)
        if record is None or run_id in seen:
            raise ValueError("unknown or duplicated interruption run")
        seen.add(run_id)
        directory = Path(record.get("path", "")).resolve()
        if directory.parent != root or Path(entry.get("run", "")).resolve() != directory or directory.name != run_id:
            raise ValueError("interruption run path/identity mismatch")
        stage = entry.get("stage")
        if stage not in {"replication", "practical"} or not run_id.startswith(PREFIXES["architecture" if stage == "replication" else "practical"]):
            raise ValueError("interruption stage mismatch")
        model = record.get("contract", {}).get("configuration", {}).get("model")
        expected_state = "interrupted by reboot" if stage == "replication" else "cancelled by user"
        if not model or entry.get("model") != model or entry.get("state") != expected_state:
            raise ValueError("interruption model/state mismatch")
        endpoint = "new_boot_epoch" if stage == "replication" else "ended_epoch"
        start, boot, charge = (entry.get(k) for k in ("started_epoch", endpoint, "charged_seconds"))
        if not all(finite(v) for v in (start, boot, charge)) or charge < 0 or boot < start or abs((boot-start)-charge) > 1e-6:
            raise ValueError("invalid interruption charge/interval")
        hashes = entry.get("source_artifacts", {})
        if not {"status.json", "metrics.json", "contract.json", "config.json"}.issubset(hashes):
            raise ValueError("interruption source hashes incomplete")
        for name, expected in hashes.items():
            artifact = directory/name
            if Path(name).name != name or not artifact.is_file() or digest(artifact) != expected:
                raise ValueError("interruption source hash mismatch: "+name)
        if not entry.get("basis") or entry.get("independent_replication") is not False:
            raise ValueError("interruption provenance incomplete")
        if stage == "practical":
            stop = Path(entry.get("stop_evidence", "")).resolve()
            if not stop.is_relative_to(root) or not stop.is_file() or digest(stop) != entry.get("stop_evidence_sha256") or read(stop).get("epoch") != boot:
                raise ValueError("user-stop evidence mismatch")
        checked.append((record, entry))
    if not checked:
        raise ValueError("empty interruption sidecar")
    for record, entry in checked:
        record["observed_state"] = record["state"]
        record["state"] = "cancelled" if entry["stage"] == "practical" else "interrupted"
        record["interruption_accounting"] = {**entry, "sidecar_path":str(path), "sidecar_sha256":digest(path), "validated":True}
    return {"path":str(path), "sha256":digest(path), "validated_runs":sorted(seen)}


def canonical(spec):
    spec = dict(spec)
    if spec.get("method") == "whitening" or (spec.get("method") == "anchored" and
            spec.get("beta") == 0 and spec.get("ridge_fraction", 0) == 0):
        spec.update(method="whitening", beta=0.0, ridge_fraction=0.0)
    return spec


def bootstrap(candidate, baseline):
    """Exact protocol block units, excluding the final partial block only in CI."""
    if not candidate or not baseline or [b.get("targets") for b in candidate] != [b.get("targets") for b in baseline]:
        return {"available": False, "reason": "missing or mismatched paired NLL blocks"}
    counts = [b["targets"] for b in candidate]
    if any(not isinstance(n, int) or n <= 0 or n > 4096 or (i < len(counts)-1 and n != 4096)
           for i,n in enumerate(counts)):
        return {"available": False, "reason": "not fixed contiguous 4096-target blocks"}
    differences = [a.get("nll_sum", float("nan"))-b.get("nll_sum", float("nan")) for a,b in zip(candidate, baseline)]
    if not all(finite(x) for x in differences):
        return {"available": False, "reason": "nonfinite block values"}
    full = math.fsum(differences)/sum(counts)
    excluded = counts[-1] if counts[-1] < 4096 else 0
    if excluded:
        differences = differences[:-1]
    if len(differences) < 2:
        return {"available": False, "reason": "fewer than two complete blocks", "full_test_difference": full,
                "excluded_partial_targets": excluded}
    generator = random.Random(11)
    samples = sorted(math.fsum(generator.choice(differences) for _ in differences)/(4096*len(differences))
                     for _ in range(2000))
    def quantile(q):
        index = (len(samples)-1)*q
        lo = int(index)
        return samples[lo]+(samples[min(lo+1,len(samples)-1)]-samples[lo])*(index-lo)
    interval = [quantile(.025), quantile(.975)]
    return {"available": True, "resamples": 2000, "block_targets": 4096, "blocks": len(differences),
            "full_test_difference": full, "bootstrap_subset_difference": math.fsum(differences)/(4096*len(differences)),
            "excluded_partial_targets": excluded, "confidence_interval_95": interval,
            "direction": "candidate_minus_baseline", "nll_advantage_under_block_uncertainty": interval[1] < 0,
            "limitation": "evaluation-block variation, not training-seed uncertainty"}


def record_hash(record):
    return record.get("metadata", {}).get("sha256") or record.get("checkpoint_sha256") or record.get("source_sha256")


def practical_same_bit_costs(records, contract):
    """Separate descriptive ratios, never a replacement for the original gate."""
    config, env = contract.get("configuration", {}), contract.get("environment", {})
    paired, excluded, indexed = [], [], {}
    workload_keys = ("mode", "prefill_tokens", "decode_tokens", "batch_size", "warmups", "measurements", "use_cache")
    for row in records:
        method = canonical(row.get("specification", {})).get("method")
        if (row.get("label"),method) not in {("baseline","whitening"),("solver","anchored")}:
            continue
        # The practical runner moves the full model to configuration.device;
        # explicit row placement/device metadata, if supplied, takes precedence.
        device = row.get("device",config.get("device"))
        placement = row.get("placement",config.get("placement", "resident" if device and not config.get("device_map") and not config.get("offload_folder") else None))
        identity = {"seed":row.get("seed"),"bits":row.get("bits"),"device":device,"placement":placement,
                    "gpu":row.get("gpu",env.get("gpu")),"compute_precision":row.get("compute_precision",config.get("precision"))}
        if any(value is None for value in identity.values()) or identity["bits"] not in (8,16):
            excluded.append({"label":row.get("label"),"seed":row.get("seed"),"reason":"missing device/placement/precision or factor-bit identity"})
            continue
        for timing in row.get("timing",[]):
            if any(key not in timing for key in workload_keys):
                excluded.append({"label":row["label"],"seed":row["seed"],"reason":"incomplete timing workload"})
                continue
            scope={**identity, **{key:timing[key] for key in workload_keys}}
            key=json.dumps(scope,sort_keys=True)
            indexed.setdefault(key,{"scope":scope,"baseline":[],"solver":[]})[row["label"]].append((row,timing))
    for group in indexed.values():
        if len(group["baseline"]) != 1 or len(group["solver"]) != 1:
            excluded.append({"scope":group["scope"],"reason":"missing or ambiguous same-bit matched pair"})
            continue
        baseline, reference=group["baseline"][0]
        candidate, measured=group["solver"][0]
        ratios={}
        values={"median_latency":(measured.get("median_seconds"),reference.get("median_seconds")),
                "p95_latency":(measured.get("p95_seconds"),reference.get("p95_seconds")),
                "peak_allocated":(measured.get("peak_allocated_bytes"),reference.get("peak_allocated_bytes")),
                "peak_reserved":(measured.get("peak_reserved_bytes"),reference.get("peak_reserved_bytes")),
                "factor_bytes":(candidate.get("factor_checkpoint_bytes"),baseline.get("factor_checkpoint_bytes")),
                "whole_bytes":(candidate.get("whole_checkpoint",{}).get("bytes"),baseline.get("whole_checkpoint",{}).get("bytes"))}
        for name,(a,b) in values.items():
            ratios[name]=a/b if finite(a) and finite(b) and a>0 and b>0 else None
        paired.append({"scope":group["scope"],"ratios":ratios,"direction":"anchor divided by same-bit whitening",
                       "placement_basis":"Explicit row metadata when present; otherwise full-model resident placement declared by the same-run configuration.",
                       "hardware_gate":False})
    return {"comparisons":paired,"excluded":excluded,"interpretation":"Descriptive same-run same-seed same-bit ratios; below 1 is lower cost. Original hardware gates retain FP16-whitening reference for both solver storage formats."}


def presentation_supersession(root, reports):
    """Select a unique verified continuation for display, never pool attempts."""
    root=Path(root).resolve()
    by_path={Path(r["path"]).resolve():r for r in reports if r.get("path")}
    candidates, warnings={},[]
    for successor in reports:
        recovery=successor.get("contract",{}).get("recovery")
        if not recovery or not successor["run_id"].startswith("architecture-replication-"):
            continue
        source_path=Path(recovery.get("source_run","")).resolve()
        # Count even invalid branches: ambiguity must never silently pick one.
        candidates.setdefault(source_path,[]).append((successor,recovery))
    retained={}
    for source_path,branches in candidates.items():
        try:
            if len(branches)!=1:
                raise ValueError("ambiguous recovery successors")
            successor,recovery=branches[0]
            source=by_path.get(source_path)
            if source is None or source_path.parent!=root or source_path.name==successor["run_id"]:
                raise ValueError("unknown or invalid recovery source")
            if recovery.get("independent_replication") is not False:
                raise ValueError("recovery must be explicitly nonindependent")
            a,b=source.get("contract",{}),successor.get("contract",{})
            for key in ("model","model_revision","dataset_revision","tokenizer_revision"):
                value=a.get("configuration",{}).get(key)
                if value is None or value!=b.get("configuration",{}).get(key):
                    raise ValueError("recovery asset pin mismatch: "+key)
            for key in ("splits","targets","baseline"):
                if not a.get(key) or a[key]!=b.get(key):
                    raise ValueError("recovery metadata mismatch: "+key)
            hashes=recovery.get("source_artifacts",{})
            if not {"metrics.json","contract.json","config.json","status.json"}.issubset(hashes):
                raise ValueError("recovery source hashes incomplete")
            for name,expected in hashes.items():
                target=source_path/name
                if Path(name).name!=name or not target.is_file() or digest(target)!=expected:
                    raise ValueError("recovery source hash mismatch: "+name)
            old=read(source_path/"metrics.json")
            new=read(Path(successor["path"])/"metrics.json")
            if not isinstance(old,list) or not old or recovery.get("reused_records")!=len(old) or new[:len(old)]!=old:
                raise ValueError("recovery rows are not the exact declared source prefix")
            retained[source["run_id"]]={"successor":successor["run_id"],"source_state":source.get("state"),
                "successor_state":successor.get("state"),"reused_records":len(old),"independent_replication":False,
                "source_artifacts":hashes,"reason":"Unique validated continuation includes every original record unchanged; original attempt remains in the evidence and manuscript appendix."}
        except (ValueError,TypeError,OSError,KeyError) as exc:
            warnings.append({"source":str(source_path),"reason":str(exc),"action":"retain all attempts in primary presentation"})
    return {"superseded":retained,"warnings":warnings,"interpretation":"Display selection only; no pooling, deletion or independent replication from reused rows. Partial successors retain their actual state and coverage."}


def summarize(records):
    groups = {}
    for row in records:
        scope = {k: row[k] for k in ("group", "rank", "rank_divisor", "recipes", "target_recipes", "split", "bits") if k in row}
        # Hardware labels are repeated observations, not independent methods.
        if "timing" in row:
            scope["timing_label"] = row.get("label")
        key = json.dumps({"specification": canonical(row.get("specification", {})), "scope": scope}, sort_keys=True)
        groups.setdefault(key, {}).setdefault(row.get("seed"), []).append(row)
    result = []
    for key, seeds in groups.items():
        retained, aliases, conflicts = [], [], []
        for seed, values in seeds.items():
            first = values[0]
            same = len(values) == 1 or (bool(record_hash(first)) and all(record_hash(v) == record_hash(first)
                and v.get("metrics") == first.get("metrics") and v.get("metadata", {}).get("factor_parameters") == first.get("metadata", {}).get("factor_parameters") for v in values[1:]))
            if same:
                retained.append(first)
            else:
                conflicts.append({"seed": seed, "raw_records": values})
            if len(values) > 1:
                aliases.append({"seed": seed, "count": len(values), "identical_evidence": same,
                                "source_ids": [v.get("id", v.get("label")) for v in values]})
        metric_summary = {}
        for name in ("nll", "ppl", "logit_kl", "primary_drift", "teacher_relative_nll_difference", "teacher_relative_ppl"):
            values = [r.get("metrics", {}).get(name, r.get(name)) for r in retained]
            values = [v for v in values if finite(v)]
            metric_summary[name] = {"mean": statistics.mean(values) if values else None,
                                    "sample_std": statistics.stdev(values) if len(values)>1 else None, "count": len(values)}
        result.append({**json.loads(key), "seeds": [r.get("seed") for r in retained],
                       "three_seeds": {r.get("seed") for r in retained} == {11,23,37},
                       "metrics": metric_summary, "records": retained, "alias_evidence": aliases,
                       "conflicts": conflicts, "promotion_disabled": True})
    return result


def paired_comparisons(groups, contract, directory):
    baseline = contract.get("baseline")
    selection = directory/"frozen-selection.json"
    if selection.is_file():
        baseline = read(selection).get("strongest_baseline", {}).get("specification", baseline)
    if not isinstance(baseline, dict):
        return []
    baseline = canonical(baseline)
    output = []
    for reference in [g for g in groups if g["specification"] == baseline]:
        for candidate in groups:
            if candidate["specification"] == baseline or candidate["scope"] != reference["scope"]:
                continue
            if candidate["specification"].get("method") == "teacher":
                continue
            ref = {r.get("seed"):r for r in reference["records"]}
            rows = []
            for row in candidate["records"]:
                seed = row.get("seed")
                if seed not in ref:
                    continue
                a,b = row.get("metrics", {}),ref[seed].get("metrics", {})
                counts_match = a.get("targets") is not None and a.get("targets") == b.get("targets")
                factor_match = row.get("metadata", {}).get("factor_parameters") is not None and row.get("metadata", {}).get("factor_parameters") == ref[seed].get("metadata", {}).get("factor_parameters")
                rows.append({"seed": seed, "targets_match": counts_match, "actual_factor_parameters_match": factor_match,
                             "nll_difference": a["nll"]-b["nll"] if finite(a.get("nll")) and finite(b.get("nll")) else None,
                             "drift_difference": a["primary_drift"]-b["primary_drift"] if finite(a.get("primary_drift")) and finite(b.get("primary_drift")) else None,
                             "uncertainty": bootstrap(a.get("nll_blocks"), b.get("nll_blocks")) if counts_match else {"available":False}})
            paired_three = {r["seed"] for r in rows} == {11,23,37}
            nll_better = paired_three and all(finite(r["nll_difference"]) and r["nll_difference"]<0 for r in rows)
            drift_better = paired_three and all(finite(r["drift_difference"]) and r["drift_difference"]<0 for r in rows)
            jointly_worse = paired_three and all(finite(r["nll_difference"]) and r["nll_difference"]>0 and
                finite(r["drift_difference"]) and r["drift_difference"]>0 for r in rows)
            output.append({"candidate":candidate["specification"], "baseline":baseline, "scope":candidate["scope"],
                           "per_seed":rows, "paired_three_seeds":paired_three,
                           "nll_better_all_three":bool(nll_better),"drift_better_all_three":bool(drift_better),
                           "nll_and_drift_worse_all_three":bool(jointly_worse),
                           "hypothesis_status":"weakened" if jointly_worse else "inconclusive", "promotion_disabled":True,
                           "interpretation":"Exploratory within-model comparison; inspect readiness, provenance and hardware separately."})
    return output


def inspect_run(directory):
    files = {p.name:{"path":str(p.resolve()), "sha256":digest(p), "bytes":p.stat().st_size}
             for p in directory.iterdir() if p.is_file() and p.suffix in {"json", ".json", ".log", ".md"}}
    status = read(directory/"status.json")
    contract = read(directory/"contract.json") if (directory/"contract.json").is_file() else {}
    raw = read(directory/"metrics.json") if (directory/"metrics.json").is_file() else None
    record = {"run_id":directory.name, "path":str(directory.resolve()), "state":status.get("state"), "status":status,
              "artifacts":files, "contract":contract, "classification":"exploratory", "promotion_disabled":True}
    if isinstance(raw,list):
        record["method_groups"] = summarize(raw)
        record["comparisons"] = paired_comparisons(record["method_groups"],contract,directory)
        if directory.name.startswith("practical-"):
            record["same_bit_cost_comparisons"] = practical_same_bit_costs(raw,contract)
        if directory.name.startswith("architecture-replication-"):
            test=[r for r in raw if r.get("split")=="test"]
            validation=[r for r in raw if r.get("split")=="validation"]
            record["cohort_coverage"]={"primary_test_records":len(test),"required_primary_test_records":8,
                "frontier_validation_records":len(validation),"required_frontier_validation_records":18,
                "validation_rank_divisors":sorted({r["rank_divisor"] for r in validation if "rank_divisor" in r}),
                "record_counts_complete":len(test)==8 and len(validation)==18,
                "note":"Record-count checks do not replace seed, target, recipe and provenance checks; an earlier 8-record cohort has no amended frontier evidence."}
    elif isinstance(raw,dict):
        record["control_metrics"] = raw
    else:
        record["missing_evidence"] = ["metrics.json"]
    for name in ("readiness.json", "native-fp8-probe.json", "teacher-audit.json", "hardware-gates.json", "ledger.json", "accounting-caveat.json", "budget-audit.json", "token-manifests.json", "validation-teacher.json"):
        if (directory/name).is_file():
            record[name.removesuffix(".json")] = read(directory/name)
    integrity = directory/"artifact-integrity.json"
    if integrity.is_file():
        checks = []
        for name, expected in read(integrity).get("artifacts", {}).items():
            target = directory/name
            checks.append({"artifact":name, "matches":target.is_file() and digest(target)==expected})
        record["integrity_checks"] = checks
    if contract:
        if contract.get("recovery"):
            record["recovery_provenance"] = contract["recovery"]
            record["reuse_interpretation"] = "Successor includes preserved source observations; attempts remain separate and are not independent replications."
        record["provenance_gaps"] = [key for key in ("configuration", "environment", "splits", "targets") if not contract.get(key)]
        record["recorded_source_hashes"] = contract.get("implementation_hashes", contract.get("environment", {}).get("source_hashes"))
        test_tokens = contract.get("splits",{}).get("test",{}).get("tokens")
        expected = test_tokens-1 if isinstance(test_tokens,int) else None
        record["full_test_expected_targets"] = expected
        if isinstance(raw,list):
            record["test_coverage_checks"] = [{"seed":r.get("seed"),"specification":r.get("specification"),
                "targets":r.get("metrics",{}).get("targets"),"full_native_test":expected is not None and r.get("metrics",{}).get("targets")==expected}
                for r in raw if r.get("split","test")=="test" and r.get("metrics")]
    return record


def budget_audit(root, reports):
    stages = {s:{"cap_seconds":cap, "recorded_charged_seconds":0.0, "attempts":[], "unknown_cost_runs":[]} for s,cap in CAPS.items()}
    for record in reports:
        stage = next((s for s,prefixes in PREFIXES.items() if record["run_id"].startswith(prefixes)),None)
        if not stage:
            continue
        status = record.get("status", {})
        elapsed = status.get("elapsed_seconds")
        basis = "terminal status elapsed wall time"
        interruption = record.get("interruption_accounting", {})
        if interruption.get("validated"):
            status = {**status, "state":"interrupted"}
            elapsed, basis = interruption["charged_seconds"], interruption["basis"]
        if status.get("state") == "waiting for a human" or (status.get("phase") == "initialization" and status.get("observed_process_exit_code") == 1):
            elapsed,basis = 0.0,"pre-CUDA preparation/import failure excluded by original accounting"
        if elapsed is None and status.get("state") in TERMINAL and record.get("path"):
            # Match the original paper_study.spent_seconds legacy fallback.
            # The added minute is an allowance, not a bound on an unlogged tail.
            log=Path(record["path"])/"stdout.log"
            try:
                lines=log.read_text(encoding="utf-8").splitlines()
                begin=datetime.fromisoformat(lines[0].split()[0])
                end=datetime.fromisoformat(lines[-1].split()[0])
                elapsed=(end-begin).total_seconds()+60
                basis="legacy first/last ISO log timestamps + 60-second allowance; estimated, matching original spent_seconds"
            except (OSError,ValueError,IndexError,TypeError):
                elapsed=None
        if not finite(elapsed) or elapsed < 0 or status.get("state") not in TERMINAL | {"waiting for a human"}:
            stages[stage]["unknown_cost_runs"].append(record["run_id"])
            elapsed = None
        if elapsed is not None:
            stages[stage]["recorded_charged_seconds"] += elapsed
        stages[stage]["attempts"].append({"run_id":record["run_id"],"state":record.get("state"),"seconds":elapsed,"basis":basis,
                                          "model":record.get("contract", {}).get("configuration", {}).get("model", status.get("model"))})
    caveats = []
    for path in root.glob("*/accounting-caveat.json"):
        caveats.append({"path":str(path.resolve()),"evidence":read(path)})
    for stage in stages.values():
        stage["recorded_within_cap"] = not stage["unknown_cost_runs"] and stage["recorded_charged_seconds"] <= stage["cap_seconds"]
    per_model = {}
    for attempt in stages["architecture"]["attempts"]:
        name = attempt.get("model") or "unknown/model-initialization"
        cap = 7200 if name in {"Qwen/Qwen3.5-0.8B","LiquidAI/LFM2.5-1.2B-Instruct"} else None
        item = per_model.setdefault(name,{"recorded_seconds":0.0,"cap_seconds":cap,"unknown_cost_runs":[]})
        if attempt["seconds"] is None:
            item["unknown_cost_runs"].append(attempt["run_id"])
        else:
            item["recorded_seconds"] += attempt["seconds"]
    return {"stages":stages,"architecture_per_model":per_model,"reboot_caveats":caveats,
            "exact_24_gpu_hour_compliance_established":False,
            "interpretation":"Elapsed-wall-time charges include failed/retried work and may include reboot downtime. The first reboot has an unbounded unlogged tail; no exact GPU-hour compliance claim."}


def manuscript_text(report, docs):
    """Render actual recorded tables; no invented completion or pooled cohort."""
    def number(value):
        return f"{value:.9g}" if finite(value) else "unmeasured"
    text=["---","bibliography: paper_protocol_v2.bib","---","",
          "# Activation-Weighted Low-Rank Compression: Likelihood, Drift, and Practical Cost", "",
          "Version: "+report["generated_utc"],"", "## Abstract", "",
          "This report examines whether matched activation-weighted low-rank approximations preserve global transformer behavior. "
          "It retains weighting/anchoring controls, full-stack and composition measurements, separate BF16/FP8 numerical controls, "
          "and the user-amended hybrid-architecture panel. Tables below are generated from retained artifacts; failed/incomplete attempts "
          "remain visible. Targeted weight-count reduction is separated from likelihood, representation drift and measured execution costs. "
          "The evidence does not establish reliable global preservation or a consistent anchoring advantage over matched whitening. "
          "Promotion remains disabled; a terminal dispatcher does not imply every scientific cohort completed.","",
          "## Study status and scope","",f"Dispatcher: **{report['orchestration'].get('state')}**. Report mode: {report['mode']}.","",
          "The amended panel uses Gemma, Qwen3.5-0.8B and LFM2.5-1.2B-Instruct; GPT-2 is excluded. "
          "Only full-attention output projections are targeted in the hybrids. Native-tokenizer PPL is compared within each model. "
          "Validation frontier ranks and primary test ranks remain separate; no selection is made using confirmation outcomes.",""]
    if report["orchestration"].get("precision") == "bfloat16" and report["orchestration"].get("max_seeds") == 2:
        text += ["### Later BF16-only practical amendment", "",
                 "After stopping the historical FP16/INT8 practical stage, the user authorized only BF16 model testing with at most two seeds. The new practical cohort has five declared conditions: a BF16 dense teacher, and BF16 whitening/anchor models for seeds 11 and 23. Native BF16 measured-output factor preparation retains the frozen objective and rank. Historical other-format and three-seed evidence is preserved, not relabeled or pooled with this cohort. Higher-precision fitting/metric accumulation is numerical computation, not another model-format test. Two seeds cannot satisfy the original three-seed promotion gate. The stopped stage remains charged against the original shared practical cap.", ""]
    if report.get("declared_omissions"):
        text += ["### Declared numerical-control omissions", "",
                 "The continuation formally omits unfinished dense_fp8_e4m3 and dense_fp8_e5m2 controls. "
                 "All seven saved controls, including failed gates, and the cancelled attempt remain preserved. "
                 "This scope closure does not claim that the original nine-control cohort completed or that native FP8 model inference was tested.", ""]
    template=(docs/"paper_protocol_v2.md").read_text(encoding="utf-8")
    text += ["## Mathematical methods and frozen constants", "",
             r"Let $W\in\mathbb{R}^{o\times i}$ and $X\in\mathbb{R}^{i\times n}$. Write $A=UV$ with rank at most $r$. The anchored objective is $\|(W-A)X\|_F^2+\beta n\|W-A\|_F^2$.", "",
             "The replacement is the attention-output projection after head outputs are combined; attention heads themselves are not pruned. Fitting is offline, and inference applies V followed by U in place of W. Anchoring adds no runtime branch or extra factor parameters: it changes only the fitting geometry.", "",
             "For each Gemma target, the dense matrix has 1152 x 1024 = 1,179,648 weights. Rank 384 stores 1152 x 384 + 384 x 1024 = 835,584 weights: 344,064 fewer, or 29.17% of that projection. Across 26 targets the reduction is 8,945,664 weights. These are targeted weight counts, not a 29.17% whole-model reduction; embeddings, MLPs and other untargeted weights remain.", "",
             r"Set $H=XX^T+\beta nI$ and $C=WH$ for exact teacher outputs. For measured outputs subtract the teacher bias first: $Y_b=Y-b\mathbf{1}^T$, then $C=Y_bX^T+\beta nW$. The extra-ridge solver adds $\lambda I$ to the geometry, with $\lambda=\rho\,\operatorname{mean}(\operatorname{eig}(H))$. For $G=H+\lambda I$, form $A_\mathrm{full}=CG^\dagger$ and truncate $A_\mathrm{full}G^{1/2}$ to rank $r$; multiply back by $G^{\dagger/2}$. Thus ridge-free full-rank anchored recovery gives $W$ on retained spectral directions, while positive extra ridge generally shrinks it.", "",
             r"The implementation computes weighted covariance, eigendecomposition and weighted truncation in FP64, then returns FP32 factors for the declared storage cast. It clamps negative numerical eigenvalues to zero and retains regularized eigenvalues strictly above $\mathrm{rcond}\,\lambda_\max(G)$. Plain SVD uses FP32. Fixed-rank diagonal ASVD uses $s_j=\max(\operatorname{mean}_t|X_{jt}|,10^{-8})^{0.5}$, truncates $W\operatorname{diag}(s)$, and scales back. Factors use the recorded unbalanced convention unless a run contract explicitly declares a separate convention.", "",
             "The predeclared validation anchor sweep is beta={0,0.03,0.1,0.3,1}; the extra-ridge fractions are rho={0,0.001,0.01} at beta=0.1. "
             "Validation NLL ties within 1e-6 prefer less regularization. Native evaluation uses context/stride 512/256, full shifted targets including the tail, "
             "and 16 disjoint 512-token calibration chunks for seeds 11/23/37. The primary Gemma attention-output rank is 384. "
             "The amended hybrid rank is floor((17/24)*out*in/(out+in)); frontier ranks are floor(min(out,in)/divisor) for validation-only divisors2/3/4. "
             "Exact observed run constants and factor precisions follow below; missing fields remain unrecorded.", "",
             "| Run | Precision | Main rank | Pseudoinverse rcond | Config beta | Config ridge fraction | Selected solver / baseline |",
             "|---|---|---:|---:|---:|---:|---|"]
    for run in report["runs"]:
        contract=run.get("contract",{})
        config=contract.get("configuration",{})
        if not run["run_id"].startswith(("regularization-","baselines-","composition-","architecture-replication-","precision-readiness-","practical-")):
            continue
        text.append(f"| {run['run_id']} | {config.get('precision','unrecorded')} | {config.get('rank','per-target recipe')} | {number(config.get('pseudoinverse_rcond'))} | {number(config.get('beta'))} | {number(config.get('ridge_fraction'))} | {json.dumps({'solver':contract.get('selected_solver'),'baseline':contract.get('baseline')},sort_keys=True)} |")
    for section,end in [("## Related work and scope of the planned evidence","## Verified bibliography and dates")]:
        if section in template:
            text += [section+"\n\n"+template.split(section,1)[1].split(end,1)[0].strip(), ""]
    text += ["## Recorded numerical controls", "",
             "FP16 SVD readiness remains failed. Later BF16/FP8 variants are separate numerical cohorts, not repaired FP16 evidence. "
             "FP8 codes are decoded before BF16 multiplication; native tiny-GEMM support does not establish native model speed.", "",
             "| Run / state | Variant | NLL | PPL | Maximum probe error | Probe gate | NLL gate |",
             "|---|---|---:|---:|---:|---|---|"]
    for run in report["runs"]:
        for variant,metric in run.get("control_metrics",{}).items():
            if not isinstance(metric,dict) or not finite(metric.get("nll")):
                continue
            text.append(f"| {run['run_id']} / {run.get('state')} | {variant} | {number(metric.get('nll'))} | {number(metric.get('ppl'))} | {number(metric.get('max_logit_error'))} | {metric.get('identity_passed',metric.get('identity_probe_passed','unrecorded'))} | {metric.get('nll_passed',metric.get('nll_identity_passed','unrecorded'))} |")
    for prefix,title in [("regularization-","Validation regularization controls"),("baselines-","Gemma full-stack component comparisons"),
                         ("composition-","Composition and group heterogeneity"),("architecture-replication-","Amended hybrid architecture and rank-frontier comparisons")]:
        text += ["", "## "+title,"", "Every row identifies its attempt and scope. Mean ± sample standard deviation is across available calibration seeds; "
                 "a single seed has no estimated seed standard deviation. Incomplete attempts and partial seed coverage do not constitute a full cohort.","",
                 "| Run / state | Specification | Split / group / rank scope | Seeds | NLL mean | NLL seed SD | Drift mean | Drift seed SD |",
                 "|---|---|---|---|---:|---:|---:|---:|"]
        suppressed=report.get("presentation_supersession",{}).get("superseded",{}) if prefix=="architecture-replication-" else {}
        matching=[r for r in report["runs"] if r["run_id"].startswith(prefix) and r["run_id"] not in suppressed]
        if not matching:
            text += ["| Unmeasured | pending | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured |"]
        for run in matching:
            for group in run.get("method_groups",[]):
                m=group["metrics"]
                scope={k:v for k,v in group["scope"].items() if k not in ("recipes","target_recipes")}
                if any(k in group["scope"] for k in ("recipes","target_recipes")):
                    scope["recipe_ranks"]=[r.get("rank") for r in group["scope"].get("recipes",group["scope"].get("target_recipes",[]))]
                text.append(f"| {run['run_id']} / {run.get('state')} | {json.dumps(group['specification'],sort_keys=True)} | {json.dumps(scope,sort_keys=True)} | {group['seeds']} | {number(m['nll']['mean'])} | {number(m['nll']['sample_std'])} | {number(m['primary_drift']['mean'])} | {number(m['primary_drift']['sample_std'])} |")
        for run in matching:
            if run.get("recovery_provenance"):
                text += ["", "Recovery provenance: "+json.dumps(run["recovery_provenance"],sort_keys=True)+". "+run["reuse_interpretation"], ""]
        if prefix=="architecture-replication-":
            for run in matching:
                for comparison in run.get("comparisons",[]):
                    if (comparison.get("scope",{}).get("split")=="test"
                            and comparison.get("candidate",{}).get("method")=="anchored"
                            and comparison.get("baseline",{}).get("method")=="whitening"
                            and comparison.get("nll_and_drift_worse_all_three")
                            and all(r.get("targets_match") and r.get("actual_factor_parameters_match") for r in comparison.get("per_seed",[]))):
                        model=run.get("contract",{}).get("configuration",{}).get("model","unrecorded model")
                        text += ["", f"Scoped finding ({run['run_id']}, {model}): at the primary test rank, anchoring has higher NLL and higher primary drift than matched whitening in all three calibration seeds. This weakens the anchor hypothesis at this measured scope; it does not establish a universal failure or count reused observations as new replications."]
                    seeds = comparison.get("per_seed", [])
                    if (comparison.get("scope",{}).get("split") == "test"
                            and comparison.get("candidate",{}).get("method") == "anchored"
                            and comparison.get("baseline",{}).get("method") == "whitening"
                            and comparison.get("paired_three_seeds") and len(seeds) == 3
                            and all(r.get("targets_match") and r.get("actual_factor_parameters_match")
                                    and finite(r.get("nll_difference")) and r["nll_difference"] < 0
                                    and finite(r.get("drift_difference")) and r["drift_difference"] > 0 for r in seeds)):
                        model = run.get("contract",{}).get("configuration",{}).get("model","unrecorded model")
                        text += ["", f"Scoped tradeoff ({run['run_id']}, {model}): at the primary test rank, anchoring lowers NLL but increases primary drift versus matched whitening in every historical calibration seed. The likelihood improvement does not establish preserved hidden-state computation or a denoising mechanism. This mixed outcome remains inconclusive under the joint criterion."]
        if prefix=="composition-":
            text += ["", "Group-dependent outcomes must not be pooled into a universal-improvement claim. "
                     "A lower full-stack NLL with unchanged/worse residual drift does not satisfy the predeclared joint gate. "
                     "Three paired seeds with both higher NLL and higher drift weaken the candidate only at that measured scope."]
    text += ["","## Practical costs and calibration","",
             "Whole-model checkpoint bytes include archive overhead; factor payload size and whole-model savings are separate quantities. "
             "Placements and precision must agree before comparing timings. Below are measured records only; missing conditions remain unmeasured.","",
             "| Run / state | Label | Seed | Bits | Whole bytes | Factor bytes | Raw NLL | Scaled NLL | Temperature | Mode | Prefill | Median s | p95 s | Allocated bytes | Reserved bytes |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    practical=False
    for run in report["runs"]:
        if not run["run_id"].startswith("practical-"):
            continue
        for group in run.get("method_groups",[]):
            for record in group["records"]:
                for timing in record.get("timing",[]):
                    practical=True
                    control=record.get("calibration",{})
                    format_label = str(record.get('bits','dense')) + ' / ' + str(record.get('precision', run.get('contract',{}).get('configuration',{}).get('precision','unrecorded')))
                    text.append(f"| {run['run_id']} / {run.get('state')} | {record.get('label')} | {record.get('seed')} | {format_label} | {number(record.get('whole_checkpoint',{}).get('bytes'))} | {number(record.get('factor_checkpoint_bytes'))} | {number(control.get('raw_test',{}).get('nll'))} | {number(control.get('scaled_test',{}).get('nll'))} | {number(control.get('fit',{}).get('temperature'))} | {timing.get('mode')} | {timing.get('prefill_tokens')} | {number(timing.get('median_seconds'))} | {number(timing.get('p95_seconds'))} | {number(timing.get('peak_allocated_bytes'))} | {number(timing.get('peak_reserved_bytes'))} |")
    if not practical:
        text += ["| Unmeasured | pending | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured |"]
    text += ["", "### Practical measurement scope", "",
             "The historical FP16/INT8 stage's hardware-gates.json, when present, uses the same seed's FP16 whitening reference for both FP16 and INT8 solver factors. Its threshold concerns median latency and peak allocated memory only; it is not a same-bit INT8 gate and does not test p95, reserved memory, serialized bytes or quality. The later user-authorized BF16-only cohort is separate: seeds 11/23, five expected conditions, no other model formats and no three-seed promotion claim. BF16 has 16 storage bits but must not be labeled FP16. Its calibration routine saves full-target aggregate NLL and ECE, not per-block losses; practical NLL points are descriptive, without paired block intervals. Historical main-quality cohorts retain their separate block uncertainty. "
             "The separate descriptive ratios below use same-run, same-seed, same-bit whitening/anchor records with matching device, declared placement, compute precision and workload. Missing or ambiguous pairs remain excluded in report.json. No original hardware gate is relabeled.", "",
             "Timings use batch size 1, 10 warmups and 30 synchronized samples. Cached 128-token decode timing excludes the initial prompt pass; prefill is measured separately at 128 and 512 tokens. Peak allocated/reserved bytes are CUDA allocator statistics including resident model and cache, not total driver GPU memory or system RAM. INT8 factors are dequantized for floating-point multiplication, so these timings do not establish native INT8-kernel acceleration.", "",
             "Whole-model reload parity covers a 128-token prompt and one cached next-token forward, not the complete evaluation corpus or arbitrary generation. The historical dense-teacher file lacks this probe; the BF16-only amendment probes its dense teacher as well as its compressed models. Historical local factor probes and BF16 whole-model probes are distinct. The practical stage cap is cooperative; terminal elapsed wall time remains the charged cost.", "",
             "### Same-bit descriptive cost ratios", "",
             "Each ratio is anchor divided by matched whitening; below 1 is lower cost. Placement is explicit row metadata where available, otherwise the full-model resident device declared in the same-run configuration. These are exploratory measured workload comparisons, not new promotion gates.", "",
             "| Run | Seed | Factor bits | Device / placement | Mode | Prefill | Decode | Median ratio | p95 ratio | Allocated ratio | Reserved ratio | Factor-byte ratio | Whole-byte ratio |",
             "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ratio_count=0
    for run in report["runs"]:
        for pair in run.get("same_bit_cost_comparisons",{}).get("comparisons",[]):
            scope, ratios=pair["scope"],pair["ratios"]
            text.append(f"| {run['run_id']} | {scope['seed']} | {scope['bits']} | {scope['device']} / {json.dumps(scope['placement'],sort_keys=True)} | {scope['mode']} | {scope['prefill_tokens']} | {scope['decode_tokens']} | "+" | ".join(number(ratios[key]) for key in ("median_latency","p95_latency","peak_allocated","peak_reserved","factor_bytes","whole_bytes"))+" |")
            ratio_count+=1
    if not ratio_count:
        text.append("| Unmeasured | pending | pending | pending | pending | pending | pending | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |")
    text += ["","## Statistical interpretation","",
             "NLL differences use 2,000 paired bootstrap resamples of fixed contiguous 4,096-target blocks. "
             "The final partial block remains in full-test NLL but is excluded from fixed-block resampling. "
             "Individual dependent tokens are not bootstrapped. Per-seed confidence intervals, exact excluded counts, matched target checks "
             "and all scoped candidate-minus-baseline differences are in report.json and report.md. These are evaluation-block intervals, not seed intervals.","",
             "## Limitations", "",
             "WikiText-2 has already informed this project, so these results are regression evidence rather than fresh-domain confirmation. No downstream-task capability or arbitrary-generation equivalence was measured. Only attention-output projections are compressed; hybrid DeltaNet/convolution mixers and all MLPs remain dense.", "",
             "Within-model comparisons match factor budgets and native evaluation targets. Differences across model cohorts cannot be attributed solely to architecture: pretrained weights, tokenizers and, for the historical Gemma reference, inference precision differ. Raw perplexities must not rank different tokenizers. The later BF16 Gemma practical cohort measures costs and calibration; it does not add new BF16 drift evidence.", "",
             "The original strict numerical-readiness failure remains binding. The two-seed BF16 practical amendment cannot satisfy a three-seed replication gate. Residual drift is a diagnostic proxy, not a mathematical certificate of global computation; lower perplexity on one corpus likewise does not establish preservation. The joint criterion is an operational research gate, not a universal theorem.", "",
             "Weight-count reduction, actual serialized bytes and measured runtime are separate outcomes. Timings are descriptive for the recorded GPU/software/workloads, and reload probes cover only the stated prompt and cached token. Budget-limited frontier points remain unmeasured. Reboot accounting, including an unbounded first-reboot log tail, prevents an exact 24 GPU-hour compliance claim.", "",
             "## Future scope: preparation and runtime optimization", "",
             "Future work should separate the cost of constructing compressed models from the cost of running them. The directions below are proposals, not measured speedups or authorization for additional experiments. BF16 should remain the initial execution format, with at most two fixed calibration seeds under the current user constraint; any broader replication requires a separately agreed protocol and budget.", "",
             "### Faster offline compression", "",
             "Profile activation capture, covariance construction, decomposition, factor transfer and checkpoint serialization separately. Cache reusable teacher activations and sufficient statistics with keys binding the exact model revision, target, token positions, calibration seed and dtype. Reuse must preserve the fitting objective and split boundaries. Batched preparation or approximate decompositions could then be evaluated against the current solver for preparation time, peak CPU/GPU memory, reconstruction error and held-out global behavior; they should not be assumed numerically equivalent.", "",
             "### Efficient factorized inference", "",
             "The deployed projection computes z=Vx followed by y=Uz, with the original bias applied once. A future BF16 implementation could investigate fused or tiled execution that avoids writing the complete rank-sized intermediate tensor to device memory, reduces launch overhead and uses suitable factor layouts. Compilation, reusable execution graphs and reusable buffers are additional candidates to profile. These changes must retain the compact factors; reconstructing and storing the full dense matrix would surrender the targeted weight-storage reduction. Fewer weights alone do not guarantee lower latency, since the factorized path introduces a dependent operation and its performance depends on shapes and workloads.", "",
             "### Runtime-aware rank and placement choices", "",
             "A separately frozen validation study could allocate ranks across projections using both error sensitivity and measured BF16 kernel cost, rather than assuming one uniform rank is optimal. It should compare against a matched total parameter or byte budget, keep test data untouched during selection, and retain a strong whitening baseline. GPU-resident and RAM-offloaded execution should be reported as separate conditions: host-device transfers, placement and intermediate buffers belong in the end-to-end cost. Extending compression to MLPs or hybrid non-attention mixers remains a separate scientific question.", "",
             "### Verification and decision rules", "",
             "For each implementation change, first check numerical agreement with the existing BF16 factorized operator and then evaluate full-model likelihood, logits and drift. Measure offline preparation and online inference separately, including prefill, cached decode, multiple prompt lengths and batch sizes, synchronized median/p95 latency, tokens per second, serialized bytes, allocator memory and independently measured total device/system memory. Compare the optimized factors with both the current factorized implementation and the dense BF16 teacher under identical workload and placement. Report speedups only where observed, alongside quality changes and uncertainty. More extensive reload/generation probes, a fresh pinned corpus and downstream tasks could test whether any benefit generalizes; preservation of global computation remains unresolved.", "",
             "## Reproducibility and incomplete-run appendix","",
             "The accompanying report.json embeds each retained contract, source hashes, package/device versions, asset revisions, "
             "calibration/split hashes, target recipes, run-specific command, raw per-seed metrics, integrity checks, artifact paths and "
             "all failed/interrupted attempts. Read errors are retained explicitly; promotion is disabled.","",
             "| Stage | Recorded charged wall seconds | Declared cap seconds | Unknown-cost attempts |","|---|---:|---:|---|"]
    for stage,audit in report["budget"]["stages"].items():
        text.append(f"| {stage} | {audit['recorded_charged_seconds']:.3f} | {audit['cap_seconds']} | {audit['unknown_cost_runs']} |")
    supersession=report.get("presentation_supersession",{})
    if supersession.get("superseded"):
        text += ["", "### Preserved interrupted attempts and reused observations", "",
                 "The main architecture table and figures show the unique validated continuation, including its actual incomplete/terminal state. The source observations are identical reused measurements, not new replications. Original attempts and full metrics remain separately recorded in report.json and report.md; all source hashes and validation warnings are in presentation_supersession.", "",
                 "| Original attempt / state | Continuation / state | Reused rows | Original coverage | Charged wall seconds |",
                 "|---|---|---:|---|---:|"]
        by_id={r["run_id"]:r for r in report["runs"]}
        for run_id,entry in supersession["superseded"].items():
            source=by_id[run_id]
            charge=source.get("interruption_accounting",{}).get("charged_seconds",source.get("status",{}).get("elapsed_seconds"))
            text.append(f"| {run_id} / {source.get('state')} | {entry['successor']} / {entry['successor_state']} | {entry['reused_records']} | {json.dumps(source.get('cohort_coverage',{}),sort_keys=True)} | {number(charge)} |")
        for run_id,entry in supersession["superseded"].items():
            text += ["", "Source provenance ("+run_id+"): "+json.dumps(entry["source_artifacts"],sort_keys=True)+".", ""]
    if supersession.get("warnings"):
        text += ["", "Presentation validation warnings: "+json.dumps(supersession["warnings"],sort_keys=True), ""]
    text += ["",report["budget"]["interpretation"],"",
             "## Conclusion","",report["decision"],"",
             "Complete negative or mixed comparisons are retained as bounded evidence. Unmeasured downstream tasks, "
             "fresh-corpus confirmation, absent seeds or hardware conditions are not implied by dispatcher completion."]
    section="## Verified bibliography and dates"
    if section in template:
        text += ["", section+"\n\n"+template.split(section,1)[1].split("## Conclusion",1)[0].strip()]
    return "\n".join(text)+"\n"


def export_documents(directory, *, exporters=None):
    """Export separate TeX source/ReportLab PDF; errors are explicit failures."""
    directory=Path(directory).resolve()
    status_path=directory/"export-status.json"
    if status_path.exists():
        raise FileExistsError(status_path)
    exports={}
    definitions=(("latex","export_paper_latex_20261014","export_report","manuscript.tex"),
                 ("pdf","export_paper_pdf_20261014","export_pdf","manuscript.pdf"))
    for kind,module_name,function_name,filename in definitions:
        try:
            if exporters is None:
                module=importlib.import_module((__package__+"." if __package__ else "")+module_name)
                callback=getattr(module,function_name)
                source=Path(module.__file__).resolve()
            else:
                callback=exporters[kind]
                source=None
            path=Path(callback(directory)).resolve()
            if path!=directory/filename or not path.is_file() or not path.stat().st_size:
                raise ValueError("export API did not return the expected nonempty artifact: "+filename)
            if kind=="pdf":
                with path.open("rb") as stream:
                    if not stream.read(5).startswith(b"%PDF-"):
                        raise ValueError("PDF artifact has an invalid header")
            exports[kind]={"state":"prepared","path":str(path),"sha256":digest(path),"bytes":path.stat().st_size,
                "format":"standalone LaTeX source, not compiler-verified" if kind=="latex" else "ReportLab PDF, not compiled from LaTeX",
                "exporter_source":str(source) if source else "injected test exporter",
                "exporter_sha256":digest(source) if source else None}
        except Exception as exc:
            exports[kind]={"state":"failed","error":type(exc).__name__+": "+str(exc),
                           "format":"standalone LaTeX source" if kind=="latex" else "ReportLab PDF"}
    result={"state":"prepared" if all(e["state"]=="prepared" for e in exports.values()) else "failed",
            "exports":exports,"latex_compilation_verified":False,
            "interpretation":"ReportLab PDF and standalone LaTeX are separate review artifacts. Compiler success requires an independent successful compiler diagnostic."}
    with status_path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="results/paper-v2")
    parser.add_argument("--orchestration",default="results/paper-v2/revised-orchestration-20261014-3")
    parser.add_argument("--snapshot",action="store_true")
    args = parser.parse_args(argv)
    root, orchestration = Path(args.root).resolve(),Path(args.orchestration).resolve()
    orchestrator = read(orchestration/"status.json")
    if not args.snapshot and orchestrator.get("state") not in TERMINAL:
        raise RuntimeError("serial revised orchestration is not terminal; wait or explicitly request --snapshot")
    reports, errors = [],[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/"status.json").is_file() or directory.name.startswith("revised-report-"):
            continue
        try:
            reports.append(inspect_run(directory))
        except Exception as exc:
            errors.append({"run_id":directory.name,"error":type(exc).__name__+": "+str(exc),"classification":"exploratory"})
    interruption_audit = None
    if orchestrator.get("interruption_accounting_path"):
        try:
            interruption_audit = apply_interruption_accounting(root,reports,orchestrator["interruption_accounting_path"])
        except Exception as exc:
            errors.append({"run_id":"interruption-sidecar","error":type(exc).__name__+": "+str(exc),"classification":"exploratory"})
    report = {"generated_utc":datetime.now(timezone.utc).isoformat(),"mode":"interim snapshot" if args.snapshot else "terminal orchestration report",
              "reporter":{"path":str(Path(__file__).resolve()),"sha256":digest(__file__),"python":sys.version,"executable":sys.executable},
              "orchestration_path":str(orchestration),"orchestration":orchestrator,"runs":reports,"read_errors":errors,
              "budget":budget_audit(root,reports+errors),"promotion_disabled":True,"hypothesis_status":"inconclusive",
              "decision":"Low-rank factors reduce targeted weight counts. Anchoring has no established consistent advantage over matched whitening on both likelihood and representation drift; reliable preservation of global computation remains unresolved. Recorded storage and runtime tradeoffs are reported separately, and all findings remain exploratory.",
              "limits":["Original FP16 SVD readiness failed; BF16/FP8 controls are separate cohorts.",
                        "FP8 storage is decoded for BF16 multiplication; a tiny native GEMM probe is not model acceleration.",
                        "Architecture panel targets only six full-attention outputs per new model; hybrid mixers remain dense.",
                        "Raw PPL cannot rank tokenizers/models. Compare within-model differences only.",
                        "No pooling across precision, placements, reboot attempts, or resumed/reused cohorts.",
                        "Downstream tasks and fresh sampled-C4 confirmation remain unmeasured unless separately evidenced."]}
    report["interruption_accounting_audit"] = interruption_audit
    report["presentation_supersession"] = presentation_supersession(root,reports)
    continuation=Path(__file__).resolve().parent/"CONTINUATION_20261014.md"
    if continuation.is_file():
        report["declared_omissions"]={"variants":["dense_fp8_e4m3","dense_fp8_e5m2"],
            "scope":"unfinished numerical side controls formally omitted; seven saved controls preserved, not a completed nine-control study",
            "source":str(continuation),"sha256":digest(continuation),"text":continuation.read_text(encoding="utf-8")}
    output=root/("revised-report-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
    output.mkdir(exist_ok=False)
    (output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    lines=["# Revised study evidence report","",f"Mode: {report['mode']}. Orchestration: {orchestrator.get('state')}.","",report["decision"],"",
           "Promotion disabled. Research hypothesis: inconclusive.","","| Attempt | Terminal/observed state | Model | Precision |","|---|---|---|---|"]
    for item in reports:
        config=item.get("contract",{}).get("configuration",{})
        lines.append(f"| {item['run_id']} | {item.get('state')} | {config.get('model','unrecorded')} | {config.get('precision','unrecorded')} |")
    lines += ["","## Accounted costs","","| Stage | Recorded wall seconds | Cap seconds | Unknown-cost attempts |","|---|---:|---:|---|"]
    for name,stage in report["budget"]["stages"].items():
        lines.append(f"| {name} | {stage['recorded_charged_seconds']:.3f} | {stage['cap_seconds']} | {', '.join(stage['unknown_cost_runs']) or 'none'} |")
    lines += ["",report["budget"]["interpretation"],"","## Interpretation limits","",*["- "+limit for limit in report["limits"]],"",
              "Full retained contracts, source/asset provenance, per-seed summaries, paired block uncertainty, integrity checks and read failures are in report.json. Each run remains a separate cohort.",
              "Report-generation errors do not change experiment terminal states or erase successful measurements."]
    lines += ["", "## Practical cost interpretation", "",
              "Historical FP16/INT8 hardware gates compare both solver factor formats against same-seed FP16 whitening; the separate BF16-only two-seed amendment uses BF16 matched comparisons, using median latency and peak allocated memory. Separate same-bit ratios require identical seed, bits, device/placement, compute precision and workload; these descriptive comparisons and exclusions are in each practical run's same_bit_cost_comparisons field in report.json. They do not change the saved gates.", "",
              "Decode timing excludes initial prompt processing; CUDA allocated/reserved memory is allocator-only, not total device or RAM use. INT8 storage is dequantized for floating-point execution. Compressed whole-model reload probes cover 128 prompt tokens plus one cached token; the measured dense checkpoint has no equivalent reload probe. Practical elapsed caps are cooperative."]
    lines += ["","## Within-model, scope-specific comparisons","",
              "Each delta is candidate minus baseline. Group outcomes remain separate; a full-stack NLL gain does not imply improved block drift or improvement at every layer group.","",
              "| Run | Candidate | Scope | Seed | NLL delta | Drift delta | Scoped hypothesis |",
              "|---|---|---|---:|---:|---:|---|"]
    for item in reports:
        for comparison in item.get("comparisons",[]):
            for row in comparison["per_seed"]:
                nll=f"{row['nll_difference']:+.9g}" if finite(row["nll_difference"]) else "missing"
                drift=f"{row['drift_difference']:+.9g}" if finite(row["drift_difference"]) else "missing"
                lines.append(f"| {item['run_id']} | {json.dumps(comparison['candidate'],sort_keys=True)} | {json.dumps(comparison['scope'],sort_keys=True)} | {row['seed']} | {nll} | {drift} | {comparison['hypothesis_status']} |")
    if errors:
        lines += ["","Read errors: "+json.dumps(errors)]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    docs=Path(__file__).resolve().parent
    (output/"manuscript.md").write_text(manuscript_text(report,docs),encoding="utf-8")
    (output/"paper_protocol_v2.bib").write_bytes((docs/"paper_protocol_v2.bib").read_bytes())
    try:
        from .plot_revised_study_20261014 import generate_figures
    except ImportError:
        from plot_revised_study_20261014 import generate_figures
    figures=generate_figures(output)
    with (output/"manuscript.md").open("a",encoding="utf-8") as stream:
        stream.write("\n## Measured scientific figures\n\n")
        for figure in figures["figures"]:
            stream.write(f"![{figure['caption']}]({figure['png']})\n\n")
        if not figures["figures"]:
            stream.write("No scientific figures available from the supplied completed cohorts. See figures/provenance.json for explicit availability reasons.\n")
        stream.write("\nFigure inputs, scope, source hashes and unavailable measurements are recorded in figures/provenance.json.\n")
        stream.write("\n## Review export formats\n\nStandalone LaTeX source and a separately generated ReportLab PDF are review artifacts. "
                     "Their preparation status and source hashes are recorded in export-status.json. "
                     "The ReportLab PDF is not compiled from LaTeX; successful TeX compilation requires a separate compiler diagnostic.\n")
    exports=export_documents(output)
    with (output/"report.md").open("a",encoding="utf-8") as stream:
        stream.write("\n## Review export status\n\n"+exports["interpretation"]+"\n\n")
        for kind,value in exports["exports"].items():
            stream.write(f"- {kind}: {value['state']}. "+(value.get("path") or value.get("error",""))+"\n")
    print(output)
    return 0 if exports["state"]=="prepared" else 1


if __name__ == "__main__":
    raise SystemExit(main())

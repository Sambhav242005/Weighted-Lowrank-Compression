"""Audit saved BF16 practical evidence on CPU, without a model forward."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import uuid

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path("results/paper-v2").resolve()
    directory = Path(args.run).resolve()
    if directory.parent != root or not directory.name.startswith("practical-bf16-"):
        raise ValueError("expected a BF16 practical run within the results root")
    read = lambda path: json.loads(Path(path).read_text())
    config, contract, status, rows = [read(directory/name) for name in ("config.json", "contract.json", "status.json", "metrics.json")]
    errors, checked = [], {}
    def check(condition, message):
        if not condition:
            errors.append(message)
    def file_check(path, expected_sha=None, expected_bytes=None):
        path = Path(path)
        if not path.is_file():
            check(False, "missing file "+str(path))
            return
        if str(path) not in checked:
            checked[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
        info = checked[str(path)]
        check(expected_sha is None or info["sha256"] == expected_sha, "SHA mismatch "+str(path))
        check(expected_bytes is None or info["bytes"] == expected_bytes, "size mismatch "+str(path))
    check(status["state"] == "completed" and len(rows) == 5 and contract["expected_records"] == 5, "five completed conditions")
    check(config["seeds"] == [11, 23] and config["precision"] == "bfloat16" and config["factor_formats"] == ["bfloat16"], "BF16 and two-seed restriction")
    check(config["model_revision"] == "dcc83ea841ab6100d6b47a070329e1ba4cf78752", "pinned Gemma revision")
    check(config["dataset_revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3", "pinned dataset revision")
    expected = {("teacher", None), ("baseline", 11), ("solver", 11), ("baseline", 23), ("solver", 23)}
    conditions = [(r["label"], r["seed"]) for r in rows]
    check(set(conditions) == expected and len(set(conditions)) == len(conditions), "exact unique authorized conditions")
    check(contract["configuration"] == config, "resolved configuration matches contract")
    check(len(contract["targets"]) == 26 and all(r["shape"] == [1152, 1024] and r["rank"] == 384 and r["factor_parameters"] == 835584 for r in contract["targets"]), "frozen target budget")
    for filename, digest in contract["environment"]["source_hashes"].items():
        file_check(filename, digest)
    file_check(config["selection_path"], contract["selection_hash"])
    manifests = contract["splits"]
    check(set(manifests["calibration"]) == {"11", "23"}, "no additional calibration seed")
    test_targets = manifests["test"]["tokens"]-1
    validation_targets = manifests["validation"]["tokens"]-1
    timing_scopes = {("prefill", 128), ("prefill", 512), ("decode", 128), ("decode", 512)}
    for row in rows:
        label, seed = row["label"], row["seed"]
        prefix = f"{label}/{seed}"
        check(row["precision"] == "bfloat16" and row["compute_precision"] == "bfloat16" and row["bits"] == 16, prefix+" BF16 format")
        folder = directory/("teacher" if label == "teacher" else f"{label}-seed-{seed}-bf16")
        control = row["calibration"]
        fit = control["fit"]
        check(fit["selection_split"] == "validation" and fit["targets"] == validation_targets, prefix+" validation-only temperature")
        check(fit["grid"] == config["temperature_grid"] and len(fit["grid"]) == 33, prefix+" frozen grid")
        check(read(folder/"frozen-temperature.json") == fit, prefix+" frozen temperature artifact")
        for key in ("raw_test", "scaled_test"):
            metric = control[key]
            check(metric["targets"] == test_targets == 292281, prefix+" full native test tail")
            check(len(metric["bins"]) == 15 and sum(b["targets"] for b in metric["bins"]) == test_targets, prefix+" complete ECE bins")
            check(all(math.isfinite(metric[k]) for k in ("nll", "ppl", "accuracy", "entropy", "ece_15")), prefix+" finite metrics")
        check(control["raw_test"]["temperature"] == 1 and control["scaled_test"]["temperature"] == fit["temperature"], prefix+" raw/frozen scale")
        timings = row["timing"]
        check(len(timings) == 4 and {(t["mode"], t["prefill_tokens"]) for t in timings} == timing_scopes, prefix+" four workloads")
        for timing in timings:
            check(timing["batch_size"] == 1 and timing["warmups"] == 10 and timing["measurements"] == 30 and timing["use_cache"] is True, prefix+" timing protocol")
            check(timing["decode_tokens"] == (128 if timing["mode"] == "decode" else 0), prefix+" decode scope")
            check(0 < timing["median_seconds"] <= timing["p95_seconds"] and 0 < timing["peak_allocated_bytes"] <= timing["peak_reserved_bytes"], prefix+" measured time/memory")
        parity = row["whole_model_reload_parity"]
        check(parity["exact"] is True and parity["before"] == parity["after"] and parity["before"]["prompt_tokens"] == 128, prefix+" exact reload probe")
        check(read(folder/"whole-model-reload-parity.json") == parity, prefix+" probe artifact")
        whole = folder/"whole-model.pt"
        file_check(whole, row["whole_checkpoint"]["sha256"], row["whole_checkpoint"]["bytes"])
        payload = torch.load(whole, map_location="cpu", weights_only=True)
        check(payload["precision"] == "bfloat16", prefix+" saved whole format")
        for name, value in payload["state_dict"].items():
            if name.endswith((".weight", ".bias")) and value.is_floating_point():
                check(value.dtype == torch.bfloat16, prefix+" saved parameter "+name)
        check(len(payload["bf16_recipes"]) == (0 if label == "teacher" else 26), prefix+" saved factor coverage")
        for name, recipe in payload["bf16_recipes"].items():
            check(recipe["storage"] == "bf16" and recipe["compute"] == "bf16", prefix+" saved factor format")
            check(payload["state_dict"][name+".left"].dtype == torch.bfloat16 and payload["state_dict"][name+".right"].dtype == torch.bfloat16, prefix+" factor tensors")
        del payload
        gc.collect()
        if label != "teacher":
            expected_spec = {"method": "whitening", "beta": 0.0, "ridge_fraction": 0.0} if label == "baseline" else {"method": "anchored", "beta": 0.03, "ridge_fraction": 0.0}
            check(row["specification"] == expected_spec, prefix+" frozen objective")
            file_check(row["checkpoint"], row["metadata"]["sha256"], row["factor_checkpoint_bytes"])
            check(row["metadata"]["factor_parameters"] == 26*835584 and row["metadata"]["precision"] == "bfloat16", prefix+" factor count and precision")
            payload = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
            check(set(payload["state"]) == {r["name"] for r in contract["targets"]}, prefix+" factor checkpoint coverage")
            for values in payload["state"].values():
                check(list(values["left"].shape) == [1152, 384] and list(values["right"].shape) == [384, 1024], prefix+" factor shapes")
                check(all(v.dtype == torch.bfloat16 and torch.isfinite(v).all().item() for v in values.values() if v is not None), prefix+" finite BF16 factors")
            del payload
    prior = config["prior_practical_charged_seconds"]
    elapsed = status["elapsed_seconds"]
    check(math.isfinite(elapsed) and elapsed >= 0 and prior+elapsed <= 14400, "recorded original practical cap")
    out = root/("bf16-practical-audit-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    result = {"state": "passed" if not errors else "errors", "run": directory.name,
              "conditions": conditions, "precision": "bfloat16", "seeds": [11, 23],
              "test_targets": test_targets, "validation_targets": validation_targets,
              "prior_charged_seconds": prior, "new_elapsed_seconds": elapsed, "cumulative_seconds": prior+elapsed,
              "files": checked, "errors": errors, "script_sha256": sha(__file__),
              "limitations": ["CPU checkpoint inspection performs no model forward or new seed test.",
                              "Exact reload covers the recorded prompt/cached-token probes, not arbitrary generation.",
                              "Practical NLL has aggregate calibration metrics, not paired block loss intervals.",
                              "Two seeds do not meet the historical three-seed promotion criterion."]}
    (out/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"report.md").write_text("# BF16 practical evidence audit\n\nState: "+result["state"]+"\n\nFive BF16 conditions, seeds 11/23 only. Full native targets, frozen calibration, checkpoint dtypes/hashes, reload probes and timing workloads checked.\n\nCumulative practical seconds: "+str(prior+elapsed)+" / 14400.\n\n"+"\n".join("- "+s for s in result["limitations"])+"\n\nErrors: "+str(errors)+"\n")
    print(json.dumps({"output": str(out), "state": result["state"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

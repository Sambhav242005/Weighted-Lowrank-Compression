"""Read-only CPU audit of the completed historical BF16 LFM cohort."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from audit_gemma_evidence_20261014 import RESULTS, digest, hashes


def main():
    directory = RESULTS/"architecture-replication-20261014-103759-8d32ab84"
    read = lambda name: json.loads((directory/name).read_text())
    rows, contract, status = read("metrics.json"), read("contract.json"), read("status.json")
    errors = []
    def check(condition, message):
        if not condition:
            errors.append(message)
    config = contract["configuration"]
    check(status["state"] == "completed" and len(rows) == 26, "completed 26-record cohort")
    check(config["model_revision"] == "0f604ada3f766f9f257460c4c9f0b5d6f69d431b", "pinned model revision")
    check(config["dataset_revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3", "pinned data revision")
    scope = lambda row: (row["split"], row["rank_divisor"], row["label"], row["seed"])
    observed = [scope(row) for row in rows]
    expected = [scope(row) for row in contract["comparison_plan"]]
    check(len(set(observed)) == 26 and set(observed) == set(expected), "all expected unique scopes")
    for filename, sha in contract["implementation_hashes"].items():
        check(digest(filename) == sha, "frozen implementation "+filename)
    for index, row in enumerate(rows):
        label = f"row {index}"
        split = row["split"]
        targets = contract["splits"][split]["tokens"]-1
        check(row["metrics"]["targets"] == targets, label+" full native targets")
        check(sum(b["targets"] for b in row["metrics"]["nll_blocks"]) == targets, label+" complete block tail")
        check(row["teacher_reference_split"] == split and row["teacher_reference_targets"] == targets, label+" teacher reference")
        check(len(row["metrics"]["residual_by_block"]) == 16, label+" all decoder block drift")
        rank = None if row["label"] == "teacher" else (725 if split == "test" else 2048//row["rank_divisor"])
        check(row["rank"] == rank, label+" frozen rank")
        recipes = row["target_recipes"]
        check(len(recipes) == 6, label+" six attention outputs")
        for recipe, block in zip(recipes, [2, 5, 8, 10, 12, 14]):
            check(recipe["name"] == f"model.layers.{block}.self_attn.out_proj" and recipe["shape"] == [2048, 2048]
                  and recipe["rank"] == rank and recipe["precision"] == "bfloat16"
                  and recipe["factor_parameters"] == (4096*rank if rank is not None else None), label+" target recipe")
        if row["label"] != "teacher":
            check(digest(row["checkpoint"]) == row["metadata"]["sha256"], label+" checkpoint SHA")
            check(row["metadata"]["factor_parameters"] == 6*4096*rank and row["metadata"]["targets"] == recipes, label+" matched factor budget")
        if row["label"] in {"baseline", "solver"}:
            spec = {"method": "whitening", "beta": 0.0, "ridge_fraction": 0.0} if row["label"] == "baseline" else {
                "method": "anchored", "beta": 0.03, "ridge_fraction": 0.0}
            check(row["specification"] == spec, label+" frozen method")
    check(status["elapsed_seconds"] <= 7200, "original per-model cap")
    out = RESULTS/("lfm-evidence-audit-"+datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    payload = {"state": "passed" if not errors else "errors", "run": directory.name,
               "records": len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
               "native_targets": {s: contract["splits"][s]["tokens"]-1 for s in ["test", "validation"]},
               "elapsed_seconds": status["elapsed_seconds"], "files_hashed": hashes, "errors": errors,
               "limitation": "Historical three-seed BF16 evidence predates the new two-seed cap; this audit performs no model forward or new seed experiment.",
               "script_sha256": digest(__file__)}
    (out/"audit.json").write_text(json.dumps(payload, indent=2)+"\n")
    (out/"report.md").write_text("# LFM immutable evidence audit\n\nState: "+payload["state"]+"\n\n26 expected records: eight test and eighteen validation. Exact pins, target counts, ranks, checkpoint hashes and frozen recipes checked.\n\n"+payload["limitation"]+"\n\nErrors: "+str(errors)+"\n")
    print(json.dumps({"output": str(out), "state": payload["state"], "errors": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

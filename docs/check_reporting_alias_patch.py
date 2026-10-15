"""CPU-only verification of the proposed patch, without changing src."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_reporting_alias_patch import proposed_source


root = Path(__file__).resolve().parents[1]
path = root/"src/paper_reporting.py"
before = hashlib.sha256(path.read_bytes()).hexdigest()
namespace = {}
source = path.read_text(encoding="utf-8")
checked_source = source if "def reduce_alias_records(records):" in source else proposed_source(source)
exec(compile(checked_source, "checked-paper-reporting", "exec"), namespace)
reduce = namespace["reduce_alias_records"]
record = dict(id="original", seed=11, specification={"method": "whitening"},
              checkpoint="one.pt", metadata={"sha256": "same-payload", "factor_parameters": 12, "serialized_bytes": 100},
              metrics={"nll": 1, "targets": 8192, "nll_blocks": [{"nll_sum": 4096, "targets": 4096}]*2})
alias = {**copy.deepcopy(record), "id": "alias", "checkpoint": "two.pt",
         "specification": {"method": "anchored", "beta": 0}}
clean, evidence = reduce([record, alias])
assert len(clean) == 1 and evidence[0]["equivalent_aliases"] and len(evidence[0]["sources"]) == 2
for mutation in ("nll", "targets", "nll_blocks", "hash", "missing_hash", "factor_parameters"):
    changed = copy.deepcopy(alias)
    if mutation == "hash":
        changed["metadata"]["sha256"] = "different"
    elif mutation == "missing_hash":
        del changed["metadata"]["sha256"]
    elif mutation == "factor_parameters":
        changed["metadata"]["factor_parameters"] += 1
    elif mutation == "nll_blocks":
        changed["metrics"]["nll_blocks"][0]["nll_sum"] += 1
    else:
        changed["metrics"][mutation] += 1
    clean, evidence = reduce([record, changed])
    assert not clean and evidence[0]["conflicting_alias_evidence"] and len(evidence[0]["raw_conflicting_records"]) == 2
with tempfile.TemporaryDirectory(prefix="report-alias-proposal-") as temporary:
    run = Path(temporary)/"baselines-alias"
    run.mkdir()
    records = [{**copy.deepcopy(item), "seed": seed} for seed in (11,23,37) for item in (record, alias)]
    for filename, data in [("metrics.json", records), ("status.json", {"state": "completed"}), ("contract.json", {})]:
        (run/filename).write_text(json.dumps(data))
    report = namespace["build_run_report"](run)
    assert len(report["method_summaries"]) == 1 and not report["comparisons"]
    assert next(iter(report["method_summaries"].values()))["seeds"] == [11,23,37]
    assert all(e["equivalent_aliases"] for e in next(iter(report["alias_evidence"].values())))
    records[-1]["metrics"]["nll"] += 1
    (run/"metrics.json").write_text(json.dumps(records))
    conflict = namespace["build_run_report"](run)
    summary = next(iter(conflict["method_summaries"].values()))
    assert summary["seeds"] == [11,23] and not summary["complete"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == before
print("Proposed alias patch: equal aliases, six conflict cases, three-seed grouping, conflict exclusion and source immutability passed.")

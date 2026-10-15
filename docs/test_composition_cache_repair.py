"""CPU checks of the proposal or installed repair; source is never edited."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


def proposed_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("composition_proposal", root/"docs/prepare_composition_cache_repair.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    module = types.ModuleType("src._composition_proposal")
    module.__package__ = "src"
    current = (root/"src/paper_followups.py").read_text(encoding="utf-8")
    # Once applied, exercise the actual implementation rather than attempting
    # to insert a second copy of the helpers into the patched source.
    installed = "def _composition_record(" in current and "def _composition_source_artifacts(" in current
    proposed = current if installed else builder.proposed_source(current)
    exec(compile(proposed, "docs-only-proposed-followups.py", "exec"), module.__dict__)
    return module


def fixture_entry(depth=12):
    names = [f"block.{index}.proj" for index in range(depth)]
    recipes = [{"name": name, "shape": [8, 8], "rank": 2, "factor_parameters": 32} for name in names]
    entry = {"id": "seed-11-solver", "seed": 11, "specification": {"method": "anchored", "beta": .1},
        "checkpoint": "retained/factors.pt", "metadata": {"targets": recipes, "precision": "float16",
        "serialized_bytes": 1234, "sha256": "checkpoint-hash"},
        "metrics": {"targets": 292281, "nll": 3.5, "nll_blocks": [{"nll_sum": 5, "targets": 4096}],
        "residual_by_block": [{"normalized_squared_error": index/100} for index in range(depth)]}}
    artifacts = {"run_id": "completed-C", "metrics_sha256": "metrics-hash",
                 "boundary_integrity_verified": True, "integrity_manifest_sha256": "integrity-hash"}
    return names, entry, artifacts


def test_only_full_stack_reuses_deep_copied_metrics():
    module = proposed_module()
    names, entry, artifacts = fixture_entry()
    untouched = copy.deepcopy(entry)
    calls = []
    fresh_metrics = {"nll": 4.0, "targets": 292281,
                     "residual_by_block": [{"normalized_squared_error": .1} for _ in names]}
    def evaluate_fresh():
        calls.append(True)
        return copy.deepcopy(fresh_metrics)
    full = module._composition_record("solver", 11, 2, tuple(range(len(names))), names, entry,
                                      artifacts, evaluate_fresh, validated_source=True)
    assert not calls and entry == untouched
    assert full["reused_from_run"] == "completed-C" and full["independent_replication"] is False
    assert full["source_record_id"] == entry["id"] and full["source_metrics_sha256"] == "metrics-hash"
    assert full["checkpoint_sha256"] == "checkpoint-hash" and full["metadata"]["serialized_bytes"] == 1234
    assert full["metadata"]["factor_parameters"] == 384 and full["metrics"]["downstream_drift"] == []
    full["metrics"]["nll_blocks"][0]["nll_sum"] = 999
    assert entry == untouched
    for group in module.composition_groups(len(names))[:-1]:
        result = module._composition_record("solver", 11, 2, group, names, entry, artifacts,
                                             evaluate_fresh, validated_source=True)
        assert result["metrics"]["nll"] == 4.0 and "reused_from_run" not in result
        assert result["metadata"]["serialized_bytes"] is None
        assert result["metadata"]["factor_parameters"] == len(group)*32
    assert len(calls) == 9


def test_fresh_validation_is_required_before_any_reuse():
    module = proposed_module()
    names, entry, artifacts = fixture_entry()
    with pytest.raises(ValueError, match="fresh teacher/data"):
        module._composition_record("solver", 11, 2, tuple(range(12)), names, entry,
                                    artifacts, lambda: pytest.fail("must not evaluate"), validated_source=False)


@pytest.mark.parametrize("changed", ["metrics.json", "contract.json", "teacher.json"])
def test_boundary_integrity_rejects_changed_source(tmp_path, changed):
    module = proposed_module()
    hashes = {}
    for name in ("metrics.json", "contract.json", "teacher.json"):
        (tmp_path/name).write_text("retained source "+name)
        hashes[name] = hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    (tmp_path/"artifact-integrity.json").write_text(json.dumps({"artifacts": hashes}))
    reference = module._composition_source_artifacts({"baseline_run_path": str(tmp_path)})
    assert reference["boundary_integrity_verified"] is True
    assert reference["metrics_sha256"] == hashes["metrics.json"]
    (tmp_path/changed).write_text("changed source")
    with pytest.raises(ValueError, match="artifact changed"):
        module._composition_source_artifacts({"baseline_run_path": str(tmp_path)})

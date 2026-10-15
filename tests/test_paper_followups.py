import math
import json
from types import SimpleNamespace

import pytest
import torch

from src.compression_core import PackedLowRankLinear
from src.paper_followups import (
    calibration_metrics, composition_groups, fit_temperature,
    frozen_methods, run_replication, save_reload_factor,
    run_followups,
    calibration_control, restore_model_checkpoint, _save_model_state,
    _baseline_checkpoints,
    dataset_content_hash,
)


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, use_cache=False):
        logits = torch.tensor([math.log(3), 0.0]).expand(*input_ids.shape, 2)
        return SimpleNamespace(logits=logits)


def test_calibration_metrics_score_partial_window_once():
    tokens = torch.zeros(773, dtype=torch.long)
    result = calibration_metrics(ConstantModel(), tokens)
    assert result["targets"] == 772
    assert result["nll"] == pytest.approx(-math.log(.75), rel=1e-6)
    assert result["accuracy"] == 1
    assert result["ece_15"] == pytest.approx(.25)
    assert sum(b["targets"] for b in result["bins"]) == 772


def test_temperature_uses_validation_and_declares_boundary():
    result = fit_temperature(ConstantModel(), torch.zeros(773, dtype=torch.long), temperatures=[.5, 1, 2])
    assert result["temperature"] == .5
    assert result["targets"] == 772
    assert result["boundary_hit"]
    assert result["selection_split"] == "validation"


def test_group_positions_deduplicate():
    groups = composition_groups(4)
    assert len(groups) == len(set(groups))
    assert (0, 1) in groups and (1, 2) in groups and (2, 3) in groups
    assert groups.count((0, 1, 2, 3)) == 1


def test_replication_freezes_settings_and_keeps_frontier_off_test():
    selection = {"selection_split": "validation", "source_run_id": "source",
                 "solver": {"method": "anchored", "beta": .03}, "baseline": {"method": "asvd"}}
    calls = []
    run_replication(selection, [(768, 768)]*12, lambda *args: calls.append(args))
    assert len(calls) == 26
    assert all(call[2] == [256]*12 for call in calls[:8])
    assert all(call[4] == "validation" for call in calls[8:])
    selection["selection_split"] = "test"
    with pytest.raises(ValueError):
        frozen_methods(selection)


@pytest.mark.parametrize("bits", [8, 16])
def test_packed_reload_is_exact_and_refuses_overwrite(tmp_path, bits):
    torch.manual_seed(7)
    layer = PackedLowRankLinear.from_factors(torch.randn(4, 2), torch.randn(2, 6), bits=bits)
    path = tmp_path / "factor.pt"
    result = save_reload_factor(layer, path, torch.randn(3, 6))
    assert result["reload_exact"] and result["bytes"] == path.stat().st_size
    with pytest.raises(FileExistsError):
        save_reload_factor(layer, path, torch.randn(3, 6))


def test_replication_dispatch_preserves_frozen_settings_and_split_contract(tmp_path, monkeypatch):
    import src.paper_benchmark as benchmark
    import src.paper_comparisons as comparisons
    selection_path = tmp_path / "selection.json"
    dataset_directory = tmp_path/"dataset"
    dataset_directory.mkdir()
    (dataset_directory/"fixture.arrow").write_bytes(b"synthetic dataset fixture")
    selection_path.write_text(json.dumps({"state": "frozen", "selection_split": "validation",
        "dataset_content_hash": dataset_content_hash(dataset_directory),
        "dataset_revision": "dataset", "precision": "float16", "rank": 384,
        "model_revision": "gemma-revision", "selected_solver": {"specification": {"method": "anchored", "beta": .03}},
        "strongest_baseline": {"specification": {"method": "asvd"}}}))
    config = {"selection_path": str(selection_path), "dataset_revision": "dataset", "precision": "float16",
              "dataset_path": str(dataset_directory),
              "seeds": [11, 23, 37], "stage_gpu_hours": {"replication": 4}, "output_root": str(tmp_path),
              "context": 512, "stride": 256, "device": "cpu", "pseudoinverse_rcond": 1e-7}
    model = ConstantModel()
    model.projection = torch.nn.Linear(768, 768)
    tokens = {"train": torch.zeros(9000, dtype=torch.long), "validation": torch.zeros(513, dtype=torch.long),
              "test": torch.zeros(773, dtype=torch.long)}
    monkeypatch.setattr(benchmark, "environment", lambda c: {"ready": True})
    monkeypatch.setattr(comparisons, "load_assets", lambda c: (model, tokens))
    monkeypatch.setattr(comparisons, "target_modules", lambda m: [("projection", model.projection)])
    monkeypatch.setattr(comparisons, "capture_inputs", lambda *args: {})
    monkeypatch.setattr(comparisons, "fit_checkpoint", lambda teacher, inputs, c, spec, path, progress: (path/"factors.pt", {"rank": c["rank"]}))
    monkeypatch.setattr(comparisons, "student_from_checkpoint", lambda *args, **kwargs: ConstantModel())
    seen = []
    def diagnostic(student, teacher, split, c, progress):
        seen.append(len(split))
        return {"nll": 1.0, "targets": len(split)-1}
    monkeypatch.setattr(comparisons, "evaluate_diagnostics", diagnostic)
    assert run_followups(config, "replication") == 0
    output = next(tmp_path.glob("replication-*"))
    metrics = json.loads((output/"metrics.json").read_text())
    assert len(metrics) == 26
    assert seen.count(773) == 7 and seen.count(513) == 18
    assert all(r["specification"].get("beta", .03) == .03 for r in metrics)
    contract = json.loads((output/"contract.json").read_text())
    assert contract["configuration"]["rank"] == 256
    assert contract["configuration"]["model_revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert json.loads((output/"status.json").read_text())["state"] == "completed"
    config["runtime_remaining_seconds"] = {"replication": -1}
    assert run_followups(config, "replication") == 1
    directories = list(tmp_path.glob("replication-*"))
    assert len(directories) == 2
    states = [json.loads((directory/"status.json").read_text())["state"] for directory in directories]
    assert sorted(states) == ["completed", "incomplete"]


def test_dataset_digest_is_location_independent_and_sensitive_to_bytes(tmp_path):
    original, copy = tmp_path/"original", tmp_path/"copy"
    original.mkdir()
    copy.mkdir()
    (original/"train.arrow").write_bytes(b"train")
    (copy/"train.arrow").write_bytes(b"train")
    assert dataset_content_hash(original) == dataset_content_hash(copy)
    (copy/"changed.arrow").write_bytes(b"changed")
    assert dataset_content_hash(original) != dataset_content_hash(copy)


def test_temperature_freeze_is_written_before_test_evaluation(tmp_path, monkeypatch):
    import src.paper_followups as followups
    frozen = tmp_path / "frozen-temperature.json"
    calls = []
    def metrics(model, tokens, **kwargs):
        assert frozen.exists()
        assert json.loads(frozen.read_text())["selection_split"] == "validation"
        calls.append(kwargs.get("temperature", 1))
        return {"targets": len(tokens)-1}
    monkeypatch.setattr(followups, "calibration_metrics", metrics)
    calibration_control(ConstantModel(), torch.zeros(513, dtype=torch.long), torch.zeros(773, dtype=torch.long),
                        freeze_path=frozen, temperatures=[.5, 1, 2])
    assert calls == [1, .5]


@pytest.mark.parametrize("bits", [8, 16])
def test_complete_checkpoint_reconstructs_without_dense_packed_weights(tmp_path, bits):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.proj = PackedLowRankLinear.from_factors(torch.randn(4, 2), torch.randn(2, 6), bits=bits)
            self.config = SimpleNamespace(to_dict=lambda: {"model_type": "test"})
        def forward(self, x):
            return self.block.proj(x)
    torch.manual_seed(11)
    model = Toy()
    probe = torch.randn(2, 6)
    before = model(probe)
    checkpoint = tmp_path / "whole-model.pt"
    _save_model_state(model, checkpoint)
    restored = restore_model_checkpoint(checkpoint, model_factory=lambda config: Toy())
    assert torch.equal(before, restored(probe))
    assert restored.block.proj.bits == bits
    assert "block.proj.weight" not in restored.state_dict()


def test_cached_baseline_requires_fresh_tokens_teacher_and_recipes(tmp_path):
    from src.compression_core import sha256_file
    selection_path = tmp_path/"selection.json"
    selection_path.write_text("{}")
    source = tmp_path/"baselines"
    source.mkdir()
    config = {"selection_path": str(selection_path), "baseline_run_path": str(source),
              "model_revision": "model", "dataset_revision": "data", "rank": 2, "precision": "float16",
              "context": 512, "stride": 256, "seeds": [11, 23, 37]}
    (source/"status.json").write_text(json.dumps({"state": "completed"}))
    (source/"config.json").write_text(json.dumps(config))
    targets = [{"name": "block.proj", "shape": [4, 6], "rank": 2, "factor_parameters": 20}]
    manifests = {split: {"tokens": 9000, "hash": split} for split in ("train", "validation", "test")}
    manifests["calibration_positions"] = {str(seed): [0, 512] for seed in config["seeds"]}
    manifests["calibration_hashes"] = {str(seed): "captured-"+str(seed) for seed in config["seeds"]}
    contract = {"selection_hash": sha256_file(selection_path), "teacher_hash": "teacher",
                "splits": manifests, "targets": targets}
    (source/"contract.json").write_text(json.dumps(contract))
    checkpoint = source/"factors.pt"
    torch.save({"block.proj": {"u": torch.ones(4, 2).half(), "v": torch.ones(2, 6).half(), "bias": None}}, checkpoint)
    frozen = {"solver": {"method": "anchored", "beta": .1}, "baseline": {"method": "asvd"}}
    records = [{"seed": seed, "specification": frozen[label], "checkpoint": str(checkpoint),
                "metadata": {"sha256": sha256_file(checkpoint), "targets": targets, "precision": "float16"}}
               for label in ("solver", "baseline") for seed in config["seeds"]]
    (source/"metrics.json").write_text(json.dumps(records))
    assert len(_baseline_checkpoints(config, {}, frozen, manifests=manifests, teacher_hash="teacher", targets=targets)) == 6
    for split in ("train", "validation", "test"):
        altered = json.loads(json.dumps(manifests))
        altered[split]["hash"] = "different"
        with pytest.raises(ValueError, match="token content changed"):
            _baseline_checkpoints(config, {}, frozen, manifests=altered, teacher_hash="teacher", targets=targets)
    altered = json.loads(json.dumps(manifests))
    altered["calibration_hashes"]["11"] = "different"
    with pytest.raises(ValueError, match="calibration token content changed"):
        _baseline_checkpoints(config, {}, frozen, manifests=altered, teacher_hash="teacher", targets=targets)
    with pytest.raises(ValueError, match="teacher differs"):
        _baseline_checkpoints(config, {}, frozen, manifests=manifests, teacher_hash="different", targets=targets)
    with pytest.raises(ValueError, match="target names/shapes/ranks"):
        _baseline_checkpoints(config, {}, frozen, manifests=manifests, teacher_hash="teacher",
                              targets=[{**targets[0], "rank": 1}])

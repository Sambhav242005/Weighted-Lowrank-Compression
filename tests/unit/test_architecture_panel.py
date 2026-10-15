import json
import copy
import math
from pathlib import Path

import pytest
import torch

from src import paper_architecture_panel as panel


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_model(kind):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Lfm2Config, Lfm2ForCausalLM
    if kind == "qwen3_5_text":
        configuration = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
            linear_num_value_heads=2, tie_word_embeddings=True,
            layer_types=["linear_attention"]*3+["full_attention"],
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                "partial_rotary_factor": .25, "mrope_section": [1, 1, 0], "mrope_interleaved": True})
        return Qwen3_5ForCausalLM(configuration).to(torch.bfloat16).eval()
    configuration = Lfm2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        block_auto_adjust_ff_dim=False, block_multiple_of=8,
        layer_types=["conv", "full_attention"])
    return Lfm2ForCausalLM(configuration).to(torch.bfloat16).eval()


def quiet(*args, **kwargs):
    pass


def test_budget_matched_rectangular_and_square_rank():
    assert panel.matched_rank((1152, 1024)) == 384
    assert panel.matched_rank((1024, 2048)) == 483
    assert panel.matched_rank((2048, 2048)) == 725
    for shape in ((1024, 2048), (2048, 2048)):
        rank = panel.matched_rank(shape)
        assert rank * sum(shape) <= 17 * shape[0] * shape[1] / 24
        assert (rank+1) * sum(shape) > 17 * shape[0] * shape[1] / 24


@pytest.mark.parametrize("kind", ["qwen3_5_text", "lfm2"])
def test_native_targets_exclude_other_mixers_and_preserve_them(tmp_path, kind):
    torch.manual_seed(19)
    teacher = tiny_model(kind)
    recipes = panel.target_recipes(teacher, enforce_pin=False)
    assert len(recipes) == 1
    target = recipes[0]
    assert target["block_index"] == (3 if kind == "qwen3_5_text" else 1)
    assert target["shape"] == ([16, 32] if kind == "qwen3_5_text" else [16, 16])
    assert "linear_attn" not in target["name"] and ".conv." not in target["name"]
    calibration = torch.randn(target["shape"][1], 40).to(torch.bfloat16)
    checkpoint, metadata = panel.fit_factors(teacher, recipes,
        {target["name"]: calibration}, {"method": "anchored", "beta": .03, "ridge_fraction": 0.0},
        tmp_path/"fit", device="cpu", progress=quiet, rcond=1e-7)
    student = panel.make_student(teacher, checkpoint)
    assert metadata["precision"] == "bfloat16"
    assert student.get_submodule(target["name"]).left.dtype == torch.bfloat16
    for name, tensor in teacher.state_dict().items():
        if not name.startswith(target["name"]+"."):
            assert torch.equal(tensor, student.state_dict()[name]), name
    assert not any(name == target["name"]+".weight" for name in student.state_dict())
    tokens = torch.arange(773) % 32
    result = panel.evaluate_panel(student, teacher, tokens, device="cpu", progress=quiet)
    assert result["targets"] == 772
    assert len(result["residual_by_block"]) == len(teacher.model.layers)
    assert all(row["tokens"] == 772 for row in result["residual_by_block"])
    assert result["local_output"][target["name"]]["tokens"] == 772
    assert sum(row["targets"] for row in result["nll_blocks"]) == 772
    assert result["logit_kl"] >= 0


@pytest.mark.parametrize("kind", ["qwen3_5_text", "lfm2"])
def test_native_fullrank_identity_and_teacher_metrics(kind):
    teacher = tiny_model(kind)
    tokens = torch.arange(19) % 32
    result = panel.readiness_probe(teacher, tokens, {"atol": .1, "rtol": .01}, progress=quiet)
    assert result["variants"]["copied"]["max_logit_error"] == 0
    assert result["variants"]["identity_bf16"]["max_logit_error"] == 0
    assert result["full_test_readiness"] is False
    dense = panel.evaluate_panel(teacher, teacher, tokens, device="cpu", progress=quiet)
    assert dense["targets"] == 18
    assert dense["logit_kl"] == 0 and dense["primary_drift"] == 0
    assert len(dense["residual_by_block"]) == len(teacher.model.layers)


def test_prior_caps_count_cancelled_failed_attempts_and_reject_unknown(tmp_path):
    names = list(panel.PINNED)
    for label, status in (("replication-old", {"state": "cancelled", "elapsed_seconds": 123}),
        ("architecture-replication-qwen", {"state": "failed", "elapsed_seconds": 80, "model": names[0]}),
        ("architecture-replication-lfm", {"state": "completed", "elapsed_seconds": 90, "model": names[1]})):
        directory = tmp_path/label
        directory.mkdir()
        (directory/"status.json").write_text(json.dumps(status))
    total, by_model = panel.prior_costs(tmp_path)
    assert total == 293 and by_model == {names[0]: 80, names[1]: 90}
    unknown = tmp_path/"architecture-replication-unknown"
    unknown.mkdir()
    with pytest.raises(RuntimeError, match="unresolved"):
        panel.prior_costs(tmp_path)


def test_model_cap_and_pins_reject_gpt2_and_revision_drift():
    config = {"precision": "bfloat16", "seeds": [11, 23, 37], "context": 512, "stride": 256,
        "calibration": {"split": "train", "chunks": 16, "chunk_tokens": 512},
        "selection_split": "validation", "evaluation_split": "test", "existing_gemma_reference": "Gemma",
        "stage_gpu_hours": {"replication": 4}, "models": [{"model": name, "model_revision": value["revision"],
            "architecture": value["architecture"], "model_path": str(Path("snapshot")/value["revision"])}
            for name, value in panel.PINNED.items()]}
    panel.validate_config(config)
    changed = json.loads(json.dumps(config))
    changed["models"][0]["model"] = "openai-community/gpt2"
    with pytest.raises(ValueError, match="GPT-2 excluded"):
        panel.validate_config(changed)
    changed = json.loads(json.dumps(config))
    changed["models"][0]["model_revision"] = "different"
    with pytest.raises(ValueError, match="pinned model"):
        panel.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["validation_frontier"] = {**panel.VALIDATION_FRONTIER, "select_on_test": True}
    with pytest.raises(ValueError, match="frontier protocol"):
        panel.validate_config(changed)


def test_independent_model_budget_limit_continues_other_model_but_blocks_advance(tmp_path, monkeypatch):
    config = {"output_root": str(tmp_path), "models": [{"model": name} for name in panel.PINNED]}
    monkeypatch.setattr(panel, "validate_config", lambda value: value)
    monkeypatch.setattr(panel, "frozen_specs", lambda value: None)
    monkeypatch.setattr(panel, "completed_model_run", lambda *args: None)
    calls = []
    def execute(config, record, allowance, prior):
        calls.append((record["model"], allowance, prior))
        output = tmp_path/f"architecture-replication-{len(calls)}"
        output.mkdir()
        limited = len(calls) == 1
        (output/"status.json").write_text(json.dumps({"state": "incomplete" if limited else "completed",
            "model": record["model"], "elapsed_seconds": 7200 if limited else 50, "budget_limited": limited}))
        return int(limited), output
    monkeypatch.setattr(panel, "run_model", execute)
    assert panel.run_panel(config) == 1
    assert len(calls) == 2 and calls[0][1] == 7200 and calls[1][1] == 7200
    status = json.loads(next(tmp_path.glob("architecture-panel-*/status.json")).read_text())
    assert status["state"] == "incomplete" and status["advance_to_practical"] is False


def test_unexpected_model_failure_stops_before_next_architecture(tmp_path, monkeypatch):
    config = {"output_root": str(tmp_path), "models": [{"model": name} for name in panel.PINNED]}
    monkeypatch.setattr(panel, "validate_config", lambda value: value)
    monkeypatch.setattr(panel, "frozen_specs", lambda value: None)
    monkeypatch.setattr(panel, "completed_model_run", lambda *args: None)
    calls = []
    def execute(config, record, allowance, prior):
        calls.append(record["model"])
        output = tmp_path/"architecture-replication-failed"
        output.mkdir()
        (output/"status.json").write_text(json.dumps({"state": "failed", "model": record["model"],
            "elapsed_seconds": 20, "budget_limited": False, "error": "synthetic NaN"}))
        return 1, output
    monkeypatch.setattr(panel, "run_model", execute)
    assert panel.run_panel(config) == 1 and len(calls) == 1


def test_frontier_plan_preserves_eight_test_cases_and_declared_native_ranks():
    baseline = {"method": "whitening", "beta": 0., "ridge_fraction": 0.}
    anchor = {"method": "anchored", "beta": .03, "ridge_fraction": 0.}
    cases = panel.panel_cases(baseline, anchor)
    assert len(cases) == 26 and len({panel.record_scope(row) for row in cases}) == 26
    assert len([row for row in cases if row["split"] == "test"]) == 8
    assert len([row for row in cases if row["split"] == "validation"]) == 18
    assert all(row["rank_divisor"] is None for row in cases[:8])
    for shape, expected in (([1024, 2048], {2: 512, 3: 341, 4: 256}),
                            ([2048, 2048], {2: 1024, 3: 682, 4: 512})):
        primary = [{"name": "model.layers.3.self_attn.o_proj", "shape": shape,
            "block_index": 3, "rank": panel.matched_rank(shape),
            "factor_parameters": panel.matched_rank(shape)*sum(shape)}]
        frozen = copy.deepcopy(primary)
        for row in cases[8:]:
            recipes = panel.case_recipes(primary, row)
            assert recipes[0]["rank"] == expected[row["rank_divisor"]]
            assert recipes[0]["factor_parameters"] == expected[row["rank_divisor"]]*sum(shape)
        assert primary == frozen
    for divisor in (2, 3, 4):
        selected = [row for row in cases if row["rank_divisor"] == divisor]
        assert {(row["label"], row["seed"]) for row in selected} == {
            (label, seed) for label in ("baseline", "solver") for seed in panel.SEEDS}
    square = [{"shape": [2048, 2048], "dense_parameters": 2048**2,
        "rank": 1024, "factor_parameters": 1024*4096}]
    storage = panel.parameter_storage(square, 1_170_340_608, checkpoint_bytes=8_400_000)
    assert storage["rank_fraction_by_target"] == [.5]
    assert storage["target_matrix_parameter_savings_fraction"] == 0
    assert storage["whole_model_parameter_savings_fraction"] == 0
    assert storage["factor_checkpoint_bytes"] == 8_400_000 and storage["whole_model_serialized_bytes"] is None


def complete_frontier_fixture(tmp_path):
    baseline = {"method": "whitening", "beta": 0., "ridge_fraction": 0.}
    anchor = {"method": "anchored", "beta": .03, "ridge_fraction": 0.}
    primary = [{"name": "model.layers.3.self_attn.o_proj", "block_index": 3,
        "shape": [16, 32], "rank": 7, "factor_parameters": 7*48,
        "dense_parameters": 512, "precision": "bfloat16"}]
    contract = {"baseline": baseline, "selected_solver": anchor, "targets": primary,
        "whole_model_parameters": 100000,
        "splits": {"test": {"tokens": 774}, "validation": {"tokens": 514}}}
    references = {split: {"targets": contract["splits"][split]["tokens"]-1,
        "nll": nll, "ppl": math.exp(nll)} for split, nll in (("test", 2.), ("validation", 1.))}
    rows = []
    for index, case in enumerate(panel.panel_cases(baseline, anchor)):
        recipes = panel.case_recipes(primary, case)
        reference = references[case["split"]]
        nll = reference["nll"] + (0. if case["label"] == "teacher" else .1)
        measured = {"targets": reference["targets"], "nll": nll, "ppl": math.exp(nll)}
        row = {**case, "rank": recipes[0]["rank"], "target_recipes": recipes, "metrics": measured,
            **panel.relative_metrics(measured, reference, case["split"])}
        if case["rank_divisor"] is not None:
            row["independent_replication"] = False
        if case["label"] != "teacher":
            recipe = recipes[0]
            checkpoint = tmp_path/f"factor-{index}.pt"
            torch.save({"targets": recipes, "specification": case["specification"],
                "storage": "bfloat16", "compute": "bfloat16", "state": {recipe["name"]: {
                    "left": torch.ones(16, recipe["rank"], dtype=torch.bfloat16),
                    "right": torch.ones(recipe["rank"], 32, dtype=torch.bfloat16), "bias": None}}}, checkpoint)
            row.update(checkpoint=str(checkpoint), metadata={"sha256": panel.sha256_file(checkpoint),
                "targets": recipes, "factor_parameters": sum(recipe["factor_parameters"] for recipe in recipes)})
        row["storage"] = panel.parameter_storage(recipes, contract["whole_model_parameters"])
        rows.append(row)
    return rows, contract, {"split": "validation", "metrics": references["validation"]}


def test_frontier_recovery_rejects_missing_scopes_wrong_reference_and_wrong_checkpoint(tmp_path):
    rows, contract, teacher = complete_frontier_fixture(tmp_path)
    assert panel.validate_completed_rows(rows, contract, teacher)
    with pytest.raises(ValueError, match="eight test and eighteen validation"):
        panel.validate_completed_rows(rows[:-1], contract, teacher)
    altered = copy.deepcopy(rows)
    altered[-1]["teacher_reference_split"] = "test"
    with pytest.raises(ValueError, match="wrong split/reference"):
        panel.validate_completed_rows(altered, contract, teacher)
    altered = copy.deepcopy(rows)
    altered[-1]["target_recipes"][0]["rank"] += 1
    with pytest.raises(ValueError, match="target provenance"):
        panel.validate_completed_rows(altered, contract, teacher)
    altered = copy.deepcopy(rows)
    altered[-1]["metrics"]["targets"] = contract["splits"]["test"]["tokens"]-1
    with pytest.raises(ValueError, match="target provenance"):
        panel.validate_completed_rows(altered, contract, teacher)
    path = Path(rows[-1]["checkpoint"])
    payload = torch.load(path, weights_only=True)
    name = rows[-1]["target_recipes"][0]["name"]
    payload["state"][name]["left"] = torch.ones(16, 1, dtype=torch.bfloat16)
    torch.save(payload, path)
    rows[-1]["metadata"]["sha256"] = panel.sha256_file(path)
    with pytest.raises(ValueError, match="factor tensor shape"):
        panel.validate_completed_rows(rows, contract, teacher)


def test_complete_model_recovery_requires_frontier_artifacts_and_all_scopes(tmp_path):
    rows, contract, teacher = complete_frontier_fixture(tmp_path)
    directory = tmp_path/"architecture-replication-complete"
    directory.mkdir()
    model, pin = next(iter(panel.PINNED.items()))
    selection = tmp_path/"selection.json"
    selection.write_text("{}")
    dataset = tmp_path/"dataset"
    dataset.mkdir()
    (dataset/"content.arrow").write_bytes(b"synthetic raw dataset")
    record = {"model": model, "model_revision": pin["revision"], "architecture": pin["architecture"]}
    config = {"output_root": str(tmp_path), "selection_path": str(selection), "dataset_path": str(dataset),
        "precision": "bfloat16", "seeds": list(panel.SEEDS), "context": 512, "stride": 256,
        "dataset_revision": "pinned-test-data", "calibration": {"split": "train", "chunks": 16, "chunk_tokens": 512},
        "selection_split": "validation", "evaluation_split": "test"}
    contract.update(configuration={**config, **record}, selection_hash=panel.sha256_file(selection),
        dataset_content_hash=panel.dataset_content_hash(dataset), implementation_hashes={},
        validation_frontier=copy.deepcopy(panel.VALIDATION_FRONTIER))
    artifacts = {"contract.json": contract, "metrics.json": rows, "validation-teacher.json": teacher,
        "teacher-audit.json": {"unchanged": True}, "readiness.json": {"classification": "exploratory"}}
    for name, value in artifacts.items():
        (directory/name).write_text(json.dumps(value))
    (directory/"status.json").write_text(json.dumps({"model": model, "state": "completed"}))
    def manifest():
        (directory/"artifact-integrity.json").write_text(json.dumps({"artifacts": {
            name: panel.sha256_file(directory/name) for name in artifacts}}))
    manifest()
    assert panel.completed_model_run(config, record) == directory
    (directory/"metrics.json").write_text(json.dumps(rows[:8]))
    manifest()
    with pytest.raises(ValueError, match="eighteen validation"):
        panel.completed_model_run(config, record)

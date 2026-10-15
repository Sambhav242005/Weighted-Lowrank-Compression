import json

import pytest

from src.paper_study import spent_seconds, run_study


def test_unknown_prior_cost_is_not_silently_reset(tmp_path):
    run = tmp_path / "regularization-old"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "failed"}))
    with pytest.raises(RuntimeError, match="prior cost unknown"):
        spent_seconds(tmp_path, "regularization")


def test_caps_count_failed_and_completed_runs(tmp_path):
    for name, state, elapsed in [("one", "failed", 12), ("two", "completed", 23)]:
        run = tmp_path / ("baselines-" + name)
        run.mkdir()
        (run / "status.json").write_text(json.dumps({"state": state, "elapsed_seconds": elapsed}))
    assert spent_seconds(tmp_path, "baselines") == 35


def test_sequential_study_freezes_paths_and_keeps_exploratory_flag(tmp_path, monkeypatch):
    from src import paper_benchmark, paper_comparisons, paper_followups, paper_reporting
    root = tmp_path / "results"
    root.mkdir()
    evidence = root / "svd-diagnosis-evidence"
    evidence.mkdir()
    (evidence / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 30}))
    config = {"output_root": str(root), "study": "local-core", "readiness_passed": False,
              "readiness_evidence": str(evidence), "stage_gpu_hours": {
                  "readiness": 1, "regularization": 4, "baselines": 7,
                  "composition": 4, "replication": 4, "practical": 4}}
    monkeypatch.setattr("src.paper_study.environment", lambda c: {"ready": True})
    calls = []
    def execute(c, stage):
        assert c["readiness_passed"] is False
        assert c["runtime_remaining_seconds"][stage] == config["stage_gpu_hours"][stage] * 3600
        if stage != "regularization":
            assert c["selection_path"].endswith("selection.json")
        if stage in {"composition", "replication", "practical"}:
            assert "baselines" in c["baseline_run_path"]
        path = paper_benchmark.create_output(c, stage)
        (path / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 1}))
        if stage == "regularization":
            (path / "selection.json").write_text("{}")
        calls.append(stage)
        return 0
    monkeypatch.setattr(paper_comparisons, "run_comparisons", execute)
    monkeypatch.setattr(paper_followups, "run_followups", execute)
    monkeypatch.setattr(paper_reporting, "report_study", lambda c, s: 0)
    assert run_study(config) == 0
    assert calls == ["regularization", "baselines", "composition", "replication", "practical"]
    status = json.loads(next(root.glob("study-*/status.json")).read_text())
    assert status["exploratory"] is True


def test_nominal_success_cannot_advance_after_stage_time_cap(tmp_path, monkeypatch):
    from src import paper_benchmark, paper_comparisons
    root = tmp_path / "results"
    root.mkdir()
    evidence = root / "svd-diagnosis-evidence"
    evidence.mkdir()
    (evidence / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 0}))
    config = {"output_root": str(root), "study": "local-core", "readiness_passed": False,
              "readiness_evidence": str(evidence), "stage_gpu_hours": {
                  "readiness": 1, "regularization": 1/3600}}
    monkeypatch.setattr("src.paper_study.environment", lambda c: {"ready": True})
    times = iter([0.0, 2.0])
    monkeypatch.setattr("src.paper_study.time.monotonic", lambda: next(times))
    def execute(c, stage):
        path = paper_benchmark.create_output(c, stage)
        (path / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 2}))
        return 0
    monkeypatch.setattr(paper_comparisons, "run_comparisons", execute)
    assert run_study(config) == 1
    assert next(root.glob("regularization-*/budget-audit.json")).exists()
    assert not list(root.glob("baselines-*"))


def test_resume_reuses_frozen_regularization_without_refitting(tmp_path, monkeypatch):
    from src import paper_benchmark, paper_comparisons, paper_followups, paper_reporting
    root = tmp_path / "results"
    root.mkdir()
    evidence = root / "svd-diagnosis-evidence"
    evidence.mkdir()
    (evidence / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 30}))
    previous = root / "regularization-preserved"
    previous.mkdir()
    (previous / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 500}))
    selection = previous / "selection.json"
    selection.write_text(json.dumps({"state": "frozen"}))
    config = {"output_root": str(root), "study": "local-core", "readiness_passed": False,
              "readiness_evidence": str(evidence), "start_stage": "baselines", "selection_path": str(selection),
              "stage_gpu_hours": {"readiness": 1, "regularization": 4, "baselines": 7,
                                  "composition": 4, "replication": 4, "practical": 4}}
    monkeypatch.setattr("src.paper_study.environment", lambda c: {"ready": True})
    called = []
    def execute(c, stage):
        assert stage != "regularization"
        path = paper_benchmark.create_output(c, stage)
        (path / "status.json").write_text(json.dumps({"state": "completed", "elapsed_seconds": 1}))
        called.append(stage)
        return 0
    monkeypatch.setattr(paper_comparisons, "run_comparisons", execute)
    monkeypatch.setattr(paper_followups, "run_followups", execute)
    monkeypatch.setattr(paper_reporting, "report_study", lambda c, s: 0)
    assert run_study(config) == 0
    assert called == ["baselines", "composition", "replication", "practical"]
    assert spent_seconds(root, "regularization") == 500

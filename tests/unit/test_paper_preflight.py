from src.paper_benchmark import dependency_errors


def test_incompatible_hub_version_is_rejected_before_model_loading(monkeypatch):
    from src import paper_benchmark
    monkeypatch.setattr(paper_benchmark.importlib.metadata, "requires", lambda owner: [
        "huggingface-hub>=0.30.0,<1.0", "optional-library; extra == 'optional'"] if owner == "transformers" else [])
    monkeypatch.setattr(paper_benchmark.importlib.metadata, "version", lambda name: "2.2.0")
    errors = dependency_errors()
    assert len(errors) == 1
    assert "huggingface-hub" in errors[0] and "2.2.0" in errors[0]

import pytest

from src.paper_reporting import (assess_promotion, paired_block_bootstrap,
                                 summarize_seeds, validate_matched_contracts, MATCH_FIELDS,
                                 build_run_report, report_study, normalize_contract, canonical_specification)


def test_matching_requires_every_contract_field_and_preserves_split_identity():
    contract = {key: "same" for key in MATCH_FIELDS}
    contract["readiness_passed"] = True
    assert validate_matched_contracts(contract, contract)["comparable"]
    changed = {**contract, "split_hashes": "different"}
    assert validate_matched_contracts(changed, contract)["mismatched"] == ["split_hashes"]
    missing = dict(contract)
    del missing["factor_parameters"]
    assert validate_matched_contracts(missing, contract)["classification"] == "exploratory"
    assert not validate_matched_contracts({**contract, "readiness_passed": False}, contract)["comparable"]


def test_calibration_hashes_are_actual_tokens_not_only_positions():
    contract = {"splits": {"calibration_positions": {"11": [0]}, "calibration_hashes": {"11": "actual"}}}
    normalized = normalize_contract(contract)
    assert normalized["calibration_token_hash"] == {"11": "actual"}
    assert normalized["calibration_positions"] == {"11": [0]}
    assert canonical_specification({"method": "anchored", "beta": 0}) == canonical_specification({"method": "whitening"})


def test_paired_bootstrap_keeps_dependence_blocks_and_reports_partial_exclusion():
    candidate = [{"nll_sum": 4096 * value, "targets": 4096} for value in [1, 2, 3]]
    baseline = [{"nll_sum": 4096 * value, "targets": 4096} for value in [2, 3, 4]]
    candidate.append({"nll_sum": 20, "targets": 2})
    baseline.append({"nll_sum": 2, "targets": 2})
    result = paired_block_bootstrap(candidate, baseline)
    assert result == paired_block_bootstrap(candidate, baseline)
    assert result["confidence_interval"] == [-1, -1]
    assert result["difference"] == pytest.approx((-4096 * 3 + 18) / (4096 * 3 + 2))
    assert result["excluded_partial_targets"] == 2
    assert result["resamples"] == 2000
    assert result["established_nll_advantage"]


def test_uncertainty_crossing_zero_does_not_establish_advantage():
    a = [{"nll_sum": 4096 * v, "targets": 4096} for v in [1, 3]]
    b = [{"nll_sum": 4096 * 2, "targets": 4096}] * 2
    assert not paired_block_bootstrap(a, b)["established_nll_advantage"]


def test_incomplete_blocks_and_nonfinite_observations_fail_safely():
    assert paired_block_bootstrap([1] * 100, [2] * 100)["confidence_interval"] is None
    with pytest.raises(ValueError, match="boundaries"):
        paired_block_bootstrap([1] * 8192, [2] * 8191)
    with pytest.raises(ValueError, match="finite"):
        paired_block_bootstrap([float("nan")], [1])
    with pytest.raises(ValueError, match="fixed-size"):
        paired_block_bootstrap([{"nll_sum": 2, "targets": 2}] * 2,
                               [{"nll_sum": 2, "targets": 2}] * 2)


def records(value):
    return [dict(seed=s, nll=value, logit_kl=value, primary_drift=value,
                 serialized_bytes=100, latency_median_ms=100, peak_memory_bytes=100)
            for s in (11, 23, 37)]


def test_promotion_requires_all_seed_drift_uncertainty_and_hardware_gates():
    candidate, baseline = records(1), records(2)
    intervals = {s: {"established_nll_advantage": True} for s in (11, 23, 37)}
    assert assess_promotion(candidate, baseline, intervals, contracts={"comparable": True, "readiness_passed": True})["promoted"]
    candidate[0]["latency_median_ms"] = 105.01
    assert not assess_promotion(candidate, baseline, intervals, contracts={"comparable": True, "readiness_passed": True})["promoted"]
    candidate[0]["latency_median_ms"] = 105
    assert assess_promotion(candidate, baseline, intervals, contracts={"comparable": True, "readiness_passed": True})["promoted"]
    del candidate[0]["primary_drift"]
    result = assess_promotion(candidate, baseline, intervals, contracts={"comparable": True, "readiness_passed": True})
    assert not result["promoted"] and "primary_drift" in result["missing_evidence"]


def test_seed_summary_keeps_missing_and_duplicate_evidence_explicit():
    summary = summarize_seeds(records(1)[:2])
    assert not summary["complete"]
    assert summary["metrics"]["nll"]["std"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        summarize_seeds(records(1) + records(1)[:1])


def test_report_preserves_runs_and_exposes_unmeasured_hardware(tmp_path):
    import json
    run = tmp_path / "baselines-example"
    run.mkdir()
    config = dict(protocol="paper-v2", model_revision="model", dataset_revision="dataset",
                  precision="float16", context=512, stride=256, evaluation_split="test", selection_split="validation",
                  readiness_passed=True, readiness_evidence="readiness-path")
    contract = dict(configuration=config, selection_hash="frozen-selection-hash", splits={s: {"hash": s} for s in ("train", "validation", "test")},
                    targets=[dict(name="projection", shape=[2,2], rank=3, factor_parameters=12)])
    contract["splits"]["calibration_positions"] = {str(s): [0] for s in (11, 23, 37)}
    contract["splits"]["calibration_token_hashes"] = {str(s): "actual-token-hash" for s in (11,23,37)}
    data = []
    for method, nll in [("asvd", 2), ("anchored", 1)]:
        for seed in (11, 23, 37):
            data.append(dict(seed=seed, specification={"method": method},
                             metrics=dict(nll=nll, logit_kl=nll, primary_drift=nll,
                                          nll_blocks=[dict(nll_sum=nll*4096, targets=4096)] * 2),
                             metadata=dict(serialized_bytes=100, factor_parameters=12, sha256=f"{method}-{seed}")))
    for name, obj in [("contract.json", contract), ("status.json", {"state": "completed"}), ("metrics.json", data)]:
        (run / name).write_text(json.dumps(obj))
    before = (run / "metrics.json").read_bytes()
    result = build_run_report(run)
    gate = result["comparisons"][0]["promotion"]
    assert not gate["promoted"]
    assert "latency_median_ms" in gate["missing_evidence"]
    assert report_study({"output_root": str(tmp_path)}) == 0
    assert report_study({"output_root": str(tmp_path)}) == 0
    assert len(list(tmp_path.glob("report-*"))) == 2
    assert (run / "metrics.json").read_bytes() == before
    practical = tmp_path / "practical-example"
    practical.mkdir()
    practical_contract = {**contract, "configuration": {**config, "baseline_run_path": str(run)}}
    timing = [dict(mode=mode, prefill_tokens=length, decode_tokens=128 if mode == "decode" else 0,
                   batch_size=1, warmups=10, measurements=30, use_cache=True,
                   median_seconds=1, peak_allocated_bytes=100, peak_reserved_bytes=100)
              for mode in ("prefill", "decode") for length in (128, 512)]
    practical_records = [dict(label=label, seed=seed, bits=16, specification={"method": method},
                              source_sha256=f"{method}-{seed}", factor_checkpoint_bytes=100,
                              whole_checkpoint={"bytes": 100}, timing=timing)
                         for method, label in [("asvd", "baseline"), ("anchored", "solver")]
                         for seed in (11, 23, 37)]
    for name, obj in [("contract.json", practical_contract), ("status.json", {"state": "completed"}),
                      ("metrics.json", practical_records)]:
        (practical/name).write_text(json.dumps(obj))
    joined = build_run_report(run, practical_directory=practical)
    assert joined["comparisons"][0]["promotion"]["promoted"]
    practical_records[-1]["timing"] = [dict(t) for t in timing]
    practical_records[-1]["timing"][0]["median_seconds"] = 1.06
    (practical/"metrics.json").write_text(json.dumps(practical_records))
    joined = build_run_report(run, practical_directory=practical)
    assert not joined["comparisons"][0]["promotion"]["promoted"]
    assert not joined["comparisons"][0]["promotion"]["all_hardware_conditions_passed"]
    scoped = tmp_path/"composition-example"
    scoped.mkdir()
    grouped = [{**record, "group": group, "rank": 3, "split": "test"} for group in ([0], [1,2]) for record in data]
    for name, obj in [("contract.json", contract), ("status.json", {"state": "completed"}),
                      ("metrics.json", grouped), ("frozen-selection.json", {"strongest_baseline": {"specification": {"method": "asvd"}}})]:
        (scoped/name).write_text(json.dumps(obj))
    result = build_run_report(scoped)
    assert len(result["comparisons"]) == 2
    assert all(r["candidate"]["_report_scope"] == r["baseline"]["_report_scope"] for r in result["comparisons"])

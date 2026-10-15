"""Recovery gates prevent repeated scopes and budget resets after a reboot."""
import copy
import json
from pathlib import Path

import pytest

from src import paper_architecture_recovery as recovery
from test_architecture_panel import complete_frontier_fixture


def test_prefix_reuse_validates_checkpoints_and_rejects_scope_gaps(tmp_path):
    rows, contract, teacher = complete_frontier_fixture(tmp_path)
    assert recovery.validate_recovery_rows(rows[:10], contract, teacher)
    for invalid in (rows[:9]+rows[10:11], rows[:10]+rows[9:10], rows[1:10]):
        with pytest.raises(ValueError, match='ordered prefix'):
            recovery.validate_recovery_rows(invalid, contract, teacher)
    altered = copy.deepcopy(rows[:10])
    altered[-1]['teacher_reference_split'] = 'test'
    with pytest.raises(ValueError, match='wrong split/reference'):
        recovery.validate_recovery_rows(altered, contract, teacher)


def test_interrupted_wall_charge_replaces_stale_status_and_adds_new_cost(tmp_path):
    model = next(iter(recovery.PINNED))
    old = tmp_path/'architecture-replication-old'
    old.mkdir()
    status = old/'status.json'
    status.write_text(json.dumps({'state': 'running', 'elapsed_seconds': 25.6, 'model': model}))
    new = tmp_path/'architecture-replication-new'
    new.mkdir()
    (new/'status.json').write_text(json.dumps({'state': 'incomplete', 'elapsed_seconds': 500., 'model': model}))
    recovery.ACTIVE_CONFIG.clear()
    with pytest.raises(RuntimeError, match='unresolved'):
        recovery.prior_costs(tmp_path)
    recovery.ACTIVE_CONFIG.update(interrupted_runs={str(old.resolve()): {
        'charged_seconds': 3647.1586377620697,
        'source_artifacts': {'status.json': recovery.sha256_file(status)}}})
    total, costs = recovery.prior_costs(tmp_path)
    assert total == costs[model] == 4147.15863776207
    status.write_text(json.dumps({'state': 'running', 'elapsed_seconds': 30., 'model': model}))
    with pytest.raises(RuntimeError, match='unresolved'):
        recovery.prior_costs(tmp_path)
    recovery.ACTIVE_CONFIG.clear()


def test_exhausted_model_does_not_borrow_or_block_other_model(tmp_path, monkeypatch):
    models = list(recovery.PINNED)
    config = {'output_root': str(tmp_path), 'models': [{'model': model} for model in models]}
    monkeypatch.setattr(recovery, 'validate_config', lambda value: None)
    monkeypatch.setattr(recovery, 'frozen_specs', lambda value: None)
    monkeypatch.setattr(recovery, 'completed_model_run', lambda *args: None)
    monkeypatch.setattr(recovery, 'prior_costs', lambda root: (7200., {models[0]: 7200., models[1]: 0.}))
    calls = []
    def execute(config, record, allowance, prior):
        calls.append((record['model'], allowance, prior))
        directory = tmp_path/'architecture-replication-second'
        directory.mkdir()
        (directory/'status.json').write_text(json.dumps({'state': 'completed'}))
        return 0, directory
    monkeypatch.setattr(recovery, 'run_model', execute)
    assert recovery.run_panel(config) == 1
    assert calls == [(models[1], 7200., 7200.)]


def test_generated_recovery_skips_before_fitting_and_keeps_full_completion_gate():
    source = Path(recovery.__file__).read_text()
    assert source.index('if record_scope(case) in recovered_scopes:') < source.index('seed, spec, split = case')
    assert 'validate_completed_rows(rows, contract, validation_teacher)' in source

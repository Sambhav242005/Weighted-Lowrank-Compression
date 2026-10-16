"""Synthetic proofs for diagnostic controls; no model/data download or optimization."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).resolve().parents[2] / "docs/diagnose_gemma270m_20261016.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_signed_identity_preserves_linear_outputs_and_unpaired_relu_can_fail():
    x = torch.tensor([[-2., -1.], [2., 1.]])
    left, right = torch.eye(2), torch.eye(2)
    assert torch.equal(diagnostic.signed_pair(x, left, right), x)
    assert not torch.equal(torch.relu(x), x)


def test_shift_preserves_linear_on_calibrated_domain_but_can_clip_outside():
    x = torch.tensor([[-2., -1.], [2., 1.]])
    left, right = torch.eye(2), torch.eye(2)
    bias = -x.min(dim=0).values + 1e-4
    assert torch.allclose(diagnostic.shifted(x, left, right, bias), x, atol=1e-6)
    unseen = torch.tensor([[-5., -5.]])
    assert not torch.allclose(diagnostic.shifted(unseen, left, right, bias), unseen)

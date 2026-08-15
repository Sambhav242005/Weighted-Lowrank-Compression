import json

import pytest
import torch

from src.compression_core import (
    CompressionRecipe,
    ModuleRecipe,
    PackedLowRankLinear,
    RunContract,
    checkpoint_has_dense_weight,
    fit_activation_weighted_low_rank,
)


def test_run_contract_captures_reproducibility_fields():
    contract = RunContract.capture(
        run_id="unit-001",
        model="tiny",
        model_revision="model-sha",
        tokenizer_revision="tokenizer-sha",
        dataset="toy",
        dataset_revision="data-sha",
        split_hashes={"calibration": "abc", "test": "def"},
        seeds=[11, 23, 37],
        command="pytest tests/unit/test_compression_core.py",
        device="cpu",
    )
    loaded = json.loads(json.dumps(contract.to_dict()))
    assert loaded["seeds"] == [11, 23, 37]
    assert loaded["split_hashes"]["test"] == "def"
    assert loaded["package_versions"]["torch"]


def test_packed_factorized_forward_matches_dense_product():
    torch.manual_seed(7)
    u = torch.randn(5, 3)
    v = torch.randn(3, 4)
    bias = torch.randn(5)
    layer = PackedLowRankLinear.from_factors(u, v, bias, bits=16)
    x = torch.randn(2, 6, 4)
    expected = x @ v.T @ u.T + bias
    assert torch.allclose(layer(x), expected, atol=2e-3, rtol=2e-3)
    assert not hasattr(layer, "weight")
    assert not checkpoint_has_dense_weight(layer.state_dict())


@pytest.mark.parametrize("bits,tolerance", [(8, 0.08), (4, 0.6)])
def test_quantized_packed_factors_round_trip(bits, tolerance):
    torch.manual_seed(9)
    u = torch.randn(6, 2)
    v = torch.randn(2, 5)
    layer = PackedLowRankLinear.from_factors(u, v, bits=bits)
    x = torch.randn(3, 5)
    expected = x @ v.T @ u.T
    assert torch.allclose(layer(x), expected, atol=tolerance, rtol=tolerance)
    assert layer.serialized_bytes() < (u.numel() + v.numel()) * 4
    assert not checkpoint_has_dense_weight(layer.state_dict())


def test_recipe_enforces_total_serialized_byte_budget():
    recipe = CompressionRecipe(
        name="tiny",
        modules=(ModuleRecipe("layer.o_proj", "packed_lowrank", rank=2, bits=4),),
        total_serialized_byte_budget=10,
    )
    assert recipe.serialized_bytes({"layer.o_proj": 9}) == 9
    with pytest.raises(ValueError, match="exceeds byte budget"):
        recipe.serialized_bytes({"layer.o_proj": 11})


def test_activation_weighted_fit_honors_bias_and_orientation():
    torch.manual_seed(13)
    weight = torch.randn(5, 5)
    bias = torch.randn(5)
    inputs = torch.randn(5, 80)
    outputs = weight @ inputs + bias[:, None]
    fitted = fit_activation_weighted_low_rank(
        weight, inputs, outputs, rank=5, beta=0.1, bias=bias, ridge_fraction=1e-8
    )
    assert fitted.shape == weight.shape
    assert torch.allclose(fitted, weight, atol=2e-3, rtol=2e-3)

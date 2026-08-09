import pytest
import torch

from src.representations import fit_fourier, fit_low_rank_product, fit_svd


def test_svd_reconstructs_known_rank_two_matrix_and_reports_metadata():
    torch.manual_seed(7)
    weight = torch.randn(6, 2) @ torch.randn(2, 5)

    result = fit_svd(weight, rank=2)
    reconstructed = result.reconstruct()

    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == weight.dtype
    assert torch.allclose(reconstructed, weight, atol=1e-5, rtol=1e-5)
    assert result.n_params == 6 * 2 + 2 + 2 * 5
    assert result.original_params == weight.numel()
    assert result.compression_ratio == pytest.approx(weight.numel() / result.n_params)
    assert result.metadata["rank"] == 2
    assert result.metadata["relative_error"] == pytest.approx(0.0, abs=1e-5)


def test_fourier_reconstructs_when_all_coefficients_are_retained():
    rows = torch.arange(4, dtype=torch.float32).view(-1, 1)
    cols = torch.arange(4, dtype=torch.float32).view(1, -1)
    weight = torch.sin(rows) + torch.cos(cols / 2)

    result = fit_fourier(weight, n_components=12)
    reconstructed = result.reconstruct()

    assert reconstructed.shape == weight.shape
    assert reconstructed.dtype == weight.dtype
    assert torch.allclose(reconstructed, weight, atol=1e-5, rtol=1e-5)
    assert result.n_params == 2 * 12 + 2
    assert result.original_params == weight.numel()
    assert result.compression_ratio == pytest.approx(weight.numel() / result.n_params)
    assert result.metadata["n_components"] == 12
    assert result.metadata["relative_error"] == pytest.approx(0.0, abs=1e-5)


def test_low_rank_product_has_deterministic_shape_and_parameter_metadata():
    torch.manual_seed(11)
    weight = torch.randn(5, 4)

    first = fit_low_rank_product(weight, rank=2, steps=4)
    second = fit_low_rank_product(weight, rank=2, steps=4)

    assert first.reconstruct().shape == weight.shape
    assert first.reconstruct().dtype == weight.dtype
    assert torch.isfinite(first.reconstruct()).all()
    assert torch.allclose(first.reconstruct(), second.reconstruct())
    assert first.n_params == 5 * 2 + 2 * 4
    assert first.original_params == weight.numel()
    assert first.compression_ratio == pytest.approx(weight.numel() / first.n_params)
    assert first.metadata["rank"] == 2
    assert torch.isfinite(torch.tensor(first.metadata["final_loss"]))

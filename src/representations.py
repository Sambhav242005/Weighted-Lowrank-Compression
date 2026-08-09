"""
Representation fitting: SVD, Fourier, Hypernetwork, Low-rank.
All operations accept a device parameter for GPU acceleration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class RepresentationResult:
    name: str
    n_params: int
    original_params: int
    compression_ratio: float
    reconstruct: Callable[[], torch.Tensor]
    metadata: dict


def fit_svd(weight: torch.Tensor, rank: int, name: str = "svd", device: str = "cpu") -> RepresentationResult:
    """Truncated SVD on GPU."""
    W = weight.float().to(device)
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]
    n_params = m * rank + rank + rank * n

    def reconstruct():
        return ((U_r * S_r.unsqueeze(0)) @ Vt_r).cpu()

    error = torch.norm(W - reconstruct().to(device), 'fro') / torch.norm(W, 'fro')
    return RepresentationResult(f"{name}_r{rank}", n_params, m*n, (m*n)/n_params, reconstruct,
                                {"rank": rank, "relative_error": float(error)})


def fit_svd_at_threshold(weight: torch.Tensor, variance_threshold: float = 0.99,
                         name: str = "svd", device: str = "cpu") -> RepresentationResult:
    W = weight.float().to(device)
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    return fit_svd(weight, rank, name, device)


def fit_fourier(weight: torch.Tensor, n_components: int, name: str = "fourier",
                device: str = "cpu") -> RepresentationResult:
    """2D Fourier - keep top-n_components coefficients."""
    W = weight.float().to(device)
    m, n = W.shape
    W_mean, W_std = W.mean(), W.std()
    W_norm = (W - W_mean) / (W_std + 1e-8)

    fft_result = torch.fft.rfft2(W_norm)
    magnitudes = torch.abs(fft_result).reshape(-1)
    k = min(n_components, magnitudes.numel())
    topk = torch.topk(magnitudes, k)
    indices = topk.indices
    values = fft_result.reshape(-1)[indices]

    n_params = 2 * k + 2

    def reconstruct():
        flat = torch.zeros(magnitudes.numel(), dtype=fft_result.dtype, device=device)
        flat[indices] = values
        return (torch.fft.irfft2(flat.reshape(fft_result.shape), s=(m, n)) * W_std + W_mean).cpu()

    W_approx = reconstruct().to(device)
    error = torch.norm(W - W_approx, 'fro') / torch.norm(W, 'fro')
    return RepresentationResult(f"{name}_k{n_components}", n_params, m*n, (m*n)/n_params, reconstruct,
                                {"n_components": n_components, "relative_error": float(error)})


def fit_fourier_at_threshold(weight: torch.Tensor, variance_threshold: float = 0.99,
                             name: str = "fourier", device: str = "cpu") -> RepresentationResult:
    W = weight.float().to(device)
    W_mean, W_std = W.mean(), W.std()
    W_norm = (W - W_mean) / (W_std + 1e-8)
    fft_result = torch.fft.rfft2(W_norm)
    magnitudes = torch.abs(fft_result).reshape(-1)
    total_energy = (magnitudes ** 2).sum()
    sorted_mag, _ = torch.sort(magnitudes, descending=True)
    cum_energy = torch.cumsum(sorted_mag ** 2, dim=0) / total_energy
    n_components = int(torch.searchsorted(cum_energy, variance_threshold)) + 1
    return fit_fourier(weight, n_components, name, device)


def fit_hypernetwork(weight: torch.Tensor, hidden_dim: int = 64, n_layers: int = 2,
                     name: str = "hypernet", device: str = "cpu", steps: int = 2000) -> RepresentationResult:
    """Tiny MLP that generates weights from (row, col) coordinates."""
    W = weight.float().to(device)
    m, n = W.shape

    grid_rows, grid_cols = torch.meshgrid(torch.linspace(0, 1, m, device=device),
                                           torch.linspace(0, 1, n, device=device), indexing='ij')
    coords = torch.stack([grid_rows.flatten(), grid_cols.flatten()], dim=1)
    targets = W.flatten()

    layers = [torch.nn.Linear(2, hidden_dim, device=device), torch.nn.Tanh()]
    for _ in range(n_layers - 1):
        layers.extend([torch.nn.Linear(hidden_dim, hidden_dim, device=device), torch.nn.Tanh()])
    layers.append(torch.nn.Linear(hidden_dim, 1, device=device))
    hypernet = torch.nn.Sequential(*layers)

    optimizer = torch.optim.Adam(hypernet.parameters(), lr=1e-3)
    for _ in range(steps):
        loss = F.mse_loss(hypernet(coords).squeeze(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    n_params = sum(p.numel() for p in hypernet.parameters())

    def reconstruct():
        with torch.no_grad():
            return hypernet(coords).squeeze().reshape(m, n).cpu()

    W_approx = reconstruct().to(device)
    error = torch.norm(W - W_approx, 'fro') / torch.norm(W, 'fro')
    return RepresentationResult(f"{name}_h{hidden_dim}", n_params, m*n, (m*n)/n_params, reconstruct,
                                {"hidden_dim": hidden_dim, "relative_error": float(error),
                                 "final_loss": float(loss.detach())})


def fit_low_rank_product(weight: torch.Tensor, rank: int, name: str = "lowrank",
                         device: str = "cpu", steps: int = 1000) -> RepresentationResult:
    """Low-rank factorization W ≈ A @ B trained with gradient descent."""
    W = weight.float().to(device)
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    A = torch.nn.Parameter(U[:, :rank] * S[:rank].unsqueeze(0))
    B = torch.nn.Parameter(Vt[:rank, :])
    optimizer = torch.optim.Adam([A, B], lr=1e-3)
    for _ in range(steps):
        loss = F.mse_loss(A @ B, W)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    n_params = m * rank + rank * n

    def reconstruct():
        return (A.detach() @ B.detach()).cpu()

    W_approx = reconstruct().to(device)
    error = torch.norm(W - W_approx, 'fro') / torch.norm(W, 'fro')
    return RepresentationResult(f"{name}_r{rank}", n_params, m*n, (m*n)/n_params, reconstruct,
                                {"rank": rank, "relative_error": float(error),
                                 "final_loss": float(loss.detach())})

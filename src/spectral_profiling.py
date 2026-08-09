"""
Spectral profiling: rank, singular value spectrum, and spectral decay analysis.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats as scipy_stats


@dataclass
class SpectralProfile:
    matrix_name: str
    shape: tuple
    rank: int  # numerical rank (above tolerance)
    singular_values: np.ndarray
    spectral_norm: float
    frobenius_norm: float
    nuclear_norm: float
    explained_variance_ratio: np.ndarray  # cumulative variance explained
    decay_type: str  # 'exponential' or 'power_law'
    decay_rate: float
    rank_at_99percent: int  # rank needed for 99% variance
    rank_at_95percent: int
    rank_at_90percent: int
    eff_rank: float  # entropy-based effective rank


def compute_effective_rank(sv: np.ndarray) -> float:
    """Compute effective rank via entropy of normalized singular values."""
    sv_norm = sv / sv.sum()
    sv_norm = sv_norm[sv_norm > 1e-15]
    entropy = -np.sum(sv_norm * np.log(sv_norm))
    return np.exp(entropy)


def fit_spectral_decay(sv: np.ndarray):
    """Determine if decay is exponential or power-law, return fit parameters."""
    sv = sv[sv > 1e-10]
    if len(sv) < 3:
        return "unknown", 0.0

    x = np.arange(1, len(sv) + 1, dtype=float)

    # Fit exponential: log(sv) = a - b*x
    log_sv = np.log(sv)
    slope_exp, intercept_exp, r_exp, _, _ = scipy_stats.linregress(x, log_sv)

    # Fit power law: log(sv) = a - b*log(x)
    log_x = np.log(x)
    slope_pl, intercept_pl, r_pl, _, _ = scipy_stats.linregress(log_x, log_sv)

    r2_exp = r_exp ** 2
    r2_pl = r_pl ** 2

    if r2_exp > r2_pl:
        return "exponential", -slope_exp
    else:
        return "power_law", -slope_pl


def profile_spectral(matrix: torch.Tensor, name: str, rank_tol: float = 1e-6) -> SpectralProfile:
    """Full spectral profile of a weight matrix."""
    W = matrix.float().cpu().numpy()
    m, n = W.shape

    # SVD
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)

    # Numerical rank
    rank = int(np.sum(sv > rank_tol * sv[0]))

    # Variance explained
    sv_sq = sv ** 2
    total_var = sv_sq.sum()
    cumvar = np.cumsum(sv_sq) / total_var

    # Ranks at thresholds
    rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1

    # Spectral decay
    decay_type, decay_rate = fit_spectral_decay(sv)

    # Effective rank
    eff_rank = compute_effective_rank(sv)

    return SpectralProfile(
        matrix_name=name,
        shape=W.shape,
        rank=rank,
        singular_values=sv,
        spectral_norm=float(sv[0]),
        frobenius_norm=float(np.linalg.norm(W, 'fro')),
        nuclear_norm=float(sv.sum()),
        explained_variance_ratio=cumvar,
        decay_type=decay_type,
        decay_rate=decay_rate,
        rank_at_99percent=rank_99,
        rank_at_95percent=rank_95,
        rank_at_90percent=rank_90,
        eff_rank=eff_rank,
    )


def profile_all_spectral(weights: dict) -> List[SpectralProfile]:
    """Run spectral profiling on all extracted weight matrices."""
    profiles = []
    for name, wm in weights.items():
        if wm.block_type in ("embedding", "unembedding"):
            continue  # skip embeddings for now
        p = profile_spectral(wm.tensor, name)
        profiles.append(p)
        print(f"  {name}: shape={p.shape}, rank={p.rank}, "
              f"decay={p.decay_type} (rate={p.decay_rate:.3f}), "
              f"eff_rank={p.eff_rank:.1f}/{min(p.shape)}")
    return profiles


def print_spectral_summary(profiles: List[SpectralProfile]):
    """Print a summary table of spectral profiles."""
    print("\n" + "=" * 90)
    print("SPECTRAL PROFILE SUMMARY")
    print("=" * 90)
    print(f"{'Name':<30} {'Shape':<14} {'Rank':<6} {'99%':<6} {'95%':<6} {'90%':<6} "
          f"{'Decay':<12} {'Rate':<8} {'EffRank':<8}")
    print("-" * 90)
    for p in profiles:
        shape_str = f"{p.shape[0]}x{p.shape[1]}"
        print(f"{p.matrix_name:<30} {shape_str:<14} {p.rank:<6} {p.rank_at_99percent:<6} "
              f"{p.rank_at_95percent:<6} {p.rank_at_90percent:<6} "
              f"{p.decay_type:<12} {p.decay_rate:<8.3f} {p.eff_rank:<8.1f}")
    print("=" * 90)

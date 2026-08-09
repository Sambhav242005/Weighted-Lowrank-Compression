"""
Statistical profiling: sparsity, weight distributions, row/column correlation, entropy.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats as scipy_stats


@dataclass
class StatisticalProfile:
    matrix_name: str
    shape: tuple
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    skewness: float
    kurtosis: float
    sparsity_1e5: float  # fraction of elements with |w| < 1e-5
    sparsity_1e3: float
    sparsity_1e2: float
    entropy_bits: float  # entropy of binned weight distribution
    row_norm_std: float  # std of row norms (if high, some rows are more important)
    col_norm_std: float  # std of column norms
    row_correlation_mean: float  # mean pairwise cosine similarity of rows
    col_correlation_mean: float  # mean pairwise cosine similarity of cols
    is_symmetric: bool
    offdiagonal_correlation: float  # correlation between upper and lower triangle (for square matrices)


def compute_entropy(x: np.ndarray, n_bins: int = 100) -> float:
    """Compute entropy of binned distribution in bits."""
    hist, _ = np.histogram(x, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = (x.max() - x.min()) / n_bins if x.max() != x.min() else 1.0
    probs = hist * bin_width
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-15)))


def compute_pairwise_cosine_similarity(matrix: np.ndarray, max_pairs: int = 500) -> float:
    """Compute mean pairwise cosine similarity of rows (or columns)."""
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    
    # Sample pairs for efficiency
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    
    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = matrix / norms
    
    similarities = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        sim = np.dot(normalized[i], normalized[j])
        similarities.append(sim)
    
    return float(np.mean(similarities))


def profile_statistical(matrix: torch.Tensor, name: str) -> StatisticalProfile:
    """Compute full statistical profile of a weight matrix."""
    W = matrix.float().cpu().numpy().flatten()
    
    # Basic statistics
    mean = float(np.mean(W))
    std = float(np.std(W))
    median = float(np.median(W))
    
    # Distribution shape
    skewness = float(scipy_stats.skew(W))
    kurtosis = float(scipy_stats.kurtosis(W))
    
    # Sparsity
    abs_w = np.abs(W)
    sparsity_1e5 = float(np.mean(abs_w < 1e-5))
    sparsity_1e3 = float(np.mean(abs_w < 1e-3))
    sparsity_1e2 = float(np.mean(abs_w < 1e-2))
    
    # Entropy
    entropy = compute_entropy(W)
    
    # Row/column analysis (unflatten)
    W_2d = matrix.float().cpu().numpy()
    row_norms = np.linalg.norm(W_2d, axis=1)
    col_norms = np.linalg.norm(W_2d, axis=0)
    
    row_norm_std = float(np.std(row_norms) / (np.mean(row_norms) + 1e-10))
    col_norm_std = float(np.std(col_norms) / (np.mean(col_norms) + 1e-10))
    
    # Correlation analysis
    row_corr = compute_pairwise_cosine_similarity(W_2d.T)  # similarity of rows
    col_corr = compute_pairwise_cosine_similarity(W_2d)     # similarity of columns
    
    # Symmetry check
    m, n = W_2d.shape
    is_symmetric = False
    offdiag_corr = 0.0
    if m == n:
        is_symmetric = np.allclose(W_2d, W_2d.T, atol=1e-4)
        # Off-diagonal correlation
        upper = W_2d[np.triu_indices(m, k=1)]
        lower = W_2d[np.tril_indices(m, k=-1)]
        if len(upper) > 10:
            offdiag_corr = float(np.corrcoef(upper, lower)[0, 1])
    
    return StatisticalProfile(
        matrix_name=name,
        shape=matrix.shape,
        mean=mean,
        std=std,
        median=median,
        min_val=float(W.min()),
        max_val=float(W.max()),
        skewness=skewness,
        kurtosis=kurtosis,
        sparsity_1e5=sparsity_1e5,
        sparsity_1e3=sparsity_1e3,
        sparsity_1e2=sparsity_1e2,
        entropy_bits=entropy,
        row_norm_std=row_norm_std,
        col_norm_std=col_norm_std,
        row_correlation_mean=row_corr,
        col_correlation_mean=col_corr,
        is_symmetric=is_symmetric,
        offdiagonal_correlation=offdiag_corr,
    )


def profile_all_statistical(weights: dict) -> List[StatisticalProfile]:
    """Run statistical profiling on all weight matrices."""
    profiles = []
    for name, wm in weights.items():
        if wm.block_type in ("embedding", "unembedding"):
            continue
        p = profile_statistical(wm.tensor, name)
        profiles.append(p)
        print(f"  {name}: std={p.std:.4f}, entropy={p.entropy_bits:.2f} bits, "
              f"skew={p.skewness:.3f}, kurt={p.kurtosis:.3f}")
    return profiles


def print_statistical_summary(profiles: List[StatisticalProfile]):
    """Print a summary of statistical profiles."""
    print("\n" + "=" * 100)
    print("STATISTICAL PROFILE SUMMARY")
    print("=" * 100)
    print(f"{'Name':<28} {'Mean':<10} {'Std':<10} {'Skew':<8} {'Kurt':<8} "
          f"{'Sp<1e-3':<8} {'Entropy':<8} {'RowVar':<8} {'ColVar':<8}")
    print("-" * 100)
    for p in profiles:
        print(f"{p.matrix_name:<28} {p.mean:<10.4f} {p.std:<10.4f} "
              f"{p.skewness:<8.3f} {p.kurtosis:<8.3f} "
              f"{p.sparsity_1e3:<8.3f} {p.entropy_bits:<8.2f} "
              f"{p.row_norm_std:<8.3f} {p.col_norm_std:<8.3f}")
    print("=" * 100)

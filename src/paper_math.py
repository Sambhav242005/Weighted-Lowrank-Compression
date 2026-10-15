"""Protocol-v2 decompositions; legacy solver remains separately identifiable."""
import torch


def factors(weight, rank):
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    return u[:, :rank] * s[:rank], vh[:rank]


def weighted_factors(weight, inputs, rank, *, beta=0.0, ridge_fraction=0.0, rcond=1e-7,
                     outputs=None, bias=None):
    """Anchored objective, optionally with measured outputs and extra ridge.

    H=XX.T+beta*n*I. Analytic cross=W*H; measured cross=Y_b*X.T+beta*n*W.
    Bias is subtracted only for explicitly supplied measured outputs.
    Extra ridge replaces H by H+lambda*I
    in the solve and truncation geometry. Singular zero-anchor directions use
    a declared spectral pseudoinverse with relative threshold rcond.
    """
    if weight.ndim != 2 or inputs.ndim != 2 or weight.shape[1] != inputs.shape[0]:
        raise ValueError("weight [out,in] and inputs [in,tokens] required")
    if not 0 < rank <= min(weight.shape) or inputs.shape[1] == 0:
        raise ValueError("invalid rank or empty calibration")
    if min(beta, ridge_fraction) < 0 or not 0 < rcond < 1:
        raise ValueError("invalid regularization")
    if not torch.isfinite(weight).all() or not torch.isfinite(inputs).all():
        raise ValueError('nonfinite weight or calibration inputs')
    if outputs is not None:
        if outputs.ndim != 2 or outputs.shape != (weight.shape[0], inputs.shape[1]):
            raise ValueError('outputs must be [out,tokens]')
        if not torch.isfinite(outputs).all():
            raise ValueError('nonfinite measured outputs')
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0] or not torch.isfinite(bias).all()):
            raise ValueError('invalid bias')
    elif bias is not None:
        raise ValueError('bias subtraction requires measured outputs')
    w, x = weight.double(), inputs.double()
    h = x @ x.T + beta * x.shape[1] * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
    cross = w @ h
    if outputs is not None:
        y = outputs.to(device=x.device, dtype=x.dtype)
        if bias is not None:
            y = y-bias.to(device=x.device, dtype=x.dtype)[:, None]
        cross = y @ x.T + beta*x.shape[1]*w
    values, vectors = torch.linalg.eigh(h)
    values = values.clamp_min(0)
    regularized = values + ridge_fraction * values.mean()
    keep = regularized > rcond * regularized.max()
    inv = torch.where(keep, regularized.clamp_min(torch.finfo(x.dtype).tiny).reciprocal(), 0)
    sqrt = regularized.sqrt()
    full = (cross @ vectors * inv) @ vectors.T
    weighted = (full @ vectors * sqrt) @ vectors.T
    u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
    inverse_half = torch.where(keep, inv.sqrt(), 0)
    left = u[:, :rank] * s[:rank]
    right = ((vh[:rank] @ vectors) * inverse_half) @ vectors.T
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise FloatingPointError("nonfinite factors")
    return left.float(), right.float()


def asvd_factors(weight, inputs, rank, *, alpha=0.5, floor=1e-8):
    scale = inputs.float().abs().mean(dim=1).clamp_min(floor).pow(alpha)
    u, v = factors(weight.float() * scale, rank)
    return u, v / scale

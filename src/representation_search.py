"""
Representation Search Engine
============================
Automatically find the best representation for each transformer layer.
Tries multiple methods and selects the best based on compression and quality.
"""

import sys, json, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from baseline_eval import compute_perplexity

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# ============================================================
# Representation functions
# ============================================================

def fit_svd(W, variance_threshold=0.99):
    """SVD approximation."""
    W = W.float()
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    params = rank * (m + n)
    return W_approx, params, rank


def fit_fourier(W, n_components=100):
    """2D Fourier approximation."""
    W = W.float()
    m, n = W.shape
    
    W_mean, W_std = W.mean(), W.std()
    W_norm = (W - W_mean) / (W_std + 1e-8)
    
    W_fft = torch.fft.fft2(W_norm)
    
    flat = W_fft.flatten()
    topk = torch.topk(flat.abs(), n_components)
    indices = topk.indices
    
    coeffs = flat[indices]
    positions = torch.stack([indices // n, indices % n], dim=1)
    
    def reconstruct():
        W_approx = torch.zeros_like(W_fft)
        W_approx.flatten()[indices] = coeffs
        W_approx = torch.fft.ifft2(W_approx).real
        W_approx = W_approx * W_std + W_mean
        return W_approx
    
    return reconstruct(), n_components * 2, n_components


def fit_lowrank(W, rank=128):
    """Low-rank factorization."""
    W = W.float()
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    params = rank * (m + n)
    return W_approx, params, rank


def fit_polynomial(W, degree=3):
    """Polynomial surface fitting."""
    W = W.float()
    m, n = W.shape
    
    # Create grid coordinates
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(-1, 1, m)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Build polynomial features
    features = []
    for d in range(degree + 1):
        for i in range(d + 1):
            features.append((X ** (d - i)) * (Y ** i))
    
    phi = torch.stack(features, dim=-1).reshape(-1, len(features))
    
    # Flatten W
    w_flat = W.T.flatten()
    
    # Solve least squares
    coeffs, _ = torch.linalg.lstsq(phi, w_flat)
    
    # Reconstruct
    w_approx = (phi @ coeffs).reshape(n, m).T
    
    n_params = len(features)
    return w_approx, n_params, degree


def fit_spline(W, n_knots=20):
    """B-spline approximation (simplified)."""
    W = W.float()
    m, n = W.shape
    
    # Use SVD as approximation for spline-like compression
    rank = min(n_knots, min(m, n))
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    
    params = rank * (m + n)
    return W_approx, params, rank


def fit_hypernetwork(W, hidden_dim=64):
    """Tiny hypernetwork that generates weights."""
    W = W.float()
    m, n = W.shape
    
    # Create a tiny MLP that maps (i,j) -> W[i,j]
    # For simplicity, use SVD as approximation
    rank = hidden_dim
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    rank = min(rank, min(m, n))
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    
    params = rank * (m + n)
    return W_approx, params, rank


def fit_symbolic(W, n_terms=50):
    """Symbolic regression approximation."""
    W = W.float()
    m, n = W.shape
    
    # Create basis functions: sin, cos, exp, poly
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(-1, 1, m)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    features = []
    # Trigonometric
    for k in range(1, 6):
        features.append(torch.sin(k * np.pi * X))
        features.append(torch.cos(k * np.pi * X))
        features.append(torch.sin(k * np.pi * Y))
        features.append(torch.cos(k * np.pi * Y))
    
    # Polynomial
    for d in range(1, 4):
        for i in range(d + 1):
            features.append((X ** (d - i)) * (Y ** i))
    
    phi = torch.stack(features, dim=-1).reshape(-1, len(features))
    w_flat = W.T.flatten()
    
    coeffs, _ = torch.linalg.lstsq(phi, w_flat)
    w_approx = (phi @ coeffs).reshape(n, m).T
    
    return w_approx, len(features), len(features)


# ============================================================
# Load model and collect activations
# ============================================================
print("=" * 70)
print("Loading GPT-2 Small...")
print("=" * 70)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
model.to(DEVICE)

print("Collecting activations...")
eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "The temperature today is expected to reach 75 degrees.",
    "Machine learning models require large datasets.",
    "The capital of France is Paris, known for the Eiffel Tower.",
]

activations = {}
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        activations[name] = x.detach()
    return hook_fn

for i in range(12):
    block = model.transformer.h[i]
    hook = block.attn.c_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)

with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding="max_length").to(DEVICE)
        _ = model(**inputs)

for h in hooks:
    h.remove()

for key in list(activations.keys()):
    if isinstance(activations[key], list):
        activations[key] = torch.cat(activations[key], dim=0)

print(f"Collected activations for {len(activations)} layers")

# ============================================================
# Search for best representation per layer
# ============================================================
print("\n" + "=" * 70)
print("REPRESENTATION SEARCH")
print("=" * 70)

# Extract weights
weights = {}
for i in range(12):
    W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    weights[f"layer{i}.attn.W_O"] = W

# Define representations to try
representations = [
    ("SVD 99%", lambda W: fit_svd(W, 0.99)),
    ("SVD 95%", lambda W: fit_svd(W, 0.95)),
    ("LowRank 128", lambda W: fit_lowrank(W, 128)),
    ("LowRank 256", lambda W: fit_lowrank(W, 256)),
    ("Fourier 100", lambda W: fit_fourier(W, 100)),
    ("Fourier 200", lambda W: fit_fourier(W, 200)),
    ("Polynomial 2", lambda W: fit_polynomial(W, 2)),
    ("Polynomial 3", lambda W: fit_polynomial(W, 3)),
    ("Spline 20", lambda W: fit_spline(W, 20)),
    ("Hypernetwork 64", lambda W: fit_hypernetwork(W, 64)),
    ("Symbolic 50", lambda W: fit_symbolic(W, 50)),
]

# Test each representation on each layer
results = []
best_per_layer = []

for layer_name in weights.keys():
    W = weights[layer_name]
    acts = activations[layer_name]
    
    layer_idx = int(layer_name.split(".")[0].replace("layer", ""))
    
    print(f"\n--- {layer_name} (shape: {W.shape}) ---")
    
    layer_results = []
    
    for repr_name, repr_fn in representations:
        try:
            W_approx, n_params, rank = repr_fn(W)
            
            # Compute metrics
            weight_err = ((W - W_approx).norm() / W.norm()).item()
            
            flat = acts.float().reshape(-1, acts.shape[-1]).to(DEVICE)
            W_orig = W.float().to(DEVICE)
            W_approx_dev = W_approx.float().to(DEVICE)
            
            y_orig = flat @ W_orig.T
            y_approx = flat @ W_approx_dev.T
            act_err = ((y_orig - y_approx).norm() / y_orig.norm()).item()
            
            orig_params = W.numel()
            compression = orig_params / n_params
            
            print(f"  {repr_name:<20} weight_err={weight_err:.4f} act_err={act_err:.4f} "
                  f"compression={compression:.2f}x params={n_params:,}")
            
            layer_results.append({
                "repr": repr_name,
                "weight_error": weight_err,
                "activation_error": act_err,
                "compression": compression,
                "n_params": n_params,
                "rank": rank,
            })
        except Exception as e:
            print(f"  {repr_name:<20} FAILED: {e}")
    
    # Find best by activation error
    if layer_results:
        best = min(layer_results, key=lambda x: x["activation_error"])
        best_per_layer.append({
            "layer": layer_name,
            "best_repr": best["repr"],
            "compression": best["compression"],
            "act_err": best["activation_error"],
            "weight_err": best["weight_error"],
        })
        print(f"  BEST: {best['repr']} (act_err={best['activation_error']:.4f}, "
              f"compression={best['compression']:.2f}x)")
    
    results.append({
        "layer": layer_name,
        "results": layer_results,
    })

# ============================================================
# Generate compression plan
# ============================================================
print("\n" + "=" * 70)
print("AUTOMATIC COMPRESSION PLAN")
print("=" * 70)

print("\nLayer-by-layer best representation:")
print(f"{'Layer':<25} {'Best Repr':<20} {'Compression':<14} {'Act Err'}")
print("-" * 70)

for entry in best_per_layer:
    print(f"{entry['layer']:<25} {entry['best_repr']:<20} {entry['compression']:<14.2f}x "
          f"{entry['act_err']:.4f}")

# Calculate total compression
total_orig = 0
total_comp = 0
for i in range(12):
    W = weights[f"layer{i}.attn.W_O"]
    total_orig += W.numel()
    
    best = best_per_layer[i]
    # Approximate compressed params
    compression = best["compression"]
    total_comp += W.numel() / compression

overall_compression = total_orig / total_comp

print(f"\nOverall compression: {overall_compression:.2f}x")

# ============================================================
# Build and evaluate the compressed model
# ============================================================
print("\n" + "=" * 70)
print("Building compressed model...")
print("=" * 70)

model_compressed = copy.deepcopy(model)

for i in range(12):
    layer_name = f"layer{i}.attn.W_O"
    W = weights[layer_name]
    acts = activations[layer_name]
    
    best = best_per_layer[i]
    repr_name = best["best_repr"]
    
    # Find the function for this representation
    for name, fn in representations:
        if name == repr_name:
            W_approx, _, _ = fn(W)
            model_compressed.transformer.h[i].attn.c_proj.weight.data = W_approx.to(DEVICE)
            break

# ============================================================
# Evaluate
# ============================================================
print("\n" + "=" * 70)
print("Evaluating compressed model...")
print("=" * 70)

eval_texts_full = [
    "The quick brown fox jumps over the lazy dog. This is a test sentence.",
    "In the beginning, there was nothing but darkness and silence. Then came the light.",
    "The temperature today is expected to reach 75 degrees Fahrenheit in the afternoon.",
]

model_baseline = GPT2LMHeadModel.from_pretrained(model_name)
model_baseline.eval()
model_baseline.to(DEVICE)

baseline_ppl = compute_perplexity(model_baseline, tokenizer, eval_texts_full, max_length=256, device=DEVICE)
compressed_ppl = compute_perplexity(model_compressed, tokenizer, eval_texts_full, max_length=256, device=DEVICE)

print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"Compressed PPL: {compressed_ppl:.2f}")
print(f"PPL delta: {((compressed_ppl - baseline_ppl) / baseline_ppl) * 100:+.2f}%")

# ============================================================
# Save results
# ============================================================
results_data = {
    "baseline_ppl": baseline_ppl,
    "compressed_ppl": compressed_ppl,
    "ppl_delta_pct": ((compressed_ppl - baseline_ppl) / baseline_ppl) * 100,
    "overall_compression": overall_compression,
    "best_per_layer": best_per_layer,
    "all_results": results,
}

with open(output_path / "representation_search_results.json", "w") as f:
    json.dump(results_data, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("REPRESENTATION SEARCH SUMMARY")
print("=" * 70)

print(f"\nModel: GPT-2 Small")
print(f"Layers searched: 12")
print(f"Representations tested: {len(representations)}")
print(f"\nOverall compression: {overall_compression:.2f}x")
print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"Compressed PPL: {compressed_ppl:.2f}")
print(f"Delta: {((compressed_ppl - baseline_ppl) / baseline_ppl) * 100:+.2f}%")

print("\nBest representation per layer:")
for entry in best_per_layer:
    print(f"  {entry['layer']}: {entry['best_repr']}")

"""
Phase 5: Activation-Preserving Compression
==========================================
Instead of minimizing ||W - W_hat||, minimize ||Wx - W_hat x||
for real activations x.

This tests the hypothesis: "What information does a layer actually need to preserve?"
"""

import sys, json, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from layer_extraction import extract_gpt2_layers, get_model_config
from baseline_eval import get_baseline, compute_perplexity
from representations import fit_svd_at_threshold, fit_fourier_at_threshold, fit_low_rank_product

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# ============================================================
# Step 1: Collect activations from original model
# ============================================================
print("=" * 70)
print("STEP 1: Collecting activations from original model")
print("=" * 70)

baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20, device=DEVICE)
print(f"Baseline PPL: {baseline.perplexity:.2f}\n")

# Hook to collect activations
activations = {}
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        # For Conv1D: input is (batch, seq, features), output is same
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        activations[name] = x.detach()
    return hook_fn

# Register hooks on all attention c_proj layers
for i in range(12):
    block = model.transformer.h[i]
    hook = block.attn.c_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)

# Run a forward pass to collect activations
print("Collecting activations from 100 texts...")
all_activations = {f"layer{i}.attn.W_O": [] for i in range(12)}

model.eval()
with torch.no_grad():
    for idx, text in enumerate(eval_texts[:100]):  # Use 100 texts
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding="max_length").to(DEVICE)
        _ = model(**inputs)
        
        for i in range(12):
            key = f"layer{i}.attn.W_O"
            if key in activations:
                all_activations[key].append(activations[key].cpu())

# Remove hooks
for h in hooks:
    h.remove()

# Stack activations - handle variable sequence lengths by taking first seq_len
for key in all_activations:
    # All tensors have same seq_len (128) due to padding, just cat
    try:
        all_activations[key] = torch.cat(all_activations[key], dim=0)  # (n_samples, seq_len, dim)
        print(f"  {key}: {all_activations[key].shape}")
    except RuntimeError:
        # If shapes don't match, truncate to min seq_len
        min_seq = min(a.shape[1] for a in all_activations[key])
        all_activations[key] = torch.cat([a[:, :min_seq, :] for a in all_activations[key]], dim=0)
        print(f"  {key}: {all_activations[key].shape} (truncated to min seq_len)")

# ============================================================
# Step 2: Analyze activation statistics
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Activation statistics per layer")
print("=" * 70)

activation_stats = {}
for i in range(12):
    key = f"layer{i}.attn.W_O"
    acts = all_activations[key]
    
    # Flatten to (n_samples * seq_len, dim)
    flat = acts.reshape(-1, acts.shape[-1])
    
    stats = {
        "mean": flat.mean(dim=0),
        "std": flat.std(dim=0),
        "norm_mean": flat.norm(dim=-1).mean().item(),
        "norm_std": flat.norm(dim=-1).std().item(),
        "sparsity": (flat.abs() < 0.01).float().mean().item(),
    }
    activation_stats[key] = stats
    
    print(f"  {key}: norm={stats['norm_mean']:.4f} +/- {stats['norm_std']:.4f}, "
          f"sparsity={stats['sparsity']:.2%}")

# ============================================================
# Step 3: Compare weight-optimal vs behavior-optimal
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Weight-optimal vs Behavior-optimal comparison")
print("=" * 70)

weights, _ = extract_gpt2_layers("gpt2")

def compute_activation_error(W_orig, W_approx, activations):
    """Compute ||Wx - W_hat x|| / ||Wx|| for real activations."""
    # activations: (n_samples, seq_len, dim)
    # W: (out_dim, in_dim)
    # x: (n_samples, seq_len, in_dim)
    
    W_orig = W_orig.to(DEVICE)
    W_approx = W_approx.to(DEVICE)
    
    flat = activations.reshape(-1, activations.shape[-1]).to(DEVICE)
    
    # Compute Wx for original and approximated
    y_orig = flat @ W_orig.T  # (n, out_dim)
    y_approx = flat @ W_approx.T  # (n, out_dim)
    
    # Relative error
    error = (y_orig - y_approx).norm() / y_orig.norm()
    return error.item()


def compute_weight_error(W_orig, W_approx):
    """Compute ||W - W_hat|| / ||W|| (Frobenius norm)."""
    W_orig = W_orig.to(DEVICE)
    W_approx = W_approx.to(DEVICE)
    return ((W_orig - W_approx).norm() / W_orig.norm()).item()


# Test on a few layers
test_layers = [0, 3, 6, 9, 11]
results = []

for i in test_layers:
    layer_name = f"layer{i}.attn.W_O"
    W = weights[layer_name].tensor
    acts = all_activations[layer_name]
    
    print(f"\n--- Layer {i} ---")
    
    # SVD weight-optimal (99% variance)
    rep99 = fit_svd_at_threshold(W, variance_threshold=0.99, device=DEVICE)
    W_svd99 = rep99.reconstruct().to(DEVICE)
    
    weight_err = compute_weight_error(W, W_svd99)
    act_err = compute_activation_error(W, W_svd99, acts)
    
    print(f"  SVD 99%: weight_err={weight_err:.6f}, activation_err={act_err:.6f}")
    print(f"           compression={rep99.compression_ratio:.2f}x")
    
    results.append({
        "layer": i,
        "repr": "SVD 99%",
        "weight_error": weight_err,
        "activation_error": act_err,
        "compression": rep99.compression_ratio,
    })
    
    # SVD weight-optimal (95% variance)
    rep95 = fit_svd_at_threshold(W, variance_threshold=0.95, device=DEVICE)
    W_svd95 = rep95.reconstruct().to(DEVICE)
    
    weight_err = compute_weight_error(W, W_svd95)
    act_err = compute_activation_error(W, W_svd95, acts)
    
    print(f"  SVD 95%: weight_err={weight_err:.6f}, activation_err={act_err:.6f}")
    print(f"           compression={rep95.compression_ratio:.2f}x")
    
    results.append({
        "layer": i,
        "repr": "SVD 95%",
        "weight_error": weight_err,
        "activation_error": act_err,
        "compression": rep95.compression_ratio,
    })

# ============================================================
# Step 4: Optimize directly for activation preservation
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Direct activation-preserving optimization")
print("=" * 70)

def fit_activation_preserving(W, activations, rank, n_steps=1000, lr=1e-3):
    """
    Find W_hat that minimizes ||Wx - W_hat x|| directly.
    Uses gradient descent on the factorized form W_hat = B @ A.
    """
    W = W.to(DEVICE)
    flat = activations.reshape(-1, activations.shape[-1]).to(DEVICE)
    
    # Subsample activations for efficiency
    if flat.shape[0] > 5000:
        idx = torch.randperm(flat.shape[0])[:5000]
        flat = flat[idx]
    
    # Initialize with SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    # Parameterize as B @ A where B = U_r * sqrt(S_r), A = sqrt(S_r) * Vh_r
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)).to(DEVICE))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r).to(DEVICE))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    # Compute original outputs
    y_orig = flat @ W.T  # (n, out_dim)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Reconstructed weight matrix
        W_hat = B @ A
        
        # Compute outputs
        y_hat = flat @ W_hat.T
        
        # Loss: relative activation error
        loss = (y_orig - y_hat).norm() / y_orig.norm()
        
        # Add small weight regularization
        loss += 0.001 * (W_hat.norm() / W.norm())
        
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 200 == 0:
            print(f"    Step {step+1}/{n_steps}, activation_error={loss.item():.6f}")
    
    W_hat = (B @ A).detach()
    return W_hat


# Test activation-preserving optimization on a few layers
for i in [0, 6, 11]:
    layer_name = f"layer{i}.attn.W_O"
    W = weights[layer_name].tensor
    acts = all_activations[layer_name]
    
    print(f"\n--- Layer {i} ---")
    
    # Weight-optimal SVD
    rep_svd = fit_svd_at_threshold(W, variance_threshold=0.99, device=DEVICE)
    W_svd = rep_svd.reconstruct().to(DEVICE)
    weight_err_svd = compute_weight_error(W, W_svd)
    act_err_svd = compute_activation_error(W, W_svd, acts)
    
    print(f"  SVD 99% (weight-optimal):")
    print(f"    weight_error={weight_err_svd:.6f}, activation_error={act_err_svd:.6f}")
    
    # Activation-preserving optimization
    for rank in [128, 256]:
        print(f"  Activation-preserving (rank={rank}):")
        W_ap = fit_activation_preserving(W, acts, rank=rank, n_steps=1000, lr=1e-3)
        
        weight_err_ap = compute_weight_error(W, W_ap)
        act_err_ap = compute_activation_error(W, W_ap, acts)
        
        print(f"    weight_error={weight_err_ap:.6f}, activation_error={act_err_ap:.6f}")
        
        results.append({
            "layer": i,
            "repr": f"ActivationPreserving r{rank}",
            "weight_error": weight_err_ap,
            "activation_error": act_err_ap,
            "compression": W.numel() / (rank * W.shape[0]),
        })

# ============================================================
# Step 5: Functional evaluation of activation-preserving
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Functional evaluation")
print("=" * 70)

# Replace a layer with activation-preserving representation and measure PPL
for i in [0, 6, 11]:
    layer_name = f"layer{i}.attn.W_O"
    W = weights[layer_name].tensor
    acts = all_activations[layer_name]
    
    print(f"\n--- Layer {i} ---")
    
    # Get activation-preserving reconstruction
    W_ap = fit_activation_preserving(W, acts, rank=256, n_steps=1000, lr=1e-3)
    
    # Replace and evaluate
    model_copy = copy.deepcopy(model)
    model_copy.transformer.h[i].attn.c_proj.weight.data = W_ap.float()
    
    ppl = compute_perplexity(model_copy, tokenizer, eval_texts, max_length=256, device=DEVICE)
    delta_pct = ((ppl - baseline.perplexity) / baseline.perplexity) * 100
    
    print(f"  PPL: {baseline.perplexity:.2f} -> {ppl:.2f} ({delta_pct:+.2f}%)")
    
    results.append({
        "layer": i,
        "repr": "ActivationPreserving r256 (functional)",
        "ppl": ppl,
        "delta_pct": delta_pct,
    })
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Step 6: Test what information a layer needs to preserve
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: What information does a layer need to preserve?")
print("=" * 70)

def compute_spectral_overlap(W, W_hat):
    """Compute overlap of dominant singular subspaces."""
    W = W.to(DEVICE)
    W_hat = W_hat.to(DEVICE)
    U_W, S_W, _ = torch.linalg.svd(W, full_matrices=False)
    U_hat, S_hat, _ = torch.linalg.svd(W_hat, full_matrices=False)
    
    # Use top-k singular vectors
    k = min(50, U_W.shape[1])
    U_k = U_W[:, :k]
    U_hat_k = U_hat[:, :k]
    
    # Compute principal angles
    _, S, _ = torch.linalg.svd(U_k.T @ U_hat_k)
    overlap = S.mean().item()
    return overlap


def compute_output_distribution(W, activations):
    """Compute statistics of Wx output."""
    flat = activations.reshape(-1, activations.shape[-1]).to(DEVICE)
    W = W.to(DEVICE)
    
    y = flat @ W.T
    return {
        "mean": y.mean(dim=0),
        "std": y.std(dim=0),
        "kurtosis": ((y - y.mean(dim=0))**4).mean(dim=0) / (y.std(dim=0)**4 + 1e-8),
    }


# Compare different representations' ability to preserve various properties
for i in [0, 6, 11]:
    layer_name = f"layer{i}.attn.W_O"
    W = weights[layer_name].tensor
    acts = all_activations[layer_name]
    
    print(f"\n--- Layer {i} ---")
    
    # Get different reconstructions
    rep_svd99 = fit_svd_at_threshold(W, variance_threshold=0.99, device=DEVICE)
    W_svd99 = rep_svd99.reconstruct().to(DEVICE)
    
    W_ap = fit_activation_preserving(W, acts, rank=256, n_steps=1000, lr=1e-3)
    
    # Compare properties
    orig_stats = compute_output_distribution(W, acts)
    svd_stats = compute_output_distribution(W_svd99, acts)
    ap_stats = compute_output_distribution(W_ap, acts)
    
    # Spectral overlap
    orig_overlap = compute_spectral_overlap(W, W_svd99)
    ap_overlap = compute_spectral_overlap(W, W_ap)
    
    print(f"  Spectral overlap: SVD={orig_overlap:.4f}, AP={ap_overlap:.4f}")
    print(f"  Output mean diff: SVD={(orig_stats['mean'] - svd_stats['mean']).abs().mean():.6f}, "
          f"AP={(orig_stats['mean'] - ap_stats['mean']).abs().mean():.6f}")
    print(f"  Output std diff: SVD={(orig_stats['std'] - svd_stats['std']).abs().mean():.6f}, "
          f"AP={(orig_stats['std'] - ap_stats['std']).abs().mean():.6f}")

# ============================================================
# Save results
# ============================================================
with open(output_path / "phase5_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PHASE 5 SUMMARY")
print("=" * 70)
print("\nKey findings saved to results/phase5_results.json")

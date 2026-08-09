"""
Finding the Invariant
=====================
Measure many properties of original vs compressed layer outputs.
Find which metric correlates best with generation quality.
"""

import sys, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from baseline_eval import compute_perplexity

# ============================================================
# Load model
# ============================================================
print("=" * 70)
print("Loading GPT-2 Small...")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
model.to(DEVICE)

# ============================================================
# Collect many activations
# ============================================================
print("Collecting activations...")
eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "The temperature today is expected to reach 75 degrees.",
    "Machine learning models require large datasets.",
    "The capital of France is Paris, known for the Eiffel Tower.",
    "Python is a popular programming language for AI.",
    "The theory of relativity was proposed by Einstein.",
    "Deep learning has revolutionized natural language processing.",
    "The Eiffel Tower was built in 1889 for the World's Fair.",
    "Quantum computing promises to solve complex problems faster.",
    "Artificial intelligence is transforming every industry.",
    "The human brain has approximately 86 billion neurons.",
    "Climate change is one of the greatest challenges of our time.",
    "The internet has fundamentally changed how we communicate.",
    "DNA contains the instructions for building living organisms.",
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

print(f"Collected activations: {activations['layer5.attn.W_O'].shape}")

# ============================================================
# SVD function
# ============================================================
def fit_svd(W, variance_threshold=0.99):
    W = W.float()
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    return W_approx, rank

# ============================================================
# Compute all metrics
# ============================================================
def compute_all_metrics(W_orig, W_comp, activations):
    """Compute comprehensive metrics between original and compressed layer."""
    
    W_orig = W_orig.float().to(DEVICE)
    W_comp = W_comp.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    # Subsample for speed
    if flat.shape[0] > 5000:
        idx = torch.randperm(flat.shape[0])[:5000]
        flat = flat[idx]
    
    # Compute outputs
    y_orig = flat @ W_orig.T  # (n, out_dim)
    y_comp = flat @ W_comp.T  # (n, out_dim)
    
    metrics = {}
    
    # 1. Basic error metrics
    metrics["mse"] = ((y_orig - y_comp) ** 2).mean().item()
    metrics["mae"] = (y_orig - y_comp).abs().mean().item()
    metrics["relative_error"] = ((y_orig - y_comp).norm() / y_orig.norm()).item()
    
    # 2. Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(y_orig, y_comp, dim=-1)
    metrics["cosine_mean"] = cos_sim.mean().item()
    metrics["cosine_std"] = cos_sim.std().item()
    metrics["cosine_min"] = cos_sim.min().item()
    
    # 3. Covariance analysis
    cov_orig = torch.cov(y_orig.T)
    cov_comp = torch.cov(y_comp.T)
    
    # Frobenius norm of covariance difference
    metrics["cov_frobenius"] = (cov_orig - cov_comp).norm().item()
    
    # Eigenvalue comparison
    eig_orig = torch.linalg.eigvalsh(cov_orig)
    eig_comp = torch.linalg.eigvalsh(cov_comp)
    
    # Keep top eigenvalues
    k = min(50, len(eig_orig))
    metrics["eigenvalue_cosine"] = torch.nn.functional.cosine_similarity(
        eig_orig[-k:], eig_comp[-k:], dim=0
    ).mean().item()
    
    # 4. CKA (Centered Kernel Alignment) similarity
    def center_kernel(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H
    
    def cka(X, Y):
        K_X = X @ X.T
        K_Y = Y @ Y.T
        K_X_centered = center_kernel(K_X)
        K_Y_centered = center_kernel(K_Y)
        
        numerator = (K_X_centered * K_Y_centered).sum()
        denominator = torch.sqrt((K_X_centered * K_X_centered).sum() * 
                                (K_Y_centered * K_Y_centered).sum())
        return (numerator / (denominator + 1e-10)).item()
    
    # Use subset for CKA (expensive)
    if flat.shape[0] > 1000:
        cka_idx = torch.randperm(flat.shape[0])[:1000]
        cka_y_orig = y_orig[cka_idx]
        cka_y_comp = y_comp[cka_idx]
    else:
        cka_y_orig = y_orig
        cka_y_comp = y_comp
    
    metrics["cka"] = cka(cka_y_orig, cka_y_comp)
    
    # 5. Jacobian analysis
    # Compute diagonal of Jacobian (sensitivity of each output to each input)
    # For linear layer W, Jacobian is just W itself
    # Compare singular values of W_orig vs W_comp
    U_o, S_o, Vh_o = torch.linalg.svd(W_orig, full_matrices=False)
    U_c, S_c, Vh_c = torch.linalg.svd(W_comp, full_matrices=False)
    
    metrics["sv_cosine"] = torch.nn.functional.cosine_similarity(
        S_o[:min(len(S_o), len(S_c))], 
        S_c[:min(len(S_o), len(S_c))], 
        dim=0
    ).item()
    
    metrics["sv_relative_error"] = ((S_o[:min(len(S_o), len(S_c))] - 
                                    S_c[:min(len(S_o), len(S_c))]).norm() / 
                                   S_o[:min(len(S_o), len(S_c))].norm()).item()
    
    # 6. Output distribution comparison
    # KL divergence (approximate)
    p_orig = torch.nn.functional.softmax(y_orig / 1.0, dim=-1)
    p_comp = torch.nn.functional.softmax(y_comp / 1.0, dim=-1)
    kl = (p_orig * (torch.log(p_orig + 1e-10) - torch.log(p_comp + 1e-10))).sum(dim=-1)
    metrics["kl_divergence"] = kl.mean().item()
    
    # 7. Token neighborhood preservation
    # For each token, check if its k-nearest neighbors are the same
    def neighborhood_preservation(Y_orig, Y_comp, k=10):
        n = Y_orig.shape[0]
        if n < k:
            return 0.0
        
        # Compute pairwise distances
        dist_orig = torch.cdist(Y_orig, Y_orig)
        dist_comp = torch.cdist(Y_comp, Y_comp)
        
        # Get k-nearest neighbors
        _, nn_orig = dist_orig.topk(k, dim=-1, largest=False)
        _, nn_comp = dist_comp.topk(k, dim=-1, largest=False)
        
        # Compute overlap
        overlap = 0
        for i in range(n):
            set_orig = set(nn_orig[i].cpu().numpy())
            set_comp = set(nn_comp[i].cpu().numpy())
            overlap += len(set_orig & set_comp) / k
        
        return overlap / n
    
    metrics["neighborhood_preservation"] = neighborhood_preservation(y_orig, y_comp, k=10)
    
    # 8. Attention pattern preservation
    # Compute attention-like similarity matrix
    attn_orig = torch.softmax(y_orig @ y_orig.T / np.sqrt(y_orig.shape[-1]), dim=-1)
    attn_comp = torch.softmax(y_comp @ y_comp.T / np.sqrt(y_comp.shape[-1]), dim=-1)
    
    metrics["attention_mse"] = ((attn_orig - attn_comp) ** 2).mean().item()
    metrics["attention_cosine"] = torch.nn.functional.cosine_similarity(
        attn_orig.flatten(), attn_comp.flatten(), dim=0
    ).item()
    
    # 9. Fisher information (approximate)
    # Fisher = E[gradient^2]
    # For linear layer, gradient of loss w.r.t. W is x * (y_true - y_pred)
    # Approximate using output variance
    metrics["fisher_orig"] = y_orig.var(dim=0).mean().item()
    metrics["fisher_comp"] = y_comp.var(dim=0).mean().item()
    metrics["fisher_ratio"] = metrics["fisher_comp"] / (metrics["fisher_orig"] + 1e-10)
    
    # 10. Manifold geometry
    # Compute principal angles between subspaces
    U_o, _, _ = torch.linalg.svd(y_orig.T, full_matrices=False)
    U_c, _, _ = torch.linalg.svd(y_comp.T, full_matrices=False)
    
    k_sub = min(50, U_o.shape[1], U_c.shape[1])
    _, S_sub, _ = torch.linalg.svd(U_o[:, :k_sub].T @ U_c[:, :k_sub])
    metrics["subspace_cosine"] = S_sub.mean().item()
    
    return metrics

# ============================================================
# Test on layer 5 (safest layer)
# ============================================================
print("\n" + "=" * 70)
print("ANALYZING LAYER 5 (SAFEST LAYER)")
print("=" * 70)

layer_idx = 5
layer_name = f"layer{layer_idx}.attn.W_O"

W_orig = model.transformer.h[layer_idx].attn.c_proj.weight.data.cpu().float()
acts = activations[layer_name]

# Test different compression levels
thresholds = [0.99, 0.95, 0.90, 0.85, 0.80]
all_metrics = []

for threshold in thresholds:
    print(f"\n--- SVD {int(threshold*100)}% ---")
    
    W_comp, rank = fit_svd(W_orig, threshold)
    
    metrics = compute_all_metrics(W_orig, W_comp, acts)
    metrics["threshold"] = threshold
    metrics["rank"] = rank
    metrics["compression"] = W_orig.numel() / (rank * (W_orig.shape[0] + W_orig.shape[1]))
    
    print(f"  Rank: {rank}, Compression: {metrics['compression']:.2f}x")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Cosine: {metrics['cosine_mean']:.4f}")
    print(f"  CKA: {metrics['cka']:.4f}")
    print(f"  Neighborhood: {metrics['neighborhood_preservation']:.4f}")
    print(f"  Subspace: {metrics['subspace_cosine']:.4f}")
    
    all_metrics.append(metrics)

# ============================================================
# Correlate with generation quality
# ============================================================
print("\n" + "=" * 70)
print("CORRELATING METRICS WITH GENERATION QUALITY")
print("=" * 70)

# Compute generation quality for each threshold
prompts = [
    "Hello",
    "What is 2+2?",
    "The meaning of life is",
    "Once upon a time",
    "In machine learning",
]

def compute_generation_quality(model, tokenizer, prompts, n_tokens=50):
    """Compute generation quality metrics."""
    all_repetition = []
    all_distinct_2 = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=n_tokens,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens = tokenizer.encode(generated)
        
        # Repetition
        if len(tokens) > 1:
            repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            repetition = repeats / (len(tokens) - 1)
        else:
            repetition = 0
        
        # Distinct-2
        if len(tokens) > 1:
            ngrams = set()
            for i in range(len(tokens) - 1):
                ngrams.add((tokens[i], tokens[i+1]))
            distinct_2 = len(ngrams) / (len(tokens) - 1)
        else:
            distinct_2 = 0
        
        all_repetition.append(repetition)
        all_distinct_2.append(distinct_2)
    
    return {
        "repetition": np.mean(all_repetition),
        "distinct_2": np.mean(all_distinct_2),
    }

# Get baseline quality
baseline_quality = compute_generation_quality(model, tokenizer, prompts)
print(f"\nBaseline: repetition={baseline_quality['repetition']:.4f}, "
      f"distinct_2={baseline_quality['distinct_2']:.4f}")

# Test each threshold
quality_results = []

for threshold in thresholds:
    print(f"\n--- Testing SVD {int(threshold*100)}% ---")
    
    model_comp = copy.deepcopy(model)
    W_comp, _ = fit_svd(W_orig, threshold)
    model_comp.transformer.h[layer_idx].attn.c_proj.weight.data = W_comp.to(DEVICE)
    
    quality = compute_generation_quality(model_comp, tokenizer, prompts)
    
    quality["threshold"] = threshold
    quality["rep_delta"] = quality["repetition"] - baseline_quality["repetition"]
    quality["distinct_delta"] = quality["distinct_2"] - baseline_quality["distinct_2"]
    
    print(f"  Repetition: {quality['repetition']:.4f} (delta: {quality['rep_delta']:+.4f})")
    print(f"  Distinct-2: {quality['distinct_2']:.4f} (delta: {quality['distinct_delta']:+.4f})")
    
    quality_results.append(quality)
    
    del model_comp
    torch.cuda.empty_cache()

# ============================================================
# Find best correlation
# ============================================================
print("\n" + "=" * 70)
print("METRIC CORRELATION ANALYSIS")
print("=" * 70)

# Compute correlations between each metric and generation quality
metric_names = ["mse", "cosine_mean", "cka", "neighborhood_preservation", 
                "subspace_cosine", "sv_cosine", "attention_cosine"]

print(f"\n{'Metric':<30} {'Correlation with Rep':<25} {'Correlation with Distinct'}")
print("-" * 80)

correlations = []

for metric_name in metric_names:
    metric_values = [m[metric_name] for m in all_metrics]
    rep_values = [q["rep_delta"] for q in quality_results]
    distinct_values = [q["distinct_delta"] for q in quality_results]
    
    # Compute correlation
    corr_rep = np.corrcoef(metric_values, rep_values)[0, 1]
    corr_dist = np.corrcoef(metric_values, distinct_values)[0, 1]
    
    print(f"{metric_name:<30} {corr_rep:<+25.4f} {corr_dist:<+.4f}")
    
    correlations.append({
        "metric": metric_name,
        "corr_rep": corr_rep,
        "corr_dist": corr_dist,
    })

# Find best metric
best_rep = max(correlations, key=lambda x: abs(x["corr_rep"]))
best_dist = max(correlations, key=lambda x: abs(x["corr_dist"]))

print(f"\nBest predictor of repetition: {best_rep['metric']} (r={best_rep['corr_rep']:.4f})")
print(f"Best predictor of diversity: {best_dist['metric']} (r={best_dist['corr_dist']:.4f})")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("INVARIANT DISCOVERY SUMMARY")
print("=" * 70)

print(f"\nLayer analyzed: {layer_name}")
print(f"Compression levels tested: {len(thresholds)}")
print(f"Metrics computed: {len(metric_names)}")

print("\nKey findings:")
print(f"1. Best predictor of repetition: {best_rep['metric']}")
print(f"2. Best predictor of diversity: {best_dist['metric']}")

print("\nThis suggests the true objective function should optimize:")
if best_rep["corr_rep"] > 0.5:
    print(f"   - {best_rep['metric']} (strong correlation with generation quality)")
else:
    print(f"   - Multiple metrics together (no single metric dominates)")

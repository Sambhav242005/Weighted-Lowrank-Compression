"""
Phase 6: Cross-Architecture Validation
========================================
Test Function-Preserving Approximation on Gemma 3 1B.

Hypothesis: If activation/function-preserving approximation consistently
outperforms weight-preserving approximation across architectures,
then current weight reconstruction objectives may be optimizing
the wrong quantity.
"""

import sys, json, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from baseline_eval import compute_perplexity

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# ============================================================
# Helper functions
# ============================================================

def compute_weight_error(W_orig, W_approx):
    """Compute ||W - W_hat|| / ||W|| (Frobenius norm)."""
    W_orig = W_orig.to(DEVICE)
    W_approx = W_approx.to(DEVICE)
    return ((W_orig - W_approx).norm() / W_orig.norm()).item()


def compute_activation_error(W_orig, W_approx, activations):
    """Compute ||Wx - W_hat x|| / ||Wx|| for real activations."""
    W_orig = W_orig.float().to(DEVICE)
    W_approx = W_approx.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    y_orig = flat @ W_orig.T
    y_approx = flat @ W_approx.T
    
    error = (y_orig - y_approx).norm() / y_orig.norm()
    return error.item()


def compute_kl_divergence(logits_orig, logits_approx):
    """Compute KL(P || P') between output distributions."""
    p_orig = torch.nn.functional.softmax(logits_orig, dim=-1)
    p_approx = torch.nn.functional.softmax(logits_approx, dim=-1)
    
    # KL(P || P') = sum(P * log(P / P'))
    kl = (p_orig * (torch.log(p_orig + 1e-10) - torch.log(p_approx + 1e-10))).sum(dim=-1)
    return kl.mean().item()


def fit_svd_at_threshold(weight, variance_threshold=0.99, device="cpu"):
    """SVD approximation preserving variance_threshold of energy."""
    W = weight.float().to(device)
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]).cpu()
    
    original_params = m * n
    compressed_params = rank * (m + n)
    
    class Result:
        pass
    r = Result()
    r.reconstruct = lambda: W_approx
    r.compression_ratio = original_params / compressed_params
    r.rank = rank
    return r


def fit_activation_preserving(W, activations, rank, n_steps=1000, lr=1e-3, device="cuda"):
    """Optimize W_hat = B @ A to minimize ||Wx - W_hat x||."""
    W = W.float().to(device)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(device)
    
    # Subsample for efficiency
    if flat.shape[0] > 5000:
        idx = torch.randperm(flat.shape[0])[:5000]
        flat = flat[idx]
    
    # Initialize with SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    y_orig = flat @ W.T
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        loss = (y_orig - y_hat).norm() / y_orig.norm()
        loss += 0.001 * (W_hat.norm() / W.norm())
        loss.backward()
        optimizer.step()
    
    W_hat = (B @ A).detach().cpu()
    
    class Result:
        pass
    r = Result()
    r.reconstruct = lambda: W_hat
    r.compression_ratio = W.numel() / (rank * W.shape[0])
    r.rank = rank
    return r


# ============================================================
# Load Gemma 3 1B
# ============================================================
print("=" * 70)
print("Loading Gemma 3 1B...")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-1b-it"
print(f"Loading {model_name}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded: {model.config}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative: google/gemma-1b-it")
    model_name = "google/gemma-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

# Get model config
config = model.config
n_layers = config.num_hidden_layers
hidden_size = config.hidden_size
intermediate_size = config.intermediate_size

print(f"\nModel: {model_name}")
print(f"Layers: {n_layers}")
print(f"Hidden size: {hidden_size}")
print(f"Intermediate size: {intermediate_size}")

# ============================================================
# Collect activations
# ============================================================
print("\n" + "=" * 70)
print("Collecting activations from Gemma 3...")
print("=" * 70)

# Test texts
eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness and silence.",
    "The temperature today is expected to reach 75 degrees Fahrenheit.",
    "Machine learning models require large datasets for training.",
    "The capital of France is Paris, known for the Eiffel Tower.",
]

# Hook to collect activations
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

# Register hooks on attention output and MLP down
# Gemma uses different naming: model.layers[i].self_attn.o_proj
# and model.layers[i].mlp.down_proj
test_layers = [0, n_layers // 2, n_layers - 1]

for i in test_layers:
    layer = model.model.layers[i]
    
    # Attention output projection
    hook = layer.self_attn.o_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)
    
    # MLP down projection
    hook = layer.mlp.down_proj.register_forward_hook(make_hook(f"layer{i}.mlp.W_down"))
    hooks.append(hook)

# Collect activations
print(f"Collecting activations from {len(eval_texts)} texts...")
all_activations = {}

model.eval()
with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding="max_length").to(DEVICE)
        _ = model(**inputs)
        
        for i in test_layers:
            for suffix in ["attn.W_O", "mlp.W_down"]:
                key = f"layer{i}.{suffix}"
                if key in activations:
                    if key not in all_activations:
                        all_activations[key] = []
                    all_activations[key].append(activations[key].cpu())

# Remove hooks
for h in hooks:
    h.remove()

# Stack activations
for key in all_activations:
    try:
        all_activations[key] = torch.cat(all_activations[key], dim=0)
        print(f"  {key}: {all_activations[key].shape}")
    except RuntimeError:
        min_seq = min(a.shape[1] for a in all_activations[key])
        all_activations[key] = torch.cat([a[:, :min_seq, :] for a in all_activations[key]], dim=0)
        print(f"  {key}: {all_activations[key].shape} (truncated)")

# ============================================================
# Extract weights
# ============================================================
print("\n" + "=" * 70)
print("Extracting weights...")
print("=" * 70)

weights = {}
for i in test_layers:
    layer = model.model.layers[i]
    
    # Attention O projection
    W_O = layer.self_attn.o_proj.weight.data.cpu().float()
    weights[f"layer{i}.attn.W_O"] = W_O
    print(f"  layer{i}.attn.W_O: {W_O.shape}")
    
    # MLP down projection
    W_down = layer.mlp.down_proj.weight.data.cpu().float()
    weights[f"layer{i}.mlp.W_down"] = W_down
    print(f"  layer{i}.mlp.W_down: {W_down.shape}")

# ============================================================
# Compare representations
# ============================================================
print("\n" + "=" * 70)
print("Comparing SVD vs Function-Preserving Approximation")
print("=" * 70)

results = []

for key in weights.keys():
    W = weights[key]
    acts = all_activations[key]
    
    print(f"\n--- {key} ---")
    
    # SVD 99%
    svd99 = fit_svd_at_threshold(W, variance_threshold=0.99, device=DEVICE)
    W_svd99 = svd99.reconstruct()
    
    weight_err_svd = compute_weight_error(W, W_svd99)
    act_err_svd = compute_activation_error(W, W_svd99, acts)
    
    print(f"  SVD 99%: weight_err={weight_err_svd:.6f}, act_err={act_err_svd:.6f}, "
          f"compression={svd99.compression_ratio:.2f}x")
    
    results.append({
        "matrix": key,
        "repr": "SVD 99%",
        "weight_error": weight_err_svd,
        "activation_error": act_err_svd,
        "compression": svd99.compression_ratio,
    })
    
    # SVD 95%
    svd95 = fit_svd_at_threshold(W, variance_threshold=0.95, device=DEVICE)
    W_svd95 = svd95.reconstruct()
    
    weight_err_svd95 = compute_weight_error(W, W_svd95)
    act_err_svd95 = compute_activation_error(W, W_svd95, acts)
    
    print(f"  SVD 95%: weight_err={weight_err_svd95:.6f}, act_err={act_err_svd95:.6f}, "
          f"compression={svd95.compression_ratio:.2f}x")
    
    results.append({
        "matrix": key,
        "repr": "SVD 95%",
        "weight_error": weight_err_svd95,
        "activation_error": act_err_svd95,
        "compression": svd95.compression_ratio,
    })
    
    # Function-Preserving (rank = same as SVD 99%)
    rank = svd99.rank
    print(f"  Fitting Function-Preserving (rank={rank})...")
    
    fp = fit_activation_preserving(W, acts, rank=rank, n_steps=1000, lr=1e-3, device=DEVICE)
    W_fp = fp.reconstruct()
    
    weight_err_fp = compute_weight_error(W, W_fp)
    act_err_fp = compute_activation_error(W, W_fp, acts)
    
    print(f"  FP r{rank}: weight_err={weight_err_fp:.6f}, act_err={act_err_fp:.6f}, "
          f"compression={fp.compression_ratio:.2f}x")
    
    results.append({
        "matrix": key,
        "repr": f"FP r{rank}",
        "weight_error": weight_err_fp,
        "activation_error": act_err_fp,
        "compression": fp.compression_ratio,
    })

# ============================================================
# Functional evaluation: replace and measure perplexity
# ============================================================
print("\n" + "=" * 70)
print("Functional evaluation: replacing layers and measuring perplexity")
print("=" * 70)

# Load model fresh for functional eval
model_func = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model_func.eval()

# Get perplexity of original
print("Computing baseline perplexity...")
# Use a simpler approach - just compute loss on eval texts
def get_perplexity(model, tokenizer, texts, max_length=128):
    total_loss = 0
    n_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(DEVICE)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            n_tokens += inputs["input_ids"].shape[1]
    return torch.exp(torch.tensor(total_loss / n_tokens)).item()

baseline_ppl = get_perplexity(model_func, tokenizer, eval_texts)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# Test each layer replacement
for key in weights.keys():
    W = weights[key]
    acts = all_activations[key]
    
    # Parse layer index and matrix type
    parts = key.split(".")
    layer_idx = int(parts[0].replace("layer", ""))
    matrix_type = parts[1]  # "attn" or "mlp"
    
    print(f"\n--- Replacing {key} ---")
    
    # SVD 99%
    svd99 = fit_svd_at_threshold(W, variance_threshold=0.99, device=DEVICE)
    W_svd99 = svd99.reconstruct()
    
    model_copy = copy.deepcopy(model_func)
    if matrix_type == "attn":
        model_copy.model.layers[layer_idx].self_attn.o_proj.weight.data = W_svd99.half().to(DEVICE)
    else:
        model_copy.model.layers[layer_idx].mlp.down_proj.weight.data = W_svd99.half().to(DEVICE)
    
    ppl_svd = get_perplexity(model_copy, tokenizer, eval_texts)
    delta_svd = ((ppl_svd - baseline_ppl) / baseline_ppl) * 100
    
    print(f"  SVD 99%: PPL={ppl_svd:.2f} ({delta_svd:+.2f}%)")
    
    results.append({
        "matrix": key,
        "repr": "SVD 99% (functional)",
        "ppl": ppl_svd,
        "delta_pct": delta_svd,
    })
    
    del model_copy
    torch.cuda.empty_cache()
    
    # Function-Preserving
    fp = fit_activation_preserving(W, acts, rank=svd99.rank, n_steps=1000, lr=1e-3, device=DEVICE)
    W_fp = fp.reconstruct()
    
    model_copy = copy.deepcopy(model_func)
    if matrix_type == "attn":
        model_copy.model.layers[layer_idx].self_attn.o_proj.weight.data = W_fp.half().to(DEVICE)
    else:
        model_copy.model.layers[layer_idx].mlp.down_proj.weight.data = W_fp.half().to(DEVICE)
    
    ppl_fp = get_perplexity(model_copy, tokenizer, eval_texts)
    delta_fp = ((ppl_fp - baseline_ppl) / baseline_ppl) * 100
    
    print(f"  FP r{svd99.rank}: PPL={ppl_fp:.2f} ({delta_fp:+.2f}%)")
    
    results.append({
        "matrix": key,
        "repr": f"FP r{svd99.rank} (functional)",
        "ppl": ppl_fp,
        "delta_pct": delta_fp,
    })
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Save results
# ============================================================
with open(output_path / "phase6_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6 SUMMARY: Cross-Architecture Validation")
print("=" * 70)

print(f"\nModel: {model_name}")
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

print("Weight Error vs Activation Error Comparison:")
print(f"  {'Matrix':<25} {'SVD Act Err':<14} {'FP Act Err':<14} {'Ratio'}")
print("  " + "-" * 60)

# Group by matrix
matrix_keys = list(weights.keys())
for key in matrix_keys:
    svd_row = next((r for r in results if r["matrix"] == key and r["repr"] == "SVD 99%"), None)
    fp_row = next((r for r in results if r["matrix"] == key and r["repr"].startswith("FP")), None)
    
    if svd_row and fp_row:
        ratio = svd_row["activation_error"] / fp_row["activation_error"]
        print(f"  {key:<25} {svd_row['activation_error']:<14.6f} {fp_row['activation_error']:<14.6f} {ratio:.2f}x")

print("\nFunctional Evaluation (PPL after replacement):")
print(f"  {'Matrix':<25} {'SVD 99%':<20} {'FP':<20}")
print("  " + "-" * 65)

for key in matrix_keys:
    svd_func = next((r for r in results if r["matrix"] == key and "SVD 99% (functional)" in r.get("repr", "")), None)
    fp_func = next((r for r in results if r["matrix"] == key and "FP" in r.get("repr", "") and "functional" in r.get("repr", "")), None)
    
    if svd_func and fp_func:
        print(f"  {key:<25} {svd_func['delta_pct']:>+8.2f}%           {fp_func['delta_pct']:>+8.2f}%")

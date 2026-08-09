"""
Compression Tradeoff Analysis
==============================
Test different compression budgets to find the sweet spot
between storage reduction and quality preservation.
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
# Helper functions
# ============================================================

def fit_svd_at_threshold(weight, variance_threshold=0.99, device="cpu"):
    """SVD approximation."""
    W = weight.float().to(device)
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    return W_approx.cpu(), rank


def fit_function_preserving(W, activations, rank, n_steps=500, lr=1e-3, device="cuda"):
    """Optimize to minimize ||Wx - W_hat x||."""
    W = W.float().to(device)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(device)
    
    if flat.shape[0] > 2000:
        idx = torch.randperm(flat.shape[0])[:2000]
        flat = flat[idx]
    
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
    
    return (B @ A).detach().cpu()


def compress_model(model, activations, attention_rank, mlp_rank, device="cuda"):
    """Compress model with given ranks."""
    model_comp = copy.deepcopy(model)
    
    compression_log = []
    total_orig = 0
    total_comp = 0
    
    for i in range(12):
        # c_attn
        W = model_comp.transformer.h[i].attn.c_attn.weight.data.cpu().float()
        acts = activations[f"layer{i}.attn.c_attn"]
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=0.99, device=device)
        model_comp.transformer.h[i].attn.c_attn.weight.data = W_svd.half().to(device)
        
        orig = W.numel()
        comp = rank * (W.shape[0] + W.shape[1])
        total_orig += orig
        total_comp += comp
        
        # W_O - function-preserving
        W = model_comp.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        acts = activations[f"layer{i}.attn.W_O"]
        W_fp = fit_function_preserving(W, acts, rank=attention_rank, n_steps=300, lr=1e-3, device=device)
        model_comp.transformer.h[i].attn.c_proj.weight.data = W_fp.half().to(device)
        
        orig = W.numel()
        comp = attention_rank * (W.shape[0] + W.shape[1])
        total_orig += orig
        total_comp += comp
        
        # MLP up
        W = model_comp.transformer.h[i].mlp.c_fc.weight.data.cpu().float()
        acts = activations[f"layer{i}.mlp.W_up"]
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=0.95, device=device)
        model_comp.transformer.h[i].mlp.c_fc.weight.data = W_svd.half().to(device)
        
        orig = W.numel()
        comp = rank * (W.shape[0] + W.shape[1])
        total_orig += orig
        total_comp += comp
        
        # MLP down
        W = model_comp.transformer.h[i].mlp.c_proj.weight.data.cpu().float()
        acts = activations[f"layer{i}.mlp.W_down"]
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=0.95, device=device)
        model_comp.transformer.h[i].mlp.c_proj.weight.data = W_svd.half().to(device)
        
        orig = W.numel()
        comp = rank * (W.shape[0] + W.shape[1])
        total_orig += orig
        total_comp += comp
    
    return model_comp, total_orig / total_comp


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
    hook = block.attn.c_attn.register_forward_hook(make_hook(f"layer{i}.attn.c_attn"))
    hooks.append(hook)
    hook = block.attn.c_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)
    hook = block.mlp.c_fc.register_forward_hook(make_hook(f"layer{i}.mlp.W_up"))
    hooks.append(hook)
    hook = block.mlp.c_proj.register_forward_hook(make_hook(f"layer{i}.mlp.W_down"))
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
# Baseline
# ============================================================
print("\n" + "=" * 70)
print("Baseline...")
print("=" * 70)

# Load fresh model in float32 for evaluation
model_baseline = GPT2LMHeadModel.from_pretrained(model_name)
model_baseline.eval()
model_baseline.to(DEVICE)

eval_texts_full = [
    "The quick brown fox jumps over the lazy dog. This is a test sentence.",
    "In the beginning, there was nothing but darkness and silence. Then came the light.",
    "The temperature today is expected to reach 75 degrees Fahrenheit in the afternoon.",
]

baseline_ppl = compute_perplexity(model_baseline, tokenizer, eval_texts_full, max_length=256, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}")

# ============================================================
# Test different compression budgets
# ============================================================
print("\n" + "=" * 70)
print("Testing compression budgets...")
print("=" * 70)

configs = [
    ("Conservative", 256, 0.99, 0.99),
    ("Moderate", 192, 0.95, 0.95),
    ("Aggressive", 128, 0.90, 0.90),
    ("Very Aggressive", 64, 0.85, 0.85),
]

results = []

for name, attn_rank, mlp_threshold, attn_threshold in configs:
    print(f"\n--- {name} (attn_rank={attn_rank}, mlp_threshold={mlp_threshold}) ---")
    
    # Modify compress_model to use threshold for MLP
    model_comp = copy.deepcopy(model)
    
    total_orig = 0
    total_comp = 0
    
    for i in range(12):
        # c_attn
        W = model_comp.transformer.h[i].attn.c_attn.weight.data.cpu().float()
        acts = activations[f"layer{i}.attn.c_attn"]
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=attn_threshold, device=DEVICE)
        model_comp.transformer.h[i].attn.c_attn.weight.data = W_svd.to(DEVICE)
        total_orig += W.numel()
        total_comp += rank * (W.shape[0] + W.shape[1])
        
        # W_O
        W = model_comp.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        acts = activations[f"layer{i}.attn.W_O"]
        W_fp = fit_function_preserving(W, acts, rank=attn_rank, n_steps=300, lr=1e-3, device=DEVICE)
        model_comp.transformer.h[i].attn.c_proj.weight.data = W_fp.to(DEVICE)
        total_orig += W.numel()
        total_comp += attn_rank * (W.shape[0] + W.shape[1])
        
        # MLP up
        W = model_comp.transformer.h[i].mlp.c_fc.weight.data.cpu().float()
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=mlp_threshold, device=DEVICE)
        model_comp.transformer.h[i].mlp.c_fc.weight.data = W_svd.to(DEVICE)
        total_orig += W.numel()
        total_comp += rank * (W.shape[0] + W.shape[1])
        
        # MLP down
        W = model_comp.transformer.h[i].mlp.c_proj.weight.data.cpu().float()
        W_svd, rank = fit_svd_at_threshold(W, variance_threshold=mlp_threshold, device=DEVICE)
        model_comp.transformer.h[i].mlp.c_proj.weight.data = W_svd.to(DEVICE)
        total_orig += W.numel()
        total_comp += rank * (W.shape[0] + W.shape[1])
    
    compression = total_orig / total_comp
    
    ppl = compute_perplexity(model_comp, tokenizer, eval_texts_full, max_length=256, device=DEVICE)
    delta_pct = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    print(f"  Compression: {compression:.2f}x")
    print(f"  PPL: {baseline_ppl:.2f} -> {ppl:.2f} ({delta_pct:+.2f}%)")
    
    results.append({
        "config": name,
        "attn_rank": attn_rank,
        "mlp_threshold": mlp_threshold,
        "compression": compression,
        "ppl": ppl,
        "delta_pct": delta_pct,
    })
    
    del model_comp
    torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPRESSION TRADEOFF SUMMARY")
print("=" * 70)

print(f"\nBaseline PPL: {baseline_ppl:.2f}")
print(f"\n{'Config':<20} {'Compression':<14} {'PPL':<10} {'Delta'}")
print("-" * 55)

for r in results:
    print(f"{r['config']:<20} {r['compression']:<14.2f}x {r['ppl']:<10.2f} {r['delta_pct']:>+8.2f}%")

# Save
with open(output_path / "compression_tradeoff.json", "w") as f:
    json.dump({
        "baseline_ppl": baseline_ppl,
        "results": results,
    }, f, indent=2)

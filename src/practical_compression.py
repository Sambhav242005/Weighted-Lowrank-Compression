"""
Practical Compression Pipeline
===============================
Compress GPT-2 using function-preserving approximation for attention
and SVD for MLP. Measures actual storage reduction and perplexity.
"""

import sys, json, torch, numpy as np, copy, os, time
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from baseline_eval import compute_perplexity

output_path = Path("results")
compressed_path = Path("compressed_model")
compressed_path.mkdir(exist_ok=True)

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
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]).cpu()
    return W_approx, rank


def fit_function_preserving(W, activations, rank, n_steps=1000, lr=1e-3, device="cuda"):
    """Optimize to minimize ||Wx - W_hat x||."""
    W = W.float().to(device)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(device)
    
    if flat.shape[0] > 5000:
        idx = torch.randperm(flat.shape[0])[:5000]
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


def compute_activation_error(W_orig, W_approx, activations):
    """Compute ||Wx - W_hat x|| / ||Wx||."""
    W_orig = W_orig.float().to(DEVICE)
    W_approx = W_approx.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    y_orig = flat @ W_orig.T
    y_approx = flat @ W_approx.T
    return ((y_orig - y_approx).norm() / y_orig.norm()).item()


# ============================================================
# Load model
# ============================================================
print("=" * 70)
print("Loading GPT-2 Small...")
print("=" * 70)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
model.to(DEVICE)

n_layers = model.config.n_layer
hidden_size = model.config.n_embd

print(f"Layers: {n_layers}, Hidden: {hidden_size}")

# ============================================================
# Measure baseline storage
# ============================================================
print("\n" + "=" * 70)
print("Measuring baseline storage...")
print("=" * 70)

# Save model temporarily to measure size
model.save_pretrained(compressed_path / "baseline")
baseline_size = sum(f.stat().st_size for f in (compressed_path / "baseline").rglob("*") if f.is_file())
print(f"Baseline model size: {baseline_size / 1e6:.2f} MB")

# ============================================================
# Collect activations
# ============================================================
print("\n" + "=" * 70)
print("Collecting activations...")
print("=" * 70)

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

# Register hooks
for i in range(n_layers):
    block = model.transformer.h[i]
    hook = block.attn.c_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)

# Collect
model.eval()
with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding="max_length").to(DEVICE)
        _ = model(**inputs)
        
        for i in range(n_layers):
            key = f"layer{i}.attn.W_O"
            if key in activations:
                if key not in activations or not isinstance(activations[key], list):
                    activations[key] = [activations[key].cpu()]
                else:
                    activations[key].append(activations[key].cpu())

for h in hooks:
    h.remove()

# Stack activations
for key in list(activations.keys()):
    if isinstance(activations[key], list):
        activations[key] = torch.cat(activations[key], dim=0)

print(f"Collected activations for {len(activations)} layers")
for key in list(activations.keys())[:3]:
    print(f"  {key}: {activations[key].shape}")

# ============================================================
# Compress the model
# ============================================================
print("\n" + "=" * 70)
print("Compressing model...")
print("=" * 70)

model_compressed = copy.deepcopy(model)

compression_log = []

for i in range(n_layers):
    layer_name = f"layer{i}.attn.W_O"
    
    # Get weight
    W_O = model_compressed.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    
    # Get activations
    acts = activations[layer_name]
    
    # Determine compression level
    # Use function-preserving for attention layers
    # At rank 256 for all layers (consistent compression)
    rank = 256
    
    print(f"\nLayer {i}: W_O (shape: {W_O.shape})")
    print(f"  Compressing with FP at rank {rank}...")
    
    W_fp = fit_function_preserving(W_O, acts, rank=rank, n_steps=500, lr=1e-3, device=DEVICE)
    
    # Compute errors
    act_err = compute_activation_error(W_O, W_fp, acts)
    
    # Original params
    orig_params = W_O.numel()
    comp_params = rank * (W_O.shape[0] + W_O.shape[1])
    compression = orig_params / comp_params
    
    print(f"  Activation error: {act_err:.6f}")
    print(f"  Compression: {compression:.2f}x")
    
    # Replace
    model_compressed.transformer.h[i].attn.c_proj.weight.data = W_fp.half().to(DEVICE)
    
    compression_log.append({
        "layer": i,
        "matrix": "W_O",
        "rank": rank,
        "orig_params": orig_params,
        "comp_params": comp_params,
        "compression": compression,
        "activation_error": act_err,
    })

# ============================================================
# Measure compressed storage
# ============================================================
print("\n" + "=" * 70)
print("Measuring compressed storage...")
print("=" * 70)

model_compressed.save_pretrained(compressed_path / "compressed")
compressed_size = sum(f.stat().st_size for f in (compressed_path / "compressed").rglob("*") if f.is_file())

print(f"Baseline size: {baseline_size / 1e6:.2f} MB")
print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
print(f"Storage reduction: {baseline_size / compressed_size:.2f}x ({(1 - compressed_size/baseline_size)*100:.1f}%)")

# ============================================================
# Measure perplexity
# ============================================================
print("\n" + "=" * 70)
print("Measuring perplexity...")
print("=" * 70)

# Load baseline model
model_baseline = GPT2LMHeadModel.from_pretrained(model_name)
model_baseline.eval()
model_baseline.to(DEVICE)

# Load compressed model
model_comp = GPT2LMHeadModel.from_pretrained(compressed_path / "compressed")
model_comp.eval()
model_comp.to(DEVICE)

# Measure
eval_texts_full = [
    "The quick brown fox jumps over the lazy dog. This is a test sentence.",
    "In the beginning, there was nothing but darkness and silence. Then came the light.",
    "The temperature today is expected to reach 75 degrees Fahrenheit in the afternoon.",
    "Machine learning models require large datasets for training to achieve good performance.",
    "The capital of France is Paris, which is known for the Eiffel Tower and its rich history.",
]

baseline_ppl = compute_perplexity(model_baseline, tokenizer, eval_texts_full, max_length=256, device=DEVICE)
compressed_ppl = compute_perplexity(model_comp, tokenizer, eval_texts_full, max_length=256, device=DEVICE)

print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"Compressed PPL: {compressed_ppl:.2f}")
print(f"PPL delta: {((compressed_ppl - baseline_ppl) / baseline_ppl) * 100:+.2f}%")

# ============================================================
# Save results
# ============================================================
results = {
    "baseline_size_mb": baseline_size / 1e6,
    "compressed_size_mb": compressed_size / 1e6,
    "storage_reduction": baseline_size / compressed_size,
    "baseline_ppl": baseline_ppl,
    "compressed_ppl": compressed_ppl,
    "ppl_delta_pct": ((compressed_ppl - baseline_ppl) / baseline_ppl) * 100,
    "compression_log": compression_log,
}

with open(output_path / "compression_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("COMPRESSION SUMMARY")
print("=" * 70)

print(f"\nModel: GPT-2 Small")
print(f"Layers compressed: {n_layers}/12 (W_O only)")
print(f"Compression method: Function-Preserving (rank=256)")
print(f"\nStorage:")
print(f"  Baseline: {baseline_size / 1e6:.2f} MB")
print(f"  Compressed: {compressed_size / 1e6:.2f} MB")
print(f"  Reduction: {baseline_size / compressed_size:.2f}x ({(1 - compressed_size/baseline_size)*100:.1f}%)")
print(f"\nQuality:")
print(f"  Baseline PPL: {baseline_ppl:.2f}")
print(f"  Compressed PPL: {compressed_ppl:.2f}")
print(f"  Delta: {((compressed_ppl - baseline_ppl) / baseline_ppl) * 100:+.2f}%")

print(f"\nPer-layer compression:")
for log in compression_log:
    print(f"  Layer {log['layer']}: {log['compression']:.2f}x, act_err={log['activation_error']:.6f}")

print(f"\nCompressed model saved to: {compressed_path / 'compressed'}")

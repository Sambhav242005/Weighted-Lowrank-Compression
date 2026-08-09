"""
Causality Test: Which Metric Actually Matters?
===============================================
Optimize each metric independently and measure generation quality.
If optimizing only attention cosine restores generation, it's causally important.
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
# Collect activations
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

# ============================================================
# Optimization functions for each metric
# ============================================================

def optimize_attention_cosine(W_orig, activations, rank, n_steps=1000, lr=1e-3):
    """Optimize W to preserve attention cosine similarity."""
    W_orig = W_orig.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    if flat.shape[0] > 3000:
        idx = torch.randperm(flat.shape[0])[:3000]
        flat = flat[idx]
    
    # Initialize with SVD
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    y_orig = flat @ W_orig.T
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        
        # Compute attention patterns
        attn_orig = torch.softmax(y_orig @ y_orig.T / np.sqrt(y_orig.shape[-1]), dim=-1)
        attn_hat = torch.softmax(y_hat @ y_hat.T / np.sqrt(y_hat.shape[-1]), dim=-1)
        
        # Loss: 1 - attention cosine
        attn_cosine = torch.nn.functional.cosine_similarity(
            attn_orig.flatten(), attn_hat.flatten(), dim=0
        )
        loss = 1 - attn_cosine
        
        loss.backward()
        optimizer.step()
    
    return (B @ A).detach().cpu()


def optimize_cka(W_orig, activations, rank, n_steps=1000, lr=1e-3):
    """Optimize W to preserve CKA similarity."""
    W_orig = W_orig.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    if flat.shape[0] > 3000:
        idx = torch.randperm(flat.shape[0])[:3000]
        flat = flat[idx]
    
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    y_orig = flat @ W_orig.T
    
    def center_kernel(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        
        # Compute CKA
        K_orig = y_orig @ y_orig.T
        K_hat = y_hat @ y_hat.T
        
        K_orig_centered = center_kernel(K_orig)
        K_hat_centered = center_kernel(K_hat)
        
        numerator = (K_orig_centered * K_hat_centered).sum()
        denominator = torch.sqrt((K_orig_centered * K_orig_centered).sum() * 
                                (K_hat_centered * K_hat_centered).sum())
        
        cka = numerator / (denominator + 1e-10)
        loss = 1 - cka
        
        loss.backward()
        optimizer.step()
    
    return (B @ A).detach().cpu()


def optimize_subspace(W_orig, activations, rank, n_steps=1000, lr=1e-3):
    """Optimize W to preserve subspace alignment."""
    W_orig = W_orig.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    if flat.shape[0] > 3000:
        idx = torch.randperm(flat.shape[0])[:3000]
        flat = flat[idx]
    
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    y_orig = flat @ W_orig.T
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        
        # Compute principal subspaces
        U_orig, _, _ = torch.linalg.svd(y_orig.T, full_matrices=False)
        U_hat, _, _ = torch.linalg.svd(y_hat.T, full_matrices=False)
        
        k = min(50, U_orig.shape[1], U_hat.shape[1])
        _, S_sub, _ = torch.linalg.svd(U_orig[:, :k].T @ U_hat[:, :k])
        
        subspace_cosine = S_sub.mean()
        loss = 1 - subspace_cosine
        
        loss.backward()
        optimizer.step()
    
    return (B @ A).detach().cpu()


def optimize_neighborhood(W_orig, activations, rank, n_steps=1000, lr=1e-3):
    """Optimize W to preserve neighborhood structure."""
    W_orig = W_orig.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    if flat.shape[0] > 2000:
        idx = torch.randperm(flat.shape[0])[:2000]
        flat = flat[idx]
    
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    y_orig = flat @ W_orig.T
    
    # Precompute original distances
    with torch.no_grad():
        dist_orig = torch.cdist(y_orig, y_orig)
        k = min(10, y_orig.shape[0] - 1)
        _, nn_orig = dist_orig.topk(k, dim=-1, largest=False)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        
        # Compute neighborhoods
        dist_hat = torch.cdist(y_hat, y_hat)
        _, nn_hat = dist_hat.topk(k, dim=-1, largest=False)
        
        # Compute overlap
        overlap = 0
        for i in range(y_orig.shape[0]):
            set_orig = set(nn_orig[i].cpu().numpy())
            set_hat = set(nn_hat[i].cpu().numpy())
            overlap += len(set_orig & set_hat) / k
        
        neighborhood_score = overlap / y_orig.shape[0]
        loss = 1 - neighborhood_score
        
        loss.backward()
        optimizer.step()
    
    return (B @ A).detach().cpu()


def optimize_combined(W_orig, activations, rank, n_steps=1000, lr=1e-3):
    """Optimize W to preserve all metrics together."""
    W_orig = W_orig.float().to(DEVICE)
    flat = activations.float().reshape(-1, activations.shape[-1]).to(DEVICE)
    
    if flat.shape[0] > 2000:
        idx = torch.randperm(flat.shape[0])[:2000]
        flat = flat[idx]
    
    U, S, Vh = torch.linalg.svd(W_orig, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    
    B = torch.nn.Parameter((U_r * torch.sqrt(S_r)))
    A = torch.nn.Parameter((torch.sqrt(S_r)[:, None] * Vh_r))
    
    optimizer = torch.optim.Adam([A, B], lr=lr)
    
    y_orig = flat @ W_orig.T
    
    def center_kernel(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        return H @ K @ H
    
    for step in range(n_steps):
        optimizer.zero_grad()
        W_hat = B @ A
        y_hat = flat @ W_hat.T
        
        # Combined loss
        loss = 0
        
        # 1. Attention cosine
        attn_orig = torch.softmax(y_orig @ y_orig.T / np.sqrt(y_orig.shape[-1]), dim=-1)
        attn_hat = torch.softmax(y_hat @ y_hat.T / np.sqrt(y_hat.shape[-1]), dim=-1)
        attn_cosine = torch.nn.functional.cosine_similarity(
            attn_orig.flatten(), attn_hat.flatten(), dim=0
        )
        loss += 0.4 * (1 - attn_cosine)
        
        # 2. CKA
        K_orig = center_kernel(y_orig @ y_orig.T)
        K_hat = center_kernel(y_hat @ y_hat.T)
        cka = ((K_orig * K_hat).sum() / 
               torch.sqrt((K_orig * K_orig).sum() * (K_hat * K_hat).sum() + 1e-10) + 1e-10)
        loss += 0.3 * (1 - cka)
        
        # 3. Subspace
        U_orig, _, _ = torch.linalg.svd(y_orig.T, full_matrices=False)
        U_hat, _, _ = torch.linalg.svd(y_hat.T, full_matrices=False)
        k_sub = min(30, U_orig.shape[1], U_hat.shape[1])
        _, S_sub, _ = torch.linalg.svd(U_orig[:, :k_sub].T @ U_hat[:, :k_sub])
        loss += 0.2 * (1 - S_sub.mean())
        
        # 4. MSE (small weight)
        loss += 0.1 * ((y_orig - y_hat) ** 2).mean()
        
        loss.backward()
        optimizer.step()
    
    return (B @ A).detach().cpu()


# ============================================================
# Generation quality metrics
# ============================================================
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

# ============================================================
# Test prompts
# ============================================================
prompts = [
    "Hello",
    "What is 2+2?",
    "The meaning of life is",
    "Once upon a time",
    "In machine learning",
]

# ============================================================
# Run causality test
# ============================================================
print("\n" + "=" * 70)
print("CAUSALITY TEST: Which Metric Actually Matters?")
print("=" * 70)

layer_idx = 5
layer_name = f"layer{layer_idx}.attn.W_O"
W_orig = model.transformer.h[layer_idx].attn.c_proj.weight.data.cpu().float()
acts = activations[layer_name]

# Get baseline
baseline_quality = compute_generation_quality(model, tokenizer, prompts)
print(f"\nBaseline: repetition={baseline_quality['repetition']:.4f}, "
      f"distinct_2={baseline_quality['distinct_2']:.4f}")

# Test each optimization objective
rank = 256
methods = [
    ("SVD 99% (baseline)", lambda W, a: fit_svd_simple(W, 0.99)),
    ("Attention Cosine", lambda W, a: optimize_attention_cosine(W, a, rank)),
    ("CKA", lambda W, a: optimize_cka(W, a, rank)),
    ("Subspace", lambda W, a: optimize_subspace(W, a, rank)),
    ("Combined (0.4*Attn + 0.3*CKA + 0.2*Sub + 0.1*MSE)", lambda W, a: optimize_combined(W, a, rank)),
]

def fit_svd_simple(W, threshold):
    W = W.float()
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, threshold)) + 1
    rank = min(rank, min(m, n))
    return (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]).cpu()

results = []

for name, optimize_fn in methods:
    print(f"\n--- {name} ---")
    
    W_comp = optimize_fn(W_orig, acts)
    
    model_comp = copy.deepcopy(model)
    model_comp.transformer.h[layer_idx].attn.c_proj.weight.data = W_comp.to(DEVICE)
    
    quality = compute_generation_quality(model_comp, tokenizer, prompts)
    
    rep_delta = quality["repetition"] - baseline_quality["repetition"]
    dist_delta = quality["distinct_2"] - baseline_quality["distinct_2"]
    
    print(f"  Repetition: {quality['repetition']:.4f} (delta: {rep_delta:+.4f})")
    print(f"  Distinct-2: {quality['distinct_2']:.4f} (delta: {dist_delta:+.4f})")
    
    results.append({
        "method": name,
        "repetition": quality["repetition"],
        "distinct_2": quality["distinct_2"],
        "rep_delta": rep_delta,
        "dist_delta": dist_delta,
    })
    
    del model_comp
    torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("CAUSALITY TEST SUMMARY")
print("=" * 70)

print(f"\nBaseline: repetition={baseline_quality['repetition']:.4f}, "
      f"distinct_2={baseline_quality['distinct_2']:.4f}")

print(f"\n{'Method':<50} {'Rep Delta':<12} {'Dist Delta':<12}")
print("-" * 74)

for r in results:
    print(f"{r['method']:<50} {r['rep_delta']:<+12.4f} {r['dist_delta']:<+12.4f}")

# Find best method
best_rep = min(results, key=lambda x: x["rep_delta"])
best_dist = max(results, key=lambda x: x["dist_delta"])

print(f"\nBest for reducing repetition: {best_rep['method']}")
print(f"Best for improving diversity: {best_dist['method']}")

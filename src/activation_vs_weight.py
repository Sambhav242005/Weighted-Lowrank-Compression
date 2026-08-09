"""
Activation-Space vs Weight-Space Compression
=============================================
Test: Is the problem SVD itself, or weight-space representation?

If we compress in activation space (compress the hidden states directly)
and find a mapping back to weights, does it work better?

This tests the core hypothesis: "The information geometry lives in
activation space, not weight space."
"""

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
eval_texts_full = eval_texts[:50]
eval_texts_short = eval_texts[:30]

def compute_perplexity(model, tokenizer, texts, max_length=256, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            labels = inputs["input_ids"].clone()
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if n_tokens == 0:
                continue
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    if total_tokens == 0:
        return float('inf')
    return float(np.exp(total_loss / total_tokens))

def collect_activations(model, tokenizer, texts, n_layers=12, max_length=128, device="cpu"):
    activations = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            activations[name] = x.detach()
        return hook_fn
    
    hooks = []
    for i in range(n_layers):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook(f"layer{i}"))
        hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return activations

baseline_ppl = compute_perplexity(model, tokenizer, eval_texts_full, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# Collect activations
print("Collecting activations...")
activations = collect_activations(model, tokenizer, eval_texts_short, device=DEVICE)

results = []

# ============================================================
# Test 1: Weight-space SVD (baseline) at various ranks
# ============================================================
print("=" * 70)
print("TEST 1: Weight-space SVD (baseline)")
print("=" * 70)

for rank in [128, 256, 384]:
    model_sub = copy.deepcopy(model)
    
    for i in range(12):
        W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, min(U.shape[1], Vt.shape[0]))
        W_sub = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
        model_sub.transformer.h[i].attn.c_proj.weight.data = W_sub
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts_full, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    compression = (768 * 768 * 12) / (768 * rank * 12)
    
    print(f"  rank={rank:3d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({"test": "weight_svd", "rank": rank, "ppl": ppl, "delta": delta, "compression": compression})
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Test 2: Activation-space compression
# Compress the activation subspace, find weight that preserves it
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Activation-space compression")
print("=" * 70)

for rank in [128, 256, 384]:
    model_sub = copy.deepcopy(model)
    
    for i in range(12):
        # Get activations at this layer
        key = f"layer{i}"
        if key not in activations:
            continue
        
        acts = activations[key]  # (n_samples, seq_len, dim)
        flat = acts.reshape(-1, acts.shape[-1]).float().to(DEVICE)
        
        if flat.shape[0] > 3000:
            idx = torch.randperm(flat.shape[0])[:3000]
            flat = flat[idx]
        
        # SVD of activation matrix to find principal subspace
        U_act, S_act, Vt_act = torch.linalg.svd(flat.T, full_matrices=False)
        
        # Keep top-rank directions
        U_r = U_act[:, :rank]  # (dim, rank) — principal directions
        
        # Original weight
        W_orig = model_sub.transformer.h[i].attn.c_proj.weight.data.float().to(DEVICE)
        
        # Find weight that maps input to activation subspace
        # W_orig projects input to output. We want W_sub that projects input
        # to the same rank-r subspace of the output.
        
        # Method: project W_orig onto the activation subspace
        # W_sub = U_r @ U_r.T @ W_orig
        W_sub = (U_r @ U_r.T @ W_orig).cpu()
        
        model_sub.transformer.h[i].attn.c_proj.weight.data = W_sub.to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts_full, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    compression = (768 * 768 * 12) / (768 * rank * 12)
    
    print(f"  rank={rank:3d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({"test": "activation_proj", "rank": rank, "ppl": ppl, "delta": delta, "compression": compression})
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Test 3: Dual-space compression
# Compress weight AND activation subspace together
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Dual-space compression (weight SVD + activation projection)")
print("=" * 70)

for rank in [128, 256]:
    model_sub = copy.deepcopy(model)
    
    for i in range(12):
        key = f"layer{i}"
        if key not in activations:
            continue
        
        acts = activations[key]
        flat = acts.reshape(-1, acts.shape[-1]).float().to(DEVICE)
        
        if flat.shape[0] > 3000:
            idx = torch.randperm(flat.shape[0])[:3000]
            flat = flat[idx]
        
        # Weight SVD
        W_orig = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        U_w, S_w, Vt_w = torch.linalg.svd(W_orig, full_matrices=False)
        k = min(rank, min(U_w.shape[1], Vt_w.shape[0]))
        W_svd = (U_w[:, :k] @ torch.diag(S_w[:k]) @ Vt_w[:k, :])
        
        # Activation projection
        U_act, S_act, Vt_act = torch.linalg.svd(flat.T, full_matrices=False)
        U_r = U_act[:, :rank].to(DEVICE)
        
        # Combine: project SVD result onto activation subspace
        W_combined = (U_r @ U_r.T @ W_svd.to(DEVICE)).cpu()
        
        model_sub.transformer.h[i].attn.c_proj.weight.data = W_combined.to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts_full, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    compression = (768 * 768 * 12) / (768 * rank * 12)
    
    print(f"  rank={rank:3d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({"test": "dual_space", "rank": rank, "ppl": ppl, "delta": delta, "compression": compression})
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Test 4: Low-rank product with activation-guided initialization
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Low-rank product with activation-guided init")
print("=" * 70)

for rank in [128, 256]:
    model_sub = copy.deepcopy(model)
    
    for i in range(12):
        key = f"layer{i}"
        if key not in activations:
            continue
        
        acts = activations[key]
        flat = acts.reshape(-1, acts.shape[-1]).float().to(DEVICE)
        
        if flat.shape[0] > 3000:
            idx = torch.randperm(flat.shape[0])[:3000]
            flat = flat[idx]
        
        W_orig = model_sub.transformer.h[i].attn.c_proj.weight.data.float().to(DEVICE)
        
        # Initialize with activation-guided decomposition
        U_act, S_act, Vt_act = torch.linalg.svd(flat.T, full_matrices=False)
        U_r = U_act[:, :rank]  # (dim, rank)
        
        # B = W_orig @ U_r (project weight onto activation subspace)
        B = W_orig @ U_r  # (dim, rank)
        
        # A = U_r.T (transpose of activation subspace)
        A = U_r.T  # (rank, dim)
        
        # W_approx = B @ A
        W_sub = (B @ A).cpu()
        
        model_sub.transformer.h[i].attn.c_proj.weight.data = W_sub.to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts_full, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    compression = (768 * 768 * 12) / (768 * rank * 12)
    
    print(f"  rank={rank:3d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({"test": "activation_guided", "rank": rank, "ppl": ppl, "delta": delta, "compression": compression})
    
    del model_sub
    torch.cuda.empty_cache()

# Save results
output_path = Path("results/activation_vs_weight.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

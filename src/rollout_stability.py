"""
Rollout Stability Analysis
===========================
Test layer fragility, blending, and rollout metrics.
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
    return W_approx

# ============================================================
# Rollout stability metrics
# ============================================================
def compute_rollout_metrics(model, tokenizer, prompts, n_tokens=100):
    """Generate text and measure stability."""
    all_metrics = []
    
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
        
        # Compute metrics
        tokens = tokenizer.encode(generated)
        
        # 1. Repetition ratio (how often tokens repeat)
        if len(tokens) > 1:
            repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
            repetition_ratio = repeats / (len(tokens) - 1)
        else:
            repetition_ratio = 0
        
        # 2. Distinct n-grams
        def distinct_ngrams(tokens, n):
            if len(tokens) < n:
                return 0
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngrams.add(tuple(tokens[i:i+n]))
            return len(ngrams)
        
        distinct_1 = distinct_ngrams(tokens, 1) / len(tokens) if tokens else 0
        distinct_2 = distinct_ngrams(tokens, 2) / max(len(tokens) - 1, 1) if len(tokens) > 1 else 0
        distinct_3 = distinct_ngrams(tokens, 3) / max(len(tokens) - 2, 1) if len(tokens) > 2 else 0
        
        # 3. Loop detection (longest repeated sequence)
        def longest_loop(tokens):
            max_loop = 0
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 50, len(tokens))):
                    # Check if tokens[i:j] repeats
                    seq = tokens[i:j]
                    if len(seq) < 2:
                        continue
                    count = 0
                    for k in range(i, len(tokens) - len(seq) + 1):
                        if tokens[k:k+len(seq)] == seq:
                            count += 1
                    if count > 1:
                        max_loop = max(max_loop, len(seq))
            return max_loop
        
        loop_length = longest_loop(tokens[:200])  # Limit for speed
        
        # 4. Entropy of token distribution
        unique, counts = np.unique(tokens, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        all_metrics.append({
            "prompt": prompt,
            "repetition_ratio": repetition_ratio,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "distinct_3": distinct_3,
            "loop_length": loop_length,
            "entropy": entropy,
            "length": len(tokens),
            "generated": generated[:200],
        })
    
    return all_metrics

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
# Experiment 1: Test each layer individually
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Layer Fragility Test")
print("=" * 70)

layer_fragility = []

for layer_idx in range(12):
    print(f"\n--- Replacing ONLY layer {layer_idx} ---")
    
    model_comp = copy.deepcopy(model)
    W = model.transformer.h[layer_idx].attn.c_proj.weight.data.cpu().float()
    W_approx = fit_svd(W, 0.99)
    model_comp.transformer.h[layer_idx].attn.c_proj.weight.data = W_approx.to(DEVICE)
    
    # Get perplexity
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the meaning of life?",
        "Python is a programming language.",
    ]
    ppl = compute_perplexity(model_comp, tokenizer, test_sentences, max_length=256, device=DEVICE)
    
    # Get rollout metrics
    metrics = compute_rollout_metrics(model_comp, tokenizer, prompts, n_tokens=50)
    
    avg_repetition = np.mean([m["repetition_ratio"] for m in metrics])
    avg_distinct_2 = np.mean([m["distinct_2"] for m in metrics])
    avg_entropy = np.mean([m["entropy"] for m in metrics])
    
    print(f"  PPL: {ppl:.2f}")
    print(f"  Avg repetition: {avg_repetition:.4f}")
    print(f"  Avg distinct-2: {avg_distinct_2:.4f}")
    print(f"  Avg entropy: {avg_entropy:.4f}")
    
    layer_fragility.append({
        "layer": layer_idx,
        "ppl": ppl,
        "repetition": avg_repetition,
        "distinct_2": avg_distinct_2,
        "entropy": avg_entropy,
    })
    
    del model_comp
    torch.cuda.empty_cache()

# ============================================================
# Experiment 2: Baseline metrics
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: Baseline Metrics")
print("=" * 70)

baseline_metrics = compute_rollout_metrics(model, tokenizer, prompts, n_tokens=50)

baseline_repetition = np.mean([m["repetition_ratio"] for m in baseline_metrics])
baseline_distinct_2 = np.mean([m["distinct_2"] for m in baseline_metrics])
baseline_entropy = np.mean([m["entropy"] for m in baseline_metrics])

print(f"Baseline repetition: {baseline_repetition:.4f}")
print(f"Baseline distinct-2: {baseline_distinct_2:.4f}")
print(f"Baseline entropy: {baseline_entropy:.4f}")

# ============================================================
# Experiment 3: Find safest layers
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: Finding Safest Layers")
print("=" * 70)

# Sort by impact (lowest repetition increase = safest)
baseline_ppl = layer_fragility[0]["ppl"]  # Should be close to original
for entry in layer_fragility:
    entry["repetition_delta"] = entry["repetition"] - baseline_repetition
    entry["entropy_delta"] = entry["entropy"] - baseline_entropy

# Sort by repetition delta (lower = safer)
safest_layers = sorted(layer_fragility, key=lambda x: x["repetition_delta"])

print("\nLayer ranking (safest to most fragile):")
print(f"{'Layer':<8} {'PPL':<10} {'Rep Delta':<12} {'Entropy Delta':<14} {'Status'}")
print("-" * 60)

for entry in safest_layers:
    status = "SAFE" if entry["repetition_delta"] < 0.01 else "FRAGILE"
    print(f"{entry['layer']:<8} {entry['ppl']:<10.2f} {entry['repetition_delta']:<+12.4f} "
          f"{entry['entropy_delta']:<+14.4f} {status}")

# ============================================================
# Experiment 4: Replace safest layers together
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: Replace Safest Layers Together")
print("=" * 70)

# Get top 3 and top 5 safest layers
top3 = [e["layer"] for e in safest_layers[:3]]
top5 = [e["layer"] for e in safest_layers[:5]]

print(f"\nTop 3 safest layers: {top3}")
print(f"Top 5 safest layers: {top5}")

# Test replacing top 3
model_comp3 = copy.deepcopy(model)
for i in top3:
    W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    W_approx = fit_svd(W, 0.99)
    model_comp3.transformer.h[i].attn.c_proj.weight.data = W_approx.to(DEVICE)

metrics3 = compute_rollout_metrics(model_comp3, tokenizer, prompts, n_tokens=50)
avg_rep3 = np.mean([m["repetition_ratio"] for m in metrics3])
avg_dist3 = np.mean([m["distinct_2"] for m in metrics3])

print(f"\nTop 3 layers:")
print(f"  Repetition: {avg_rep3:.4f} (baseline: {baseline_repetition:.4f})")
print(f"  Distinct-2: {avg_dist3:.4f} (baseline: {baseline_distinct_2:.4f})")

del model_comp3
torch.cuda.empty_cache()

# Test replacing top 5
model_comp5 = copy.deepcopy(model)
for i in top5:
    W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    W_approx = fit_svd(W, 0.99)
    model_comp5.transformer.h[i].attn.c_proj.weight.data = W_approx.to(DEVICE)

metrics5 = compute_rollout_metrics(model_comp5, tokenizer, prompts, n_tokens=50)
avg_rep5 = np.mean([m["repetition_ratio"] for m in metrics5])
avg_dist5 = np.mean([m["distinct_2"] for m in metrics5])

print(f"\nTop 5 layers:")
print(f"  Repetition: {avg_rep5:.4f} (baseline: {baseline_repetition:.4f})")
print(f"  Distinct-2: {avg_dist5:.4f} (baseline: {baseline_distinct_2:.4f})")

del model_comp5
torch.cuda.empty_cache()

# ============================================================
# Experiment 5: Blending original and compressed
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: Blending Original and Compressed")
print("=" * 70)

alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
blend_results = []

for alpha in alphas:
    print(f"\n--- Alpha={alpha} (0=compressed, 1=original) ---")
    
    model_blend = copy.deepcopy(model)
    
    for i in range(12):
        W_orig = model.transformer.h[i].attn.c_proj.weight.data
        W_comp = fit_svd(W_orig.cpu().float(), 0.99).to(DEVICE)
        
        # Blend: W' = alpha * W_orig + (1-alpha) * W_comp
        W_blended = alpha * W_orig + (1 - alpha) * W_comp
        model_blend.transformer.h[i].attn.c_proj.weight.data = W_blended
    
    # Get metrics
    metrics_blend = compute_rollout_metrics(model_blend, tokenizer, prompts, n_tokens=50)
    avg_rep_blend = np.mean([m["repetition_ratio"] for m in metrics_blend])
    avg_dist_blend = np.mean([m["distinct_2"] for m in metrics_blend])
    
    print(f"  Repetition: {avg_rep_blend:.4f}")
    print(f"  Distinct-2: {avg_dist_blend:.4f}")
    
    blend_results.append({
        "alpha": alpha,
        "repetition": avg_rep_blend,
        "distinct_2": avg_dist_blend,
    })
    
    del model_blend
    torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("ROLLOUT STABILITY SUMMARY")
print("=" * 70)

print(f"\nBaseline:")
print(f"  Repetition: {baseline_repetition:.4f}")
print(f"  Distinct-2: {baseline_distinct_2:.4f}")

print(f"\nSafest layers (lowest repetition increase):")
for entry in safest_layers[:5]:
    print(f"  Layer {entry['layer']}: rep_delta={entry['repetition_delta']:+.4f}")

print(f"\nBlending results:")
print(f"  {'Alpha':<8} {'Repetition':<12} {'Distinct-2'}")
for r in blend_results:
    print(f"  {r['alpha']:<8} {r['repetition']:<12.4f} {r['distinct_2']:.4f}")

print("\nConclusion:")
print("- Some layers are much safer to compress than others")
print("- Blending may help preserve generation quality")
print("- PPL and generation quality can diverge significantly")

"""
Targeted Layer 0 Fix
====================
Layer 0 is the bottleneck not because of its spectrum, but because its error
propagates through all 12 layers and amplifies.

Strategy: Use HIGHER rank for layer 0 (the cheap layers elsewhere can afford it)
and normal rank for layers 1-11.

Also test: Does keeping layer 0 exact (no compression) allow aggressive
compression of all other layers?
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
eval_texts = eval_texts[:50]

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

def svd_reconstruct(W, rank):
    W = W.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, min(U.shape[1], Vt.shape[0]))
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()

# Baseline
baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

results = []

# ============================================================
# Test 1: Higher rank for layer 0
# ============================================================
print("=" * 70)
print("TEST 1: Higher rank for layer 0, rank=128 for others")
print("=" * 70)

for layer0_rank in [128, 256, 384, 512, 768]:
    model_sub = copy.deepcopy(model)
    
    # Layer 0: higher rank
    W0 = model_sub.transformer.h[0].attn.c_proj.weight.data.cpu().float()
    model_sub.transformer.h[0].attn.c_proj.weight.data = svd_reconstruct(W0, layer0_rank).to(DEVICE)
    
    # Layers 1-11: rank=128
    for i in range(1, 12):
        Wi = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(Wi, 128).to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    # Calculate total params
    total_params = 768 * layer0_rank + 768 * 768 * 11  # layer0 B + 11 layers SVD
    original_params = 768 * 768 * 12
    compression = original_params / (768 * layer0_rank + 768 * 128 * 11)
    
    print(f"  L0_rank={layer0_rank:4d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({
        "test": "layer0_higher_rank",
        "layer0_rank": layer0_rank,
        "ppl": ppl,
        "ppl_delta_pct": delta,
        "compression": compression,
    })
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Test 2: Keep layer 0 exact, compress everything else aggressively
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Layer 0 exact, compress layers 1-11 at various ranks")
print("=" * 70)

for other_rank in [64, 32, 16, 8, 4]:
    model_sub = copy.deepcopy(model)
    
    # Layer 0: exact (no compression)
    # Layers 1-11: aggressive compression
    for i in range(1, 12):
        Wi = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(Wi, other_rank).to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    total_params = 768 * 768 + 768 * other_rank * 11  # layer0 exact + 11 layers SVD
    original_params = 768 * 768 * 12
    compression = original_params / total_params
    
    print(f"  Other_rank={other_rank:3d}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({
        "test": "layer0_exact",
        "other_rank": other_rank,
        "ppl": ppl,
        "ppl_delta_pct": delta,
        "compression": compression,
    })
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Test 3: Budget-neutral comparison
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Budget-neutral — same total params, different allocation")
print("=" * 70)

# Target: ~3x compression overall = ~589,824 params per layer average
# Total original: 768*768*12 = 7,077,888
# Target total: ~2,359,296

# Strategy A: Uniform rank=128 for all layers
# Strategy B: Layer 0 rank=256, others rank=115
# Strategy C: Layer 0 rank=384, others rank=103
# Strategy D: Layer 0 exact (768), others rank=98

strategies = [
    ("Uniform rank=128", [(i, 128) for i in range(12)]),
    ("L0=256, others=115", [(0, 256)] + [(i, 115) for i in range(1, 12)]),
    ("L0=384, others=103", [(0, 384)] + [(i, 103) for i in range(1, 12)]),
    ("L0=exact, others=98", [(0, 768)] + [(i, 98) for i in range(1, 12)]),
]

for name, ranks in strategies:
    model_sub = copy.deepcopy(model)
    
    for layer_idx, rank in ranks:
        Wi = model_sub.transformer.h[layer_idx].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[layer_idx].attn.c_proj.weight.data = svd_reconstruct(Wi, rank).to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    total_params = sum(768 * r for _, r in ranks)
    compression = (768 * 768 * 12) / total_params
    
    print(f"  {name:25s}: PPL={ppl:.2f} ({delta:+.2f}%), compression={compression:.1f}x")
    results.append({
        "test": "budget_neutral",
        "strategy": name,
        "ranks": ranks,
        "ppl": ppl,
        "ppl_delta_pct": delta,
        "compression": compression,
    })
    
    del model_sub
    torch.cuda.empty_cache()

# Save results
output_path = Path("results/layer0_fix.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

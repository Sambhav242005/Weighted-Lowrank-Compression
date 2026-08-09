"""
Phase B Summary: Optimal Compression Across Ratios
===================================================
Test inverse eff_rank allocation vs uniform at 2x, 3x, 4x, 6x compression.
This is the definitive experiment comparing our approach to naive SVD.
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

baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# Get effective ranks
eff_ranks = {}
for i in range(12):
    W = model.transformer.h[i].attn.c_proj.weight.data.float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    eff_ranks[i] = torch.exp(entropy).item()

# ============================================================
# Test across compression ratios
# ============================================================

compression_ratios = [2, 3, 4, 6]
all_results = []

for ratio in compression_ratios:
    print(f"\n{'='*70}")
    print(f"COMPRESSION RATIO: {ratio}x")
    print(f"{'='*70}")
    
    avg_rank = 768 // ratio
    total_budget = avg_rank * 12
    
    # Strategy 1: Uniform
    uniform_ranks = {i: avg_rank for i in range(12)}
    
    # Strategy 2: Inverse eff_rank
    inverse_weights = {i: 1/eff_ranks[i] for i in range(12)}
    total_weight = sum(inverse_weights.values())
    inverse_ranks = {i: max(4, int(total_budget * inverse_weights[i] / total_weight)) for i in range(12)}
    
    # Strategy 3: Layer 0 special (give it rank=avg_rank*2, reduce others)
    special_ranks = {i: avg_rank for i in range(12)}
    special_ranks[0] = min(768, avg_rank * 2)
    # Redistribute from others
    extra = special_ranks[0] - avg_rank
    reduction = extra // 11
    for i in range(1, 12):
        special_ranks[i] = max(4, special_ranks[i] - reduction)
    
    strategies = [
        ("Uniform", uniform_ranks),
        ("Inverse eff_rank", inverse_ranks),
        ("L0 special", special_ranks),
    ]
    
    for name, ranks in strategies:
        model_sub = copy.deepcopy(model)
        for i in range(12):
            W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
            model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, ranks[i]).to(DEVICE)
        
        ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        total_params = sum(768 * ranks[i] for i in range(12))
        actual_ratio = (768 * 768 * 12) / total_params
        
        print(f"  {name:20s}: ranks={[ranks[i] for i in range(12)]}")
        print(f"  {'':20s}  PPL={ppl:.2f} ({delta:+.2f}%), ratio={actual_ratio:.1f}x")
        
        all_results.append({
            "ratio": ratio,
            "strategy": name,
            "ranks": ranks,
            "ppl": ppl,
            "ppl_delta_pct": delta,
            "actual_ratio": actual_ratio,
        })
        
        del model_sub
        torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("PHASE B SUMMARY: Optimal Compression Results")
print(f"{'='*70}")

print(f"\nBaseline PPL: {baseline_ppl:.2f}\n")

print(f"{'Ratio':<8} {'Strategy':<20} {'PPL':<10} {'Delta':<12} {'Actual Ratio':<14}")
print("-" * 64)

for r in all_results:
    print(f"{r['ratio']:<8} {r['strategy']:<20} {r['ppl']:<10.2f} {r['ppl_delta_pct']:<+12.2f} {r['actual_ratio']:<14.1f}")

# Find best strategy per ratio
print(f"\n{'Best strategy per ratio:'}")
for ratio in compression_ratios:
    ratio_results = [r for r in all_results if r["ratio"] == ratio]
    best = min(ratio_results, key=lambda x: x["ppl_delta_pct"])
    print(f"  {ratio}x: {best['strategy']} ({best['ppl_delta_pct']:+.2f}%)")

# Save results
output_path = Path("results/phase_b_summary.json")
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")

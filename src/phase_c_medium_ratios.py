"""
Phase C Part 2: GPT-2 Medium at higher compression ratios
Test 2x, 3x, 4x, 6x on GPT-2 Medium to see if the model is more robust.
"""

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE)
model.eval()

n_layers = len(model.transformer.h)
hidden = model.config.n_embd

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50][:50]

def compute_perplexity(model, tokenizer, texts, max_length=256, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            labels = inputs["input_ids"].clone()
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if n_tokens == 0: continue
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')

def svd_reconstruct(W, rank):
    W = W.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, min(U.shape[1], Vt.shape[0]))
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()

baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"GPT-2 Medium Baseline PPL: {baseline_ppl:.2f}")

# Effective ranks
eff_ranks = {}
for i in range(n_layers):
    W = model.transformer.h[i].attn.c_proj.weight.data.float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    eff_ranks[i] = torch.exp(entropy).item()

# Test across ratios
all_results = []
for ratio in [2, 3, 4, 6]:
    print(f"\n{'='*70}")
    print(f"COMPRESSION RATIO: {ratio}x (rank={hidden//ratio})")
    print(f"{'='*70}")
    
    rank = hidden // ratio
    
    # Uniform
    model_sub = copy.deepcopy(model)
    for i in range(n_layers):
        W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, rank).to(DEVICE)
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"  Uniform (rank={rank:3d}): PPL={ppl:.2f} ({delta:+.2f}%)")
    all_results.append({"ratio": ratio, "strategy": "uniform", "rank": rank, "ppl": ppl, "delta_pct": delta})
    del model_sub; torch.cuda.empty_cache()
    
    # Compare with GPT-2 Small at same ratio
    small_rank = 768 // ratio
    print(f"  (GPT-2 Small at rank={small_rank} was +{['+5.50%', '+21.36%', '+152.12%', '+261.81%'][[2,3,4,6].index(ratio)]})")

print(f"\n{'='*70}")
print("SUMMARY: GPT-2 Medium vs Small Robustness")
print(f"{'='*70}")
print(f"{'Ratio':<8} {'Medium PPL Delta':<20} {'Small PPL Delta':<20} {'Ratio'}")
print("-" * 70)

small_deltas = {2: 5.50, 3: 21.36, 4: 152.12, 6: 261.81}
for r in all_results:
    small_d = small_deltas[r["ratio"]]
    improvement = small_d / max(abs(r["delta_pct"]), 0.01)
    print(f"{r['ratio']:<8} {r['delta_pct']:<+20.2f} {small_d:<+20.2f} {improvement:.1f}x more robust")

# Save
output_path = Path("results/phase_c_medium_ratios.json")
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {output_path}")

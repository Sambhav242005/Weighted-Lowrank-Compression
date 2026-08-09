"""
Test GPT-2 Large (774M) to see if robustness scales with model size.
If Large is even more robust, it's a scaling issue.
If Large shows similar degradation, it's a fundamental SVD problem.
"""

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0)}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(DEVICE)
model.eval()

n_layers = len(model.transformer.h)
hidden = model.config.n_embd
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"GPT-2 Large: {params:.0f}M params, {n_layers} layers, {hidden} hidden")

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

def svd_compress(model, rank):
    m = copy.deepcopy(model)
    for i in range(n_layers):
        W = m.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, min(U.shape[1], Vt.shape[0]))
        m.transformer.h[i].attn.c_proj.weight.data = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
    return m

baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# Effective ranks
eff_ranks = []
for i in range(n_layers):
    W = model.transformer.h[i].attn.c_proj.weight.data.float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    eff_ranks.append(torch.exp(entropy).item())

print("Effective ranks:")
for i, er in enumerate(eff_ranks):
    print(f"  Layer {i:2d}: {er:.1f}")
print(f"  Mean: {np.mean(eff_ranks):.1f}")

# Test at 2x, 3x, 4x, 6x
results = []
for ratio in [2, 3, 4, 6]:
    rank = hidden // ratio
    m = svd_compress(model, rank)
    ppl = compute_perplexity(m, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results.append({"ratio": ratio, "rank": rank, "ppl": ppl, "delta": delta})
    print(f"  {ratio}x (rank={rank:4d}): PPL={ppl:.2f} ({delta:+.2f}%)")
    del m; torch.cuda.empty_cache()

# Compare all three models
print(f"\n{'='*70}")
print("SCALING COMPARISON: Small (124M) vs Medium (355M) vs Large (774M)")
print(f"{'='*70}")

small = {2: 5.50, 3: 21.36, 4: 152.12, 6: 261.81}
medium = {2: 2.63, 3: 9.91, 4: 20.71, 6: 48.60}

print(f"\n{'Ratio':<8} {'Small':<15} {'Medium':<15} {'Large':<15} {'Trend'}")
print("-"*65)

for r in results:
    s = small[r["ratio"]]
    m = medium[r["ratio"]]
    l = r["delta"]
    
    # Is it getting better or worse?
    if l < m:
        trend = "IMPROVING"
    elif l > m * 2:
        trend = "DEGRADING"
    else:
        trend = "similar"
    
    print(f"{r['ratio']:<8} {s:<+15.2f} {m:<+15.2f} {l:<+15.2f} {trend}")

# Save
output = {
    "model": "gpt2-large",
    "params": params,
    "n_layers": n_layers,
    "hidden": hidden,
    "baseline_ppl": baseline_ppl,
    "eff_ranks": eff_ranks,
    "results": results,
}
with open("results/phase_c_large.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/phase_c_large.json")

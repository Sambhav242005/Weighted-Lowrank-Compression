"""
Phase C: Test on GPT-2 Medium
==============================
Validate if our findings from GPT-2 Small generalize to a larger model.
GPT-2 Medium: 24 layers, 1024 hidden, 16 heads
"""

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(DEVICE)
model.eval()

n_layers = len(model.transformer.h)
hidden = model.config.n_embd
n_heads = model.config.n_head
print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {n_layers} layers, {hidden} hidden, {n_heads} heads")

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

# ============================================================
# Step 1: Baseline
# ============================================================
baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"\nBaseline PPL: {baseline_ppl:.2f}")

# ============================================================
# Step 2: Effective ranks
# ============================================================
eff_ranks = {}
for i in range(n_layers):
    W = model.transformer.h[i].attn.c_proj.weight.data.float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    eff_ranks[i] = torch.exp(entropy).item()

print(f"\nEffective ranks:")
for i in range(n_layers):
    print(f"  Layer {i:2d}: eff_rank = {eff_ranks[i]:.1f}")

# ============================================================
# Step 3: Per-layer compression sensitivity
# ============================================================
print(f"\n{'='*70}")
print("STEP 3: Per-layer compression sensitivity (rank=128)")
print(f"{'='*70}")

layer_ppls = {}
for i in range(n_layers):
    model_sub = copy.deepcopy(model)
    W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, 128).to(DEVICE)
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    layer_ppls[i] = {"ppl": ppl, "delta_pct": delta, "eff_rank": eff_ranks[i]}
    print(f"  Layer {i:2d}: PPL={ppl:.2f} ({delta:+.2f}%), eff_rank={eff_ranks[i]:.1f}")
    del model_sub
    torch.cuda.empty_cache()

# Correlation
deltas = [layer_ppls[i]["delta_pct"] for i in range(n_layers)]
ranks = [layer_ppls[i]["eff_rank"] for i in range(n_layers)]
correlation = np.corrcoef(deltas, ranks)[0, 1]
print(f"\nCorrelation (eff_rank vs damage): r = {correlation:.3f}")

# ============================================================
# Step 4: Optimal allocation at 3x compression
# ============================================================
print(f"\n{'='*70}")
print("STEP 4: Optimal allocation at 3x compression")
print(f"{'='*70}")

total_budget = hidden * n_layers // 3  # 3x compression

# Uniform
uniform_ranks = {i: hidden // 3 for i in range(n_layers)}

# Inverse eff_rank
inverse_weights = {i: 1/eff_ranks[i] for i in range(n_layers)}
total_weight = sum(inverse_weights.values())
inverse_ranks = {i: max(4, int(total_budget * inverse_weights[i] / total_weight)) for i in range(n_layers)}

# Layer 0 special
special_ranks = {i: hidden // 3 for i in range(n_layers)}
special_ranks[0] = min(hidden, (hidden // 3) * 2)
extra = special_ranks[0] - hidden // 3
reduction = extra // (n_layers - 1)
for i in range(1, n_layers):
    special_ranks[i] = max(4, special_ranks[i] - reduction)

strategies = [
    ("Uniform", uniform_ranks),
    ("Inverse eff_rank", inverse_ranks),
    ("L0 special", special_ranks),
]

results = []
for name, ranks in strategies:
    model_sub = copy.deepcopy(model)
    for i in range(n_layers):
        W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, ranks[i]).to(DEVICE)
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    total_params = sum(hidden * ranks[i] for i in range(n_layers))
    actual_ratio = (hidden * hidden * n_layers) / total_params
    
    print(f"  {name:20s}: PPL={ppl:.2f} ({delta:+.2f}%), ratio={actual_ratio:.1f}x")
    results.append({"strategy": name, "ppl": ppl, "delta_pct": delta, "ranks": list(ranks.values())})
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Step 5: Drift profile
# ============================================================
print(f"\n{'='*70}")
print("STEP 5: Drift profile at rank=128")
print(f"{'='*70}")

import torch.nn.functional as F

def get_activations(model, text, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    activations = {}
    hooks = []
    for i in range(n_layers):
        def hook_fn(module, input, output, layer_idx=i):
            activations[layer_idx] = output[0].detach().cpu().float()
        hooks.append(model.transformer.h[i].register_forward_hook(hook_fn))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return activations

drifts = {}
sample_text = eval_texts[0] if eval_texts else "The quick brown fox jumps over the lazy dog."
acts_orig = get_activations(model, sample_text, tokenizer, DEVICE)

model_sub = copy.deepcopy(model)
for i in range(n_layers):
    W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, 128).to(DEVICE)

acts_comp = get_activations(model_sub, sample_text, tokenizer, DEVICE)

for i in range(n_layers):
    if i in acts_orig and i in acts_comp:
        cos_sim = F.cosine_similarity(
            acts_orig[i].reshape(1, -1),
            acts_comp[i].reshape(1, -1)
        ).item()
        mse = F.mse_loss(acts_orig[i], acts_comp[i]).item()
        drifts[i] = {"cosine_sim": cos_sim, "mse": mse}
        print(f"  Layer {i:2d}: cos={cos_sim:.4f}, MSE={mse:.6f}")

del model_sub, acts_orig, acts_comp
torch.cuda.empty_cache()

# ============================================================
# Save results
# ============================================================
output = {
    "model": "gpt2-medium",
    "n_layers": n_layers,
    "hidden": hidden,
    "n_heads": n_heads,
    "baseline_ppl": baseline_ppl,
    "eff_ranks": eff_ranks,
    "layer_sensitivity": layer_ppls,
    "correlation_eff_rank_damage": correlation,
    "allocation_results": results,
    "drift_at_rank128": drifts,
}

output_path = Path("results/phase_c_medium.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print("PHASE C SUMMARY: GPT-2 Medium")
print(f"{'='*70}")
print(f"Baseline PPL: {baseline_ppl:.2f}")
print(f"Correlation (eff_rank vs damage): r = {correlation:.3f}")
print(f"\nBest strategy at 3x:")
best = min(results, key=lambda x: x["ppl"])
print(f"  {best['strategy']}: PPL={best['ppl']:.2f} ({best['delta_pct']:+.2f}%)")
print(f"\nResults saved to {output_path}")

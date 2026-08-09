"""
Optimal Budget Allocator
========================
Use the sensitivity predictor to allocate rank budget optimally.

Strategy:
- Measure eff_rank for each layer
- Predict PPL impact for each rank allocation
- Find allocation that minimizes total PPL at given total budget
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

# ============================================================
# Step 1: Get effective ranks
# ============================================================
print("=" * 70)
print("STEP 1: Effective ranks")
print("=" * 70)

eff_ranks = {}
for i in range(12):
    W = model.transformer.h[i].attn.c_proj.weight.data.float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
    eff_ranks[i] = torch.exp(entropy).item()
    print(f"  Layer {i:2d}: eff_rank = {eff_ranks[i]:.1f}")

# ============================================================
# Step 2: Measure rank-PPL relationship for each layer
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Rank-PPL curves for each layer")
print("=" * 70)

rank_ppl_curves = {}
ranks_to_test = [16, 32, 64, 128, 192, 256, 384, 512, 768]

for layer_idx in [0, 5, 10]:  # Sample 3 layers to build curve
    print(f"\n  Layer {layer_idx}:")
    curve = []
    
    for rank in ranks_to_test:
        model_sub = copy.deepcopy(model)
        W = model_sub.transformer.h[layer_idx].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[layer_idx].attn.c_proj.weight.data = svd_reconstruct(W, rank).to(DEVICE)
        
        ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
        delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
        
        curve.append({"rank": rank, "ppl_delta": delta})
        print(f"    rank={rank:4d}: PPL delta = {delta:+.2f}%")
        
        del model_sub
        torch.cuda.empty_cache()
    
    rank_ppl_curves[f"layer{layer_idx}"] = curve

# ============================================================
# Step 3: Build interpolation model
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Interpolation model")
print("=" * 70)

# For each layer, fit a curve: PPL_delta = f(rank)
# Use log-linear interpolation
from scipy.interpolate import interp1d

interpolators = {}
for layer_idx in [0, 5, 10]:
    curve = rank_ppl_curves[f"layer{layer_idx}"]
    ranks = [c["rank"] for c in curve]
    deltas = [c["ppl_delta"] for c in curve]
    
    # Log-linear interpolation
    log_ranks = np.log(ranks)
    interp = interp1d(log_ranks, deltas, kind='linear', fill_value='extrapolate')
    interpolators[layer_idx] = interp
    
    print(f"  Layer {layer_idx}: fitted curve")

# ============================================================
# Step 4: Optimize budget allocation
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Budget optimization")
print("=" * 70)

# Target: 3x compression overall
# Total params: 768 * 768 * 12 = 7,077,888
# Target: 7,077,888 / 3 = 2,359,296
# Average rank: 2,359,296 / (768 * 12) = 256

target_avg_rank = 256
total_budget = target_avg_rank * 12  # 3072 total rank units

def predict_total_ppl_delta(ranks_dict):
    """Predict total PPL delta for a given rank allocation."""
    total_delta = 0
    for layer_idx in range(12):
        rank = ranks_dict[layer_idx]
        if layer_idx in interpolators:
            log_rank = np.log(max(rank, 1))
            delta = float(interpolators[layer_idx](log_rank))
        else:
            # For layers not fitted, use linear interpolation from neighbors
            # Simple approximation: delta = k * (1/rank - 1/768)
            # where k depends on eff_rank
            k = 100 * (768 / eff_ranks[layer_idx])  # Higher k for lower eff_rank
            delta = k * (1/rank - 1/768) * 100
        total_delta += delta
    return total_delta

# Strategy 1: Uniform
uniform_ranks = {i: target_avg_rank for i in range(12)}
uniform_delta = predict_total_ppl_delta(uniform_ranks)
print(f"\n  Strategy 1: Uniform rank={target_avg_rank}")
print(f"    Predicted total PPL delta: {uniform_delta:+.2f}%")

# Strategy 2: Inverse eff_rank (give low eff_rank layers more budget)
inverse_weights = {i: 1/eff_ranks[i] for i in range(12)}
total_weight = sum(inverse_weights.values())
inverse_ranks = {i: int(total_budget * inverse_weights[i] / total_weight) for i in range(12)}
inverse_delta = predict_total_ppl_delta(inverse_ranks)
print(f"\n  Strategy 2: Inverse eff_rank allocation")
print(f"    Ranks: {[inverse_ranks[i] for i in range(12)]}")
print(f"    Predicted total PPL delta: {inverse_delta:+.2f}%")

# Strategy 3: Quadratic (give very low eff_rank layers much more)
quad_weights = {i: (1/eff_ranks[i])**2 for i in range(12)}
total_weight = sum(quad_weights.values())
quad_ranks = {i: max(16, int(total_budget * quad_weights[i] / total_weight)) for i in range(12)}
quad_delta = predict_total_ppl_delta(quad_ranks)
print(f"\n  Strategy 3: Quadratic inverse eff_rank")
print(f"    Ranks: {[quad_ranks[i] for i in range(12)]}")
print(f"    Predicted total PPL delta: {quad_delta:+.2f}%")

# Strategy 4: Position-aware (layers 0,11 get more)
position_weights = {i: 1.0 for i in range(12)}
position_weights[0] = 3.0  # Layer 0 is most sensitive
position_weights[11] = 1.5  # Layer 11 is moderately sensitive
total_weight = sum(position_weights.values())
position_ranks = {i: int(total_budget * position_weights[i] / total_weight) for i in range(12)}
position_delta = predict_total_ppl_delta(position_ranks)
print(f"\n  Strategy 4: Position-aware (L0=3x, L11=1.5x)")
print(f"    Ranks: {[position_ranks[i] for i in range(12)]}")
print(f"    Predicted total PPL delta: {position_delta:+.2f}%")

# Strategy 5: Combined (eff_rank + position)
combined_weights = {i: (1/eff_ranks[i]) * position_weights[i] for i in range(12)}
total_weight = sum(combined_weights.values())
combined_ranks = {i: max(16, int(total_budget * combined_weights[i] / total_weight)) for i in range(12)}
combined_delta = predict_total_ppl_delta(combined_ranks)
print(f"\n  Strategy 5: Combined (eff_rank + position)")
print(f"    Ranks: {[combined_ranks[i] for i in range(12)]}")
print(f"    Predicted total PPL delta: {combined_delta:+.2f}%")

# ============================================================
# Step 5: Validate best strategies
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Validation (top 3 strategies)")
print("=" * 70)

strategies = [
    ("Uniform", uniform_ranks),
    ("Inverse eff_rank", inverse_ranks),
    ("Quadratic", quad_ranks),
    ("Position-aware", position_ranks),
    ("Combined", combined_ranks),
]

strategies.sort(key=lambda x: predict_total_ppl_delta(x[1]))

for name, ranks in strategies[:3]:
    print(f"\n  {name}:")
    print(f"    Ranks: {[ranks[i] for i in range(12)]}")
    print(f"    Total params: {sum(768 * ranks[i] for i in range(12)):,}")
    print(f"    Compression: {768*768*12 / sum(768*ranks[i] for i in range(12)):.1f}x")
    
    # Actually measure
    model_sub = copy.deepcopy(model)
    for i in range(12):
        W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        model_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, ranks[i]).to(DEVICE)
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    actual_delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    predicted_delta = predict_total_ppl_delta(ranks)
    
    print(f"    Predicted PPL delta: {predicted_delta:+.2f}%")
    print(f"    Actual PPL delta: {actual_delta:+.2f}%")
    
    del model_sub
    torch.cuda.empty_cache()

# Save results
output = {
    "baseline_ppl": baseline_ppl,
    "eff_ranks": eff_ranks,
    "rank_ppl_curves": rank_ppl_curves,
    "strategies": {
        name: {
            "ranks": ranks,
            "predicted_delta": predict_total_ppl_delta(ranks),
        }
        for name, ranks in strategies
    },
}

output_path = Path("results/optimal_budget.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")

"""
Compression Sensitivity Predictor
=================================
Can we predict PPL degradation from weight properties BEFORE compressing?

Key insight from error propagation model:
- Jacobian norms predict RELATIVE ordering of layer sensitivity
- But absolute prediction requires accounting for nonlinear contraction

Practical approach: Empirical correlation between weight properties and PPL impact.
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

baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# ============================================================
# Step 1: Extract weight features for each layer
# ============================================================
print("=" * 70)
print("STEP 1: Extracting weight features")
print("=" * 70)

features = {}
for i in range(12):
    block = model.transformer.h[i]
    
    # Weight matrices
    W_O = block.attn.c_proj.weight.data.float()
    W_c_attn = block.attn.c_attn.weight.data.float()
    W_Q = W_c_attn[:, :768]
    W_K = W_c_attn[:, 768:1536]
    W_V = W_c_attn[:, 1536:]
    W_up = block.mlp.c_fc.weight.data.float()
    W_down = block.mlp.c_proj.weight.data.float()
    
    # Layer norm parameters
    ln1_w = block.ln_1.weight.data.float()
    ln1_b = block.ln_1.bias.data.float()
    ln2_w = block.ln_2.weight.data.float()
    ln2_b = block.ln_2.bias.data.float()
    
    # Spectral norms
    _, S_O, _ = torch.linalg.svd(W_O, full_matrices=False)
    _, S_Q, _ = torch.linalg.svd(W_Q, full_matrices=False)
    _, S_K, _ = torch.linalg.svd(W_K, full_matrices=False)
    _, S_V, _ = torch.linalg.svd(W_V, full_matrices=False)
    _, S_up, _ = torch.linalg.svd(W_up, full_matrices=False)
    _, S_down, _ = torch.linalg.svd(W_down, full_matrices=False)
    
    # Effective rank (entropy-based)
    def effective_rank(S):
        s_norm = S / S.sum()
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
        return torch.exp(entropy).item()
    
    # Variance captured by top-k
    total_var = (S_O ** 2).sum()
    cumvar = torch.cumsum(S_O ** 2, dim=0) / total_var
    var_top64 = float(cumvar[63] * 100) if len(cumvar) > 63 else 0
    var_top128 = float(cumvar[127] * 100) if len(cumvar) > 127 else 0
    
    # Weight statistics
    weight_norm = W_O.norm().item()
    weight_mean = W_O.mean().item()
    weight_std = W_O.std().item()
    
    # Layer norm statistics
    ln1_scale = ln1_w.mean().item()
    ln2_scale = ln2_w.mean().item()
    
    # MLP expansion ratio
    mlp_ratio = S_up[0].item() / S_O[0].item()
    
    features[f"layer{i}"] = {
        "W_O_spectral_norm": S_O[0].item(),
        "W_O_eff_rank": effective_rank(S_O),
        "W_O_var_top64": var_top64,
        "W_O_var_top128": var_top128,
        "W_O_weight_norm": weight_norm,
        "W_O_weight_mean": weight_mean,
        "W_O_weight_std": weight_std,
        "W_Q_spectral_norm": S_Q[0].item(),
        "W_K_spectral_norm": S_K[0].item(),
        "W_V_spectral_norm": S_V[0].item(),
        "W_up_spectral_norm": S_up[0].item(),
        "W_down_spectral_norm": S_down[0].item(),
        "mlp_ratio": mlp_ratio,
        "ln1_scale": ln1_scale,
        "ln2_scale": ln2_scale,
        "effective_jacobian_norm": 1 + S_O[0].item() + S_down[0].item() * S_up[0].item(),
    }
    
    print(f"  Layer {i:2d}: W_O_norm={S_O[0]:.2f}, eff_rank={effective_rank(S_O):.0f}, "
          f"var_top128={var_top128:.1f}%, E_J={features[f'layer{i}']['effective_jacobian_norm']:.0f}")

# ============================================================
# Step 2: Measure actual PPL impact for each layer
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Measuring actual PPL impact")
print("=" * 70)

actual_impact = {}
for i in range(12):
    model_sub = copy.deepcopy(model)
    W = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = 128
    W_sub = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
    model_sub.transformer.h[i].attn.c_proj.weight.data = W_sub
    
    ppl = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    
    actual_impact[f"layer{i}"] = {
        "ppl": ppl,
        "ppl_delta_pct": delta,
    }
    
    print(f"  Layer {i:2d}: PPL={ppl:.2f} ({delta:+.2f}%)")
    
    del model_sub
    torch.cuda.empty_cache()

# ============================================================
# Step 3: Correlation analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Correlation analysis")
print("=" * 70)

# Prepare data for correlation
layer_indices = list(range(12))
actual_deltas = [actual_impact[f"layer{i}"]["ppl_delta_pct"] for i in layer_indices]

# Features to correlate
feature_names = [
    "W_O_spectral_norm", "W_O_eff_rank", "W_O_var_top128",
    "W_O_weight_norm", "effective_jacobian_norm",
    "W_up_spectral_norm", "W_down_spectral_norm", "mlp_ratio",
]

correlations = {}
for fname in feature_names:
    values = [features[f"layer{i}"][fname] for i in layer_indices]
    
    # Pearson correlation
    corr = np.corrcoef(values, actual_deltas)[0, 1]
    correlations[fname] = corr
    
    print(f"  {fname:25s}: r = {corr:+.3f}")

# Find best predictor
best_feature = max(correlations, key=lambda k: abs(correlations[k]))
print(f"\n  Best predictor: {best_feature} (r = {correlations[best_feature]:+.3f})")

# ============================================================
# Step 4: Build prediction model
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Linear prediction model")
print("=" * 70)

# Simple linear regression: PPL_delta = a * feature + b
from numpy.polynomial import polynomial as P

X = np.array([features[f"layer{i}"][best_feature] for i in layer_indices])
y = np.array(actual_deltas)

# Fit linear model
coeffs = np.polyfit(X, y, 1)
predictions = np.polyval(coeffs, X)

print(f"\n  Model: PPL_delta = {coeffs[0]:.4f} * {best_feature} + {coeffs[1]:.4f}")
print(f"\n  {'Layer':<8} {'Feature':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
print("-" * 56)

for i in layer_indices:
    feat_val = features[f"layer{i}"][best_feature]
    actual = actual_deltas[i]
    predicted = predictions[i]
    error = predicted - actual
    print(f"  {i:<8} {feat_val:<12.2f} {actual:<+12.2f} {predicted:<+12.2f} {error:<+12.2f}")

# R-squared
ss_res = np.sum((y - predictions) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\n  R-squared: {r_squared:.3f}")

# ============================================================
# Step 5: Multi-feature prediction
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Multi-feature prediction (top 3)")
print("=" * 70)

# Select top 3 features
top_features = sorted(correlations, key=lambda k: abs(correlations[k]), reverse=True)[:3]
print(f"  Top features: {top_features}")

# Build multi-feature model
X_multi = np.array([[features[f"layer{i}"][f] for f in top_features] for i in layer_indices])
coeffs_multi = np.linalg.lstsq(X_multi, y, rcond=None)[0]
predictions_multi = X_multi @ coeffs_multi

print(f"\n  {'Layer':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
print("-" * 44)

for i in layer_indices:
    actual = actual_deltas[i]
    predicted = predictions_multi[i]
    error = predicted - actual
    print(f"  {i:<8} {actual:<+12.2f} {predicted:<+12.2f} {error:<+12.2f}")

ss_res_multi = np.sum((y - predictions_multi) ** 2)
r_squared_multi = 1 - (ss_res_multi / ss_tot)
print(f"\n  R-squared (multi-feature): {r_squared_multi:.3f}")

# ============================================================
# Step 6: Practical recommendation
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Practical recommendation")
print("=" * 70)

# Rank layers by predicted sensitivity
layer_sensitivity = []
for i in layer_indices:
    predicted_delta = predictions_multi[i]
    layer_sensitivity.append((i, predicted_delta))

layer_sensitivity.sort(key=lambda x: x[1])

print(f"\n  Layer sensitivity ranking (most to least sensitive):")
for rank, (layer, delta) in enumerate(layer_sensitivity):
    print(f"    {rank+1:2d}. Layer {layer:2d}: predicted PPL delta = {delta:+.2f}%")

# Optimal budget allocation
print(f"\n  Budget allocation strategy:")
print(f"    - Give layers 0, 11 higher rank (they're most sensitive)")
print(f"    - Give layers 1-10 lower rank (they're more robust)")
print(f"    - Total budget: same as uniform rank=128")

# Save results
output = {
    "features": features,
    "actual_impact": actual_impact,
    "correlations": correlations,
    "best_feature": best_feature,
    "linear_model": {"slope": coeffs[0], "intercept": coeffs[1], "r_squared": r_squared},
    "multi_feature_model": {
        "features": top_features,
        "coefficients": coeffs_multi.tolist(),
        "r_squared": r_squared_multi,
    },
    "layer_sensitivity": [(l, float(d)) for l, d in layer_sensitivity],
}

output_path = Path("results/compression_sensitivity.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")

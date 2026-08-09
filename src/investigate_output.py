"""
Investigate WHY compressed models generate different output.
Look at: attention patterns, hidden states, probability distributions, token-level shifts.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

def svd_compress(model, rank):
    m = copy.deepcopy(model)
    for i in range(12):
        W = m.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        k = min(rank, min(U.shape[1], Vt.shape[0]))
        m.transformer.h[i].attn.c_proj.weight.data = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
    return m

model_256 = svd_compress(model, 256)
model_128 = svd_compress(model, 128)

prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

# ============================================================
# 1. Attention Pattern Comparison
# ============================================================
print("="*80)
print("1. ATTENTION PATTERN COMPARISON")
print("="*80)

def get_attention_patterns(model, inputs):
    attn_patterns = {}
    attn_outputs = {}
    hooks = []
    
    def hook_fn(module, input, output, layer_idx):
        # output is (attn_output, attn_weights, past_key_values)
        # attn_weights: (batch, heads, seq, seq)
        if len(output) > 1 and output[1] is not None:
            attn_patterns[layer_idx] = output[1].detach().cpu().float()
        attn_outputs[layer_idx] = output[0].detach().cpu().float()
    
    for i in range(12):
        hooks.append(model.transformer.h[i].attn.register_forward_hook(
            lambda mod, inp, out, idx=i: hook_fn(mod, inp, out, idx)
        ))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return attn_patterns, attn_outputs

print("\nCollecting attention patterns...")
attn_orig, _ = get_attention_patterns(model, inputs)
attn_256, _ = get_attention_patterns(model_256, inputs)
attn_128, _ = get_attention_patterns(model_128, inputs)

for i in range(12):
    if i in attn_orig and i in attn_256 and i in attn_128:
        orig = attn_orig[i]  # (1, heads, seq, seq)
        comp256 = attn_256[i]
        comp128 = attn_128[i]
        
        # Compare attention entropy (focus distribution)
        def attention_entropy(attn):
            # attn: (1, heads, seq, seq)
            attn = attn + 1e-10
            entropy = -(attn * torch.log(attn)).sum(dim=-1).mean()
            return entropy.item()
        
        ent_orig = attention_entropy(orig)
        ent_256 = attention_entropy(comp256)
        ent_128 = attention_entropy(comp128)
        
        # Compare attention pattern similarity
        cos_sim_256 = F.cosine_similarity(orig.reshape(1,-1), comp256.reshape(1,-1)).item()
        cos_sim_128 = F.cosine_similarity(orig.reshape(1,-1), comp128.reshape(1,-1)).item()
        
        print(f"  Layer {i:2d}: entropy orig={ent_orig:.4f} 256={ent_256:.4f} 128={ent_128:.4f} | cos 256={cos_sim_256:.4f} 128={cos_sim_128:.4f}")

# ============================================================
# 2. Hidden State Drift Per Layer
# ============================================================
print("\n" + "="*80)
print("2. HIDDEN STATE DRIFT (token-by-token)")
print("="*80)

def get_hidden_states(model, inputs):
    hiddens = {}
    hooks = []
    for i in range(12):
        def hook_fn(module, input, output, layer_idx=i):
            hiddens[layer_idx] = output[0].detach().cpu().float()
        hooks.append(model.transformer.h[i].register_forward_hook(hook_fn))
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return hiddens

h_orig = get_hidden_states(model, inputs)
h_256 = get_hidden_states(model_256, inputs)
h_128 = get_hidden_states(model_128, inputs)

print(f"\nPrompt tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
print(f"{'Layer':<8} {'Cos@256':<12} {'Cos@128':<12} {'MSE@256':<12} {'MSE@128':<12} {'Drift Direction'}")
print("-"*70)

prev_cos_256 = 1.0
prev_cos_128 = 1.0

for i in range(12):
    if i in h_orig and i in h_256 and i in h_128:
        cos_256 = F.cosine_similarity(h_orig[i].reshape(1,-1), h_256[i].reshape(1,-1)).item()
        cos_128 = F.cosine_similarity(h_orig[i].reshape(1,-1), h_128[i].reshape(1,-1)).item()
        mse_256 = F.mse_loss(h_orig[i], h_256[i]).item()
        mse_128 = F.mse_loss(h_orig[i], h_128[i]).item()
        
        # Drift acceleration
        accel_256 = prev_cos_256 - cos_256
        accel_128 = prev_cos_128 - cos_128
        
        direction = ""
        if accel_256 > 0.005 or accel_128 > 0.01:
            direction = "ACCELERATING"
        elif cos_256 < 0.99 or cos_128 < 0.98:
            direction = "diverging"
        
        print(f"  {i:<6} {cos_256:<12.6f} {cos_128:<12.6f} {mse_256:<12.6f} {mse_128:<12.6f} {direction}")
        prev_cos_256 = cos_256
        prev_cos_128 = cos_128

# ============================================================
# 3. Output Logit Distribution Comparison
# ============================================================
print("\n" + "="*80)
print("3. NEXT-TOKEN PROBABILITY DISTRIBUTION")
print("="*80)

def get_logits(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0, -1, :]  # last token logits

logits_orig = get_logits(model, inputs)
logits_256 = get_logits(model_256, inputs)
logits_128 = get_logits(model_128, inputs)

probs_orig = F.softmax(logits_orig, dim=-1)
probs_256 = F.softmax(logits_256, dim=-1)
probs_128 = F.softmax(logits_128, dim=-1)

# Top predictions
print(f"\nPrompt: '{prompt}'")
print(f"\nTop 10 predictions for next token:\n")

for name, probs in [("ORIGINAL", probs_orig), ("RANK=256", probs_256), ("RANK=128", probs_128)]:
    topk = torch.topk(probs, 10)
    tokens = [tokenizer.decode([idx]) for idx in topk.indices]
    print(f"  {name}:")
    for tok, prob in zip(tokens, topk.values):
        print(f"    '{tok}' ({prob:.4f})")
    print()

# KL divergence between distributions
kl_256 = F.kl_div(probs_256.log(), probs_orig, reduction='sum').item()
kl_128 = F.kl_div(probs_128.log(), probs_orig, reduction='sum').item()
print(f"KL divergence from original:")
print(f"  Rank=256: {kl_256:.4f}")
print(f"  Rank=128: {kl_128:.4f}")

# ============================================================
# 4. Which layer causes most output drift?
# ============================================================
print("\n" + "="*80)
print("4. PER-LAYER OUTPUT CONTRIBUTION TO DRIFT")
print("="*80)

# Ablation: compress one layer at a time, measure output change
print("\nCompressing one layer at a time (rank=128), measuring KL divergence:")
layer_kl = []

for target_layer in range(12):
    m = copy.deepcopy(model)
    W = m.transformer.h[target_layer].attn.c_proj.weight.data.cpu().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(128, min(U.shape[1], Vt.shape[0]))
    m.transformer.h[target_layer].attn.c_proj.weight.data = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
    
    logits_ablated = get_logits(m, inputs)
    probs_ablated = F.softmax(logits_ablated, dim=-1)
    kl = F.kl_div(probs_ablated.log(), probs_orig, reduction='sum').item()
    layer_kl.append(kl)
    
    print(f"  Layer {target_layer:2d}: KL={kl:.4f}")
    del m; torch.cuda.empty_cache()

# Find most critical layers
sorted_layers = sorted(enumerate(layer_kl), key=lambda x: x[1], reverse=True)
print(f"\nMost critical layers (highest KL when compressed):")
for layer, kl in sorted_layers[:3]:
    print(f"  Layer {layer}: KL={kl:.4f}")

# ============================================================
# 5. Weight spectrum comparison
# ============================================================
print("\n" + "="*80)
print("5. WEIGHT SPECTRUM: WHAT GETS LOST?")
print("="*80)

for i in [0, 5, 11]:
    W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    
    # What fraction of energy is in top-k components?
    total_energy = S.sum()
    cumulative = S.cumsum(dim=0) / total_energy
    
    r256_energy = cumulative[255].item() if len(cumulative) > 255 else 1.0
    r128_energy = cumulative[127].item() if len(cumulative) > 127 else 1.0
    
    print(f"  Layer {i:2d}: rank=256 retains {r256_energy*100:.1f}% energy, rank=128 retains {r128_energy*100:.1f}% energy")
    print(f"           Top 10 singular values: {S[:10].tolist()}")

# ============================================================
# 6. GELU nonlinearity impact
# ============================================================
print("\n" + "="*80)
print("6. GELU NONLINEARITY: HOW IT AMPLIFIES DRIFT")
print("="*80)

def get_pre_post_gelu(model, inputs):
    pre_gelu = {}
    post_gelu = {}
    hooks = []
    
    for i in range(12):
        mlp = model.transformer.h[i].mlp
        def hook_pre(module, input, layer_idx=i):
            pre_gelu[layer_idx] = input[0].detach().cpu().float()
        def hook_post(module, input, output, layer_idx=i):
            post_gelu[layer_idx] = output.detach().cpu().float()
        hooks.append(mlp.c_fc.register_forward_hook(hook_pre))
        hooks.append(mlp.c_proj.register_forward_hook(hook_post))
    
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return pre_gelu, post_gelu

print("Comparing MLP outputs (post-GELU)...")
pre_o, post_o = get_pre_post_gelu(model, inputs)
pre_c, post_c = get_pre_post_gelu(model_256, inputs)

for i in range(12):
    if i in post_o and i in post_c:
        cos = F.cosine_similarity(post_o[i].reshape(1,-1), post_c[i].reshape(1,-1)).item()
        # Check how many neurons are "dead" (near zero) in compressed
        dead_orig = (post_o[i].abs() < 0.01).float().mean().item()
        dead_comp = (post_c[i].abs() < 0.01).float().mean().item()
        print(f"  Layer {i:2d}: cos={cos:.4f}, dead neurons orig={dead_orig*100:.1f}% comp={dead_comp*100:.1f}%")

# Save summary
output = {
    "prompt": prompt,
    "layer_ablation_kl": layer_kl,
    "critical_layers": [l for l, _ in sorted_layers[:3]],
    "kl_divergence": {"rank_256": kl_256, "rank_128": kl_128},
}

with open("results/output_drift_analysis.json", 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results/output_drift_analysis.json")

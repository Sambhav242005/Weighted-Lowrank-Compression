"""
Phase D: Test Gemma 3 1B
========================
Different architecture (not GPT-2), larger model (1B params).
Tests: effective ranks, per-layer sensitivity, compression ratios.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0)}")

# Load Gemma 3 1B
print("Loading Gemma 3 1B...")
model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)
model.eval()

# Model info
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {params:.0f}M params")

# Find transformer layers - Gemma 3 uses different structure
# Check model structure
print(f"\nModel structure:")
for name, module in model.named_modules():
    if "layers" in name and name.count(".") <= 3:
        print(f"  {name}: {type(module).__name__}")
    if name.count(".") > 3:
        break

# Get the layer structure
# Gemma 3 typically has: model.layers[i].self_attn, model.layers[i].mlp
n_layers = None
for name, _ in model.named_modules():
    if "layers." in name and ".self_attn" in name:
        parts = name.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        if n_layers is None or layer_idx + 1 > n_layers:
            n_layers = layer_idx + 1

if n_layers is None:
    # Try to infer from config
    n_layers = model.config.num_hidden_layers
    
print(f"Number of layers: {n_layers}")

# Find the projection weights
# Gemma 3 uses: self_attn.o_proj (output projection)
# and mlp.up_proj, mlp.down_proj
sample_weight = None
for name, param in model.named_parameters():
    if "o_proj" in name and "layers.0" in name:
        sample_weight = param.data
        print(f"Found projection: {name}, shape={param.shape}")
        break

if sample_weight is None:
    print("ERROR: Could not find projection weights")
    # List all parameters to understand structure
    print("\nAll parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    sys.exit(1)

hidden = sample_weight.shape[0]
print(f"Hidden dimension: {hidden}")

# Dataset
print("\nLoading dataset...")
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

def svd_compress_projection(model, rank, proj_name="o_proj"):
    """Compress only the output projection of attention"""
    m = copy.deepcopy(model)
    for i in range(n_layers):
        # Find the projection weight
        for name, param in m.named_parameters():
            if f"layers.{i}.self_attn.{proj_name}" in name:
                W = param.data.cpu().float().numpy()
                W_tensor = torch.from_numpy(W)
                U, S, Vt = torch.linalg.svd(W_tensor, full_matrices=False)
                k = min(rank, min(U.shape[1], Vt.shape[0]))
                W_compressed = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE)
                param.data = W_compressed.half()
                break
    return m

# Baseline
print("\nComputing baseline perplexity...")
baseline_ppl = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}")

# Effective ranks
print("\nEffective ranks (o_proj):")
eff_ranks = []
for i in range(n_layers):
    for name, param in model.named_parameters():
        if f"layers.{i}.self_attn.o_proj" in name:
            W = param.data.cpu().float()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            s_norm = S / S.sum()
            entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
            er = torch.exp(entropy).item()
            eff_ranks.append(er)
            print(f"  Layer {i:2d}: {er:.1f}")
            break

print(f"  Mean: {np.mean(eff_ranks):.1f}")

# Test at 2x, 3x, 4x, 6x
print(f"\n{'='*70}")
print("COMPRESSION TESTS")
print(f"{'='*70}")

results = []
for ratio in [2, 3, 4, 6]:
    rank = hidden // ratio
    print(f"\nTesting {ratio}x compression (rank={rank})...")
    m = svd_compress_projection(model, rank)
    ppl = compute_perplexity(m, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results.append({"ratio": ratio, "rank": rank, "ppl": ppl, "delta": delta})
    print(f"  PPL={ppl:.2f} ({delta:+.2f}%)")
    del m; torch.cuda.empty_cache()

# Compare with GPT-2
print(f"\n{'='*70}")
print("COMPARISON: Gemma 3 1B vs GPT-2 Models")
print(f"{'='*70}")

gpt2_small = {2: 5.50, 3: 21.36, 4: 152.12, 6: 261.81}
gpt2_medium = {2: 2.63, 3: 9.91, 4: 20.71, 6: 48.60}
gpt2_large = {2: 2.10, 3: 7.45, 4: 14.40, 6: 33.71}

print(f"\n{'Ratio':<8} {'GPT-2 Small':<15} {'GPT-2 Medium':<15} {'GPT-2 Large':<15} {'Gemma 3 1B':<15}")
print("-"*65)

for r in results:
    s = gpt2_small[r["ratio"]]
    m = gpt2_medium[r["ratio"]]
    l = gpt2_large[r["ratio"]]
    g = r["delta"]
    print(f"{r['ratio']:<8} {s:<+15.2f} {m:<+15.2f} {l:<+15.2f} {g:<+15.2f}")

# Per-layer sensitivity test
print(f"\n{'='*70}")
print("PER-LAYER SENSITIVITY (rank=128)")
print(f"{'='*70}")

layer_ppls = []
for i in range(n_layers):
    m = copy.deepcopy(model)
    for name, param in m.named_parameters():
        if f"layers.{i}.self_attn.o_proj" in name:
            W = param.data.cpu().float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            k = min(128, min(U.shape[1], Vt.shape[0]))
            param.data = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE).half()
            break
    ppl = compute_perplexity(m, tokenizer, eval_texts, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    layer_ppls.append({"layer": i, "ppl": ppl, "delta": delta, "eff_rank": eff_ranks[i]})
    print(f"  Layer {i:2d}: PPL={ppl:.2f} ({delta:+.2f}%), eff_rank={eff_ranks[i]:.1f}")
    del m; torch.cuda.empty_cache()

# Save results
output = {
    "model": "gemma-3-1b-it",
    "params": params,
    "n_layers": n_layers,
    "hidden": hidden,
    "baseline_ppl": baseline_ppl,
    "eff_ranks": eff_ranks,
    "compression_results": results,
    "per_layer_sensitivity": layer_ppls,
}

output_path = Path("results/phase_d_gemma3_1b.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {output_path}")

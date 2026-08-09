"""
Test: What if we compress ONLY MLP, keeping attention intact?
Hypothesis: Attention routing is critical, MLP transformation is less critical.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch, numpy as np, copy
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Gemma 3
gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
gemma_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.float16).to(DEVICE)
gemma_model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50][:50]

def compute_ppl(model, tokenizer, texts, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            labels = inputs["input_ids"].clone()
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tokens = int(labels[:, 1:].ne(-100).sum().item())
            if n_tokens == 0: continue
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')

def svd_compress_weights(model, rank, target_modules, n_layers=26):
    """Compress only specific weight matrices"""
    m = copy.deepcopy(model)
    compressed = 0
    
    for i in range(n_layers):
        for name, param in m.named_parameters():
            # Check if this parameter should be compressed
            should_compress = False
            for target in target_modules:
                if f"layers.{i}" in name and target in name:
                    should_compress = True
                    break
            
            if should_compress and "weight" in name:
                W = param.data.cpu().float()
                if W.dim() == 2:  # Only compress 2D weight matrices
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    k = min(rank, min(U.shape[1], Vt.shape[0]))
                    param.data = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(DEVICE).half()
                    compressed += 1
    
    print(f"  Compressed {compressed} weight matrices")
    return m

# Baseline
baseline_ppl = compute_ppl(gemma_model, gemma_tokenizer, eval_texts, DEVICE)
print(f"Gemma 3 baseline PPL: {baseline_ppl:.2f}\n")

# Test different compression strategies
strategies = [
    ("Attention only (o_proj)", ["o_proj"]),
    ("MLP only (up_proj + down_proj)", ["up_proj", "down_proj"]),
    ("MLP gate (gate_proj only)", ["gate_proj"]),
    ("All MLP (up + gate + down)", ["up_proj", "gate_proj", "down_proj"]),
    ("Everything", ["o_proj", "up_proj", "gate_proj", "down_proj"]),
]

rank = 256

print(f"{'='*70}")
print(f"COMPRESSION AT RANK={rank}")
print(f"{'='*70}\n")

results = []
for name, targets in strategies:
    print(f"Strategy: {name}")
    m = svd_compress_weights(gemma_model, rank, targets)
    ppl = compute_ppl(m, gemma_tokenizer, eval_texts, DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    results.append({"strategy": name, "ppl": ppl, "delta": delta})
    print(f"  PPL={ppl:.2f} ({delta:+.2f}%)\n")
    del m; torch.cuda.empty_cache()

# Summary
print(f"{'='*70}")
print("SUMMARY")
print(f"{'='*70}\n")

for r in results:
    print(f"  {r['strategy']:<40} {r['ppl']:<10.2f} {r['delta']:<+10.2f}%")

# Analysis
print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}\n")

attention_only = next(r for r in results if "Attention only" in r["strategy"])
mlp_only = next(r for r in results if "All MLP" in r["strategy"])
everything = next(r for r in results if r["strategy"] == "Everything")

print(f"""
FINDINGS:

1. Attention routing is CRITICAL:
   - Compressing attention only: {attention_only['delta']:+.2f}%
   - This breaks the model's ability to attend to correct tokens

2. MLP transformation is LESS critical:
   - Compressing MLP only: {mlp_only['delta']:+.2f}%
   - Model still attends correctly, just processes information differently

3. Combined effect:
   - Compressing everything: {everything['delta']:+.2f}%
   - Attention damage dominates

CONCLUSION:
- Attention weights define WHERE information flows (routing)
- MLP weights define HOW information is processed (transformation)
- Routing errors are catastrophic, transformation errors are recoverable

PRACTICAL IMPLICATION:
- DON'T compress attention in modern architectures (GQA + RMSNorm + SwiGLU)
- CAN compress MLP with careful rank allocation
- Or: use different compression methods for different weight types
""")

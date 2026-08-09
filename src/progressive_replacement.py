"""
Progressive Layer Replacement Test
===================================
Replace 1 layer at a time and generate text to see exactly where it breaks.
"""

import sys, torch, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from baseline_eval import compute_perplexity

# ============================================================
# Load model
# ============================================================
print("=" * 70)
print("Loading GPT-2 Small...")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
model.to(DEVICE)

# ============================================================
# Collect activations
# ============================================================
print("Collecting activations...")
eval_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "The temperature today is expected to reach 75 degrees.",
]

activations = {}
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        activations[name] = x.detach()
    return hook_fn

for i in range(12):
    block = model.transformer.h[i]
    hook = block.attn.c_proj.register_forward_hook(make_hook(f"layer{i}.attn.W_O"))
    hooks.append(hook)

with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding="max_length").to(DEVICE)
        _ = model(**inputs)

for h in hooks:
    h.remove()

for key in list(activations.keys()):
    if isinstance(activations[key], list):
        activations[key] = torch.cat(activations[key], dim=0)

print(f"Collected activations for {len(activations)} layers")

# ============================================================
# SVD function
# ============================================================
def fit_svd(W, variance_threshold=0.99):
    W = W.float()
    m, n = W.shape
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    total_var = (S ** 2).sum()
    cumvar = torch.cumsum(S ** 2, dim=0) / total_var
    rank = int(torch.searchsorted(cumvar, variance_threshold)) + 1
    rank = min(rank, min(m, n))
    W_approx = (U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :])
    return W_approx

# ============================================================
# Test prompts
# ============================================================
prompts = [
    "Hello",
    "What is 2+2?",
    "The meaning of life is",
    "Once upon a time",
    "In machine learning",
]

# ============================================================
# Progressive replacement
# ============================================================
print("\n" + "=" * 70)
print("PROGRESSIVE LAYER REPLACEMENT")
print("=" * 70)

# Get baseline perplexity
test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the meaning of life?",
    "Python is a programming language.",
]

baseline_ppl = compute_perplexity(model, tokenizer, test_sentences, max_length=256, device=DEVICE)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

# Test each number of layers
for n_replace in range(0, 13):
    print(f"\n{'='*70}")
    print(f"REPLACING {n_replace}/12 LAYERS (SVD 99%)")
    print(f"{'='*70}")
    
    # Create compressed model
    model_comp = copy.deepcopy(model)
    
    for i in range(n_replace):
        W = model.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        W_approx = fit_svd(W, 0.99)
        model_comp.transformer.h[i].attn.c_proj.weight.data = W_approx.to(DEVICE)
    
    # Compute perplexity
    ppl = compute_perplexity(model_comp, tokenizer, test_sentences, max_length=256, device=DEVICE)
    delta = ((ppl - baseline_ppl) / baseline_ppl) * 100
    print(f"PPL: {baseline_ppl:.2f} -> {ppl:.2f} ({delta:+.2f}%)")
    
    # Generate text for each prompt
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model_comp.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {generated[:100]}...")
    
    # Clean up
    del model_comp
    torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("PROGRESSIVE REPLACEMENT SUMMARY")
print("=" * 70)
print(f"\nBaseline PPL: {baseline_ppl:.2f}")
print("\nSummary:")
print("- 0 layers: Original model")
print("- 1-3 layers: Should still generate coherent text")
print("- 4-6 layers: Start seeing repetition and templates")
print("- 7-9 layers: Severe degradation")
print("- 10-12 layers: Complete collapse")

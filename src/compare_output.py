"""
Show raw GPT-2 output before and after compression.
Compare: original vs SVD rank=256 (3x) vs SVD rank=128 (6x)
"""

import torch, copy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        W_compressed = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()
        m.transformer.h[i].attn.c_proj.weight.data = W_compressed.to(DEVICE)
    return m

def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Compress
print("Compressing...")
model_256 = svd_compress(model, 256)
model_128 = svd_compress(model, 128)

prompts = [
    "The future of artificial intelligence is",
    "In a distant galaxy, scientists discovered",
    "The quick brown fox",
    "The meaning of life is",
]

print("\n" + "="*80)
for prompt in prompts:
    print(f"\nPROMPT: {prompt}")
    print("-"*80)
    
    out_orig = generate(model, tokenizer, prompt)
    out_256 = generate(model_256, tokenizer, prompt)
    out_128 = generate(model_128, tokenizer, prompt)
    
    print(f"ORIGINAL:      {out_orig}")
    print(f"RANK=256 (3x): {out_256}")
    print(f"RANK=128 (6x): {out_128}")
    print()

# Interactive mode
print("\n" + "="*80)
print("INTERACTIVE MODE (type 'quit' to exit)")
print("="*80)

while True:
    prompt = input("\nPrompt> ").strip()
    if prompt.lower() in ('quit', 'exit', 'q'):
        break
    if not prompt:
        continue
    
    print(f"\nORIGINAL:      {generate(model, tokenizer, prompt)}")
    print(f"RANK=256 (3x): {generate(model_256, tokenizer, prompt)}")
    print(f"RANK=128 (6x): {generate(model_128, tokenizer, prompt)}")

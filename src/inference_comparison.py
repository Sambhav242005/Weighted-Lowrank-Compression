"""
Inference Comparison
====================
Run actual text generation on real prompts with original vs compressed models.
"""

import sys, torch
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "CPU"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# Load models
# ============================================================
print("=" * 70)
print("Loading models...")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Original model
model_original = GPT2LMHeadModel.from_pretrained("gpt2")
model_original.eval()
model_original.to(DEVICE)

# Load compressed model if it exists
compressed_path = Path("compressed_model_aggressive/compressed")
if compressed_path.exists():
    model_compressed = GPT2LMHeadModel.from_pretrained(str(compressed_path))
    model_compressed.eval()
    model_compressed.to(DEVICE)
    has_compressed = True
    print("Loaded compressed model")
else:
    has_compressed = False
    print("No compressed model found")

print("Loaded original model")

# ============================================================
# Test prompts
# ============================================================
prompts = [
    # Simple
    "Hello",
    "What is 2+2?",
    "The capital of France is",
    
    # Medium
    "Once upon a time",
    "The theory of relativity states that",
    "In machine learning, gradient descent is",
    
    # Complex
    "Write a short story about a robot learning to feel emotions.",
    "Explain the difference between supervised and unsupervised learning.",
    "The impact of artificial society can be",
]

# ============================================================
# Generate text
# ============================================================
print("\n" + "=" * 70)
print("TEXT GENERATION COMPARISON")
print("=" * 70)

for prompt in prompts:
    print(f"\n{'='*70}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*70}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Original model
    with torch.no_grad():
        outputs = model_original.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[ORIGINAL GPT-2]")
    print(generated)
    
    # Compressed model
    if has_compressed:
        with torch.no_grad():
            outputs = model_compressed.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[COMPRESSED GPT-2]")
        print(generated)

# ============================================================
# Perplexity comparison
# ============================================================
print("\n" + "=" * 70)
print("PERPLEXITY COMPARISON")
print("=" * 70)

test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "What is the meaning of life?",
    "Python is a programming language used for artificial intelligence.",
    "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
    "Machine learning algorithms require large amounts of data to train effectively.",
]

def compute_perplexity(model, tokenizer, texts):
    total_loss = 0
    n_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            n_tokens += inputs["input_ids"].shape[1]
    return torch.exp(torch.tensor(total_loss / n_tokens)).item()

ppl_original = compute_perplexity(model_original, tokenizer, test_sentences)
print(f"Original GPT-2 PPL: {ppl_original:.2f}")

if has_compressed:
    ppl_compressed = compute_perplexity(model_compressed, tokenizer, test_sentences)
    print(f"Compressed GPT-2 PPL: {ppl_compressed:.2f}")
    print(f"Delta: {((ppl_compressed - ppl_original) / ppl_original) * 100:+.2f}%")

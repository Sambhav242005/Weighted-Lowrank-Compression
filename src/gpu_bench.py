"""Tight GPU test - verify GPU is used during inference."""
import torch, sys, time
sys.path.insert(0, '.')

DEVICE = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load model directly to GPU
print("Loading model to GPU...")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

print(f"Model on: {next(model.parameters()).device}")

# Simple forward pass - should spike GPU
print("\nRunning 100 forward passes...")
text = "The quick brown fox jumps over the lazy dog. " * 10
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)

start = time.time()
with torch.no_grad():
    for i in range(100):
        out = model(**inputs)
elapsed = time.time() - start

print(f"100 forward passes: {elapsed:.2f}s ({100/elapsed:.1f} iter/s)")
print(f"Output device: {out.logits.device}")
print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1e6:.1f} MB")
print("\nCheck Task Manager -> GPU should show activity now.")

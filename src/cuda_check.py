"""CUDA diagnostic: check every step to see what's on GPU vs CPU."""

import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("STEP 1: CUDA Availability")
print("=" * 60)
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.get_device_capability(0): {torch.cuda.get_device_capability(0)}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cudnn version: {torch.backends.cudnn.version()}")
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")

print("\n" + "=" * 60)
print("STEP 2: Basic tensor operations on GPU")
print("=" * 60)
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = a @ b
    print(f"Matrix multiply device: {c.device}")
    print(f"Result on GPU: {c.is_cuda}")
    del a, b, c
    torch.cuda.empty_cache()
else:
    print("NO CUDA - cannot test")

print("\n" + "=" * 60)
print("STEP 3: Model loading and device check")
print("=" * 60)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded. Default device: next(model.parameters()).device")

model_gpu = model.to("cuda")
print(f"After .to('cuda'): {next(model_gpu.parameters()).device}")

# Check a few layers
for name, param in list(model_gpu.named_parameters())[:5]:
    print(f"  {name}: device={param.device}, dtype={param.dtype}")

print("\n" + "=" * 60)
print("STEP 4: Forward pass on GPU")
print("=" * 60)
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
print(f"Input device: {inputs['input_ids'].device}")

with torch.no_grad():
    outputs = model_gpu(**inputs)
print(f"Output logits device: {outputs.logits.device}")
print(f"Output logits shape: {outputs.logits.shape}")

print("\n" + "=" * 60)
print("STEP 5: Weight extraction device")
print("=" * 60)
from layer_extraction import extract_gpt2_layers
weights, _ = extract_gpt2_layers("gpt2")
layer_name = "layer0.attn.W_O"
w = weights[layer_name].tensor
print(f"Extracted weight device: {w.device}")
print(f"Extracted weight shape: {w.shape}")

print("\n" + "=" * 60)
print("STEP 6: SVD fitting device")
print("=" * 60)
from representations import fit_svd_at_threshold
svd_rep = fit_svd_at_threshold(w, variance_threshold=0.99)
recon = svd_rep.reconstruct()
print(f"SVD reconstruction device: {recon.device}")
print(f"SVD reconstruction shape: {recon.shape}")

print("\n" + "=" * 60)
print("STEP 7: Substitution - weight swap device")
print("=" * 60)
import copy
model_copy = copy.deepcopy(model_gpu)
block = model_copy.transformer.h[0]
print(f"Before swap - c_proj.weight device: {block.attn.c_proj.weight.device}")

# Move reconstructed to GPU
recon_gpu = recon.to("cuda")
print(f"Reconstructed weight device: {recon_gpu.device}")

block.attn.c_proj.weight.data = recon_gpu
print(f"After swap - c_proj.weight device: {block.attn.c_proj.weight.device}")

# Forward pass with substituted weight
with torch.no_grad():
    out = model_copy(**inputs)
print(f"Substituted model output device: {out.logits.device}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If any step shows CPU when it should be GPU, that's the problem.")

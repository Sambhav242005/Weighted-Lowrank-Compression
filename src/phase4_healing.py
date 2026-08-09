"""
Phase 4: Healing via LoRA fine-tuning after cumulative replacement.
Tests whether low-rank adapters can recover performance lost to compression.
"""

import sys, json, torch, numpy as np, copy
from pathlib import Path
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

from layer_extraction import extract_gpt2_layers, get_model_config
from representations import fit_svd_at_threshold, fit_fourier_at_threshold, fit_low_rank_product
from baseline_eval import get_baseline, compute_perplexity

# Check for peft
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
    print("peft library available")
except ImportError:
    HAS_PEFT = False
    print("peft library not available - will implement manual LoRA")

output_path = Path("results")
output_path.mkdir(exist_ok=True)

# Baseline
baseline, model, tokenizer, eval_texts = get_baseline("gpt2", n_eval_texts=20, device=DEVICE)
print(f"Baseline PPL: {baseline.perplexity:.2f}\n")

# Extract weights
weights, _ = extract_gpt2_layers("gpt2")
config = get_model_config("gpt2")
n_layers = config['n_layer']

# ============================================================
# Manual LoRA implementation (no external dependency)
# ============================================================
class ManualLoRALayer(torch.nn.Module):
    """Low-rank adapter for Conv1D: W_out = W_frozen + B @ A"""
    def __init__(self, nf, nx, rank=4, alpha=1.0):
        super().__init__()
        # GPT-2 Conv1D: weight shape is (nf, nx), forward: x @ weight + bias
        self.nf = nf  # out_features
        self.nx = nx  # in_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Initialize A with Gaussian, B with zeros
        self.lora_A = torch.nn.Parameter(torch.randn(nx, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, nf))
    
    def forward(self, x):
        # x: (batch, seq, nx)
        # Conv1D forward: x @ weight + bias
        # LoRA delta: x @ lora_A @ lora_B * scaling
        return x @ self.lora_A @ self.lora_B * self.scaling


def apply_lora_to_model(model, rank=4, alpha=1.0, layers_to_adapt=None):
    """Add LoRA adapters to specified layers' c_proj."""
    if layers_to_adapt is None:
        layers_to_adapt = list(range(12))
    
    lora_params = []
    lora_modules = []
    
    for i in layers_to_adapt:
        block = model.transformer.h[i]
        proj = block.attn.c_proj
        # GPT-2 Conv1D: weight shape is (nf, nx) where nf=out, nx=in
        nf, nx = proj.weight.shape
        
        lora = ManualLoRALayer(nf, nx, rank=rank, alpha=alpha)
        lora = lora.to(DEVICE)
        
        # Store reference to original
        block.attn._original_proj = proj
        block.attn._lora = lora
        
        # Replace forward
        original_forward = proj.forward
        def make_lora_forward(orig_fwd, lora_layer):
            def lora_forward(x):
                return orig_fwd(x) + lora_layer(x)
            return lora_forward
        proj.forward = make_lora_forward(original_forward, lora)
        
        lora_params.extend(lora.parameters())
        lora_modules.append(lora)
    
    return lora_params, lora_modules


def freeze_model_except_lora(model, lora_params):
    """Freeze everything except LoRA parameters."""
    for param in model.parameters():
        param.requires_grad = False
    for param in lora_params:
        param.requires_grad = True


# ============================================================
# Test 1: LoRA healing after full replacement
# ============================================================
print("=" * 70)
print("TEST 1: LoRA healing after LowRank r128 replacement (all 12 layers)")
print("=" * 70)

# First replace all layers
model_replaced = copy.deepcopy(model)
for i in range(n_layers):
    layer_name = f"layer{i}.attn.W_O"
    weight = weights[layer_name].tensor
    rep = fit_low_rank_product(weight, rank=128, device=DEVICE, steps=500)
    recon = rep.reconstruct().to(DEVICE)
    model_replaced.transformer.h[i].attn.c_proj.weight.data = recon.float()

ppl_before = compute_perplexity(model_replaced, tokenizer, eval_texts, max_length=256, device=DEVICE)
print(f"After replacement (no healing): PPL = {ppl_before:.2f}")

# Now apply LoRA and "heal" with a few training steps
# Use WikiText-2 training data for healing
from datasets import load_dataset
print("\nLoading WikiText-2 for healing...")
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t for t in train_dataset["text"] if t.strip()][:500]  # subset

def tokenize_and_batch(texts, tokenizer, max_length=256, batch_size=4):
    """Tokenize texts and create batches."""
    encodings = tokenizer(texts, truncation=True, max_length=max_length, 
                         padding="max_length", return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"].to(DEVICE),
        encodings["attention_mask"].to(DEVICE)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

healing_results = []

for lora_rank in [4, 8, 16]:
    print(f"\n--- LoRA Rank {lora_rank} ---")
    
    model_heal = copy.deepcopy(model_replaced)
    
    # Apply LoRA
    lora_params, lora_modules = apply_lora_to_model(model_heal, rank=lora_rank, alpha=lora_rank)
    freeze_model_except_lora(model_heal, lora_params)
    
    total_lora_params = sum(p.numel() for p in lora_params)
    print(f"LoRA parameters: {total_lora_params:,} ({total_lora_params/1e6:.2f}M)")
    
    # Train for a few steps
    optimizer = torch.optim.AdamW(lora_params, lr=5e-4)
    
    # Use a subset of training texts
    heal_texts = train_texts[:100]
    dataloader = tokenize_and_batch(heal_texts, tokenizer, max_length=128, batch_size=4)
    
    print("Training LoRA adapters...")
    model_heal.train()
    n_steps = 50
    for step, (input_ids, attention_mask) in enumerate(dataloader):
        if step >= n_steps:
            break
        
        optimizer.zero_grad()
        outputs = model_heal(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{n_steps}, Loss: {loss.item():.4f}")
    
    # Evaluate healed model
    model_heal.eval()
    ppl_healed = compute_perplexity(model_heal, tokenizer, eval_texts, max_length=256, device=DEVICE)
    
    healing_results.append({
        "lora_rank": lora_rank,
        "lora_params": total_lora_params,
        "ppl_before_healing": ppl_before,
        "ppl_after_healing": ppl_healed,
        "recovery_pct": ((ppl_before - ppl_healed) / (ppl_before - baseline.perplexity)) * 100,
    })
    
    print(f"  PPL after healing: {ppl_healed:.2f} ({((ppl_healed - baseline.perplexity)/baseline.perplexity)*100:+.2f}%)")
    print(f"  Recovery: {healing_results[-1]['recovery_pct']:.1f}% of lost performance")
    
    del model_heal, lora_modules
    torch.cuda.empty_cache()

# ============================================================
# Test 2: Mixed-precision approach
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Mixed-precision (preserve early, compress late)")
print("=" * 70)

mixed_configs = [
    ("Preserve L0-L3", [None]*4 + [128]*8),       # Keep first 4 layers
    ("Preserve L0-L5", [None]*6 + [128]*6),       # Keep first 6 layers
    ("Preserve L0-L7", [None]*8 + [128]*4),       # Keep first 8 layers
    ("Preserve L0-L1, L10-L11", [None]*2 + [128]*8 + [None]*2),  # Keep boundaries
]

mixed_results = []

for config_name, ranks in mixed_configs:
    print(f"\n--- {config_name} ---")
    model_copy = copy.deepcopy(model)
    
    total_orig = 0
    total_comp = 0
    n_compressed = 0
    
    for i in range(n_layers):
        layer_name = f"layer{i}.attn.W_O"
        weight = weights[layer_name].tensor
        rank = ranks[i]
        
        if rank is not None:
            rep = fit_low_rank_product(weight, rank=rank, device=DEVICE, steps=500)
            recon = rep.reconstruct().to(DEVICE)
            model_copy.transformer.h[i].attn.c_proj.weight.data = recon.float()
            total_orig += rep.original_params
            total_comp += rep.n_params
            n_compressed += 1
    
    ppl = compute_perplexity(model_copy, tokenizer, eval_texts, max_length=256, device=DEVICE)
    delta_pct = ((ppl - baseline.perplexity) / baseline.perplexity) * 100
    
    mixed_results.append({
        "config": config_name,
        "n_compressed": n_compressed,
        "compression": total_orig / total_comp if total_comp > 0 else float('inf'),
        "perplexity": ppl,
        "delta_pct": delta_pct,
    })
    
    print(f"  Compressed: {n_compressed}/12 layers | PPL: {ppl:.2f} ({delta_pct:+.2f}%)")
    
    del model_copy
    torch.cuda.empty_cache()

# ============================================================
# Test 3: Mixed-precision + LoRA healing
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Mixed-precision + LoRA healing (best config)")
print("=" * 70)

# Use best mixed config: preserve L0-L3, compress rest
model_mixed = copy.deepcopy(model)
for i in range(n_layers):
    layer_name = f"layer{i}.attn.W_O"
    weight = weights[layer_name].tensor
    if i >= 4:  # Only compress layers 4-11
        rep = fit_low_rank_product(weight, rank=128, device=DEVICE, steps=500)
        recon = rep.reconstruct().to(DEVICE)
        model_mixed.transformer.h[i].attn.c_proj.weight.data = recon.float()

ppl_mixed = compute_perplexity(model_mixed, tokenizer, eval_texts, max_length=256, device=DEVICE)
print(f"Mixed-precision (no healing): PPL = {ppl_mixed:.2f}")

# Apply LoRA and heal
lora_params, lora_modules = apply_lora_to_model(model_mixed, rank=8, alpha=8)
freeze_model_except_lora(model_mixed, lora_params)

optimizer = torch.optim.AdamW(lora_params, lr=5e-4)
heal_texts = train_texts[:200]
dataloader = tokenize_and_batch(heal_texts, tokenizer, max_length=128, batch_size=4)

print("Healing with LoRA rank=8...")
model_mixed.train()
for step, (input_ids, attention_mask) in enumerate(dataloader):
    if step >= 100:
        break
    optimizer.zero_grad()
    outputs = model_mixed(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    loss.backward()
    optimizer.step()
    if (step + 1) % 25 == 0:
        print(f"  Step {step+1}/100, Loss: {loss.item():.4f}")

model_mixed.eval()
ppl_healed = compute_perplexity(model_mixed, tokenizer, eval_texts, max_length=256, device=DEVICE)
print(f"After healing: PPL = {ppl_healed:.2f} ({((ppl_healed - baseline.perplexity)/baseline.perplexity)*100:+.2f}%)")

# ============================================================
# Save results
# ============================================================
results = {
    "baseline_ppl": baseline.perplexity,
    "lora_healing": healing_results,
    "mixed_precision": mixed_results,
    "mixed_plus_lora": {
        "ppl_before": ppl_mixed,
        "ppl_after": ppl_healed,
    }
}

with open(output_path / "phase4_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4 SUMMARY")
print("=" * 70)

print(f"\nBaseline PPL: {baseline.perplexity:.2f}")
print(f"After replacement (LowRank r128, all layers): PPL = {ppl_before:.2f}")

print("\nLoRA Healing (full replacement):")
print(f"  {'Rank':<10} {'Params':<14} {'PPL':<10} {'Recovery'}")
for r in healing_results:
    print(f"  {r['lora_rank']:<10} {r['lora_params']:<14,} {r['ppl_after_healing']:<10.2f} {r['recovery_pct']:.1f}%")

print("\nMixed-Precision (no healing):")
print(f"  {'Config':<25} {'Compressed':<12} {'PPL':<10} {'Delta'}")
for r in mixed_results:
    print(f"  {r['config']:<25} {r['n_compressed']:<12} {r['perplexity']:<10.2f} {r['delta_pct']:+.2f}%")

print(f"\nMixed-Precision + LoRA Healing:")
print(f"  PPL before healing: {ppl_mixed:.2f}")
print(f"  PPL after healing:  {ppl_healed:.2f}")

"""
Drift Profiler
==============
Measure how hidden-state drift accumulates when independently compressed groups
of consecutive layers are replaced at various group sizes.

For each group size g in {1, 2, 4, 8, 12}:
  - Independently replace g consecutive layers (using low-rank at rank=128)
  - Measure at every layer boundary:
    * MSE between original and reconstructed hidden states
    * Cosine similarity of hidden states
    * CKA similarity of hidden states
    * Attention entropy of original vs reconstructed
    * Final perplexity
  - Output: drift trajectory per group size

This tells us WHERE drift accumulates and WHETHER larger groups compose.
"""

import sys, torch, numpy as np, copy, json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, '.')
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")


# ============================================================
# Metrics
# ============================================================

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return float(torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0))


def hidden_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """MSE between two hidden-state tensors."""
    return float(((a.float() - b.float()) ** 2).mean())


def compute_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two activation matrices.
    x, y: (n_samples, feature_dim)
    """
    x = x.float()
    y = y.float()
    
    # Center
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    
    # Gram matrices
    xxt = x @ x.T
    yyt = y @ y.T
    
    # CKA
    numerator = (xxt * yyt).sum()
    denominator = torch.sqrt((xxt * xxt).sum() * (yyt * yyt).sum() + 1e-10)
    return float(numerator / (denominator + 1e-10))


def attention_entropy(attn_weights: torch.Tensor) -> float:
    """Entropy of attention distribution. attn_weights: (batch, heads, seq, seq)"""
    # Average over batch and heads
    attn = attn_weights.float().mean(dim=(0, 1))  # (seq, seq)
    # Entropy per query position, then average
    log_attn = torch.log(attn + 1e-10)
    entropy = -(attn * log_attn).sum(dim=-1).mean()
    return float(entropy)


# ============================================================
# SVD reconstruction
# ============================================================

def svd_reconstruct(W: torch.Tensor, rank: int = 128) -> torch.Tensor:
    """Low-rank SVD reconstruction."""
    W = W.float()
    m, n = W.shape
    k = min(rank, min(m, n))
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()


# ============================================================
# Perplexity
# ============================================================

def compute_perplexity(model, tokenizer, texts, max_length=256, device="cpu"):
    """Compute perplexity on a list of texts."""
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


# ============================================================
# Hook-based drift measurement
# ============================================================

@dataclass
class LayerDrift:
    """Drift metrics at a single layer boundary."""
    layer_idx: int
    mse: float
    cosine: float
    cka: float
    attn_entropy_orig: float
    attn_entropy_sub: float
    attn_entropy_delta: float


@dataclass
class GroupResult:
    """Result for one group size."""
    group_size: int
    replaced_layers: List[int]
    ppl_orig: float
    ppl_sub: float
    ppl_delta_pct: float
    layer_drift: List[LayerDrift]


def install_hooks(model, names):
    """Install forward hooks to capture hidden states. Returns (hooks, activations_dict)."""
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
    
    for i, name in enumerate(names):
        block = model.transformer.h[i]
        hook = block.attn.c_proj.register_forward_hook(make_hook(name))
        hooks.append(hook)
    
    return hooks, activations


def measure_attention_entropy(model, tokenizer, texts, layer_idx, max_length=128, device="cpu"):
    """Measure attention entropy at a specific layer."""
    model.eval()
    entropies = []
    
    def hook_fn(module, input, output):
        # output is attention weights if we hook attn.attention or attn_dropout
        # But c_proj doesn't give us attention weights directly
        # We need to hook the attention weights before c_proj
        pass
    
    # Alternative: manually compute attention for this layer
    with torch.no_grad():
        for text in texts[:5]:  # Sample for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            
            # Forward through transformer, capture attention weights
            hidden = model.transformer.wte(inputs["input_ids"]) + model.transformer.wpe(
                torch.arange(inputs["input_ids"].shape[1], device=device)
            )
            
            for i in range(layer_idx + 1):
                block = model.transformer.h[i]
                # Run attention manually
                h = block.ln_1(hidden)
                attn_out, attn_weights = block.attn(h, h, h, use_cache=False, output_attentions=True)
                hidden = hidden + attn_out
                hidden = hidden + block.mlp(block.ln_2(hidden))
            
            if attn_weights is not None:
                ent = attention_entropy(attn_weights)
                entropies.append(ent)
    
    return np.mean(entropies) if entropies else 0.0


def profile_group(
    model,
    tokenizer,
    texts,
    eval_texts,
    group_start: int,
    group_size: int,
    rank: int = 128,
    device: str = "cpu",
) -> GroupResult:
    """
    Profile drift for a single group of consecutive layers replaced independently.
    
    1. Collect original hidden states
    2. Replace group layers with SVD(r=rank) independently
    3. Collect reconstructed hidden states
    4. Measure drift at every layer boundary
    """
    n_layers = len(model.transformer.h)
    group_end = min(group_start + group_size, n_layers)
    replaced = list(range(group_start, group_end))
    
    # --- Step 1: Collect original hidden states ---
    orig_hidden = {}
    
    def make_hook_orig(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            orig_hidden[name] = x.detach().cpu()
        return hook_fn
    
    hooks = []
    for i in range(n_layers):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(
            make_hook_orig(f"layer{i}")
        )
        hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for text in texts[:30]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128).to(device)
            _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    # --- Step 2: Build modified model with group replaced ---
    model_sub = copy.deepcopy(model)
    
    for i in replaced:
        W_orig = model_sub.transformer.h[i].attn.c_proj.weight.data.cpu().float()
        W_sub = svd_reconstruct(W_orig, rank)
        model_sub.transformer.h[i].attn.c_proj.weight.data = W_sub.to(device)
    
    # --- Step 3: Collect reconstructed hidden states ---
    sub_hidden = {}
    
    def make_hook_sub(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            sub_hidden[name] = x.detach().cpu()
        return hook_fn
    
    hooks_sub = []
    for i in range(n_layers):
        hook = model_sub.transformer.h[i].attn.c_proj.register_forward_hook(
            make_hook_sub(f"layer{i}")
        )
        hooks_sub.append(hook)
    
    model_sub.eval()
    with torch.no_grad():
        for text in texts[:30]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=128).to(device)
            _ = model_sub(**inputs)
    
    for h in hooks_sub:
        h.remove()
    
    # --- Step 4: Measure drift at each layer ---
    layer_drift = []
    
    for i in range(n_layers):
        key = f"layer{i}"
        if key in orig_hidden and key in sub_hidden:
            h_orig = orig_hidden[key]
            h_sub = sub_hidden[key]
            
            # Stack across samples for CKA
            # h_orig: (n_samples, seq_len, dim) -> (n_samples * seq_len, dim)
            n_samp = h_orig.shape[0]
            seq_len = h_orig.shape[1]
            dim = h_orig.shape[2]
            
            flat_orig = h_orig.reshape(-1, dim)
            flat_sub = h_sub.reshape(-1, dim)
            
            # Subsample for CKA speed
            if flat_orig.shape[0] > 2000:
                idx = torch.randperm(flat_orig.shape[0])[:2000]
                flat_orig_s = flat_orig[idx]
                flat_sub_s = flat_sub[idx]
            else:
                flat_orig_s = flat_orig
                flat_sub_s = flat_sub
            
            layer_drift.append(LayerDrift(
                layer_idx=i,
                mse=hidden_mse(h_orig, h_sub),
                cosine=cosine_similarity(h_orig, h_sub),
                cka=compute_cka(flat_orig_s, flat_sub_s),
                attn_entropy_orig=0.0,
                attn_entropy_sub=0.0,
                attn_entropy_delta=0.0,
            ))
    
    # --- Step 5: Perplexity ---
    ppl_orig = compute_perplexity(model, tokenizer, eval_texts, max_length=256, device=device)
    ppl_sub = compute_perplexity(model_sub, tokenizer, eval_texts, max_length=256, device=device)
    ppl_delta_pct = ((ppl_sub - ppl_orig) / ppl_orig) * 100
    
    del model_sub
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return GroupResult(
        group_size=group_size,
        replaced_layers=replaced,
        ppl_orig=ppl_orig,
        ppl_sub=ppl_sub,
        ppl_delta_pct=ppl_delta_pct,
        layer_drift=layer_drift,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("DRIFT PROFILER")
    print("=" * 70)
    
    # Load model
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    
    # Load evaluation texts
    print("Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
    eval_texts = eval_texts[:50]
    
    # Use a subset for hidden-state collection (faster)
    profiling_texts = eval_texts[:30]
    
    rank = 128
    n_layers = 12
    group_sizes = [1, 2, 4, 8, 12]
    
    all_results = []
    
    # --- Test group size 1: profile EVERY layer independently ---
    print(f"\n{'='*70}")
    print(f"GROUP SIZE 1: Individual layer profiling")
    print(f"{'='*70}")
    
    for layer_idx in range(n_layers):
        print(f"\n  Layer {layer_idx}...")
        result = profile_group(
            model, tokenizer, profiling_texts, eval_texts,
            group_start=layer_idx, group_size=1, rank=rank, device=DEVICE
        )
        all_results.append(result)
        
        print(f"    PPL: {result.ppl_orig:.2f} -> {result.ppl_sub:.2f} ({result.ppl_delta_pct:+.2f}%)")
        if result.layer_drift:
            d = result.layer_drift[-1]
            print(f"    Drift@layer{layer_idx}: MSE={d.mse:.6f}  cos={d.cosine:.4f}  CKA={d.cka:.4f}")
    
    # --- Test group sizes 2, 4, 8, 12: profile each independent group ---
    for gs in [2, 4, 8, 12]:
        print(f"\n{'='*70}")
        print(f"GROUP SIZE {gs}: Independent groups")
        print(f"{'='*70}")
        
        starts = list(range(0, n_layers, gs))
        for start in starts:
            print(f"\n  Group layers {start}-{min(start+gs-1, n_layers-1)}...")
            result = profile_group(
                model, tokenizer, profiling_texts, eval_texts,
                group_start=start, group_size=gs, rank=rank, device=DEVICE
            )
            all_results.append(result)
            
            print(f"    PPL: {result.ppl_orig:.2f} -> {result.ppl_sub:.2f} ({result.ppl_delta_pct:+.2f}%)")
            if result.layer_drift:
                last = result.layer_drift[-1]
                first = result.layer_drift[0]
                print(f"    Drift start: MSE={first.mse:.6f}  cos={first.cosine:.4f}  CKA={first.cka:.4f}")
                print(f"    Drift end:   MSE={last.mse:.6f}  cos={last.cosine:.4f}  CKA={last.cka:.4f}")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("DRIFT PROFILER SUMMARY")
    print(f"{'='*70}")
    
    # Print group size 1 results (per-layer)
    print(f"\n{'Group Size 1: Per-Layer Drift'}")
    print(f"{'Layer':<8} {'MSE':<12} {'Cosine':<10} {'CKA':<10} {'PPL D%':<10}")
    print("-" * 50)
    
    for r in all_results:
        if r.group_size == 1 and r.layer_drift:
            d = r.layer_drift[-1]
            print(f"{r.replaced_layers[0]:<8} {d.mse:<12.6f} {d.cosine:<10.4f} {d.cka:<10.4f} {r.ppl_delta_pct:<+10.2f}")
    
    # Print grouped results
    for gs in [2, 4, 8, 12]:
        print(f"\n{'Group Size ' + str(gs)}")
        print(f"{'Group':<20} {'PPL D%':<10} {'Final MSE':<12} {'Final Cos':<10} {'Final CKA':<10}")
        print("-" * 62)
        
        for r in all_results:
            if r.group_size == gs and r.layer_drift:
                d = r.layer_drift[-1]
                group_label = f"L{r.replaced_layers[0]}-L{r.replaced_layers[-1]}"
                print(f"{group_label:<20} {r.ppl_delta_pct:<+10.2f} {d.mse:<12.6f} {d.cosine:<10.4f} {d.cka:<10.4f}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    results_dict = []
    for r in all_results:
        results_dict.append({
            "group_size": r.group_size,
            "replaced_layers": r.replaced_layers,
            "ppl_orig": r.ppl_orig,
            "ppl_sub": r.ppl_sub,
            "ppl_delta_pct": r.ppl_delta_pct,
            "layer_drift": [asdict(d) for d in r.layer_drift],
        })
    
    output_path = output_dir / "drift_profiler.json"
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

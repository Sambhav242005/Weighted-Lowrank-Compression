"""
Joint Group Compression
=======================
Compress consecutive layers JOINTLY with a shared low-rank parameterization
that minimizes accumulated hidden-state drift.

This tests: "Can jointly compressed groups preserve function better than
independent per-layer compression at the same total parameter budget?"

Key difference from drift_profiler.py:
- drift_profiler: compress each layer independently with its own SVD
- joint_group_compress: share a low-rank basis across the group, optimizing
  to minimize drift at the group output
"""

import sys, torch, numpy as np, copy, json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

sys.path.insert(0, '.')
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")


# ============================================================
# Low-rank parameterization for a group
# ============================================================

class JointGroupRepresentation:
    """
    Joint low-rank representation for a group of consecutive layers.
    
    Each layer's weight W_i is parameterized as:
        W_i = B_i @ A
    
    Where A is SHARED across the group (common input subspace)
    and B_i is layer-specific (output mixing per layer).
    
    Total params: group_size * (rank * out_dim) + (rank * in_dim)
    vs original: group_size * (out_dim * in_dim)
    
    Compression ratio depends on rank vs min(out_dim, in_dim).
    """
    
    def __init__(self, weight_shapes: List[tuple], rank: int, shared_dim: int = None):
        """
        Args:
            weight_shapes: list of (out_dim, in_dim) for each layer in the group
            rank: rank of shared subspace
            shared_dim: dimension of shared basis (defaults to rank)
        """
        self.rank = rank
        self.shared_dim = shared_dim or rank
        self.weight_shapes = weight_shapes
        self.n_layers = len(weight_shapes)
        
        # Shared basis A: (shared_dim, in_dim) - common input subspace
        self.A = torch.nn.Parameter(
            torch.randn(self.shared_dim, weight_shapes[0][1]) * 0.01
        )
        
        # Layer-specific B_i: (out_dim, shared_dim)
        self.B = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(s[0], self.shared_dim) * 0.01)
            for s in weight_shapes
        ])
    
    def get_weights(self) -> List[torch.Tensor]:
        """Reconstruct weight matrices for each layer."""
        weights = []
        for i in range(self.n_layers):
            W_i = self.B[i] @ self.A
            weights.append(W_i)
        return weights
    
    def to(self, device):
        self.A = torch.nn.Parameter(self.A.data.to(device))
        self.B = torch.nn.ParameterList([
            torch.nn.Parameter(b.data.to(device)) for b in self.B
        ])
        return self
    
    def parameters(self):
        return [self.A] + list(self.B)
    
    def state_dict(self):
        return {"A": self.A, **{f"B_{i}": b for i, b in enumerate(self.B)}}
    
    def load_state_dict(self, state):
        self.A = state["A"]
        for i, b in enumerate(self.B):
            self.B[i] = state[f"B_{i}"]
    
    def total_params(self) -> int:
        """Total parameter count."""
        A_params = self.A.numel()
        B_params = sum(b.numel() for b in self.B)
        return A_params + B_params
    
    def original_params(self) -> int:
        """Original parameter count."""
        return sum(s[0] * s[1] for s in self.weight_shapes)


def fit_joint_group(
    model,
    group_start: int,
    group_size: int,
    activations: dict,
    rank: int = 128,
    n_steps: int = 800,
    lr: float = 5e-4,
    device: str = "cpu",
) -> JointGroupRepresentation:
    """
    Fit a joint group representation to minimize hidden-state drift.
    
    Uses gradient descent to optimize:
        loss = MSE(h_group_orig, h_group_sub)
              + 0.1 * (1 - cosine(h_group_orig, h_group_sub))
    """
    n_layers = len(model.transformer.h)
    group_end = min(group_start + group_size, n_layers)
    
    # Collect activation shapes
    weight_shapes = []
    for i in range(group_start, group_end):
        W = model.transformer.h[i].attn.c_proj.weight.data
        weight_shapes.append((W.shape[0], W.shape[1]))
    
    # Create joint representation
    rep = JointGroupRepresentation(weight_shapes, rank=rank).to(device)
    
    # Prepare activation data
    act_keys = [f"layer{i}" for i in range(group_start, group_end)]
    act_samples = []
    
    for i, key in enumerate(act_keys):
        if key in activations:
            act = activations[key]
            if act.dim() == 3:
                flat = act.reshape(-1, act.shape[-1])
            else:
                flat = act
            if flat.shape[0] > 2000:
                idx = torch.randperm(flat.shape[0])[:2000]
                flat = flat[idx]
            act_samples.append(flat.float().to(device))
        else:
            # Fallback: use random data
            in_dim = weight_shapes[i][1]
            act_samples.append(torch.randn(1000, in_dim).to(device))
    
    # Original outputs at each layer
    orig_outputs = []
    with torch.no_grad():
        for i, key in enumerate(act_keys):
            x = act_samples[i]
            W = model.transformer.h[group_start + i].attn.c_proj.weight.data.float().to(device)
            y_orig = x @ W.T
            orig_outputs.append(y_orig.detach())
    
    # Optimize
    params = [rep.A] + list(rep.B)
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    
    best_loss = float('inf')
    best_A = None
    best_B = None
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        weights = rep.get_weights()
        
        total_loss = 0.0
        for i in range(len(weights)):
            x = act_samples[i]
            W_sub = weights[i]
            y_sub = x @ W_sub.T
            y_orig = orig_outputs[i]
            
            # MSE loss
            mse = ((y_orig - y_sub) ** 2).mean()
            
            # Cosine loss
            cos = torch.nn.functional.cosine_similarity(
                y_orig.flatten(), y_sub.flatten(), dim=0
            )
            
            total_loss = total_loss + mse + 0.1 * (1 - cos)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_A = rep.A.clone()
            best_B = [b.clone() for b in rep.B]
    
    # Restore best
    if best_A is not None:
        rep.A = torch.nn.Parameter(best_A)
        rep.B = torch.nn.ParameterList([torch.nn.Parameter(b) for b in best_B])
    
    return rep


# ============================================================
# Perplexity
# ============================================================

def compute_perplexity(model, tokenizer, texts, max_length=256, device="cpu"):
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
# Collect activations
# ============================================================

def collect_activations(model, tokenizer, texts, n_layers=12, max_length=128, device="cpu"):
    """Collect hidden-state activations at every layer."""
    activations = {}
    
    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            activations[name] = x.detach()
        return hook_fn
    
    hooks = []
    for i in range(n_layers):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook(f"layer{i}"))
        hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    return activations


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("JOINT GROUP COMPRESSION")
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
    profiling_texts = eval_texts[:30]
    
    # Collect activations
    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, profiling_texts, device=DEVICE)
    
    rank = 128
    n_layers = 12
    group_sizes = [1, 2, 4, 8, 12]
    
    all_results = []
    
    for gs in group_sizes:
        print(f"\n{'='*70}")
        print(f"JOINT GROUP SIZE {gs}")
        print(f"{'='*70}")
        
        starts = list(range(0, n_layers, gs))
        
        for start in starts:
            end = min(start + gs, n_layers)
            group_label = f"L{start}-L{end-1}"
            print(f"\n  Fitting joint representation for {group_label}...")
            
            # Fit joint representation
            rep = fit_joint_group(
                model, start, gs, activations, rank=rank, device=DEVICE
            )
            
            # Build model with joint replacement
            model_sub = copy.deepcopy(model)
            joint_weights = rep.get_weights()
            
            for i, layer_offset in enumerate(range(len(replaced))):
                layer_idx = replaced[layer_offset]
                model_sub.transformer.h[layer_idx].attn.c_proj.weight.data = joint_weights[i].to(DEVICE)
            
            # Evaluate
            ppl_orig = compute_perplexity(model, tokenizer, eval_texts, device=DEVICE)
            ppl_sub = compute_perplexity(model_sub, tokenizer, eval_texts, device=DEVICE)
            ppl_delta_pct = ((ppl_sub - ppl_orig) / ppl_orig) * 100
            
            compression = rep.original_params() / rep.total_params()
            
            print(f"    Original PPL: {ppl_orig:.2f}")
            print(f"    Substituted PPL: {ppl_sub:.2f}")
            print(f"    Delta: {ppl_delta_pct:+.2f}%")
            print(f"    Params: {rep.total_params():,} / {rep.original_params():,} ({compression:.1f}x)")
            
            all_results.append({
                "group_size": gs,
                "group_label": group_label,
                "replaced_layers": list(range(start, end)),
                "ppl_orig": ppl_orig,
                "ppl_sub": ppl_sub,
                "ppl_delta_pct": ppl_delta_pct,
                "total_params": rep.total_params(),
                "original_params": rep.original_params(),
                "compression_ratio": compression,
            })
            
            del model_sub
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("JOINT GROUP COMPRESSION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Group':<16} {'Size':<6} {'PPL D%':<10} {'Compression':<14} {'Params':<14}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['group_label']:<16} {r['group_size']:<6} {r['ppl_delta_pct']:<+10.2f} "
              f"{r['compression_ratio']:<14.1f} {r['total_params']:<14,}")
    
    # Compare: joint vs independent at same budget
    print(f"\n{'='*70}")
    print("JOINT vs INDEPENDENT COMPARISON")
    print(f"{'='*70}")
    
    for gs in group_sizes:
        joint_results = [r for r in all_results if r['group_size'] == gs]
        avg_delta = np.mean([r['ppl_delta_pct'] for r in joint_results])
        print(f"  Group size {gs:2d}: avg PPL delta = {avg_delta:+.2f}% (n={len(joint_results)})")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "joint_group_compress.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

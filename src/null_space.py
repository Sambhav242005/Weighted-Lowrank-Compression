"""
Activation null space analysis: determine what percentage of weight matrix
column space actually operates on the activation distribution.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class NullSpaceProfile:
    matrix_name: str
    n_active_directions: int  # directions with significant activation projection
    n_total_directions: int
    null_space_fraction: float  # fraction of column space in null space
    activation_variance_explained: np.ndarray  # per-direction variance
    top_directions_used: int  # directions accounting for 99% of activation energy


def collect_activations(
    model: GPT2LMHeadModel,
    tokenizer,
    n_samples: int = 50,
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Collect activations from all layers on calibration data."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    activations = {}
    hooks = []
    
    # Register hooks to capture inputs to each layer
    layer_inputs = {}
    
    def make_hook(name, store_dict):
        def hook_fn(module, input, output):
            # For GPT-2 blocks, input[0] is the hidden state
            store_dict[name] = input[0].detach()
        return hook_fn
    
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_hook(make_hook(f"layer{i}", layer_inputs)))
    
    # Collect samples
    all_hidden = {f"layer{i}": [] for i in range(model.config.n_layer)}
    
    texts = []
    for item in dataset:
        if len(texts) >= n_samples:
            break
        text = item["text"].strip()
        if len(text) > 10:
            texts.append(text)
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=max_length).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            
            # Also capture via hooks
            _ = model(**inputs)
            
            # Store hidden states
            hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
            for i in range(model.config.n_layer):
                # hidden_states[i+1] is output of layer i
                hs = hidden_states[i+1].squeeze(0)  # [seq_len, n_embd]
                all_hidden[f"layer{i}"].append(hs.cpu())
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Concatenate across samples
    result = {}
    for key, tensors in all_hidden.items():
        if tensors:
            result[key] = torch.cat(tensors, dim=0)  # [total_tokens, n_embd]
    
    return result


def profile_null_space(
    weight: torch.Tensor,
    activations: torch.Tensor,
    name: str,
    threshold: float = 0.01,
) -> NullSpaceProfile:
    """
    Analyze null space relationship between weight matrix and activations.
    
    For a weight matrix W with columns w_i, we check which columns
    have significant projection onto the activation space.
    """
    W = weight.float()  # [n_out, n_in]
    A = activations.float()  # [n_samples, n_in]
    
    n_in = W.shape[1]
    
    # Center activations
    A_centered = A - A.mean(dim=0, keepdim=True)
    
    # Compute covariance of activations
    cov = (A_centered.T @ A_centered) / A_centered.shape[0]
    
    # SVD of activations to find principal directions
    U_a, S_a, Vt_a = torch.linalg.svd(A_centered, full_matrices=False)
    
    # For each column of W, measure how much it aligns with active activation directions
    # Project each weight column onto activation principal components
    W_proj = W @ Vt_a.T  # [n_out, min(n_samples, n_in)]
    
    # Energy of weight projection per activation direction
    direction_energy = (W_proj ** 2).sum(dim=0)  # [min(n_samples, n_in)]
    total_energy = direction_energy.sum()
    
    cum_energy = torch.cumsum(direction_energy, dim=0) / total_energy
    n_active = int(torch.searchsorted(cum_energy, 1.0 - threshold)) + 1
    n_top_99 = int(torch.searchsorted(cum_energy, 0.99)) + 1
    
    null_frac = 1.0 - (n_active / n_in)
    
    return NullSpaceProfile(
        matrix_name=name,
        n_active_directions=n_active,
        n_total_directions=n_in,
        null_space_fraction=float(null_frac),
        activation_variance_explained=cum_energy.numpy(),
        top_directions_used=n_top_99,
    )


def profile_all_null_spaces(
    weights: dict,
    model: GPT2LMHeadModel,
    tokenizer,
    device: str = "cpu",
    n_samples: int = 50,
) -> List[NullSpaceProfile]:
    """Run null space analysis on all weight matrices."""
    print("Collecting activations on calibration data...")
    activations = collect_activations(model, tokenizer, n_samples=n_samples, device=device)
    
    profiles = []
    for name, wm in weights.items():
        if wm.block_type in ("embedding", "unembedding"):
            continue
        if wm.name not in activations:
            continue
        
        # For attention weights, we need activations of matching dimension
        act = activations.get(f"layer{wm.layer_idx}")
        if act is None:
            continue
        
        # Adjust dimensions for Q/K/V/O vs MLP
        if wm.block_type == "attention":
            if wm.role in ("W_Q", "W_K", "W_V"):
                # Input dim is n_embd
                pass
            elif wm.role == "W_O":
                # Output dim is n_embd, input is n_embd
                pass
        elif wm.block_type == "mlp":
            if wm.role == "W_down":
                # W_down is [n_embd, 4*n_embd], activations are [n_embd]
                pass
            elif wm.role == "W_up":
                # W_up is [4*n_embd, n_embd], activations are [n_embd]
                act = act  # acts are [n_tokens, n_embd], W_up is [4*n_embd, n_embd]
        
        p = profile_null_space(wm.tensor, act, name)
        profiles.append(p)
        print(f"  {name}: null_space={p.null_space_fraction:.1%}, "
              f"active_dirs={p.n_active_directions}/{p.n_total_directions}")
    
    return profiles


def print_null_space_summary(profiles: List[NullSpaceProfile]):
    """Print a summary of null space profiles."""
    print("\n" + "=" * 70)
    print("NULL SPACE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Name':<30} {'Matrix Size':<14} {'Active Dirs':<14} {'Null Space %':<14}")
    print("-" * 70)
    for p in profiles:
        print(f"{p.matrix_name:<30} {p.n_total_directions:<14} "
              f"{p.n_active_directions:<14} {p.null_space_fraction:<14.1%}")
    print("=" * 70)

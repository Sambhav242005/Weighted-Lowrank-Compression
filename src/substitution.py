"""
Layer substitution and functional testing.
Swap a weight matrix into the model and measure impact on outputs.
"""

import torch
import torch.nn.functional as F
import copy
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, '.')
try:
    from .representations import RepresentationResult
    from .baseline_eval import compute_perplexity, compute_logit_divergence
except ImportError:  # Preserve direct execution from the src directory.
    from representations import RepresentationResult
    from baseline_eval import compute_perplexity, compute_logit_divergence


@dataclass
class SubstitutionResult:
    layer_name: str
    representation: str
    original_params: int
    compressed_params: int
    compression_ratio: float
    relative_frobenius_error: float
    perplexity: float
    perplexity_delta: float
    perplexity_delta_pct: float
    logit_divergence: Dict
    top1_agreement: float


def get_layer_by_name(model: GPT2LMHeadModel, layer_name: str):
    """Get a specific layer module by name."""
    parts = layer_name.split('.')
    
    # Map weight names to model modules
    # layer{i}.attn.W_Q -> model.transformer.h[i].attn.c_attn (Q portion)
    # layer{i}.attn.W_O -> model.transformer.h[i].attn.c_proj
    # layer{i}.mlp.W_up -> model.transformer.h[i].mlp.c_fc
    # layer{i}.mlp.W_down -> model.transformer.h[i].mlp.c_proj
    
    layer_idx = int(parts[0].replace('layer', ''))
    block_type = parts[1]  # 'attn' or 'mlp'
    weight_role = parts[2]  # 'W_Q', 'W_K', 'W_V', 'W_O', 'W_up', 'W_down'
    
    block = model.transformer.h[layer_idx]
    
    if block_type == 'attn':
        if weight_role == 'W_O':
            return block.attn.c_proj, 'weight'
        elif weight_role in ('W_Q', 'W_K', 'W_V'):
            return block.attn.c_attn, 'weight', weight_role
    elif block_type == 'mlp':
        if weight_role == 'W_up':
            return block.mlp.c_fc, 'weight'
        elif weight_role == 'W_down':
            return block.mlp.c_proj, 'weight'
    
    return None


def substitute_weight_in_model(
    model: GPT2LMHeadModel,
    layer_name: str,
    new_weight: torch.Tensor,
) -> GPT2LMHeadModel:
    """
    Create a copy of the model with one weight matrix replaced.
    Handles the fused c_attn case for Q/K/V.
    """
    model_copy = copy.deepcopy(model)
    # Get target device from model
    target_device = next(model_copy.parameters()).device
    new_weight = new_weight.to(target_device)
    
    parts = layer_name.split('.')
    layer_idx = int(parts[0].replace('layer', ''))
    block_type = parts[1]
    weight_role = parts[2]
    
    block = model_copy.transformer.h[layer_idx]
    
    if block_type == 'attn':
        if weight_role == 'W_O':
            block.attn.c_proj.weight.data = new_weight.float()
        elif weight_role in ('W_Q', 'W_K', 'W_V'):
            # c_attn.weight is [n_embd, 3*n_embd] (fused Q,K,V)
            # We need to replace only the relevant slice
            n_embd = model.config.n_embd
            current = block.attn.c_attn.weight.data.clone()
            
            idx_map = {'W_Q': 0, 'W_K': 1, 'W_V': 2}
            idx = idx_map[weight_role]
            
            current[:, idx*n_embd:(idx+1)*n_embd] = new_weight.float()
            block.attn.c_attn.weight.data = current
    elif block_type == 'mlp':
        if weight_role == 'W_up':
            block.mlp.c_fc.weight.data = new_weight.float()
        elif weight_role == 'W_down':
            block.mlp.c_proj.weight.data = new_weight.float()
    
    return model_copy


def test_substitution(
    original_model: GPT2LMHeadModel,
    layer_name: str,
    representation: RepresentationResult,
    tokenizer,
    eval_texts: List[str],
    baseline_perplexity: float,
    baseline_logits: Optional[torch.Tensor] = None,
    max_length: int = 256,
    device: str = "cpu",
) -> SubstitutionResult:
    """
    Test substituting a representation into a layer and measure impact.
    """
    # Get original weight for error computation
    parts = layer_name.split('.')
    layer_idx = int(parts[0].replace('layer', ''))
    block_type = parts[1]
    weight_role = parts[2]
    
    block = original_model.transformer.h[layer_idx]
    if block_type == 'attn':
        if weight_role == 'W_O':
            original_weight = block.attn.c_proj.weight.data
        elif weight_role == 'W_Q':
            n_embd = original_model.config.n_embd
            original_weight = block.attn.c_attn.weight.data[:, :n_embd]
        elif weight_role == 'W_K':
            n_embd = original_model.config.n_embd
            original_weight = block.attn.c_attn.weight.data[:, n_embd:2*n_embd]
        elif weight_role == 'W_V':
            n_embd = original_model.config.n_embd
            original_weight = block.attn.c_attn.weight.data[:, 2*n_embd:]
    elif block_type == 'mlp':
        if weight_role == 'W_up':
            original_weight = block.mlp.c_fc.weight.data
        elif weight_role == 'W_down':
            original_weight = block.mlp.c_proj.weight.data
    
    # Reconstruct and compute Frobenius error
    reconstructed = representation.reconstruct()
    # Move to same device as original weight
    target_device = original_weight.device
    reconstructed = reconstructed.to(target_device)
    frob_error = torch.norm(original_weight.float() - reconstructed.float(), 'fro')
    frob_original = torch.norm(original_weight.float(), 'fro')
    relative_error = float(frob_error / frob_original)
    
    # Substitute into model
    substituted_model = substitute_weight_in_model(original_model, layer_name, reconstructed)
    substituted_model = substituted_model.to(device)
    
    # Compute perplexity
    new_perplexity = compute_perplexity(substituted_model, tokenizer, eval_texts, max_length, device)
    perplexity_delta = new_perplexity - baseline_perplexity
    perplexity_delta_pct = (perplexity_delta / baseline_perplexity) * 100
    
    # Compute logit divergence
    logit_div = {}
    top1_agreement = 0.0
    
    if baseline_logits is not None:
        with torch.no_grad():
            # Get logits from substituted model on a sample
            sample_text = eval_texts[0]
            inputs = tokenizer(sample_text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(device)
            
            orig_out = original_model(**inputs)
            sub_out = substituted_model(**inputs)
            
            # Shift for next-token prediction
            orig_logits = orig_out.logits[:, :-1, :].cpu()
            sub_logits = sub_out.logits[:, :-1, :].cpu()
            
            logit_div = compute_logit_divergence(orig_logits, sub_logits)
            top1_agreement = logit_div["top1_agreement"]
    
    print(f"  {layer_name} ({representation.name}):")
    print(f"    Compression: {representation.compression_ratio:.2f}x "
          f"({representation.original_params:,} -> {representation.n_params:,} params)")
    print(f"    Frobenius error: {relative_error:.4f}")
    print(f"    Perplexity: {baseline_perplexity:.2f} -> {new_perplexity:.2f} "
          f"(+{perplexity_delta_pct:.2f}%)")
    if logit_div:
        print(f"    Top-1 agreement: {top1_agreement:.4f}")
        print(f"    KL divergence: {logit_div['kl_orig_to_sub']:.4f}")
    
    # Free memory
    del substituted_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return SubstitutionResult(
        layer_name=layer_name,
        representation=representation.name,
        original_params=representation.original_params,
        compressed_params=representation.n_params,
        compression_ratio=representation.compression_ratio,
        relative_frobenius_error=relative_error,
        perplexity=new_perplexity,
        perplexity_delta=perplexity_delta,
        perplexity_delta_pct=perplexity_delta_pct,
        logit_divergence=logit_div,
        top1_agreement=top1_agreement,
    )

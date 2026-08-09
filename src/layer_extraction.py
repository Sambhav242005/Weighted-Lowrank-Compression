"""
Layer extraction module for GPT-2 style transformers.
Extracts individual weight matrices from attention and MLP blocks.
"""

import torch
from transformers import GPT2LMHeadModel
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WeightMatrix:
    name: str
    tensor: torch.Tensor
    layer_idx: int
    block_type: str  # 'attention' or 'mlp'
    role: str  # e.g. 'W_Q', 'W_K', 'W_V', 'W_O', 'W_up', 'W_down', 'W_gate'


def extract_gpt2_layers(model_name: str = "gpt2") -> Dict[str, WeightMatrix]:
    """Load a GPT-2 model and extract all weight matrices."""
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    weights = {}
    n_layers = model.config.n_layer

    for i in range(n_layers):
        block = model.transformer.h[i]

        # Attention weights — GPT-2 uses a fused c_attn (Q,K,V combined)
        # Shape: [n_embd, 3*n_embd] — we split it
        c_attn = block.attn.c_attn.weight.data  # [n_embd, 3*n_embd]
        n_embd = model.config.n_embd
        W_Q = c_attn[:, :n_embd].clone()
        W_K = c_attn[:, n_embd:2*n_embd].clone()
        W_V = c_attn[:, 2*n_embd:].clone()
        W_O = block.attn.c_proj.weight.data.clone()

        weights[f"layer{i}.attn.W_Q"] = WeightMatrix(
            name=f"layer{i}.attn.W_Q", tensor=W_Q,
            layer_idx=i, block_type="attention", role="W_Q"
        )
        weights[f"layer{i}.attn.W_K"] = WeightMatrix(
            name=f"layer{i}.attn.W_K", tensor=W_K,
            layer_idx=i, block_type="attention", role="W_K"
        )
        weights[f"layer{i}.attn.W_V"] = WeightMatrix(
            name=f"layer{i}.attn.W_V", tensor=W_V,
            layer_idx=i, block_type="attention", role="W_V"
        )
        weights[f"layer{i}.attn.W_O"] = WeightMatrix(
            name=f"layer{i}.attn.W_O", tensor=W_O,
            layer_idx=i, block_type="attention", role="W_O"
        )

        # MLP weights — GPT-2 uses c_fc (up) and c_proj (down)
        W_up = block.mlp.c_fc.weight.data.clone()
        W_down = block.mlp.c_proj.weight.data.clone()

        weights[f"layer{i}.mlp.W_up"] = WeightMatrix(
            name=f"layer{i}.mlp.W_up", tensor=W_up,
            layer_idx=i, block_type="mlp", role="W_up"
        )
        weights[f"layer{i}.mlp.W_down"] = WeightMatrix(
            name=f"layer{i}.mlp.W_down", tensor=W_down,
            layer_idx=i, block_type="mlp", role="W_down"
        )

    # Token embeddings and position embeddings
    weights["token_emb"] = WeightMatrix(
        name="token_emb",
        tensor=model.transformer.wte.weight.data.clone(),
        layer_idx=-1, block_type="embedding", role="token_emb"
    )
    weights["pos_emb"] = WeightMatrix(
        name="pos_emb",
        tensor=model.transformer.wpe.weight.data.clone(),
        layer_idx=-1, block_type="embedding", role="pos_emb"
    )

    # Unembedding (LM head shares weights with token embedding in GPT-2)
    # but we'll treat it separately for clarity
    weights["unembedding"] = WeightMatrix(
        name="unembedding",
        tensor=model.lm_head.weight.data.clone(),
        layer_idx=-1, block_type="unembedding", role="unembedding"
    )

    print(f"Extracted {len(weights)} weight matrices from {n_layers} layers")
    print(f"  Attention: {4 * n_layers} matrices (Q,K,V,O per layer)")
    print(f"  MLP: {2 * n_layers} matrices (up, down per layer)")
    print(f"  Embeddings: token + position")
    print(f"  Unembedding: 1 matrix")

    return weights, model


def get_model_config(model_name: str = "gpt2") -> dict:
    """Return key config parameters."""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    config = model.config
    return {
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "n_ctx": config.n_ctx,
        "vocab_size": config.vocab_size,
        "n_embd_per_head": config.n_embd // config.n_head,
    }

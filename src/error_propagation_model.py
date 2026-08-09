"""
Error Propagation Model
=======================
Measure the Jacobian of each transformer layer to predict error amplification.

Hypothesis: Layer 0's error amplifies because its Jacobian has spectral norm > 1,
while other layers' Jacobians have spectral norm < 1 (contraction).

Method: Approximate Jacobian-vector product via finite differences:
  J @ v ≈ (f(x + eps*v) - f(x)) / eps

Then use power iteration to estimate spectral norm.
"""

import torch, numpy as np, copy, json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
eval_texts = eval_texts[:30]

# ============================================================
# Step 1: Collect hidden states at each layer boundary
# ============================================================
print("=" * 70)
print("STEP 1: Collecting hidden states")
print("=" * 70)

hidden_states = {}
hooks = []

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        hidden_states[name] = x.detach()
    return hook_fn

for i in range(12):
    hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook(f"layer{i}"))
    hooks.append(hook)

# Also hook the embedding output
embedding_hook = model.transformer.wte.register_forward_hook(make_hook("embedding"))
hooks.append(embedding_hook)

with torch.no_grad():
    for text in eval_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        _ = model(**inputs)

for h in hooks:
    h.remove()

# Get a single sample for Jacobian computation
sample_text = eval_texts[0]
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

# ============================================================
# Step 2: Compute Jacobian spectral norm at each layer
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Computing Jacobian spectral norms")
print("=" * 70)

def compute_jacobian_spectral_norm(model, inputs, layer_idx, n_power_iter=20, eps=1e-3):
    """
    Estimate spectral norm of the Jacobian of layer layer_idx.
    
    The Jacobian J maps perturbation in hidden state h to perturbation in output:
      J = d(h_{l+1}) / d(h_l)
    
    We estimate ||J||_2 via power iteration on the Jacobian-vector product.
    """
    model.eval()
    dim = 768  # GPT-2 hidden dim
    
    # Get the clean hidden state at layer input
    clean_hidden = {}
    def make_hook_clean(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            clean_hidden[name] = x.detach()
        return hook_fn
    
    hooks = []
    for i in range(12):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook_clean(f"layer{i}"))
        hooks.append(hook)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    h_clean = clean_hidden[f"layer{layer_idx}"].clone()
    
    # Power iteration to estimate spectral norm
    v = torch.randn(dim, device=DEVICE)
    v = v / v.norm()
    
    singular_values = []
    
    for iteration in range(n_power_iter):
        # Forward pass with perturbation: h_perturbed = h_clean + eps * v
        # We need to run the model with this perturbed input at layer layer_idx
        
        # Hook to inject perturbation
        perturbed_output = {}
        
        def make_perturbation_hook(target_layer, perturbation):
            def hook_fn(module, input, output):
                # This hook is on the NEXT layer, capturing the perturbed input
                pass
            return hook_fn
        
        # Actually, we need to perturb the input to layer layer_idx
        # and measure the output of layer layer_idx
        
        # Method: run model, collect all hidden states, then compute
        # the forward function of layer layer_idx
        
        # Get the block
        block = model.transformer.h[layer_idx]
        
        # Clean forward through the block
        h_in = h_clean.squeeze(0)  # (seq_len, dim)
        
        # We need to compute J @ v where J = d(block(h)) / d(h)
        # Use finite differences:
        # J @ v ≈ (block(h + eps*v) - block(h)) / eps
        
        # But we can't directly perturb the hidden state during forward pass
        # Instead, we'll use the hook mechanism
        
        # Store the perturbation
        perturbation = eps * v.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        
        # Run with perturbation injected at layer layer_idx
        # We'll use a forward hook on the layer's input to add perturbation
        
        injection_done = [False]
        def inject_perturbation(module, input, output):
            if not injection_done[0]:
                injection_done[0] = True
                if isinstance(input, tuple):
                    x = input[0]
                else:
                    x = input
                # Add perturbation
                return (x + perturbation,) if isinstance(input, tuple) else x + perturbation
            return output
        
        # Register hook to inject perturbation at the INPUT of layer layer_idx
        # (which is the output of the previous layer's c_proj)
        if layer_idx == 0:
            # For layer 0, perturb the embedding output
            target_module = model.transformer.wte
        else:
            # For other layers, perturb the output of previous layer's c_proj
            target_module = model.transformer.h[layer_idx - 1].attn.c_proj
        
        hook_inject = target_module.register_forward_hook(inject_perturbation)
        
        # Also need to capture the OUTPUT of layer layer_idx
        output_captured = [None]
        def capture_output(module, input, output):
            output_captured[0] = output
            return output
        
        hook_capture = model.transformer.h[layer_idx].register_forward_hook(capture_output)
        
        # Run forward
        with torch.no_grad():
            _ = model(**inputs)
        
        hook_inject.remove()
        hook_capture.remove()
        
        # Get the output perturbation
        if output_captured[0] is not None:
            h_out_perturbed = output_captured[0]
            if isinstance(h_out_perturbed, tuple):
                h_out_perturbed = h_out_perturbed[0]
            h_out_perturbed = h_out_perturbed.detach()
        
        # Clean output
        clean_output = {}
        def make_clean_output_hook(name):
            def hook_fn(module, input, output):
                clean_output[name] = output.detach()
            return hook_fn
        
        hook_clean = model.transformer.h[layer_idx].register_forward_hook(make_clean_output_hook("clean"))
        
        with torch.no_grad():
            _ = model(**inputs)
        
        hook_clean.remove()
        
        h_out_clean = clean_output["clean"]
        if isinstance(h_out_clean, tuple):
            h_out_clean = h_out_clean[0]
        
        # Compute J @ v ≈ (h_out_perturbed - h_out_clean) / eps
        j_v = (h_out_perturbed - h_out_clean) / eps
        
        # Flatten to vector
        j_v_flat = j_v.reshape(-1)
        
        # Compute norm
        norm = j_v_flat.norm().item()
        singular_values.append(norm)
        
        # Update v for next iteration (normalize and use for next power iteration)
        v = j_v_flat[:dim] / (j_v_flat[:dim].norm() + 1e-10)
    
    # Spectral norm is the largest singular value
    spectral_norm = max(singular_values) if singular_values else 0.0
    
    return spectral_norm, singular_values


# ============================================================
# Alternative: Simpler Jacobian estimation via weight matrices
# ============================================================

def estimate_jacobian_from_weights(model, layer_idx):
    """
    Estimate Jacobian spectral norm from the weight matrices.
    
    For a transformer block:
      h_out = h_in + Attn(LN(h_in)) + MLP(LN(h_in + Attn(LN(h_in))))
    
    The Jacobian is approximately:
      J ≈ I + dAttn/dh + dMLP/dh
    
    For the attention output projection (c_proj):
      J_attn ≈ W_O (the output projection weight)
    
    The spectral norm of W_O gives a lower bound on the Jacobian norm.
    """
    block = model.transformer.h[layer_idx]
    
    # Get weight matrices (GPT-2 Conv1D: weights are (in_features, out_features))
    W_O = block.attn.c_proj.weight.data.float()  # (768, 768)
    
    # c_attn combines Q, K, V: weight is (768, 2304) — split along output dim
    W_c_attn = block.attn.c_attn.weight.data.float()  # (768, 2304)
    W_Q = W_c_attn[:, :768]      # (768, 768)
    W_K = W_c_attn[:, 768:1536]  # (768, 768)
    W_V = W_c_attn[:, 1536:]     # (768, 768)
    
    # MLP weights
    W_up = block.mlp.c_fc.weight.data.float()      # (768, 3072)
    W_down = block.mlp.c_proj.weight.data.float()   # (3072, 768)
    
    # Spectral norms
    _, S_O, _ = torch.linalg.svd(W_O, full_matrices=False)
    _, S_K, _ = torch.linalg.svd(W_K, full_matrices=False)
    _, S_V, _ = torch.linalg.svd(W_V, full_matrices=False)
    _, S_Q, _ = torch.linalg.svd(W_Q, full_matrices=False)
    _, S_up, _ = torch.linalg.svd(W_up, full_matrices=False)
    _, S_down, _ = torch.linalg.svd(W_down, full_matrices=False)
    
    # Jacobian approximation
    # Attention: J_attn ≈ W_O (simplified)
    # MLP: J_mlp ≈ W_down @ W_up (but with GELU nonlinearity)
    # Total: J ≈ I + J_attn + J_mlp
    
    spectral_norm_O = S_O[0].item()
    spectral_norm_K = S_K[0].item()
    spectral_norm_V = S_V[0].item()
    spectral_norm_Q = S_Q[0].item()
    spectral_norm_up = S_up[0].item()
    spectral_norm_down = S_down[0].item()
    
    # Effective Jacobian norm (simplified)
    # J ≈ I + W_O + W_down @ W_up
    # ||J|| ≈ 1 + ||W_O|| + ||W_down|| * ||W_up||
    effective_jacobian_norm = 1 + spectral_norm_O + spectral_norm_down * spectral_norm_up
    
    return {
        "W_O_norm": spectral_norm_O,
        "W_K_norm": spectral_norm_K,
        "W_V_norm": spectral_norm_V,
        "W_Q_norm": spectral_norm_Q,
        "W_up_norm": spectral_norm_up,
        "W_down_norm": spectral_norm_down,
        "effective_jacobian_norm": effective_jacobian_norm,
        "residual_scale": spectral_norm_O,  # How much the residual stream scales
    }


# ============================================================
# Step 3: Measure error propagation rate empirically
# ============================================================

def measure_error_propagation(model, tokenizer, text, target_layer, rank=128, device="cpu"):
    """
    Compress target_layer and measure error at each subsequent layer.
    Returns the error trajectory.
    """
    model.eval()
    
    # Get clean hidden states
    clean_hidden = {}
    def make_hook_clean(name):
        def hook_fn(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            clean_hidden[name] = x.detach().cpu()
        return hook_fn
    
    hooks = []
    for i in range(12):
        hook = model.transformer.h[i].attn.c_proj.register_forward_hook(make_hook_clean(f"layer{i}"))
        hooks.append(hook)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    # Build compressed model
    model_sub = copy.deepcopy(model)
    W = model_sub.transformer.h[target_layer].attn.c_proj.weight.data.cpu().float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, min(U.shape[1], Vt.shape[0]))
    W_sub = (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).to(device)
    model_sub.transformer.h[target_layer].attn.c_proj.weight.data = W_sub
    
    # Get compressed hidden states
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
    for i in range(12):
        hook = model_sub.transformer.h[i].attn.c_proj.register_forward_hook(make_hook_sub(f"layer{i}"))
        hooks_sub.append(hook)
    
    with torch.no_grad():
        _ = model_sub(**inputs)
    
    for h in hooks_sub:
        h.remove()
    
    # Compute error trajectory
    errors = []
    for i in range(target_layer, 12):
        key = f"layer{i}"
        if key in clean_hidden and key in sub_hidden:
            h_clean = clean_hidden[key].float()
            h_sub = sub_hidden[key].float()
            mse = ((h_clean - h_sub) ** 2).mean().item()
            cos = torch.nn.functional.cosine_similarity(
                h_clean.flatten(), h_sub.flatten(), dim=0
            ).item()
            errors.append({"layer": i, "mse": mse, "cosine": cos})
    
    del model_sub
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return errors


# ============================================================
# Main: Run all measurements
# ============================================================

print("\n" + "=" * 70)
print("JACOBIAN SPECTRAL NORMS (from weights)")
print("=" * 70)

jacobian_data = {}
for i in range(12):
    data = estimate_jacobian_from_weights(model, i)
    jacobian_data[f"layer{i}"] = data
    print(f"  Layer {i:2d}: W_O={data['W_O_norm']:.2f}, "
          f"W_up={data['W_up_norm']:.2f}, W_down={data['W_down_norm']:.2f}, "
          f"Effective_J={data['effective_jacobian_norm']:.2f}")

print("\n" + "=" * 70)
print("ERROR PROPAGATION (empirical)")
print("=" * 70)

propagation_data = {}
for target_layer in [0, 1, 5, 10]:
    print(f"\n  Compressing layer {target_layer}...")
    errors = measure_error_propagation(model, tokenizer, eval_texts[0], target_layer, rank=128, device=DEVICE)
    propagation_data[f"layer{target_layer}"] = errors
    
    for e in errors:
        marker = " <-- source" if e["layer"] == target_layer else ""
        print(f"    Layer {e['layer']:2d}: MSE={e['mse']:.6f}  cos={e['cosine']:.4f}{marker}")

# ============================================================
# Step 4: Predict vs observe
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION vs OBSERVATION")
print("=" * 70)

for target_layer in [0, 1, 5, 10]:
    j_data = jacobian_data[f"layer{target_layer}"]
    prop_errors = propagation_data[f"layer{target_layer}"]
    
    # Predicted amplification: product of Jacobian norms from target to end
    predicted_amp = 1.0
    for i in range(target_layer, 12):
        predicted_amp *= jacobian_data[f"layer{i}"]["residual_scale"]
    
    # Observed amplification: ratio of final error to initial error
    if prop_errors:
        initial_mse = prop_errors[0]["mse"]
        final_mse = prop_errors[-1]["mse"]
        observed_amp = final_mse / (initial_mse + 1e-10)
    else:
        observed_amp = 0
    
    print(f"\n  Layer {target_layer}:")
    print(f"    Predicted Jacobian product: {predicted_amp:.2f}")
    print(f"    Observed error amplification: {observed_amp:.2f}")
    print(f"    W_O norm: {j_data['W_O_norm']:.2f}")
    print(f"    Effective Jacobian: {j_data['effective_jacobian_norm']:.2f}")

# ============================================================
# Save results
# ============================================================

output = {
    "jacobian_data": jacobian_data,
    "propagation_data": {k: v for k, v in propagation_data.items()},
}

output_path = Path("results/error_propagation_model.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {output_path}")

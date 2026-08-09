"""
Drift-Aware Weighted Low-Rank Compression -- EXTENDED (family + architecture)
===============================================================================
Sweep 1/2 result on GPT-2 Small attn.c_proj (rank 256, one-sided 3x):
  plain SVD control +21.34%  vs  weighted fit alpha=0 beta=0.1  +3.58%.

This script tests generality along the two axes AGENTS.md requires:
  E1. matrix family:  GPT-2 Small mlp.c_fc (768->3072) + mlp.c_proj (3072->768)
  E2. architecture:   Gemma 3 1B attn o_proj + mlp gate/up/down_proj
      (google/gemma-3-1b-it, fp16 -- same revision/dtype as phase_d_gemma3.py)

Method (winner config of sweeps 1/2): teacher-input weighted fit (alpha=0),
weight-space anchor beta=0.1, closed-form M* = C(G+lam I)^-1 followed by
G-weighted rank-r truncation; Conv1D bias subtracted before fit (bias stays
in the module). Control per family: plain weight-space SVD at the same rank.

Rank convention: one-sided 3x, r = min(in,out)//3 (repo convention, matches
phase5/c_proj sweeps). Storage ratio reported per matrix.

Eval protocol identical to prior runs (WikiText-2 test, 50 texts >50 chars,
max_length 256). Calibration: WikiText-2 TRAIN split, per-module capture.

Output: results/drift_aware_extended.json (new file; nothing overwritten).
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RIDGE_FRAC = 0.01
EPS_REL = 1e-3
ALPHA, BETA = 0.0, 0.1
RATIO = 3
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

results = {}

# ---------------- generic building blocks ----------------
def svd_reconstruct(M, rank):
    M = M.float()
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

def weighted_low_rank_fit(G, C, rank):
    """Closed-form optimum M* = C(G+lam I)^-1, G-weighted rank-r truncation."""
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > EPS_REL * evals.max()
    if int(keep.sum()) < rank:
        keep = torch.ones_like(keep, dtype=torch.bool)
    Q = evecs[:, keep]
    Gk = Q.t() @ G @ Q
    Ck = C @ Q
    lam = RIDGE_FRAC * Gk.diagonal().mean()
    evals_k, evecs_k = torch.linalg.eigh(Gk)
    Gk_inv = evecs_k @ torch.diag(1.0 / (evals_k + lam)) @ evecs_k.t()
    Gk_half = evecs_k @ torch.diag((evals_k + lam).sqrt()) @ evecs_k.t()
    Mstar = Ck @ Gk_inv
    Uz, _, _ = torch.linalg.svd(Mstar @ Gk_half, full_matrices=False)
    r = min(rank, Uz.shape[1])
    return Uz[:, :r] @ (Uz[:, :r].t() @ Mstar) @ Q.t()

def capture_module(m, mod, chunks):
    """Single-module (input, output) column-matrix capture over chunks."""
    xs, ys = [], []
    def hook(_m, inp, out):
        xs.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).t().float())
        ys.append(out.detach().reshape(-1, out.shape[-1]).t().float())
    h = mod.register_forward_hook(hook)
    with torch.no_grad():
        for c in chunks:
            m(input_ids=c.unsqueeze(0).to(DEVICE))
    h.remove()
    return torch.cat(xs, 1), torch.cat(ys, 1)

def make_eval(texts, tok):
    def compute_perplexity(m, max_length=256):
        m.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in texts:
                inputs = tok(text, return_tensors="pt", truncation=True,
                             max_length=max_length).to(DEVICE)
                labels = inputs["input_ids"].clone()
                if "attention_mask" in inputs:
                    labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
                n_tok = int(labels[:, 1:].ne(-100).sum().item())
                if n_tok == 0:
                    continue
                out = m(**inputs, labels=labels)
                total_loss += out.loss.item() * n_tok
                total_tokens += n_tok
        return float(np.exp(total_loss / total_tokens)) if total_tokens else float("inf")
    return compute_perplexity

def fit_module(X_t, Y_t, M, bias, r):
    """Teacher-input weighted fit (alpha=0) with anchor beta; returns M_hat."""
    Y_t = Y_t - bias.unsqueeze(1)
    n = X_t.shape[1]
    d = X_t.shape[0]
    G = X_t @ X_t.t() + BETA * n * torch.eye(d, device=DEVICE)
    C = Y_t @ X_t.t() + BETA * n * M
    return weighted_low_rank_fit(G, C, r)

# ---------------- E1: GPT-2 Small MLP family ----------------
print("\n===== E1: GPT-2 Small MLP (c_fc 768->3072, c_proj 3072->768), one-sided 3x =====")
tok_g = GPT2Tokenizer.from_pretrained("gpt2")
tok_g.pad_token = tok_g.eos_token
model_g = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:50]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok_g("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks_g = [ids[i * 512:(i + 1) * 512] for i in range(64)]

ppl_g = make_eval(eval_texts, tok_g)
base_g = ppl_g(model_g)
print(f"GPT-2 baseline PPL: {base_g:.2f}")
results["gpt2_mlp"] = {"baseline_ppl": base_g, "variants": {}}

def gpt2_targets(m):
    """(module, rank) pairs; Conv1D map M = weight.T acts on column inputs."""
    ts = []
    for i in range(12):
        fc = m.transformer.h[i].mlp.c_fc      # Conv1D [768, 3072]
        pj = m.transformer.h[i].mlp.c_proj    # Conv1D [3072, 768]
        ts.append((fc, 768 // RATIO))
        ts.append((pj, 768 // RATIO))
    return ts

def eval_gpt2(name, build_fn):
    t0 = time.time()
    m_sub = copy.deepcopy(model_g)
    build_fn(m_sub)
    ppl = ppl_g(m_sub)
    delta = (ppl - base_g) / base_g * 100
    del m_sub
    torch.cuda.empty_cache()
    results["gpt2_mlp"]["variants"][name] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {name:28s} PPL={ppl:8.2f} delta={delta:+7.2f}%  ({time.time()-t0:.0f}s)")

def build_gpt2_svd(m_sub):
    for mod, r in gpt2_targets(m_sub):
        M = mod.weight.data.float().t()
        mod.weight.data = svd_reconstruct(M, r).t().to(mod.weight.dtype)

def build_gpt2_weighted(m_sub):
    for (mod_o, r), (mod_s, _) in zip(gpt2_targets(model_g), gpt2_targets(m_sub)):
        X_t, Y_t = capture_module(model_g, mod_o, chunks_g)
        M = mod_o.weight.data.float().t()
        bias = mod_o.bias.data.float()
        M_hat = fit_module(X_t, Y_t, M, bias, r)
        # Conv1D forward is x_row @ weight -> weight = M^T
        mod_s.weight.data = M_hat.t().to(mod_s.weight.dtype)
        del X_t, Y_t, M_hat
        torch.cuda.empty_cache()

print("E1 control: plain SVD (matched rank)")
eval_gpt2("plain_svd", build_gpt2_svd)
print(f"E1 candidate: weighted fit alpha={ALPHA} beta={BETA}")
eval_gpt2(f"weighted_a{ALPHA}_b{BETA}", build_gpt2_weighted)

del model_g
torch.cuda.empty_cache()

# ---------------- E2: Gemma 3 1B ----------------
print("\n===== E2: Gemma 3 1B (o_proj, gate/up/down_proj), one-sided 3x =====")
tok_gm = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model_gm = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", torch_dtype=torch.float16).to(DEVICE).eval()

ids = tok_gm("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks_gm = [ids[i * 512:(i + 1) * 512] for i in range(16)]

ppl_gm = make_eval(eval_texts, tok_gm)
base_gm = ppl_gm(model_gm)
print(f"Gemma3-1B baseline PPL: {base_gm:.2f}")
results["gemma3_1b"] = {"baseline_ppl": base_gm, "variants": {}}

def gemma_targets(m):
    """nn.Linear: y = Wx is already column convention. M = weight.data."""
    ts = []
    for blk in m.model.layers:
        for lin in (blk.self_attn.o_proj, blk.mlp.gate_proj,
                    blk.mlp.up_proj, blk.mlp.down_proj):
            ts.append((lin, min(lin.weight.shape) // RATIO))
    return ts

def eval_gemma(name, build_fn):
    t0 = time.time()
    m_sub = copy.deepcopy(model_gm)
    build_fn(m_sub)
    ppl = ppl_gm(m_sub)
    delta = (ppl - base_gm) / base_gm * 100
    del m_sub
    torch.cuda.empty_cache()
    results["gemma3_1b"]["variants"][name] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {name:28s} PPL={ppl:8.2f} delta={delta:+7.2f}%  ({time.time()-t0:.0f}s)")

def build_gemma_svd(m_sub):
    for mod, r in gemma_targets(m_sub):
        mod.weight.data = svd_reconstruct(mod.weight.data, r).to(mod.weight.dtype)

def build_gemma_weighted(m_sub):
    for (mod_o, r), (mod_s, _) in zip(gemma_targets(model_gm), gemma_targets(m_sub)):
        X_t, Y_t = capture_module(model_gm, mod_o, chunks_gm)
        M = mod_o.weight.data.float()
        bias = mod_o.bias.data.float() if getattr(mod_o, "bias", None) is not None \
            else torch.zeros(M.shape[0], device=DEVICE)
        M_hat = fit_module(X_t, Y_t, M, bias, r)
        mod_s.weight.data = M_hat.to(mod_s.weight.dtype)
        del X_t, Y_t, M_hat
        torch.cuda.empty_cache()

print("E2 control: plain SVD (matched rank)")
eval_gemma("plain_svd", build_gemma_svd)
print(f"E2 candidate: weighted fit alpha={ALPHA} beta={BETA}")
eval_gemma(f"weighted_a{ALPHA}_b{BETA}", build_gemma_weighted)

# ---------------- save ----------------
out_path = Path("results/drift_aware_extended.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
for fam, fr in results.items():
    for name, r in fr["variants"].items():
        print(f"  {fam}/{name:28s} {r['ppl_delta_pct']:+.2f}%")

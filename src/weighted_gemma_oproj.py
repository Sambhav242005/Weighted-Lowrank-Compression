"""
Gemma 3 1B o_proj-only: weighted fit vs plain SVD (cross-architecture check)
=============================================================================
Phase D reference (results/phase_d_gemma3_1b.json): o_proj-only plain SVD at
one-sided 3x (rank 384) gave +71.95% PPL. This run reproduces the SVD control
in-protocol and tests the sweep winner (weighted fit, alpha=0, beta=0.1).

Protocol matches drift_aware_extended.py E2: google/gemma-3-1b-it fp16,
WikiText-2 TEST 50 texts eval, TRAIN calibration (16x512 tokens),
rank = 1152//3 = 384 on every o_proj.

Output: results/weighted_gemma_oproj.json (new file; nothing overwritten).
"""

import json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RIDGE_FRAC = 0.01
EPS_REL = 1e-3
BETA = 0.1
RANK = 1152 // 3
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", torch_dtype=torch.float16).to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:50]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * 512:(i + 1) * 512] for i in range(16)]

def compute_perplexity(m, max_length=256):
    m.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
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

base_ppl = compute_perplexity(model)
print(f"Baseline PPL: {base_ppl:.2f}  (phase D reported 70.40)")

def svd_reconstruct(M, rank):
    M = M.float()
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

def weighted_low_rank_fit(G, C, rank):
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

def capture_module(m, mod):
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

results = {"baseline_ppl": base_ppl, "rank": RANK, "variants": {}}

def run(method):
    t0 = time.time()
    m_sub = copy.deepcopy(model)
    for blk_o, blk_s in zip(model.model.layers, m_sub.model.layers):
        mod_o, mod_s = blk_o.self_attn.o_proj, blk_s.self_attn.o_proj
        M = mod_o.weight.data.float()          # nn.Linear: y = W x (column conv)
        if method == "plain_svd":
            M_hat = svd_reconstruct(M, RANK)
        else:
            X_t, Y_t = capture_module(model, mod_o)
            bias = mod_o.bias.data.float() if getattr(mod_o, "bias", None) is not None \
                else torch.zeros(M.shape[0], device=DEVICE)
            Y_t = Y_t - bias.unsqueeze(1)
            n = X_t.shape[1]
            G = X_t @ X_t.t() + BETA * n * torch.eye(X_t.shape[0], device=DEVICE)
            C = Y_t @ X_t.t() + BETA * n * M
            M_hat = weighted_low_rank_fit(G, C, RANK)
            del X_t, Y_t, G, C
        mod_s.weight.data = M_hat.to(mod_s.weight.dtype)
        del M_hat
        torch.cuda.empty_cache()
    ppl = compute_perplexity(m_sub)
    delta = (ppl - base_ppl) / base_ppl * 100
    del m_sub
    torch.cuda.empty_cache()
    results["variants"][method] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {method:12s} PPL={ppl:9.2f} delta={delta:+9.2f}%  ({time.time()-t0:.0f}s)")

print("Control: plain SVD (phase D reference +71.95%)")
run("plain_svd")
print(f"Candidate: weighted fit alpha=0 beta={BETA}")
run(f"weighted_b{BETA}")

out_path = Path("results/weighted_gemma_oproj.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

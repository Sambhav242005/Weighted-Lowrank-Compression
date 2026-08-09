"""
One-Family-at-a-Time Weighted Low-Rank Fit (GPT-2 Small, one-sided 3x)
=======================================================================
Follow-up to drift_aware_extended.py, which showed that compressing ALL MLP
matrices simultaneously at 3x breaks the model in both SVD and weighted
variants (weighted still ~30% relatively better).

Per the decision tree: split the study by matrix family. Variants here
(plain SVD control vs weighted fit alpha=0 beta=0.1, winner of sweeps 1/2):
  V1 mlp.c_proj only   (MLP output projections, 12 matrices)
  V2 mlp.c_fc only     (MLP input projections, 12 matrices)
  V3 attn.c_attn only  (attention input projection 768->2304, 12 matrices)
  V4 attn.c_attn + attn.c_proj  (full attention block, 24 matrices)

Everything else identical to prior runs: gpt2 default revision, WikiText-2
(TRAIN calibration 64x512, TEST 50 texts eval), rank = min(in,out)//3.

Output: results/weighted_one_family.json (new file; nothing overwritten).
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RIDGE_FRAC = 0.01
EPS_REL = 1e-3
BETA = 0.1
RATIO = 3
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:50]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * 512:(i + 1) * 512] for i in range(64)]

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
print(f"Baseline PPL: {base_ppl:.2f}")

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

# family selectors: return list of (orig_mod, sub_mod) Conv1D pairs
def fam_mlp_cproj(m_o, m_s):
    return [(m_o.transformer.h[i].mlp.c_proj, m_s.transformer.h[i].mlp.c_proj)
            for i in range(12)]

def fam_mlp_cfc(m_o, m_s):
    return [(m_o.transformer.h[i].mlp.c_fc, m_s.transformer.h[i].mlp.c_fc)
            for i in range(12)]

def fam_attn_cattn(m_o, m_s):
    return [(m_o.transformer.h[i].attn.c_attn, m_s.transformer.h[i].attn.c_attn)
            for i in range(12)]

def fam_attn_full(m_o, m_s):
    return fam_attn_cattn(m_o, m_s) + \
        [(m_o.transformer.h[i].attn.c_proj, m_s.transformer.h[i].attn.c_proj)
         for i in range(12)]

results = {"baseline_ppl": base_ppl, "variants": {}}

def run_family(fname, selector):
    print(f"\n===== {fname} =====")
    for method in ("plain_svd", "weighted"):
        t0 = time.time()
        m_sub = copy.deepcopy(model)
        pairs = selector(model, m_sub)
        for mod_o, mod_s in pairs:
            M = mod_o.weight.data.float().t()          # Conv1D map M = W^T
            r = min(M.shape) // RATIO
            if method == "plain_svd":
                M_hat = svd_reconstruct(M, r)
            else:
                X_t, Y_t = capture_module(model, mod_o)
                bias = mod_o.bias.data.float()
                Y_t = Y_t - bias.unsqueeze(1)
                n = X_t.shape[1]
                G = X_t @ X_t.t() + BETA * n * torch.eye(X_t.shape[0], device=DEVICE)
                C = Y_t @ X_t.t() + BETA * n * M
                M_hat = weighted_low_rank_fit(G, C, r)
                del X_t, Y_t, G, C
            mod_s.weight.data = M_hat.t().to(mod_s.weight.dtype)
            del M_hat
            torch.cuda.empty_cache()
        ppl = compute_perplexity(m_sub)
        delta = (ppl - base_ppl) / base_ppl * 100
        del m_sub
        torch.cuda.empty_cache()
        key = f"{fname}/{method}"
        results["variants"][key] = {"ppl": ppl, "ppl_delta_pct": delta}
        print(f"  {method:12s} PPL={ppl:9.2f} delta={delta:+9.2f}%  ({time.time()-t0:.0f}s)")

run_family("mlp_cproj", fam_mlp_cproj)
run_family("mlp_cfc", fam_mlp_cfc)
run_family("attn_c_attn", fam_attn_cattn)
run_family("attn_full", fam_attn_full)

out_path = Path("results/weighted_one_family.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
for k, r in results["variants"].items():
    print(f"  {k:28s} {r['ppl_delta_pct']:+.2f}%")

"""
Robustness check for the GPT-2 c_proj headline (+3.58%):
rebuild with DISJOINT calibration chunks (64-127 instead of 0-63).
Same method (alpha=0, beta=0.1), same eval protocol.
Output: results/drift_aware_svd_robustness.json (new file).
"""
import json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda"
RIDGE_FRAC, EPS_REL, BETA, RANK, N_LAYERS = 0.01, 1e-3, 0.1, 256, 12

tok = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:50]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * 512:(i + 1) * 512] for i in range(64, 128)]  # OFFSET

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

def weighted_low_rank_fit(G, C, rank):
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > EPS_REL * evals.max()
    if int(keep.sum()) < rank:
        keep = torch.ones_like(keep, dtype=torch.bool)
    Q = evecs[:, keep]
    Gk, Ck = Q.t() @ G @ Q, C @ Q
    lam = RIDGE_FRAC * Gk.diagonal().mean()
    evals_k, evecs_k = torch.linalg.eigh(Gk)
    Gk_inv = evecs_k @ torch.diag(1.0 / (evals_k + lam)) @ evecs_k.t()
    Gk_half = evecs_k @ torch.diag((evals_k + lam).sqrt()) @ evecs_k.t()
    Mstar = Ck @ Gk_inv
    Uz, _, _ = torch.linalg.svd(Mstar @ Gk_half, full_matrices=False)
    r = min(rank, Uz.shape[1])
    return Uz[:, :r] @ (Uz[:, :r].t() @ Mstar) @ Q.t()

base = compute_perplexity(model)
print(f"Baseline PPL: {base:.2f}")

m_sub = copy.deepcopy(model)
for i in range(N_LAYERS):
    mod_o = model.transformer.h[i].attn.c_proj
    mod_s = m_sub.transformer.h[i].attn.c_proj
    xs, ys = [], []
    def hook(_m, inp, out):
        xs.append(inp[0].detach().reshape(-1, 768).t().float())
        ys.append(out.detach().reshape(-1, 768).t().float())
    h = mod_o.register_forward_hook(hook)
    with torch.no_grad():
        for c in chunks:
            model(input_ids=c.unsqueeze(0).to(DEVICE))
    h.remove()
    X_t, Y_t = torch.cat(xs, 1), torch.cat(ys, 1)
    M = mod_o.weight.data.float().t()
    Y_t = Y_t - mod_o.bias.data.float().unsqueeze(1)
    n = X_t.shape[1]
    G = X_t @ X_t.t() + BETA * n * torch.eye(768, device=DEVICE)
    C = Y_t @ X_t.t() + BETA * n * M
    mod_s.weight.data = weighted_low_rank_fit(G, C, RANK).t().to(mod_s.weight.dtype)
    del X_t, Y_t, G, C, xs, ys
    torch.cuda.empty_cache()

ppl = compute_perplexity(m_sub)
delta = (ppl - base) / base * 100
print(f"Weighted c_proj r{RANK} (offset calib): PPL={ppl:.2f} delta={delta:+.2f}%")

out = {"baseline_ppl": base, "weighted_ppl": ppl, "ppl_delta_pct": delta,
       "calibration": "wikitext-2 train chunks 64-127 (offset robustness)"}
with open(Path("results/drift_aware_svd_robustness.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/drift_aware_svd_robustness.json")

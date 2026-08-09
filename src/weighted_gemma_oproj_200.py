"""
Verify Gemma o_proj weighted-fit result on a larger eval set (200 texts).
The 50-text eval gave -4.45% (better than baseline) -- check it is not noise.
Same build as src/weighted_gemma_oproj.py; nothing overwritten.
Output: results/weighted_gemma_oproj_200.json
"""
import json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda"
RIDGE_FRAC, EPS_REL, BETA, RANK = 0.01, 1e-3, 0.1, 1152 // 3

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", torch_dtype=torch.float16).to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:200]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * 512:(i + 1) * 512] for i in range(16)]

def compute_perplexity(m, texts, max_length=256):
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

print(f"Eval on {len(eval_texts)} texts")
base = compute_perplexity(model, eval_texts)
print(f"Baseline PPL: {base:.2f}")

m_sub = copy.deepcopy(model)
for blk_o, blk_s in zip(model.model.layers, m_sub.model.layers):
    mod_o, mod_s = blk_o.self_attn.o_proj, blk_s.self_attn.o_proj
    M = mod_o.weight.data.float()
    X_t, Y_t = capture_module(model, mod_o)
    n = X_t.shape[1]
    G = X_t @ X_t.t() + BETA * n * torch.eye(X_t.shape[0], device=DEVICE)
    C = Y_t @ X_t.t() + BETA * n * M
    mod_s.weight.data = weighted_low_rank_fit(G, C, RANK).to(mod_s.weight.dtype)
    del X_t, Y_t, G, C
    torch.cuda.empty_cache()

ppl = compute_perplexity(m_sub, eval_texts)
delta = (ppl - base) / base * 100
print(f"Weighted o_proj r{RANK}: PPL={ppl:.2f} delta={delta:+.2f}%")

out = {"n_eval_texts": len(eval_texts), "baseline_ppl": base,
       "weighted_ppl": ppl, "ppl_delta_pct": delta}
with open(Path("results/weighted_gemma_oproj_200.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Saved results/weighted_gemma_oproj_200.json")

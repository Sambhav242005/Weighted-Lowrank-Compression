"""
Compression-ratio frontier for the weighted fit on GPT-2 attn.c_proj.
Ranks: 384 (2x), 256 (3x, known +3.58%), 192 (4x), 128 (6x), 96 (8x).
Controls: plain SVD at the same ranks. alpha=0, beta=0.1, calib chunks 0-63.
Output: results/weighted_ratio_frontier.json (new file).
"""
import json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

DEVICE = "cuda"
RIDGE_FRAC, EPS_REL, BETA, N_LAYERS = 0.01, 1e-3, 0.1, 12

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

def svd_reconstruct(M, rank):
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

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

# precompute teacher stats per layer once (G, C without anchor)
print("Capturing calibration stats...")
stats = []
for i in range(N_LAYERS):
    mod = model.transformer.h[i].attn.c_proj
    xs, ys = [], []
    def hook(_m, inp, out):
        xs.append(inp[0].detach().reshape(-1, 768).t().float())
        ys.append(out.detach().reshape(-1, 768).t().float())
    h = mod.register_forward_hook(hook)
    with torch.no_grad():
        for c in chunks:
            model(input_ids=c.unsqueeze(0).to(DEVICE))
    h.remove()
    X, Y = torch.cat(xs, 1), torch.cat(ys, 1)
    Y = Y - mod.bias.data.float().unsqueeze(1)
    G = X @ X.t()
    C = Y @ X.t()
    stats.append((mod.weight.data.float().t(), G, C, X.shape[1]))
    del X, Y, xs, ys
    torch.cuda.empty_cache()

results = {"baseline_ppl": base, "variants": {}}

for rank, ratio in [(384, 2), (256, 3), (192, 4), (128, 6), (96, 8)]:
    for method in ("plain_svd", "weighted"):
        t0 = time.time()
        m_sub = copy.deepcopy(model)
        for i in range(N_LAYERS):
            M, G, C, n = stats[i]
            mod_s = m_sub.transformer.h[i].attn.c_proj
            if method == "plain_svd":
                M_hat = svd_reconstruct(M, rank)
            else:
                Ga = G + BETA * n * torch.eye(768, device=DEVICE)
                Ca = C + BETA * n * M
                M_hat = weighted_low_rank_fit(Ga, Ca, rank)
            mod_s.weight.data = M_hat.t().to(mod_s.weight.dtype)
            del M_hat
        ppl = compute_perplexity(m_sub)
        delta = (ppl - base) / base * 100
        del m_sub
        torch.cuda.empty_cache()
        key = f"r{rank}_{ratio}x_{method}"
        results["variants"][key] = {"ppl": ppl, "ppl_delta_pct": delta}
        print(f"  {key:24s} PPL={ppl:9.2f} delta={delta:+9.2f}%  ({time.time()-t0:.0f}s)")

out_path = Path("results/weighted_ratio_frontier.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

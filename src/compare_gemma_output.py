"""
Output comparison: Gemma 3 1B baseline vs weighted-fit o_proj (3x, r384).
==========================================================================
The weighted fit gave -4.45% / -7.34% PPL (better than baseline). This script
looks at the OUTPUTS directly to characterize what changed:

  1. Logit divergence on 30 WikiText-2 test texts (teacher forcing):
     KL(orig||sub), KL(sub||orig), cosine similarity, top-1/top-5 agreement,
     softmax entropy of both models.
  2. Greedy generation side-by-side on 5 fixed prompts (64 new tokens).

Same build as src/weighted_gemma_oproj.py (alpha=0, beta=0.1, calib 16x512
train tokens). Output: results/compare_gemma_output.json (new file).
"""
import json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEVICE = "cuda"
RIDGE_FRAC, EPS_REL, BETA, RANK = 0.01, 1e-3, 0.1, 1152 // 3

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-1b-it", torch_dtype=torch.float16).to(DEVICE).eval()

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:30]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * 512:(i + 1) * 512] for i in range(16)]

# ---------------- build weighted o_proj student (same as prior runs) ------
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

print("Building weighted o_proj student...")
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

# ---------------- 1. logit divergence (teacher forcing) --------------------
print("Computing logit divergence on 30 texts...")
agg = {"kl_o2s": [], "kl_s2o": [], "cos": [], "top1": [], "top5": [],
       "ent_o": [], "ent_s": [], "n_tokens": 0}
with torch.no_grad():
    for text in eval_texts:
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=256).to(DEVICE)
        lo = model(**inputs).logits[:, :-1, :].float().cpu()
        ls = m_sub(**inputs).logits[:, :-1, :].float().cpu()
        po, ps = F.softmax(lo, -1), F.softmax(ls, -1)
        agg["kl_o2s"].append(float((po * (po.clamp_min(1e-12).log() - ps.clamp_min(1e-12).log())).sum(-1).mean()))
        agg["kl_s2o"].append(float((ps * (ps.clamp_min(1e-12).log() - po.clamp_min(1e-12).log())).sum(-1).mean()))
        agg["cos"].append(float(F.cosine_similarity(lo[0], ls[0], -1).mean()))
        t_o, t_s = lo.argmax(-1), ls.argmax(-1)
        agg["top1"].append(float((t_o == t_s).float().mean()))
        top5_o = lo.topk(5, -1).indices
        agg["top5"].append(float((top5_o == t_s.unsqueeze(-1)).any(-1).float().mean()))
        agg["ent_o"].append(float(-(po * po.clamp_min(1e-12).log()).sum(-1).mean()))
        agg["ent_s"].append(float(-(ps * ps.clamp_min(1e-12).log()).sum(-1).mean()))
        agg["n_tokens"] += lo.shape[1]

div = {k: float(np.mean(v)) for k, v in agg.items() if k != "n_tokens"}
div["n_tokens"] = agg["n_tokens"]
print(f"  KL(orig||sub)={div['kl_o2s']:.4f}  KL(sub||orig)={div['kl_s2o']:.4f}")
print(f"  cosine={div['cos']:.4f}  top1={div['top1']:.4f}  top5={div['top5']:.4f}")
print(f"  entropy orig={div['ent_o']:.4f}  sub={div['ent_s']:.4f}")

# ---------------- 2. greedy generation side-by-side ------------------------
prompts = [
    "The history of artificial intelligence begins",
    "In a letter to the editor, the scientist explained that",
    "The football match ended in controversy when",
    "According to the latest economic report,",
    "The old lighthouse keeper remembered the night",
]
print("\nGreedy generation comparison (64 new tokens):")
gens = []
for p in prompts:
    enc = tok(p, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out_o = model.generate(**enc, max_new_tokens=64, do_sample=False)
        out_s = m_sub.generate(**enc, max_new_tokens=64, do_sample=False)
    txt_o = tok.decode(out_o[0], skip_special_tokens=True)
    txt_s = tok.decode(out_s[0], skip_special_tokens=True)
    gens.append({"prompt": p, "baseline": txt_o, "weighted": txt_s})
    print(f"\n--- PROMPT: {p}")
    print(f"BASELINE: ...{txt_o[len(p):]}")
    print(f"WEIGHTED: ...{txt_s[len(p):]}")

res = {"model": "google/gemma-3-1b-it", "method": "weighted o_proj r384",
       "logit_divergence": div, "generations": gens}
with open(Path("results/compare_gemma_output.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\nSaved results/compare_gemma_output.json")

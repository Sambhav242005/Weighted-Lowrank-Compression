"""
Drift-Aware Sequential Low-Rank Compression (QEP idea adapted to SVD)
=====================================================================
Hypothesis: the 3x full-stack failure (+21.36% PPL) is driven by fitting each
layer on TEACHER inputs while at inference it receives DRIFTED student inputs.

Method (adapted from QEP, arXiv 2504.09629, for low-rank replacement):
  Compress layers sequentially, layer 0 -> 11. For layer l, collect
    X_t : inputs to c_proj from the TEACHER model (original weights)
    X_s : inputs to c_proj from the STUDENT model (layers < l compressed)
  The drift-aware objective is min ||W X_t - W_hat X_s||_F, solved DIRECTLY
  as a weighted low-rank fit by alternating optimization with an orthonormal
  subspace factor (avoids the U/V scale degeneracy of naive ALS).
  A weight-space anchor beta*||W - W_hat||_F^2 keeps the problem well-posed
  in input-unobserved directions (blends beta pseudo-identity into the Gram):
    G(alpha,beta) = (1-alpha) X_t X_t^T + alpha X_s X_s^T + beta*n*I
    C(alpha,beta) = (1-alpha) Y_t X_t^T + alpha Y_t X_s^T + beta*n*W
  Solution: closed-form full-rank optimum M* = C G^{-1}, then rank-r
  truncation weighted by G^{1/2} (SVD of M* G^{1/2}).
    alpha=0 -> teacher-input fit, alpha=1 -> fully drift-aware.

  RUN 2 LESSON (recorded): without the anchor the weighted fit is degenerate
  (unobserved directions are free) -> +8000-11000% PPL. Plain weight-space
  SVD works because it implicitly regularizes every direction.

  RUN 4 LESSONS (recorded):
  (a) Conv1D c_proj has a BIAS: captured outputs Y = WX + b. Fitting Y with
      a rank-limited W_hat while leaving b in the module double-counts the
      bias (capture sanity ||WX-Y||/||Y|| ~ 0.22). Subtract b before fitting.
  (b) Alternating subspace iteration DIVERGES (||M-W||2 grows 9.9 -> 23.6
      over 4 iters): the Z = M P G^{1/2} update refits only to the current
      solution's energy, ignoring C. Use the closed-form + weighted
      truncation instead.

Matched baseline reproduced in-run: plain weight-space SVD rank=256 (prior
measured +21.36%, phase_b_summary.json). Same model/dataset/eval protocol.
Calibration uses WikiText-2 TRAIN split (never the eval split).

Output: results/drift_aware_svd.json (new file; no prior artifact touched).
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANK = 256            # 3x compression of 768x768 c_proj
N_LAYERS = 12
D = 768
CALIB_CHUNKS = 64     # 64 x 512 tokens = 32768 token-columns
CHUNK_LEN = 512
RIDGE_FRAC = 0.01     # GPTQ-style damping: lambda = frac * mean(diag(G))
FIT_ITERS = 4         # (legacy; closed-form solver no longer iterates)
EPS_REL = 1e-3        # eigenspace restriction: keep eigvals > eps * max
N_CALIB = CALIB_CHUNKS * CHUNK_LEN

print(f"Device: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")

# ---------------- eval protocol (identical to prior runs) ----------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [item["text"].strip() for item in dataset if len(item["text"].strip()) > 50]
eval_texts = eval_texts[:50]

def compute_perplexity(m, texts, max_length=256):
    m.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
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
    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_loss / total_tokens))

def svd_reconstruct(W, rank):
    W = W.float()
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]

# ---------------- calibration data (train split only) ----------------
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
tok = tokenizer("\n\n".join(train_texts), return_tensors="pt")
ids = tok["input_ids"][0]
calib_chunks = [ids[i * CHUNK_LEN:(i + 1) * CHUNK_LEN]
                for i in range(CALIB_CHUNKS)]
print(f"Calibration: {len(calib_chunks)} chunks x {CHUNK_LEN} tokens (train split)")

# ---------------- activation capture ----------------
def collect_cproj_io(m):
    """Return per-layer (inputs, outputs) of attn.c_proj over calib chunks."""
    ios = {i: ([], []) for i in range(N_LAYERS)}
    handles = []
    for i in range(N_LAYERS):
        def mk_hook(li):
            def hook(mod, inp, out):
                # c_proj input: [1, T, 768] -> flatten tokens to columns
                ios[li][0].append(inp[0].detach().reshape(-1, D).t())
                ios[li][1].append(out.detach().reshape(-1, D).t())
            return hook
        handles.append(m.transformer.h[i].attn.c_proj.register_forward_hook(mk_hook(i)))
    with torch.no_grad():
        for c in calib_chunks:
            m(input_ids=c.unsqueeze(0).to(DEVICE))
    for h in handles:
        h.remove()
    return {i: (torch.cat(v, 1), torch.cat(o, 1)) for i, (v, o) in ios.items()}

def grams_and_cross(X_t, Y_t, X_s, alpha, W, beta, bias):
    """Blended Gram G and cross-correlation C with weight-space anchor beta.
    Bias is subtracted from outputs: the module keeps its bias after
    write-back, so only the linear part WX is fitted."""
    d = X_t.shape[0]
    n = X_t.shape[1]
    Y_t = Y_t - bias.unsqueeze(1)
    G_s = X_s @ X_s.t()
    C = Y_t @ X_s.t()
    if alpha < 1.0:
        G_t = X_t @ X_t.t()
        C_t = Y_t @ X_t.t()
        G = (1 - alpha) * G_t + alpha * G_s
        C = (1 - alpha) * C_t + alpha * C
    else:
        G = G_s
    # weight-space anchor: beta*n pseudo-observations of (identity, W)
    G = G + beta * n * torch.eye(d, device=G.device)
    C = C + beta * n * W
    return G, C

def direct_low_rank_fit(G, C, rank, iters=FIT_ITERS):
    """Solve min_{rank(M)<=r} tr(M G M^T) - 2 tr(M C).
    Closed form: full-rank optimum M* = C (G + lam I)^{-1}, then rank-r
    truncation in the G-weighted norm via SVD of M* G^{1/2}, restricted to
    the eigenspace of G with relative eigenvalue > EPS_REL (EoRA-style)."""
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > EPS_REL * evals.max()
    if keep.sum() < rank:   # anchor may lift all directions; fall back to full
        keep = torch.ones_like(keep, dtype=torch.bool)
    Q = evecs[:, keep]                       # [d, m]
    Gk = Q.t() @ G @ Q                       # [m, m]
    Ck = C @ Q                               # [d, m]
    lam = RIDGE_FRAC * Gk.diagonal().mean()
    evals_k, evecs_k = torch.linalg.eigh(Gk)
    Gk_inv = evecs_k @ torch.diag(1.0 / (evals_k + lam)) @ evecs_k.t()
    Gk_half = evecs_k @ torch.diag((evals_k + lam).sqrt()) @ evecs_k.t()
    Mstar = Ck @ Gk_inv                      # full-rank optimum in eigenspace
    Z = Mstar @ Gk_half                      # G-weighted view
    Uz, _, _ = torch.linalg.svd(Z, full_matrices=False)
    r = min(rank, Uz.shape[1])
    P = Uz[:, :r].t() @ Mstar                # optimal rows inside left subspace
    return Uz[:, :r] @ P @ Q.t()             # map back to full space

# ---------------- run variants ----------------
baseline_ppl = compute_perplexity(model, eval_texts)
print(f"Baseline PPL: {baseline_ppl:.2f}\n")

results = {"baseline_ppl": baseline_ppl, "rank": RANK, "variants": {}}

def eval_variant(name, build_fn):
    t0 = time.time()
    m_sub = copy.deepcopy(model)
    build_fn(m_sub)
    ppl = compute_perplexity(m_sub, eval_texts)
    delta = (ppl - baseline_ppl) / baseline_ppl * 100
    del m_sub
    torch.cuda.empty_cache()
    results["variants"][name] = {"ppl": ppl, "ppl_delta_pct": delta,
                                 "runtime_s": round(time.time() - t0, 1)}
    print(f"  {name:24s} PPL={ppl:8.2f}  delta={delta:+7.2f}%  ({time.time()-t0:.0f}s)")

# Variant A: matched control -- plain weight-space SVD (prior +21.36%)
def build_plain_svd(m_sub):
    for i in range(N_LAYERS):
        W = m_sub.transformer.h[i].attn.c_proj.weight.data
        m_sub.transformer.h[i].attn.c_proj.weight.data = svd_reconstruct(W, RANK)
print("Variant A: plain SVD (matched control)")
eval_variant("plain_svd", build_plain_svd)

# Variants B-D: sequential drift-aware direct weighted low-rank fit
teacher_ios = collect_cproj_io(model)

def make_sequential(alpha, beta):
    def build(m_sub):
        for i in range(N_LAYERS):
            X_t, Y_t = teacher_ios[i]
            W = model.transformer.h[i].attn.c_proj.weight.data.float().t()  # map M = W^T
            bias = model.transformer.h[i].attn.c_proj.bias.data.float()
            if i == 0:
                X_s = X_t  # no upstream compression yet
            else:
                student_ios = collect_cproj_io(m_sub)
                X_s = student_ios[i][0]
            G, C = grams_and_cross(X_t, Y_t, X_s, alpha, W, beta, bias)
            M_hat = direct_low_rank_fit(G, C, RANK)  # column-space map: y_col = M x_col
            # GPT-2 Conv1D forward is x_row @ weight, so weight = M^T
            m_sub.transformer.h[i].attn.c_proj.weight.data = M_hat.t().to(m_sub.transformer.h[i].attn.c_proj.weight.dtype)
            del X_s, G, C
            torch.cuda.empty_cache()
    return build

for alpha, beta in [(0.0, 0.3), (0.0, 1.0), (0.0, 3.0), (1.0, 0.3), (1.0, 1.0), (1.0, 3.0)]:
    name = f"drift_aware_a{alpha}_b{beta}"
    print(f"Variant: sequential drift-aware direct fit alpha={alpha} beta={beta}")
    eval_variant(name, make_sequential(alpha, beta))

# ---------------- save ----------------
out_path = Path("results/drift_aware_svd_sweep2.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
for name, r in results["variants"].items():
    print(f"  {name:24s} {r['ppl_delta_pct']:+.2f}%")

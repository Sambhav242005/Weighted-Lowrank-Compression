"""
HYBRID RESIDUAL SPLIT (SVDQuant-style) on Qwen2.5-7B o_proj
================================================================
Per-matrix hybrid: W_hat = U @ V + Q_int4(W - U @ V)
  - low-rank branch (fp16 factors) absorbs the structure/outliers
  - residual is group-wise symmetric int4-quantized (group=128, input dim)
  - branch rank r=256 (small, as in SVDQuant), NOT the aggressive r=1194

One-variable test: weighted-fit branch vs plain-SVD branch at IDENTICAL
storage. Reference: pure int4 on o_proj (cheaper storage).

Per-matrix storage (d=3584, r=256, g=128):
  branch fp16 : 2*r*d*2B          =  3.67 MB
  residual    : d*d*0.5B + scales =  6.62 MB
  hybrid total                      = 10.29 MB -> 2.50x  (vs 25.69 MB dense)
  pure int4                         =  6.62 MB -> 3.88x

Decision rule:
  weighted_branch < svd_branch (matched storage)  -> our fit is the better branch
  weighted_branch <= pure_int4                    -> hybrid beats cheaper option

Matched protocol: fp16 device_map=auto, WikiText-2 TEST 50 texts / TRAIN
16x512 calib, alpha=0 beta=0.1. Output: results/hybrid_residual_split.json
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import set_module_tensor_to_device
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RIDGE_FRAC, EPS_REL, BETA = 0.01, 1e-3, 0.1
HIDDEN, N_LAYERS = 3584, 28
BRANCH_RANK, GROUP, BITS = 256, 128, 4
N_CHUNKS, CHUNK_LEN = 16, 512
print(f"Device: {torch.cuda.get_device_name(0)}")

SNAP = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots").iterdir())[0]
with open(SNAP / "model.safetensors.index.json") as f:
    WEIGHT_MAP = json.load(f)["weight_map"]

def read_weight(name):
    with safe_open(SNAP / WEIGHT_MAP[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()

tok = AutoTokenizer.from_pretrained(MODEL)
print("Loading model (fp16, device_map=auto)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto").eval()
print(f"Loaded in {time.time()-t0:.0f}s")

def write_weight(name, tensor):
    li = int(name.split(".")[2])
    anchor = model.model.layers[li].input_layernorm.weight.device
    device = anchor if anchor.type != "meta" else torch.device("cpu")
    set_module_tensor_to_device(model, name, device,
                                value=tensor.to(torch.float16), dtype=torch.float16)

test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
eval_texts = [t.strip() for t in test_ds["text"] if len(t.strip()) > 50][:50]
train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * CHUNK_LEN:(i + 1) * CHUNK_LEN] for i in range(N_CHUNKS)]

def compute_perplexity(m, max_length=256):
    m.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for text in eval_texts:
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=max_length)
            inputs = {k: v.to(next(m.parameters()).device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tok = int(labels[:, 1:].ne(-100).sum().item())
            if n_tok == 0:
                continue
            out = m(**inputs, labels=labels)
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
    return float(np.exp(total_loss / total_tokens)) if total_tokens else float("inf")

print("Baseline PPL...")
t0 = time.time()
base = compute_perplexity(model)
print(f"Baseline PPL: {base:.2f}  ({time.time()-t0:.0f}s)")

# ------------- single-pass calibration capture -------------
print("Capturing calibration IO for all 28 o_proj...")
store = {i: ([], []) for i in range(N_LAYERS)}
handles = []
for i in range(N_LAYERS):
    def mk_hook(li):
        def hook(_m, inp, out):
            store[li][0].append(inp[0].detach().reshape(-1, HIDDEN).t().cpu())
            store[li][1].append(out.detach().reshape(-1, HIDDEN).t().cpu())
        return hook
    handles.append(model.model.layers[i].self_attn.o_proj.register_forward_hook(mk_hook(i)))
with torch.no_grad():
    for c in chunks:
        model(input_ids=c.unsqueeze(0).to(next(model.parameters()).device))
for h in handles:
    h.remove()
print("Captured.")

# ------------- quantizer + fits -------------
def quant_groups(X, bits=BITS, group=GROUP):
    """Group-wise symmetric int quantize + dequantize along input dim."""
    qmax = 2 ** (bits - 1) - 1
    out_d, in_d = X.shape
    Xg = X.reshape(out_d, in_d // group, group)
    scale = Xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(Xg / scale).clamp(-qmax, qmax)
    return (q * scale).reshape(out_d, in_d)

def weighted_factors(G, C, rank, device):
    G, C = G.to(device), C.to(device)
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
    U, V = Uz[:, :r], (Uz[:, :r].t() @ Mstar) @ Q.t()
    return U.cpu(), V.cpu()

def svd_factors(M, rank):
    U, S, Vt = torch.linalg.svd(M.to("cuda"), full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return U[:, :k].cpu(), (torch.diag(S[:k]) @ Vt[:k, :]).cpu()

orig_w, variants = {}, {}
print(f"Fitting all 28 o_proj (branch rank {BRANCH_RANK})...")
for i in range(N_LAYERS):
    M = read_weight(f"model.layers.{i}.self_attn.o_proj.weight")
    orig_w[i] = M.clone()
    X = torch.cat(store[i][0], 1).float()
    Y = torch.cat(store[i][1], 1).float()
    n = X.shape[1]
    G = X @ X.t() + BETA * n * torch.eye(HIDDEN)
    C = Y @ X.t() + BETA * n * M

    # reference: pure int4
    variants.setdefault("pure_int4", {})[i] = quant_groups(M)

    # hybrid with SVD branch (control)
    Us, Vs = svd_factors(M, BRANCH_RANK)
    variants.setdefault("hybrid_svd_branch", {})[i] = (
        Us @ Vs + quant_groups(M - Us @ Vs))

    # hybrid with weighted-fit branch (candidate)
    Uw, Vw = weighted_factors(G, C, BRANCH_RANK, "cuda")
    variants.setdefault("hybrid_weighted_branch", {})[i] = (
        Uw @ Vw + quant_groups(M - Uw @ Vw))

    del X, Y, G, C, Us, Vs, Uw, Vw
    torch.cuda.empty_cache()
del store
print("Fits done.")

results = {"baseline_ppl": base, "branch_rank": BRANCH_RANK,
           "residual": f"int{BITS} symmetric, group={GROUP}", "variants": {}}

def restore():
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", orig_w[i])

for key in ("pure_int4", "hybrid_svd_branch", "hybrid_weighted_branch"):
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", variants[key][i])
    t0 = time.time()
    ppl = compute_perplexity(model)
    delta = (ppl - base) / base * 100
    restore()
    results["variants"][key] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {key:24s} PPL={ppl:9.2f} delta={delta:+8.2f}%  ({time.time()-t0:.0f}s)")

# ------------- storage accounting (per matrix) -------------
d, r, g = HIDDEN, BRANCH_RANK, GROUP
dense = d * d * 2
hybrid = (2 * r * d * 2) + (d * d * 0.5 + d * (d // g) * 2)
pure = d * d * 0.5 + d * (d // g) * 2
results["storage_per_matrix_MB"] = {
    "dense_fp16": round(dense / 1e6, 2),
    "hybrid": round(hybrid / 1e6, 2),
    "pure_int4": round(pure / 1e6, 2),
    "hybrid_ratio": round(dense / hybrid, 2),
    "pure_int4_ratio": round(dense / pure, 2),
}
d_svd = results["variants"]["hybrid_svd_branch"]["ppl_delta_pct"]
d_wt = results["variants"]["hybrid_weighted_branch"]["ppl_delta_pct"]
d_q4 = results["variants"]["pure_int4"]["ppl_delta_pct"]
results["decision"] = {
    "weighted_beats_svd_branch": bool(d_wt < d_svd),
    "weighted_beats_pure_int4": bool(d_wt < d_q4),
}
print(f"\nDecision: weighted>sVD-branch={results['decision']['weighted_beats_svd_branch']}, "
      f"weighted>pure_int4={results['decision']['weighted_beats_pure_int4']}")

out_path = Path("results/hybrid_residual_split.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")

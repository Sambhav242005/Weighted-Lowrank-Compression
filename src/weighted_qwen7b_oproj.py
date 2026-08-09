"""
7B VALIDATION: Qwen2.5-7B-Instruct o_proj, weighted fit vs plain SVD, fp16
===========================================================================
Pre-registered decision rule (from redundancy-vs-drift analysis):
  delta <= 10%  -> supports redundancy hypothesis (compression scales)
  delta >= 20%  -> falsifies scaling trend
  10-20%        -> inconclusive

Protocol matched to prior runs: WikiText-2 TEST 50 texts >50 chars,
max_length 256; calibration WikiText-2 TRAIN 16x512 tokens; targets = all
28 o_proj (hidden 3584), rank 1194 (one-sided 3x); winner config alpha=0,
beta=0.1, closed-form + G-weighted truncation.

fp16 + device_map="auto" (15GB weights > 12GB VRAM -> partial CPU offload).
In-place weight swap with CPU backup (no deepcopy of a 15GB model).

Run contract:
  model: Qwen/Qwen2.5-7B-Instruct rev a09a35458c702b33eeacc393d103063234e8bc28
  dtype: fp16 | device: cuda:0 + cpu offload, device_map=auto default budget
  RUN 1 NOTE: explicit max_memory {11GiB GPU, 16GiB CPU} left layers on the
  meta device -> 'Cannot copy out of meta tensor' at fit time. Removed caps.
  dataset: wikitext-2-raw-v1 (train calib / test eval)

Output: results/weighted_qwen7b_oproj.json (new file; nothing overwritten).
"""
import json, time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import set_module_tensor_to_device
from datasets import load_dataset

print(f"Device: {torch.cuda.get_device_name(0)}")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
RIDGE_FRAC, EPS_REL, BETA = 0.01, 1e-3, 0.1
HIDDEN, N_LAYERS = 3584, 28
RANK = HIDDEN // 3
N_CHUNKS, CHUNK_LEN = 16, 512

# safetensors index for direct weight reads (offloaded layers report meta)
SNAP = sorted((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots").iterdir())[0]
with open(SNAP / "model.safetensors.index.json") as f:
    WEIGHT_MAP = json.load(f)["weight_map"]

def read_weight(name):
    """Read a weight from the safetensors shard directly (bypasses accelerate
    meta placeholders left by CPU offload)."""
    with safe_open(SNAP / WEIGHT_MAP[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()

def write_weight(name, tensor):
    """Write a weight into the dispatched model, hook-safe for offloaded layers.
    Places it where the layer actually lives (GPU layer vs CPU-offloaded)."""
    li = int(name.split(".")[2])
    anchor = model.model.layers[li].input_layernorm.weight.device
    device = anchor if anchor.type != "meta" else torch.device("cpu")
    set_module_tensor_to_device(model, name, device,
                                value=tensor.to(torch.float16), dtype=torch.float16)

tok = AutoTokenizer.from_pretrained(MODEL)
print("Loading model (fp16, device_map=auto)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto").eval()
print(f"Loaded in {time.time()-t0:.0f}s")

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
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)
            n_tok = int(labels[:, 1:].ne(-100).sum().item())
            if n_tok == 0:
                continue
            out = m(**inputs, labels=labels)
            total_loss += out.loss.item() * n_tok
            total_tokens += n_tok
    return float(np.exp(total_loss / total_tokens)) if total_tokens else float("inf")

print("Baseline PPL (this is slow with offload)...")
t0 = time.time()
base = compute_perplexity(model)
print(f"Baseline PPL: {base:.2f}  ({time.time()-t0:.0f}s)")

# ------------- single-pass calibration capture on ALL o_proj -------------
print("Capturing calibration IO for all 28 o_proj...")
store = {i: ([], []) for i in range(N_LAYERS)}
handles = []
for i in range(N_LAYERS):
    def mk_hook(li):
        def hook(_m, inp, out):
            # keep fp16 on CPU to bound memory (cast to fp32 when fitting)
            store[li][0].append(inp[0].detach().reshape(-1, HIDDEN).t().cpu())
            store[li][1].append(out.detach().reshape(-1, HIDDEN).t().cpu())
        return hook
    handles.append(model.model.layers[i].self_attn.o_proj.register_forward_hook(mk_hook(i)))
t0 = time.time()
with torch.no_grad():
    for c in chunks:
        model(input_ids=c.unsqueeze(0).to(next(model.parameters()).device))
for h in handles:
    h.remove()
print(f"Captured in {time.time()-t0:.0f}s")

# ------------- per-layer fits -------------
def weighted_low_rank_fit(G, C, rank, device):
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
    M_hat = Uz[:, :r] @ (Uz[:, :r].t() @ Mstar) @ Q.t()
    return M_hat.cpu()

def svd_reconstruct(M, rank):
    M = M.to("cuda")
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()

svd_w, weighted_w = {}, {}
orig_w = {}
print("Fitting all 28 o_proj (SVD + weighted)...")
for i in range(N_LAYERS):
    name = f"model.layers.{i}.self_attn.o_proj.weight"
    mod = model.model.layers[i].self_attn.o_proj
    M = read_weight(name)                                 # fp32 CPU, meta-safe
    orig_w[i] = M.clone()
    svd_w[i] = svd_reconstruct(M, RANK)
    X = torch.cat(store[i][0], 1).float()
    Y = torch.cat(store[i][1], 1).float()
    if getattr(mod, "bias", None) is not None:
        Y = Y - mod.bias.data.detach().float().cpu().unsqueeze(1)
    n = X.shape[1]
    G = X @ X.t() + BETA * n * torch.eye(HIDDEN)
    C = Y @ X.t() + BETA * n * M
    weighted_w[i] = weighted_low_rank_fit(G, C, RANK, "cuda")
    del X, Y, G, C
    torch.cuda.empty_cache()
del store
print("Fits done.")

results = {"baseline_ppl": base, "rank": RANK, "variants": {}}

def apply(weights):
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", weights[i])

def restore():
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", orig_w[i])

for name, weights in (("plain_svd", svd_w), ("weighted_b0.1", weighted_w)):
    apply(weights)
    t0 = time.time()
    ppl = compute_perplexity(model)
    delta = (ppl - base) / base * 100
    restore()
    results["variants"][name] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {name:16s} PPL={ppl:9.2f} delta={delta:+8.2f}%  ({time.time()-t0:.0f}s)")

d_svd = results["variants"]["plain_svd"]["ppl_delta_pct"]
d_wt = results["variants"]["weighted_b0.1"]["ppl_delta_pct"]
results["decision_rule"] = {
    "weighted_delta_pct": d_wt,
    "gate": "<=10% supports redundancy / >=20% falsifies / else inconclusive",
    "verdict": ("supports" if d_wt <= 10 else
                "falsifies" if d_wt >= 20 else "inconclusive")}
print(f"\nVerdict vs decision rule: {results['decision_rule']['verdict']}")

out_path = Path("results/weighted_qwen7b_oproj.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")

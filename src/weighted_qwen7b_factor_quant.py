"""
FACTORS-QUANTIZED HYBRID: Qwen2.5-7B o_proj, weighted fit rank 1194 (3x),
then symmetric per-row quantization of the factors U, V (int8 / int4).
================================================================================
Question: does quantizing the low-rank factors preserve the weighted fit's
quality, and what storage does the hybrid reach?

Storage per square d x d matrix at rank r = d/3:
  fp16 factors: 2*r*2d params*2B -> 1.5x | int8 factors -> 3x | int4 -> 6x
Whole-model: o_proj = 5.09% of Qwen2.5-7B params.

Matched protocol (same as weighted_qwen7b_oproj.py): fp16 device_map=auto,
WikiText-2 TEST 50 texts max_length 256, calib TRAIN 16x512, alpha=0
beta=0.1 closed-form. Offloaded weights handled via safetensors reads +
set_module_tensor_to_device.

Output: results/weighted_qwen7b_factor_quant.json (new file).
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
RANK = HIDDEN // 3
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

# ------------- single-pass calibration capture on ALL o_proj -------------
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
t0 = time.time()
with torch.no_grad():
    for c in chunks:
        model(input_ids=c.unsqueeze(0).to(next(model.parameters()).device))
for h in handles:
    h.remove()
print(f"Captured in {time.time()-t0:.0f}s")

# ------------- fits -------------
def weighted_low_rank_factors(G, C, rank, device):
    """Return (U, V) factors: M_hat = U @ V, rank-r closed-form weighted fit."""
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
    Uz, S_, _ = torch.linalg.svd(Mstar @ Gk_half, full_matrices=False)
    r = min(rank, Uz.shape[1])
    U = Uz[:, :r]
    V = (U.t() @ Mstar) @ Q.t()
    return U.cpu(), V.cpu()

def svd_reconstruct(M, rank):
    M = M.to("cuda")
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()

def quant_deq(X, bits):
    """Symmetric per-row quantize + dequantize."""
    qmax = 2 ** (bits - 1) - 1
    scale = X.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(X / scale).clamp(-qmax, qmax) * scale

orig_w, variants = {}, {}
print("Fitting all 28 o_proj...")
for i in range(N_LAYERS):
    name = f"model.layers.{i}.self_attn.o_proj.weight"
    mod = model.model.layers[i].self_attn.o_proj
    M = read_weight(name)
    orig_w[i] = M.clone()
    X = torch.cat(store[i][0], 1).float()
    Y = torch.cat(store[i][1], 1).float()
    if getattr(mod, "bias", None) is not None:
        Y = Y - mod.bias.data.detach().float().cpu().unsqueeze(1)
    n = X.shape[1]
    G = X @ X.t() + BETA * n * torch.eye(HIDDEN)
    C = Y @ X.t() + BETA * n * M
    U, V = weighted_low_rank_factors(G, C, RANK, "cuda")
    for key, Mhat in (
        ("weighted_fp16f", U @ V),
        ("weighted_int8f", quant_deq(U, 8) @ quant_deq(V, 8)),
        ("weighted_int4f", quant_deq(U, 4) @ quant_deq(V, 4)),
    ):
        variants.setdefault(key, {})[i] = Mhat
    variants.setdefault("plain_svd", {})[i] = svd_reconstruct(M, RANK)
    del X, Y, G, C, U, V
    torch.cuda.empty_cache()
del store
print("Fits done.")

results = {"baseline_ppl": base, "rank": RANK, "variants": {}}

def restore():
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", orig_w[i])

for key in ("plain_svd", "weighted_fp16f", "weighted_int8f", "weighted_int4f"):
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", variants[key][i])
    t0 = time.time()
    ppl = compute_perplexity(model)
    delta = (ppl - base) / base * 100
    restore()
    results["variants"][key] = {"ppl": ppl, "ppl_delta_pct": delta}
    print(f"  {key:18s} PPL={ppl:9.2f} delta={delta:+8.2f}%  ({time.time()-t0:.0f}s)")

# ------------- storage accounting -------------
d, r = HIDDEN, RANK
orig_bytes = d * d * 2
def factor_bytes(bits):  # codes + fp16 per-row scales (negligible)
    codes = r * 2 * d * bits / 8
    scales = (r + d) * 2
    return codes + scales
results["storage"] = {
    "per_matrix_bytes_orig_fp16": orig_bytes,
    "fp16_factors": {"per_matrix_ratio": round(orig_bytes / factor_bytes(16), 2)},
    "int8_factors": {"per_matrix_ratio": round(orig_bytes / factor_bytes(8), 2)},
    "int4_factors": {"per_matrix_ratio": round(orig_bytes / factor_bytes(4), 2)},
    "o_proj_share_of_model_pct": 5.09,
}

out_path = Path("results/weighted_qwen7b_factor_quant.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")

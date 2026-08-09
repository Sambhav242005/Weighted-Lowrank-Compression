"""
GENERATION COMPARISON: Qwen2.5-7B baseline vs compressed o_proj variants.
================================================================================
Three solvable prompts (arithmetic word problem, factual QA, coding task),
greedy decoding via the chat template, across 4 variants:
  baseline | weighted int8 factors (+3.80%) | weighted int4 factors (+8.22%)
  | plain SVD (+26.31%, the failing control)

Same fit/protocol as weighted_qwen7b_factor_quant.py (fp16 device_map=auto,
WikiText-2 TRAIN calib 16x512, rank 1194, alpha=0 beta=0.1). Autoregressive
generation is slow under CPU offload (~1.4s/token) -> expect ~25-40 min.

Output: results/compare_qwen7b_output.json (new file).
"""
import json, time
from pathlib import Path
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
MAX_NEW_TOKENS = 100

PROMPTS = [
    "A farmer has 3 fields. Field A yields 240 kg of wheat per hectare, "
    "Field B yields 310 kg per hectare, and Field C yields 185 kg per hectare. "
    "If the farmer plants 4 hectares of A, 3 hectares of B and 5 hectares of C, "
    "how many kg of wheat does he harvest in total? Show your reasoning briefly.",
    "Why is the sky blue? Answer in two or three sentences.",
    "Write a Python function `is_palindrome(s)` that returns True if the "
    "string s reads the same forwards and backwards (ignore case and "
    "non-alphanumeric characters). Then show one example call.",
]

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

train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = [t.strip() for t in train_ds["text"] if len(t.strip()) > 100]
ids = tok("\n\n".join(train_texts), return_tensors="pt")["input_ids"][0]
chunks = [ids[i * CHUNK_LEN:(i + 1) * CHUNK_LEN] for i in range(N_CHUNKS)]

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

# ------------- fits -------------
def weighted_low_rank_factors(G, C, rank, device):
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
    return Uz[:, :r].cpu(), (Uz[:, :r].t() @ Mstar @ Q.t()).cpu()

def quant_deq(X, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = X.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(X / scale).clamp(-qmax, qmax) * scale

def svd_reconstruct(M, rank):
    M = M.to("cuda")
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, U.shape[1], Vt.shape[0])
    return (U[:, :k] @ torch.diag(S[:k]) @ Vt[:k, :]).cpu()

orig_w, variants = {}, {}
print("Fitting all 28 o_proj...")
for i in range(N_LAYERS):
    mod = model.model.layers[i].self_attn.o_proj
    M = read_weight(f"model.layers.{i}.self_attn.o_proj.weight")
    orig_w[i] = M.clone()
    X = torch.cat(store[i][0], 1).float()
    Y = torch.cat(store[i][1], 1).float()
    n = X.shape[1]
    G = X @ X.t() + BETA * n * torch.eye(HIDDEN)
    C = Y @ X.t() + BETA * n * M
    U, V = weighted_low_rank_factors(G, C, RANK, "cuda")
    variants.setdefault("weighted_int8f", {})[i] = quant_deq(U, 8) @ quant_deq(V, 8)
    variants.setdefault("weighted_int4f", {})[i] = quant_deq(U, 4) @ quant_deq(V, 4)
    variants.setdefault("plain_svd", {})[i] = svd_reconstruct(M, RANK)
    del X, Y, G, C, U, V
    torch.cuda.empty_cache()
del store
print("Fits done.")

# ------------- generation -------------
def generate(prompt):
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

def restore():
    for i in range(N_LAYERS):
        write_weight(f"model.layers.{i}.self_attn.o_proj.weight", orig_w[i])

results = {"prompts": PROMPTS, "max_new_tokens": MAX_NEW_TOKENS,
           "decoding": "greedy, chat template", "variants": {}}

order = [("baseline", None), ("weighted_int8f", "weighted_int8f"),
         ("weighted_int4f", "weighted_int4f"), ("plain_svd", "plain_svd")]
for label, key in order:
    print(f"\n===== {label} =====")
    if key:
        for i in range(N_LAYERS):
            write_weight(f"model.layers.{i}.self_attn.o_proj.weight", variants[key][i])
    gens = []
    for p in PROMPTS:
        t0 = time.time()
        g = generate(p)
        gens.append(g)
        print(f"  [prompt {gens.index(g)+1}, {time.time()-t0:.0f}s] {g[:120]}...")
    if key:
        restore()
    results["variants"][label] = gens
    # incremental save so per-variant results survive even if a later run dies
    Path("results/compare_qwen7b_output.json").write_text(
        json.dumps(results, indent=2))
    print(f"  [saved incrementally after {label}]")

out_path = Path("results/compare_qwen7b_output.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")

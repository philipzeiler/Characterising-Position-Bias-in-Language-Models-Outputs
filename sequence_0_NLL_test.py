"""
Sliding-window NLL on Pile-validation doc 0
==========================================

• Loads Pythia-1.4B (FP16 weights, FP32 logits)
• Downloads/streams pietrolesci/pile-validation
• Builds an (len(doc), 2048) matrix where entry [i, j] is
  the NLL of token i when it sat at position j (0-based, left=0)
• Saves the matrix in HDF5 for later analysis
"""

import os, random, numpy as np, torch, h5py, torch.nn.functional as F
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ─── Reproducibility ────────────────────────────────────────────────────────
seed = 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.deterministic    = True
torch.backends.cudnn.benchmark        = False

CTX = 2048                       # Pythia context
MODEL_ID   = "EleutherAI/pythia-1.4b-deduped"
REVISION   = "step143000"

# ─── Load model & tokenizer (FP16 → FP32 logits) ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION,
            torch_dtype=torch.float16).to(device).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

print(f"Model loaded on {device}, dtype {next(model.parameters()).dtype}")

# ─── Load validation split ──────────────────────────────────────────────────
val_ds = load_dataset(
    "pietrolesci/pile-validation",
    split="validation",
    verification_mode="no_checks"   # <- replace old kw-arg
)
print("Validation documents:", len(val_ds))      # ~214 k
print(val_ds[0]["text"][:120], "…")              # sanity peek



# ─── Pick doc 0 and tokenise -------------------------------------------------
doc0_text = val_ds[0]["text"]
doc0_ids  = tok(doc0_text, return_tensors="pt").input_ids[0]
L = len(doc0_ids)
print(f"Doc-0 length: {L} tokens")

# ─── Prepare output matrix ---------------------------------------------------
nll_mat = np.full((L, CTX), np.nan, dtype=np.float32)

# ─── Helper: compute NLL for a tensor of IDs ---------------------------------
def nll_for_seq(seq_ids: torch.Tensor) -> np.ndarray:
    """
    seq_ids: shape (ctx_len,) on device
    Returns per-token negative log-likelihood (next-token)
             as 1-d numpy array length ctx_len-1
    """
    with torch.inference_mode():
        logits = model(seq_ids[:-1][None]).logits.float()
        logp   = F.log_softmax(logits, dim=-1)
    tgt = seq_ids[1:]
    nll = (-logp[0, torch.arange(len(tgt)), tgt]).cpu().numpy()
    return nll  # len = ctx_len-1

# ─── Sliding window evaluation ----------------------------------------------
for right_edge in range(1, L + 1):
    left_edge  = max(0, right_edge - CTX)
    window_ids = doc0_ids[left_edge:right_edge]

    pad_len = CTX - len(window_ids)
    if pad_len > 0:
        rand_idx   = random.randint(0, len(val_ds) - 1)
        rand_ids   = tok(val_ds[rand_idx]["text"], return_tensors="pt").input_ids[0]
        buffer_ids = rand_ids[-pad_len:] if len(rand_ids) >= pad_len else \
                     torch.cat([rand_ids,
                                tok.eos_token_id * torch.ones(pad_len-len(rand_ids), dtype=torch.long)])
        window_ids = torch.cat([buffer_ids, window_ids])
        print(f"[shift {right_edge:6d}] pad {pad_len:4d} ← doc {rand_idx}")

    window_ids = window_ids.to(device)
    nll_vec = nll_for_seq(window_ids)        # length 2047

    # write every real token’s NLL to the matrix
    for k in range(1, CTX):                  # skip k==0 (no target)
        if k <= pad_len:                     # skip buffer
            continue
        g_idx = left_edge + k
        if g_idx < L:
            nll_mat[g_idx, k] = nll_vec[k-1]   # store at column k

# ─── Save as HDF5 ------------------------------------------------------------
os.makedirs("nll_matrices", exist_ok=True)
h5_path = "nll_matrices/doc0_nll_ctx2048.h5"
with h5py.File(h5_path, "w") as h5f:
    h5f.create_dataset("nll", data=nll_mat, compression="gzip")
print("Matrix saved to", h5_path)

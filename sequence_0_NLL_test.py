"""
Sliding-window NLL for Pile-validation document 0
=================================================

• Source data:  pietrolesci/pile-validation  (471 MB parquet, no train split)
• Model:        EleutherAI/pythia-1.4b-deduped  – FP16 weights, FP32 logits
• Task:         For every token *t* in doc-0 and every context position p∈[0,2047],
                compute NLL(t | ctx position p).  Buffer on the left is filled
                with the tail of a *random* validation document and discarded
                from the final matrix.
• Storage:      HDF5 file  nll_matrices/doc0_ctx2048.h5   dataset name="nll"
"""

import os, random, numpy as np, torch, h5py, torch.nn.functional as F
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ─── 0.  Reproducibility & deterministic GPU maths ──────────────────────────
seed = 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

CTX = 2048
MODEL_ID   = "EleutherAI/pythia-1.4b-deduped"
REVISION   = "step143000"

# ─── 1.  Load model & tokenizer ─────────────────────────────────────────────
dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (GPTNeoXForCausalLM
         .from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float16)
         .to(dev).eval())
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] Model on {dev}, param dtype = {next(model.parameters()).dtype}")

# ─── 2.  Load Pile-validation (no integrity checks needed) ──────────────────
val_ds = load_dataset("pietrolesci/pile-validation",
                      split="validation",
                      verification_mode="no_checks")    # :contentReference[oaicite:0]{index=0}
print("[INFO] Validation rows:", len(val_ds))

# ─── 3.  Grab document 0 and tokenise ---------------------------------------
doc0 = val_ds[0]["text"]
ids  = tok(doc0, return_tensors="pt").input_ids[0]
L    = len(ids)
print(f"[INFO] Doc-0 length = {L} tokens")

# ─── 4.  Prepare output matrix  (L × 2 048) ---------------------------------
nll_mat = np.full((L, CTX), np.nan, dtype=np.float32)

# helper ---------------------------------------------------------
def nll_for_seq(seq: torch.Tensor) -> torch.Tensor:
    """Return per-token NLL (len = len(seq)-1) on *device*."""
    with torch.inference_mode():
        logits = model(seq[:-1][None]).logits.float()
        logp   = F.log_softmax(logits, -1)
    nll = -logp[0, torch.arange(seq.size(0)-1, device=seq.device), seq[1:]]
    return nll                       # 1-D tensor, dtype FP32
# ----------------------------------------------------------------

# ─── 5.  Sliding-window evaluation  (2 048 shifts) ---------------------------
for pos_last in range(CTX):                             # 0 … 2047
    pos_first = pos_last - (L - 1)                      # may be negative
    pad_len   = max(0, -pos_first)

    # take tail-slice of doc-0 that fits into current window
    tail_len   = min(pos_last + 1, L)
    tail_start = L - tail_len
    tail_ids   = ids[tail_start:]                       # shape (tail_len,)

    # left buffer from random document
    if pad_len:
        rand_j    = random.randint(0, len(val_ds) - 1)
        rand_ids  = tok(val_ds[rand_j]["text"], return_tensors="pt").input_ids[0]
        buf_ids   = rand_ids[-pad_len:] if len(rand_ids) >= pad_len else \
                    torch.cat([rand_ids,
                               tok.eos_token_id *
                               torch.ones(pad_len - len(rand_ids),
                                          dtype=torch.long)])
        window_ids = torch.cat([buf_ids, tail_ids])
        print(f"[shift {pos_last:4d}] pad {pad_len:4d}  ← doc {rand_j}")
    else:
        window_ids = tail_ids
        print(f"[shift {pos_last:4d}] pad    0   (doc fully inside window)")

    # sanity: ensure window length == CTX
    if window_ids.size(0) < CTX:
        # right-pad with BOS/EOS (won’t be used for learning)
        extras = torch.full((CTX - window_ids.size(0),),
                            tok.eos_token_id, dtype=torch.long)
        window_ids = torch.cat([extras, window_ids])

    # move to GPU, compute NLL
    window_ids = window_ids.to(dev)
    nll_vec    = nll_for_seq(window_ids)                # len 2047

    # ── print human-readable window (first 200 chars to stay sane) ─────────
    ctx_preview = tok.decode(window_ids[:200])
    print("   context-preview:", ctx_preview.replace("\n", " ")[:200], "…")

    # ── store every real token’s NLL in matrix ─────────────────────────────
    for k in range(pad_len, pad_len + tail_len):        # real tokens only
        g_idx   = (L - tail_len) + (k - pad_len)        # global doc index
        nll_mat[g_idx, k] = nll_vec[k-1].item()         # k-1 aligns to nll_vec

# ─── 6.  Save matrix as HDF5 -------------------------------------------------
os.makedirs("nll_matrices", exist_ok=True)
with h5py.File("nll_matrices/doc0_ctx2048.h5", "w") as f:
    f.create_dataset("nll", data=nll_mat, compression="gzip")
print("[INFO] Matrix written → nll_matrices/doc0_ctx2048.h5")

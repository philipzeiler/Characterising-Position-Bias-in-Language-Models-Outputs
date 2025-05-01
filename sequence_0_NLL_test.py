# ─────────────────────────────────────────────────────────────────────────────
#  Sliding-window NLL for Pile-validation document-0  (full 2 048+L-1 shifts)
# ─────────────────────────────────────────────────────────────────────────────
import os, random, numpy as np, torch, h5py, torch.nn.functional as F
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── 0. Determinism & math flags ─────────────────────────────────────────────
seed = 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)                # :contentReference[oaicite:0]{index=0}
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark  = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32      = False

CTX       = 2048
MODEL_ID  = "EleutherAI/pythia-1.4b-deduped"
REVISION  = "step143000"

# ── 1. Model & tokenizer ────────────────────────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.float16
        ).to(dev).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] model on {dev} – weights dtype {next(model.parameters()).dtype}")

# ── 2. Validation split ------------------------------------------------------
val_ds = load_dataset(
            "pietrolesci/pile-validation",
            split="validation",
            verification_mode="no_checks"      # replaces deprecated flag  :contentReference[oaicite:1]{index=1}
         )
print("[INFO] validation rows:", len(val_ds))

# ── 3. Document-0 ------------------------------------------------------------
doc0_ids = tok(val_ds[0]["text"], return_tensors="pt").input_ids[0]
L        = len(doc0_ids)
print(f"[INFO] doc-0 length = {L} tokens")

nll_mat = np.full((L, CTX), np.nan, dtype=np.float32)

# helper ----------------------------------------------------------------------
def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(seq_ids[:-1][None]).logits.float()
        lp     = F.log_softmax(logits, -1)
    tgt = seq_ids[1:]
    return -lp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]  # FP32
# -----------------------------------------------------------------------------

TOTAL_SHIFTS = CTX + L - 1                         # 2 048 + L − 1 shifts
for s in range(TOTAL_SHIFTS):
    # position of last doc token in the window
    pos_last = s
    pos_first = pos_last - (L - 1)
    pad_len   = max(0, -pos_first)

    # slice of doc-0 appearing in the window
    tail_len   = min(pos_last + 1, L)
    tail_start = L - tail_len
    tail_ids   = doc0_ids[tail_start:]

    # left buffer if needed
    if pad_len:
        r_idx   = random.randint(0, len(val_ds) - 1)
        r_ids   = tok(val_ds[r_idx]["text"], return_tensors="pt").input_ids[0]
        buf_ids = r_ids[-pad_len:] if len(r_ids) >= pad_len else \
                  torch.cat([r_ids,
                             tok.eos_token_id * torch.ones(pad_len-len(r_ids),
                                                           dtype=torch.long)])
        window  = torch.cat([buf_ids, tail_ids])
        print(f"[shift {s:5d}] pad {pad_len:4d} ← doc {r_idx}")
    else:
        window  = tail_ids
        print(f"[shift {s:5d}] pad    0  (doc fully inside window)")

    # right-pad with EOS to get exact CTX length
    if window.size(0) < CTX:
        window = torch.cat([
            torch.full((CTX-window.size(0),), tok.eos_token_id, dtype=torch.long),
            window
        ])

    # show full window compressed to ASCII (first 80 chars)
    ascii_preview = tok.decode(window).replace("\n", " ")[:80]
    print("          window:", ascii_preview, "…")

    # compute NLLs
    nll_vec = nll_for(window.to(dev))           # length CTX-1 (FP32 tensor)

    # fill matrix for real tokens only
    for k in range(pad_len, pad_len + tail_len):
        doc_idx = (L - tail_len) + (k - pad_len)
        nll_mat[doc_idx, k] = nll_vec[k-1].item()

# ── 4. Save ------------------------------------------------------------------
os.makedirs("nll_matrices", exist_ok=True)
with h5py.File("nll_matrices/doc0_ctx2048.h5", "w") as f:
    f.create_dataset("nll", data=nll_mat, compression="gzip")
print("[INFO] matrix saved to nll_matrices/doc0_ctx2048.h5")

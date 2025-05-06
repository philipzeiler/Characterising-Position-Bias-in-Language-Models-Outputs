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
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark  = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32      = False

CTX       = 2048
MODEL_ID  = "EleutherAI/pythia-1.4b"
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
            verification_mode="no_checks"      # replaces deprecated flag
         )
print("[INFO] validation rows:", len(val_ds))

# ── 2.a  Global‑snake settings ────────────────────────────────────────────
N_DOCS       = 1000          # process *this many* docs from the shuffled list
SHUFFLE_SEED = 123           # reproducible permutation of the 214 k rows

# shuffle reproducibly -------------------------------------------------------
doc_order = list(range(len(val_ds)))
rng = random.Random(SHUFFLE_SEED)
rng.shuffle(doc_order)

# limit to N_DOCS and build contiguous token stream -------------------------
stream_tokens = []
lengths       = []                 # keep each doc length for slicing later
for idx in doc_order[:N_DOCS]:
    ids = tok(val_ds[idx]["text"], return_tensors="pt").input_ids[0]
    stream_tokens.append(ids)
    lengths.append(len(ids))
stream = torch.cat(stream_tokens)   # shape (Σ lengths,)
print(f"[INFO] snake length = {len(stream):,} tokens "
      f"({N_DOCS} docs concatenated)")

# helper ----------------------------------------------------------------------
def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(seq_ids[:-1][None]).logits.float()
        lp     = F.log_softmax(logits, -1)
    tgt = seq_ids[1:]
    return -lp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]  # FP32
# -----------------------------------------------------------------------------

# copy of "helper" code, slightly adjusted by tiago to illustrate changes that I need to make in order to do "Batching" (implement later)
#def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
#    with torch.inference_mode():
#        seq_ids[np.new_axis(),:].repeat(5, 1)
#        logits = model(seq_ids[:, :-1][None]).logits.float()
#        lp     = F.log_softmax(logits, -1)
#    tgt = seq_ids[:, 1:]
#    return -lp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]  # FP32
# -----------------------------------------------------------------------------

# ── 3. slide the full stream -------------------------------------------------
# ── 3. slide the full stream -------------------------------------------------
cursor = 0
for doc_i, doc_len in enumerate(lengths):
    nll_mat = np.full((doc_len, CTX), np.nan, dtype=np.float32)

    for local_pos in range(doc_len + CTX - 1):
        global_pos_last = cursor + local_pos
        w_start = max(0, global_pos_last - CTX + 1)
        window  = stream[w_start:global_pos_last + 1]

        # pad on the left if the window is still shorter than CTX
        if window.size(0) < CTX:
            pad = torch.full((CTX - window.size(0),),
                             tok.eos_token_id, dtype=torch.long)
            window = torch.cat([pad, window])

        # ── NEW: human‑readable preview (first 120 decoded chars) ──────────
        preview = tok.decode(window[:120]).replace("\n", " ")
        print(f"[doc {doc_i:5d} | shift {local_pos:5d}] "
              f"left‑edge 0‑119: {preview}")

        # ---------- NLL computation unchanged ------------------------------
        nll_vec = nll_for(window.to(dev))

        # store NLLs for tokens belonging to *this* doc only
        token_global_idx = global_pos_last
        token_local_idx  = token_global_idx - cursor
        if 0 <= token_local_idx < doc_len:
            col = CTX - 1
            nll_mat[token_local_idx, col] = nll_vec[col-1].item()
            back = min(col, token_local_idx)
            if back:
                nll_mat[token_local_idx-back:token_local_idx,
                        col-back:col] = nll_vec[col-back-1:col-1].cpu().numpy()
    # … saving code unchanged …


    # save matrix for this document -----------------------------------------
    out_dir = "nll_matrices_snake"
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(f"{out_dir}/doc{doc_i:06d}_ctx2048.h5", "w") as f:
        f.create_dataset("nll", data=nll_mat, compression="gzip")
    print(f"[INFO] wrote doc{doc_i:06d}  ({doc_len} tokens)")

    cursor += doc_len             # advance to next document in the stream





    # preview the first 120 characters of the *document portion*
    #doc_preview = tok.decode(doc_slice[:120]).replace("\n", " ")
    #print(f"[shift {s:5d}] buf {buf_len:4d} |doc preview| {doc_preview}")

    #old preview, which always shows the left edge of the context window:
    preview = tok.decode(window[:120]).replace("\n", " ")
    print("    left-edge 0-119:", preview)
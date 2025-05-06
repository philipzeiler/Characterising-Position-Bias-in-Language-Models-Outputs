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

# ── 2.a  Padding strategy toggle and generation of fixed buffer ──────────────
USE_RANDOM_BUFFER  = False       # ⇦ flip to False for a fixed 2 048-token buffer
FIXED_PAD_PATH     = "fixed_pad.npy"   # cache for deterministic runs

if not USE_RANDOM_BUFFER:
    if os.path.exists(FIXED_PAD_PATH):
        fixed_buf = torch.from_numpy(np.load(FIXED_PAD_PATH))
        print(f"[INFO] loaded fixed buffer ({len(fixed_buf)} tokens)")
    else:
        parts, remain = [], CTX
        while remain > 0:
            r_idx  = random.randint(0, len(val_ds) - 1)
            r_ids  = tok(val_ds[r_idx]["text"], return_tensors="pt").input_ids[0]
            take   = r_ids[-remain:] if len(r_ids) >= remain else r_ids
            parts.insert(0, take)
            remain -= len(take)
        fixed_buf = torch.cat(parts)          # exactly CTX tokens
        np.save(FIXED_PAD_PATH, fixed_buf.cpu().numpy()) #comment this if you never want to save your buffer
        print(f"[INFO] built & cached fixed buffer ({len(fixed_buf)} tokens)")

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

# copy of "helper" code, slightly adjusted by tiago to illustrate changes that I need to make in order to do "Batching" (implement later)
#def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
#    with torch.inference_mode():
#        seq_ids[np.new_axis(),:].repeat(5, 1)
#        logits = model(seq_ids[:, :-1][None]).logits.float()
#        lp     = F.log_softmax(logits, -1)
#    tgt = seq_ids[:, 1:]
#    return -lp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]  # FP32
# -----------------------------------------------------------------------------

TOTAL_SHIFTS = CTX + L - 1          # run ~2 386 shifts for L=338
for s in range(TOTAL_SHIFTS):
    pos_last = s
    pos_first = pos_last - (L - 1)
    # ── window geometry (offset-based) ──────────────────────────────────────
    # w < 0  : document tail is still growing in from the right
    # w >= 0 : whole document is inside and slides right while shrinking
    w = s - (L - 1)                         # offset of doc’s first token

    doc_left  = max(0, -w)                  # first doc index inside window
    doc_right = min(L, CTX - w)             # one-past-last doc index
    doc_slice = doc0_ids[doc_left:doc_right]
    slice_len = doc_right - doc_left        # length of doc_slice
    left_start = doc_left                   # keep old variable name for write-loop

    buf_len = max(0, w)                     # EOS / random buffer on the left
    pad_len = CTX - buf_len - slice_len     # EOS on the right  ( ≥ 0 )

    # ---- build / retrieve left buffer --------------------------------------
    if buf_len > 0:
        if USE_RANDOM_BUFFER:
            # build fresh buffer every shift (current behaviour)
            parts, remain = [], buf_len
            while remain > 0:
                r_idx = random.randint(0, len(val_ds) - 1)
                r_ids = tok(val_ds[r_idx]["text"],
                            return_tensors="pt").input_ids[0]
                take  = r_ids[-remain:] if len(r_ids) >= remain else r_ids
                parts.insert(0, take)
                remain -= len(take)
            buf_ids = torch.cat(parts)
        else:
            # slice the pre-built fixed buffer
            buf_ids = fixed_buf[-buf_len:]
        print(f"[shift {s:5d}] buf {buf_len:4d}")
    else:
        buf_ids = torch.empty(0, dtype=torch.long)


    # assemble window:  [buffer | doc_slice | right-pad]
    eos_right = torch.full((pad_len,), tok.eos_token_id, dtype=torch.long)
    window    = torch.cat([buf_ids, doc_slice, eos_right])

    # preview the first 120 characters of the *document portion*
    #doc_preview = tok.decode(doc_slice[:120]).replace("\n", " ")
    #print(f"[shift {s:5d}] buf {buf_len:4d} |doc preview| {doc_preview}")

    #old preview, which always shows the left edge of the context window:
    preview = tok.decode(window[:120]).replace("\n", " ")
    print("    left-edge 0-119:", preview)


    # compute NLLs
    nll_vec = nll_for(window.to(dev))           # length CTX-1

    # ---- store NLLs for every token in *doc_slice* -------------------------
    for j, tok_id in enumerate(doc_slice):
        k       = buf_len + j
        doc_idx = left_start + j
        nll_mat[doc_idx, k] = nll_vec[k-1].item()

# ── 4. Save ------------------------------------------------------------------
os.makedirs("nll_matrices", exist_ok=True)
with h5py.File("nll_matrices/doc0_ctx2048v2.h5", "w") as f:
    f.create_dataset("nll", data=nll_mat, compression="gzip")
print("[INFO] matrix saved to nll_matrices/doc0_ctx2048v2.h5")

# ─────────────────────────────────────────────────────────────────────────────
#  Sliding‑window NLL for N docs of The Pile‑validation (giant “snake” stream)
# ─────────────────────────────────────────────────────────────────────────────
import os, random, numpy as np, torch, h5py, torch.nn.functional as F
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ─── 0. Reproducibility & math flags ────────────────────────────────────────
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

# ─── 1. Model & tokenizer ───────────────────────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.float16
        ).to(dev).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] model on {dev} – weights dtype {next(model.parameters()).dtype}")

# ─── 2. Validation split ────────────────────────────────────────────────────
val_ds = load_dataset("pietrolesci/pile-validation",
                      split="validation",
                      verification_mode="no_checks")
print("[INFO] validation rows:", len(val_ds))

# ─── 2.a  Global‑snake settings ─────────────────────────────────────────────
N_DOCS       = 10            # number of docs to process
SHUFFLE_SEED = 123           # reproducible permutation

doc_order = list(range(len(val_ds)))
rng = random.Random(SHUFFLE_SEED); rng.shuffle(doc_order)

stream_tokens, lengths = [], []
for idx in doc_order[:N_DOCS]:
    ids = tok(val_ds[idx]["text"], return_tensors="pt").input_ids[0]
    stream_tokens.append(ids)
    lengths.append(len(ids))
stream = torch.cat(stream_tokens)                    # 1‑D tensor
print(f"[INFO] snake length = {len(stream):,} tokens  ({N_DOCS} docs)")

# ─── helper: per‑window NLL vector ──────────────────────────────────────────
def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(seq_ids[:-1][None]).logits.float()
        logp   = F.log_softmax(logits, -1)
    tgt = seq_ids[1:]
    return -logp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]

# ─── 3. Slide the full stream ───────────────────────────────────────────────
cursor = 0
out_dir = "nll_matrices_snake"; os.makedirs(out_dir, exist_ok=True)

for doc_i, doc_len in enumerate(lengths):
    nll_mat = np.full((doc_len, CTX), np.nan, dtype=np.float32)

    for local_pos in range(doc_len + CTX - 1):
        global_last = cursor + local_pos           # index of last token shown
        w_start     = global_last - CTX + 1

        # build exact 2048‑token window (pad only at very beginning)
        if w_start < 0:
            pad  = torch.full((-w_start,), tok.eos_token_id, dtype=torch.long)
            window = torch.cat([pad, stream[0:global_last + 1]])
            w_start = 0
        else:
            window = stream[w_start:global_last + 1]   # already 2 048 tokens

        # human‑readable preview
        preview = tok.decode(window[:120]).replace("\n", " ")
        print(f"[doc {doc_i:5d} | shift {local_pos:5d}] left‑edge 0‑119: {preview}")

        # NLLs for this window
        nll_vec = nll_for(window.to(dev))               # length 2047

        # rows/cols that belong to the current document
        doc_left_idx  = max(0, w_start - cursor)
        doc_right_idx = min(doc_len, global_last + 1 - cursor)
        if doc_right_idx > doc_left_idx:
            rows = np.arange(doc_left_idx, doc_right_idx)
            cols = rows - (global_last - CTX + 1 - cursor)
            nll_mat[rows, cols] = nll_vec[cols - 1].cpu().numpy()

    # save matrix for this document
    with h5py.File(f"{out_dir}/doc{doc_i:06d}_ctx2048.h5", "w") as f:
        f.create_dataset("nll", data=nll_mat, compression="gzip")
    print(f"[INFO] wrote doc{doc_i:06d}  ({doc_len} tokens)")
    cursor += doc_len






    # preview the first 120 characters of the *document portion*
    #doc_preview = tok.decode(doc_slice[:120]).replace("\n", " ")
    #print(f"[shift {s:5d}] buf {buf_len:4d} |doc preview| {doc_preview}")

    #old preview, which always shows the left edge of the context window:
    #preview = tok.decode(window[:120]).replace("\n", " ")
    #print("    left-edge 0-119:", preview)

# copy of "helper" code, slightly adjusted by tiago to illustrate changes that I need to make in order to do "Batching" (implement later)
#def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
#    with torch.inference_mode():
#        seq_ids[np.new_axis(),:].repeat(5, 1)
#        logits = model(seq_ids[:, :-1][None]).logits.float()
#        lp     = F.log_softmax(logits, -1)
#    tgt = seq_ids[:, 1:]
#    return -lp[0, torch.arange(len(tgt), device=seq_ids.device), tgt]  # FP32
# -----------------------------------------------------------------------------
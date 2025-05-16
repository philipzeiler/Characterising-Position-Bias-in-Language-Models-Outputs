#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Multi‑document “snake” sliding‑window NLL evaluation
# Evaluates the first n_docs documents of the Pile‑validation split
# in a single 2 048‑token context window without random padding.
# ─────────────────────────────────────────────────────────────────────────────
import os, random, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── 0. Reproducibility & math flags ──────────────────────────────────────────
seed      = 42          # model‑side RNG seed
doc_seed  = 43       # controls doc‑order shuffling
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark      = False
torch.backends.cudnn.deterministic  = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# ── 1. Hyper‑parameters ─────────────────────────────────────────────────────
CTX        = 2048
MODEL_ID   = "EleutherAI/pythia-1.4b"
REVISION   = "step143000"
n_docs     = 10        # ← evaluate this many documents

# ── 2. Model & tokenizer ────────────────────────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.float16
        ).to(dev).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] model on {dev} – weights dtype {next(model.parameters()).dtype}")

# ── 3. Dataset & randomised doc order ───────────────────────────────────────
val_ds     = load_dataset("pietrolesci/pile-validation",
                          split="validation",
                          verification_mode="no_checks")
num_rows   = len(val_ds)
doc_order  = list(range(num_rows))
random.Random(doc_seed).shuffle(doc_order)      # deterministic
print(f"[INFO] validation rows = {num_rows}")
print("[DEBUG] first 100 shuffled doc indices:",
      doc_order[:100])

# ── 4. Helper: token‑wise NLL in FP16 ───────────────────────────────────────
def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
    """
    seq_ids – 1‑D tensor of length CTX on the same device as `model`
    returns – (CTX‑1,) tensor of per‑token NLL in float16
    """
    with torch.inference_mode():                              # :contentReference[oaicite:6]{index=6}
        logits = model(seq_ids[:-1][None]).logits             # [1,T,V] FP16
        logp  = F.log_softmax(logits.float(), -1)             # stab → FP32 then back
    tgt = seq_ids[1:]
    nll = F.nll_loss(logp.squeeze(0), tgt,
                     reduction="none").half()                 # :contentReference[oaicite:7]{index=7}
    return nll.cpu()                                          # leave GPU ASAP

# ── 5. “Snake” evaluation loop ──────────────────────────────────────────────
eos_id        = tok.eos_token_id
window_ids    = [eos_id] * CTX                       # left = index 0, right = 2047
window_meta   = [None]  * CTX                        # parallel list of (doc_id, tok_idx)
active_docs   = {}                                   # doc_id → state dict
current_doc   = None                                 # doc currently feeding tokens
docs_entered  = 0
docs_done     = 0
shift         = 0

os.makedirs("nll_matrices", exist_ok=True)

while True:
    # ── 5‑A. Supply one new token at the *left* edge ────────────────────────
    if current_doc is None and docs_entered < n_docs:
        # start a fresh document
        doc_id     = doc_order[docs_entered]
        docs_entered += 1
        tokens      = tok(val_ds[doc_id]["text"],
                          return_tensors="pt").input_ids[0]
        L           = len(tokens)
        mat_fp16    = np.full((L, CTX), np.nan, dtype=np.float16)
        active_docs[doc_id] = {
            "tokens":  tokens,      # torch 1‑D cpu long
            "pointer": L - 1,       # last token first
            "matrix":  mat_fp16,
        }
        current_doc = doc_id
        print(f"[INFO] enter doc {doc_id} (len={L}) – docs_entered={docs_entered}")

    if current_doc is not None:
        state      = active_docs[current_doc]
        tok_idx    = state["pointer"]
        token_id   = int(state["tokens"][tok_idx])
        meta_left  = (current_doc, tok_idx)
        state["pointer"] -= 1
        if state["pointer"] < 0:                # all tokens emitted
            current_doc = None
    else:
        token_id  = eos_id
        meta_left = None

    # shift the window right by one
    window_ids  = [token_id]   + window_ids[:-1]
    window_meta = [meta_left]  + window_meta[:-1]

    # ── 5‑B. Compute NLL for this window ────────────────────────────────────
    seq_tensor = torch.tensor(window_ids, dtype=torch.long, device=dev)
    nll_vec    = nll_for(seq_tensor).numpy()          # (CTX‑1,) float16 on CPU

    # ── 5‑C. Write NLLs into per‑doc matrices ───────────────────────────────
    for k in range(1, CTX):                 # k is *target* position
        meta = window_meta[k]
        if meta is None:
            continue
        doc_id, tok_idx = meta
        active_docs[doc_id]["matrix"][tok_idx, k] = nll_vec[k-1]

    # ── 5‑D. Detect docs that have completely left the window ───────────────
    finished = [d for d in active_docs
                if all(m is None or m[0] != d for m in window_meta)]
    for d in finished:
        out_path = f"nll_matrices/doc{d}_ctx{CTX}_fp16.h5"
        with h5py.File(out_path, "w") as f:                       # :contentReference[oaicite:8]{index=8}
            f.create_dataset("nll",
                             data=active_docs[d]["matrix"],
                             compression="gzip")
        print(f"[INFO] saved doc {d} → {out_path}")
        del active_docs[d]
        docs_done += 1

    # ── 5‑E. Progress logging every 1 k shifts ─────────────────────────────
    if shift % 1000 == 0:
        snake_size = sum(m is not None for m in window_meta)
        print(f"[DBG] shift={shift:7d}  snake_size={snake_size:4d}  "
              f"open_docs={len(active_docs)}")

    # ── 5‑F. Termination condition ─────────────────────────────────────────
    if docs_done == n_docs and all(m is None for m in window_meta):
        assert len(active_docs) == 0, "all docs should be closed"
        print("[INFO] evaluation complete – all requested docs processed")
        break

    shift += 1

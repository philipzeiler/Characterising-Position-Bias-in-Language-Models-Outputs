#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Variable-length “snake” NLL evaluation – RIGHT-append version
# Same logic/structures as previous file, tokens now enter from the RIGHT.
# ─────────────────────────────────────────────────────────────────────────────
import os, random, itertools, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── 0. Determinism flags ────────────────────────────────────────────────────
seed, doc_seed = 42, 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark      = False
torch.backends.cudnn.deterministic  = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# ── 1. Hyper‑parameters ─────────────────────────────────────────────────────
CTX        = 2048            # window length
MODEL_ID   = "EleutherAI/pythia-1.4b"
REVISION   = "step143000"
n_docs     = 5              # evaluate this many docs

# ── 2. Model & tokenizer ────────────────────────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
            MODEL_ID, revision=REVISION, torch_dtype=torch.float16
        ).to(dev).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] model on {dev} – dtype {next(model.parameters()).dtype}")

# ── 3. Dataset & shuffled doc order ─────────────────────────────────────────
val_ds   = load_dataset("pietrolesci/pile-validation",
                        split="validation",
                        verification_mode="no_checks")
doc_order = list(range(len(val_ds)))
random.Random(doc_seed).shuffle(doc_order)                  # deterministic order through random seed
print("[DEBUG] first 100 shuffled doc ids:", doc_order[:100])

# ── 4. Helper: FP16 per-token NLL ───────────────────────────────────────────
def nll_for(seq_ids: torch.Tensor) -> torch.Tensor:
    """
    seq_ids  – 1‑D LongTensor length CTX (on device)
    returns  – (CTX‑1,) Float16 CPU tensor of NLLs
    """
    with torch.inference_mode():
        logits = model(seq_ids[:-1][None]).logits           # [1,T,V] FP16
        lp     = torch.log_softmax(logits.float(), -1)      # FP32 for stability
    tgt = seq_ids[1:]
    return torch.nn.functional.nll_loss(                    # token‑wise NLL
            lp.squeeze(0), tgt, reduction="none").half().cpu()

# ── 5. Variable-length snake initialisation ─────────────────────────────────
eos_id      = tok.eos_token_id
snake_ids   = deque([eos_id] * (CTX - 1))                   # 2047 EOS tokens - this is the buffer at the start of the snake (may change in the future)
snake_meta  = deque([None]  * (CTX - 1))                    # parallel metadata
active_docs = {}                                            # doc_id → state
docs_entered = docs_done = shift = 0

os.makedirs("nll_matrices", exist_ok=True)

while True:
    # ── 5‑A. Ensure snake ≥ 2048 tokens ─────────────────────────────────────
    if len(snake_ids) <= CTX - 1:
        if docs_entered < n_docs:                           # add new document
            doc_id = doc_order[docs_entered]; docs_entered += 1
            tokens = tok(val_ds[doc_id]["text"],
                          return_tensors="pt").input_ids[0]
            L      = len(tokens)
            active_docs[doc_id] = {
                "tokens": tokens,                            # tokens currently unused but may be useful in the future
                "matrix": np.full((L, CTX-1), np.nan,
                                  dtype=np.float16),
            }
            # APPEND entire doc on the right (chronological order)
            snake_ids.extend(tokens.tolist())
            snake_meta.extend((doc_id, idx) for idx in range(L))
            print(f"[INFO] entered doc {doc_id} (len={L}); docs_entered={docs_entered}")
        else:
            snake_ids.append(eos_id)            # pad on the right
            snake_meta.append(None)

    # 5-B. Build 2 048-token context (LEFT-most chunk) ----------------------
    window_ids  = list(itertools.islice(snake_ids, 0, CTX))
    window_meta = list(itertools.islice(snake_meta, 0, CTX))
    seq_tensor  = torch.tensor(window_ids, dtype=torch.long, device=dev)
    nll_vec     = nll_for(seq_tensor).numpy()

    # 5-C. Scatter NLLs into per-doc matrices -------------------------------
    for k in range(1, CTX):
        meta = window_meta[k]
        if meta is None:
            continue
        doc_id, tok_idx = meta
        active_docs[doc_id]["matrix"][tok_idx, k-1] = nll_vec[k-1]

    # 5-D. Slide window: pop one token from the LEFT ------------------------
    snake_ids.popleft()
    snake_meta.popleft()

    # 5-E. Close & save docs that left the snake ----------------------------
    current_doc_ids = {m[0] for m in snake_meta if m is not None}
    finished = [d for d in active_docs if d not in current_doc_ids]
    for d in finished:
        out_path = f"nll_matrices/doc{d}_fp16.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("nll",
                             data=active_docs[d]["matrix"],
                             compression="gzip")
        print(f"[INFO] saved doc {d} → {out_path}")
        del active_docs[d]; docs_done += 1

    # 5-F. Progress logging every 1 k shifts -------------------------------
    if shift % 1000 == 0:
        print(f"[DBG] shift={shift:7d}  snake_len={len(snake_ids):5d}  "
              f"open_docs={len(active_docs)}")

    # 5-G. Terminate ---------------------------------------------------------
    if docs_done == n_docs and all(m is None for m in snake_meta):
        assert not active_docs
        print("[INFO] evaluation complete – all requested docs processed")
        break

    shift += 1

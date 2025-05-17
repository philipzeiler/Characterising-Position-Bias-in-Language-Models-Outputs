#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Variable‑length “snake” NLL evaluation
# Prepends entire docs when snake ≤ 2047 tokens; computes NLL on last 2048;
# pops one token from the right each step; repeats until all docs processed.
# ─────────────────────────────────────────────────────────────────────────────
import os, random, itertools, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── 0. Determinism flags ────────────────────────────────────────────────────
seed, doc_seed = 42, 420
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark      = False
torch.backends.cudnn.deterministic  = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# ── 1. Hyper-parameters ─────────────────────────────────────────────────────
CTX        = 2048   # window length
BATCH      = 7      # batch size
MODEL_ID   = "EleutherAI/pythia-1.4b"
REVISION   = "step143000"
n_docs     = 500              # evaluate this many docs

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
random.Random(doc_seed).shuffle(doc_order)                  # deterministic
print("[DEBUG] first 100 shuffled doc ids:", doc_order[:100])

# ── 4. Helper: FP16 batched NLL ─────────────────────────────────────────────
def nll_for(batch_ids: torch.Tensor) -> torch.Tensor:
    """
    T = CTX-1
    B = BATCH
    batch_ids - [B, CTX] LongTensor on GPU
    returns   - [B, T]   Float16 on CPU
    """
    with torch.inference_mode():
        logits = model(batch_ids[:, :-1]).logits         # [B,T,V] FP16
        lp = torch.log_softmax(logits.float(), -1).permute(0, 2, 1)  # shape [B, V, T]
    tgt = batch_ids[:, 1:]
    nll = F.nll_loss(lp, tgt, reduction="none").half()   # [B,T] FP16
    return nll.cpu()

# ── 5. Variable-length snake initialisation ─────────────────────────────────
eos_id      = tok.eos_token_id
snake_ids   = deque([eos_id] * (CTX - 1))                   # 2047 EOS
snake_meta  = deque([None]  * (CTX - 1))                    # parallel metadata
active_docs = {}                                            # doc_id → state
docs_entered = docs_done = shift = 0

os.makedirs("nll_matrices_2", exist_ok=True)

while True:
    # ── 5-A. Ensure snake ≥ 2048 tokens ─────────────────────────────────────
    while len(snake_ids) < CTX + BATCH - 1:
        if docs_entered < n_docs:
            doc_id = doc_order[docs_entered]; docs_entered += 1
            tokens = tok(val_ds[doc_id]["text"],
                          return_tensors="pt").input_ids[0]
            L      = len(tokens)
            active_docs[doc_id] = {
                "tokens": tokens,                            # tokens currently unused but may be useful in the future
                "matrix": np.full((L, CTX-1), np.nan,
                                  dtype=np.float16),
            }
            # prepend entire doc (last token first) using extendleft
            snake_ids.extendleft(int(t) for t in reversed(tokens))
            snake_meta.extendleft((doc_id, idx)
                                   for idx in reversed(range(L)))
            print(f"[INFO] entered doc {doc_id} (len={L}); "
                  f"docs_entered={docs_entered}")
        else:                                               # only EOS padding
            snake_ids.appendleft(eos_id)
            snake_meta.appendleft(None)

    # ── 5-B. Build *batched* windows (adjacent offsets) ─────────────────────
    windows_ids  = []
    windows_meta = []
    for off in range(BATCH):
        left  = len(snake_ids) - CTX - off
        right = len(snake_ids) - off
        windows_ids.append(list(itertools.islice(snake_ids, left, right)))
        windows_meta.append(list(itertools.islice(snake_meta, left, right)))

    batch_tensor = torch.tensor(windows_ids, dtype=torch.long, device=dev)
    nll_batch    = nll_for(batch_tensor).numpy()         # [B, 2047]

    # ── 5-C. Scatter losses into per-doc matrices ───────────────────────────
    for b in range(BATCH):
        meta_slice = windows_meta[b]
        for k in range(1, CTX):
            meta = meta_slice[k]
            if meta is None:
                continue
            doc_id, tok_idx = meta
            active_docs[doc_id]["matrix"][tok_idx, k-1] = nll_batch[b, k-1]

    # ── 5-D. Slide right: pop BATCH tokens from the tail ────────────────────
    for _ in range(BATCH):
        snake_ids.pop()
        snake_meta.pop()

    # ── 5‑E. Close & save docs that left the snake entirely ─────────────────
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

    # ── 5‑F. Logging shifts ───────────────────────────────────────
    if shift % 300 == 0:
        print(f"[DBG] shift={shift*BATCH}  snake_len={len(snake_ids):5d}  "
              f"open_docs={len(active_docs)}")

    # ── 5‑G. Terminate when all docs processed & window pure EOS ────────────
    if docs_done == n_docs and all(m is None for m in snake_meta):
        assert not active_docs
        print("[INFO] evaluation complete - all requested docs processed")
        break

    shift += 1

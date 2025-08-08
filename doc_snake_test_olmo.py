#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Variable‑length “snake” NLL evaluation
# Prepends entire docs when snake ≤ 2047 tokens; computes NLL on last 2048;
# pops one token from the right each step; repeats until all docs processed.
# ─────────────────────────────────────────────────────────────────────────────
import os, random, itertools, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os.path as _osp

# ── 0. Determinism flags ────────────────────────────────────────────────────
seed, doc_seed = 42, 753    #set these for reproducibility doc_seed=753 for almost all, except for second gpu in some olmo runs where it was 153
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark      = False
torch.backends.cudnn.deterministic  = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

# ── 1. Hyper-parameters ─────────────────────────────────────────────────────
CTX        = 4096   # window length, should not be changed
BATCH      = 1      # batch size (maybe set to power of 2), higher consumes more vram, had limited speedup in my testing
MODEL_SIZE = "0425-1B"
#REVISION   = "step143000"    #Ill stick to the latest for now
n_docs     = 50000            # evaluate this many docs
QUICKSTART_FROM = 500          # if code was interrupted, restart by entering how many files you already have
ADD_EOD = True               # set True to append eos_id to every document

#CLUSTER CHANGES:
ROOT = "D:/NLL_matrices_olmo2/seed753"

MODEL_ID   = f"allenai/OLMo-2-{MODEL_SIZE}"

if(ADD_EOD):                #if we add EOS token, store those results separately
    MODEL_SIZE=f"{MODEL_SIZE}_EOD"
# ── 2. Model & tokenizer ────────────────────────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            #revision=REVISION, 
            torch_dtype=torch.bfloat16 #olmo2 models used bloat16 during training
        ).to(dev).eval()

tok   = AutoTokenizer.from_pretrained(MODEL_ID, 
                                      #revision=REVISION
                                      )
print(f"[INFO] model on {dev} - dtype {next(model.parameters()).dtype}")

# ── 3. Dataset & shuffled doc order ─────────────────────────────────────────
val_ds   = load_dataset("pietrolesci/pile-validation",
                        split="validation",
                        verification_mode="no_checks")
doc_order = list(range(len(val_ds)))
random.Random(doc_seed).shuffle(doc_order)                  # deterministic
print("[DEBUG] first 100 shuffled doc ids:", doc_order[:100])

# ── 4. Helper: FP16 batched NLL ─────────────────────────────────────────────
"""?
    T = CTX-1
    B = BATCH
    V = Vocabulary size
    batch_ids - [B, CTX] LongTensor on GPU
    returns   - [B, T]   Float16 on CPU
"""

def nll_for(batch_ids: torch.Tensor,
            attn_mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(batch_ids[:, :-1],
                       attention_mask=attn_mask[:, :-1]).logits
        lp = torch.log_softmax(logits.float(), -1).permute(0, 2, 1)
        tgt = batch_ids[:, 1:]
        nll = F.nll_loss(lp, tgt, reduction="none").half()
    return nll.cpu()

def is_interesting(P_d: int) -> bool:
    """Return True if P_d in -128…128  OR  abs(P_d) is a power of two ≤4096."""
    if -128 <= P_d <= 128:
        return True
    a = abs(P_d)
    return (a <= 4096) and ((a & (a - 1)) == 0)

# ── 5. Variable-length snake initialisation ─────────────────────────────────
eos_id      = tok.eos_token_id
snake_ids   = deque([eos_id] * (CTX - 1))                   # 2047 EOS
snake_meta  = deque([None]  * (CTX - 1))                    # parallel metadata
active_docs = {}                                            # doc_id → state
docs_entered = docs_done = shift = 0

os.makedirs(f"{ROOT}/{MODEL_SIZE}", exist_ok=True)

while True:
    # ── 5-A. Ensure snake ≥ 2048 tokens ─────────────────────────────────────
    while len(snake_ids) < CTX + BATCH - 1:
        if docs_entered < n_docs:
            doc_id = doc_order[docs_entered]; docs_entered += 1
            tokens = tok(val_ds[doc_id]["text"],
                          return_tensors="pt").input_ids[0]
            
            if ADD_EOD:
                tokens = torch.cat([    tokens,
                                        torch.tensor([eos_id],
                                        dtype=tokens.dtype)])
            L      = len(tokens)
            active_docs[doc_id] = {
            #    "tokens": tokens,                            # tokens currently unused but may be useful in the future
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

    if(QUICKSTART_FROM < docs_entered):
        # ── 5-B. Build *batched* windows (adjacent offsets) ─────────────────────
        windows_ids  = []
        windows_meta = []
        max_len = 0
        for off in range(BATCH):
            # full CTX-sized slice as before -----------------------------------
            left  = len(snake_ids) - CTX - off
            right = len(snake_ids) - off
            ids   = list(itertools.islice(snake_ids,  left, right))
            meta  = list(itertools.islice(snake_meta, left, right))
            
                    # ------- locate right-most interesting document in this slice -----
            doc_last = {}                         # doc_id → last interesting pos                                    ### NEW
            for pos, m in enumerate(meta):
                if m is None:
                    continue
                doc_id, tok_idx = m
                P_t = pos + 1           # 1 … CTX
                P_d = P_t - tok_idx
                if is_interesting(P_d):
                    doc_last[doc_id] = max(pos, doc_last.get(doc_id, -1))    # keep extending until farthest match

            if not doc_last:                           # no interesting doc
                continue                                  # ⇒ skip this batch item

            # truncate slice right after the doc end  --------------------------
            last_keep = max(doc_last.values()) 
            keep = last_keep + 1                          # inclusive index
            windows_ids.append(ids[:keep])
            windows_meta.append(meta[:keep])
            max_len = max(max_len, keep)                  ### NEW

        # ---------- skip whole forward pass if nothing interesting ------------
        if windows_ids:                               ### NEW

            # ---------- right-pad to rectangular tensor ---------------------------
            orig_lens = [len(w) for w in windows_ids] 
            PAD = tok.eos_token_id
            for i in range(len(windows_ids)):
                pad_len = max_len - len(windows_ids[i])
                if pad_len:
                    windows_ids[i].extend([PAD] * pad_len)
                    windows_meta[i].extend([None] * pad_len)

            attn_mask = [[1]*orig + [0]*(max_len-orig)
                        for orig in orig_lens]

            batch_tensor = torch.tensor(windows_ids, dtype=torch.long, device=dev)
            mask_tensor  = torch.tensor(attn_mask,  dtype=torch.long, device=dev)

            nll_batch = nll_for(batch_tensor, mask_tensor).numpy()


            # ── 5-C. Scatter losses into per-doc matrices ───────────────────────────
            for b in range(len(windows_ids)):          # len == actual batch
                meta_slice = windows_meta[b]
                for k, meta in enumerate(meta_slice[1:], start=1):  # skip k=0
                    if meta is None:
                        continue
                    doc_id, tok_idx = meta             # tok_idx is in doc
                    active_docs[doc_id]["matrix"][tok_idx, k-1] = nll_batch[b, k-1]

                #We could implement following speedup, but time spent on saving into files is minimal so not worth the effort.
                #active_docs[doc_id]["matrix"][tok_idx, :CTX] = nll_batch[b, :CTX].where()
        #else:
        #    print("not important")
    # ── 5-D. Slide right: pop BATCH tokens from the tail ────────────────────
    for _ in range(BATCH):
        snake_ids.pop()
        snake_meta.pop()

    # ── 5‑E. Close & save docs that left the snake entirely ─────────────────
    current_doc_ids = {m[0] for m in snake_meta if m is not None}
    finished = [d for d in active_docs if d not in current_doc_ids]
    for d in finished:        
        out_path = f"{ROOT}/{MODEL_SIZE}/doc{d}.h5"

        # --- skip writing if file already exists -------------------
        if _osp.exists(out_path):
            print(f"[SKIP] {out_path} already on disk - keeping previous result")
        else:
            with h5py.File(out_path, "w") as f:
                f.create_dataset("nll",
                                data=active_docs[d]["matrix"],
                                compression="gzip")
            print(f"[INFO] saved doc {d} → {out_path}")

        # always remove in-memory copy to free RAM
        del active_docs[d]
        docs_done += 1

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
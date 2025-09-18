#!/usr/bin/env python3
# GPT-2 “snake” NLL evaluation (FP32 end-to-end)
# Produces per-document .h5 files with dataset "nll" of shape (L, CTX-1)

import os, random, itertools, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import os.path as _osp

# ── 0) Determinism ──────────────────────────────────────────────────────────
seed, doc_seed = 42, 753
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── 1) Run settings (GPT-2 only) ────────────────────────────────────────────
MODEL_ID   = "gpt2-xl"   # one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl
WEBTEXT_VALID = r"D:/datasets/webtext.test.jsonl"  # downloaded from OpenAI bucket
OUT_ROOT   = f"D:/NLL_matrices/{MODEL_ID}"

CTX        = 1024           # GPT-2 context window (per HF config)  ⟵ key difference vs Pythia
BATCH      = 1              # adjust for VRAM
n_docs     = 5000            # number of docs to evaluate
QUICKSTART_FROM = 1910         # resume if needed (# of docs already saved)
ADD_EOD    = True           # append <|endoftext|> to every document (mirrors training delimiter)

# ── 2) Model & tokenizer (FP32 everywhere) ──────────────────────────────────
dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(dev).eval()
tok   = AutoTokenizer.from_pretrained(MODEL_ID)

# GPT-2 uses <|endoftext|> as EOS/BOS (id 50256). Set pad→eos to avoid warnings if padding is ever used.
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print(f"[INFO] {MODEL_ID} on {dev} | dtype={next(model.parameters()).dtype} | CTX={CTX}")

# ── 3) Dataset: WebText validation JSONL ────────────────────────────────────
if not os.path.exists(WEBTEXT_VALID):
    raise FileNotFoundError(f"Missing WebText valid set at: {WEBTEXT_VALID}")
val_ds = load_dataset("json", data_files={"validation": WEBTEXT_VALID}, split="validation")
doc_order = list(range(len(val_ds)))
random.Random(doc_seed).shuffle(doc_order)
print("[DEBUG] first 100 shuffled doc ids:", doc_order[:100])

# ── 4) FP32 batched NLL helper ──────────────────────────────────────────────
def nll_for(batch_ids: torch.Tensor) -> torch.Tensor:
    """
    Input : [B, CTX] int64
    Output: [B, CTX-1] float32 NLL per position (token t predicts t+1)
    NOTE: model and math stay FP32 for numerical stability.
    """
    with torch.inference_mode():
        logits = model(batch_ids[:, :-1]).logits            # [B, T, V] FP32
        logp   = torch.log_softmax(logits, dim=-1).permute(0, 2, 1)  # [B, V, T]
        tgt    = batch_ids[:, 1:]                           # [B, T]
        nll    = F.nll_loss(logp, tgt, reduction="none")    # FP32
    return nll.cpu()                                        # keep as float32 for now

# ── 5) Variable-length snake ────────────────────────────────────────────────
eos_id      = tok.eos_token_id
snake_ids   = deque([eos_id] * (CTX - 1))   # left pad with EOS to start
snake_meta  = deque([None]  * (CTX - 1))    # (doc_id, tok_idx) or None
active_docs = {}
docs_entered = docs_done = shift = 0

os.makedirs(OUT_ROOT, exist_ok=True)

while True:
    # 5-A) Ensure we can build BATCH adjacent CTX windows
    while len(snake_ids) < CTX + BATCH - 1:
        if docs_entered < n_docs:
            doc_id = doc_order[docs_entered]; docs_entered += 1
            text = val_ds[doc_id]["text"]

            # Important: GPT-2 pipelines typically add <|endoftext|> between docs, not auto-BOS/EOS.
            # We tokenize with NO extra specials, then (optionally) append EOS explicitly.
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if ADD_EOD:
                ids = torch.cat([ids, torch.tensor([eos_id], dtype=ids.dtype)])

            L = int(ids.numel())
            active_docs[doc_id] = {
                # Store as float16 on disk (to match your merger), but compute in FP32.
                "matrix": np.full((L, CTX-1), np.nan, dtype=np.float16),
            }

            # Prepend entire doc (last token first)
            snake_ids.extendleft(int(t) for t in reversed(ids))
            snake_meta.extendleft((doc_id, idx) for idx in reversed(range(L)))
            print(f"[INFO] entered doc {doc_id} (len={L}); docs_entered={docs_entered}")
        else:
            # Only EOS padding once all requested docs entered
            snake_ids.appendleft(eos_id)
            snake_meta.appendleft(None)

    if QUICKSTART_FROM < docs_entered:
        # 5-B) Build batched CTX windows at offsets [0..BATCH-1]
        windows_ids, windows_meta = [], []
        for off in range(BATCH):
            left  = len(snake_ids) - CTX - off
            right = len(snake_ids) - off
            windows_ids.append(list(itertools.islice(snake_ids,  left, right)))
            windows_meta.append(list(itertools.islice(snake_meta, left, right)))

        batch_tensor = torch.tensor(windows_ids, dtype=torch.long, device=dev)
        nll_batch    = nll_for(batch_tensor).numpy().astype(np.float32)  # [B, CTX-1] FP32

        # 5-C) Scatter into per-doc matrices (cast to f16 when storing)
        for b in range(BATCH):
            meta_slice = windows_meta[b]
            for k in range(1, CTX):                # positions 1..CTX-1
                meta = meta_slice[k]
                if meta is None: continue
                doc_id, tok_idx = meta
                # write as float16 to match your downstream merger format
                active_docs[doc_id]["matrix"][tok_idx, k-1] = np.float16(nll_batch[b, k-1])

    # 5-D) Slide snake to the right by BATCH tokens (pop from tail)
    for _ in range(BATCH):
        snake_ids.pop()
        snake_meta.pop()

    # 5-E) Persist docs that fully left the snake
    current_doc_ids = {m[0] for m in snake_meta if m is not None}
    finished = [d for d in active_docs if d not in current_doc_ids]
    for d in finished:
        mat = active_docs[d]["matrix"]                 # ← existing object

        # ---- skip any file containing NaNs ----------------------- ▲
        if np.isnan(mat).any():
            print(f"[SKIP] doc {d} contains NaNs - not saving")
            del active_docs[d]                          # still free RAM
            docs_done += 1
            continue

        out_path = f"{OUT_ROOT}/doc{d}.h5"
        if _osp.exists(out_path):
            print(f"[SKIP] {out_path} already on disk")
        else:
            with h5py.File(out_path, "w") as f:
                f.create_dataset("nll", data=active_docs[d]["matrix"], compression="gzip")
            print(f"[INFO] saved doc {d} → {out_path}")
        del active_docs[d]
        docs_done += 1

    if shift % 300 == 0:
        print(f"[DBG] shift={shift*BATCH}  snake_len={len(snake_ids):5d}  open_docs={len(active_docs)}")

    # 5-F) Terminate when all docs processed and window is pure EOS
    if docs_done == n_docs and all(m is None for m in snake_meta):
        assert not active_docs
        print("[INFO] evaluation complete")
        break

    shift += 1

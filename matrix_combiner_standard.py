#!/usr/bin/env python3
# Merge per-document NLL matrices into one big (N_tokens × (CTX-1)) file
#  – document order follows deterministic shuffle with doc_seed

import os, h5py, numpy as np, random
from datasets import load_dataset
from tqdm import tqdm

# ---------------- small switch ---------------------------------------------
USE_GPT2       = False  # ← set True for GPT-2 merges
doc_seed       = 753

# --- Pythia defaults (unchanged) -------------------------------------------
MODEL_SIZE     = "2.8B"
SRC_DIR        = f"D:/NLL_matrices/{MODEL_SIZE}_deduped_EOD"
DST_FILE       = f"D:/NLL_matrices/{MODEL_SIZE}_deduped_EOD_merged.h5"
CTX            = 2048                      # → 2047 cols

# --- GPT-2 overrides (edit paths as needed) --------------------------------
# GPT-2 uses a 1024-token window → 1023 columns
GPT2_SRC_DIR   = r"D:/NLL_matrices/gpt2-xl"
GPT2_DST_FILE  = r"D:/NLL_matrices/gpt2-xl_merged.h5"
WEBTEXT_VALID  = r"D:/datasets/webtext.test.jsonl"  # from OpenAI repo

# ---------------- set up according to switch --------------------------------
if USE_GPT2:
    CTX      = 1024  # GPT-2 context length (n_positions / n_ctx = 1024)
    SRC_DIR  = GPT2_SRC_DIR
    DST_FILE = GPT2_DST_FILE

# chunking (row stripes); unchanged logic
CHUNK_ROWS   = 1024
CHUNK_SHAPE  = (CHUNK_ROWS, CTX-1)

# ---------------- document order (same as evaluation) -----------------------
if USE_GPT2:
    # Load WebText validation JSONL; same indexing as your GPT-2 evaluator
    val_ds = load_dataset("json",
                          data_files={"validation": WEBTEXT_VALID},
                          split="validation")
else:
    val_ds = load_dataset("pietrolesci/pile-validation",
                          split="validation",
                          verification_mode="no_checks")

doc_order = list(range(len(val_ds)))
random.Random(doc_seed).shuffle(doc_order)

paths = [os.path.join(SRC_DIR, f"doc{doc_id}.h5")
         for doc_id in doc_order
         if os.path.exists(os.path.join(SRC_DIR, f"doc{doc_id}.h5"))]

if not paths:
    raise RuntimeError(f"no .h5 files found in expected order under {SRC_DIR}")

print(f"[INFO] CTX={CTX} → expecting {CTX-1} columns; files in: {SRC_DIR}")

# ---------------- first pass: total token count ----------------------------
token_counts = []
for fp in tqdm(paths, desc="Counting tokens"):
    with h5py.File(fp, "r") as f:
        token_counts.append(f["nll"].shape[0])   # rows = tokens

N_docs         = len(paths)
N_tokens_total = int(np.sum(token_counts))
print(f"[INFO] {N_docs:,} docs → {N_tokens_total:,} rows in super-matrix")

# ---------------- create big file -----------------------------------------
with h5py.File(DST_FILE, "w") as big:
    nll_ds = big.create_dataset(
        "nll",
        shape=(N_tokens_total, CTX-1),
        dtype="f2",
        chunks=CHUNK_SHAPE,
        compression="gzip",
        compression_opts=4
    )
    ptr_ds = big.create_dataset("doc_ptr", (N_docs + 1,), "i8")

    cursor = 0
    for i, fp in enumerate(tqdm(paths, desc="Merging")):
        with h5py.File(fp, "r") as f:
            mat = f["nll"][...]          # dense (L, CTX-1) float16
        L = mat.shape[0]

        # (Optional sanity) assert column count matches CTX-1
        if mat.shape[1] != CTX - 1:
            raise ValueError(f"{fp}: expected {CTX-1} cols, found {mat.shape[1]}")

        nll_ds[cursor:cursor+L, :] = mat
        ptr_ds[i] = cursor
        cursor += L

    ptr_ds[-1] = cursor
    print(f"[DONE] wrote {cursor:,} rows × {CTX-1} cols → {DST_FILE}")

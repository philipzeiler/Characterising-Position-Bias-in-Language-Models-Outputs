#!/usr/bin/env python3
# Merge per-document NLL matrices into one big (N_tokens × 2047) file
#  – document order follows deterministic shuffle with doc_seed = 753

import os, glob, h5py, numpy as np
from datasets import load_dataset
from tqdm import tqdm, trange
import random

# ---------------- parameters -----------------------------------------------
MODEL_SIZE = "1.4B_deduped"
SRC_DIR    = f"D:/NLL_matrices/{MODEL_SIZE}_EOD"
DST_FILE   = f"D:/NLL_matrices/{MODEL_SIZE}_EOD_merged.h5"

CTX          = 2048          # window length → 2047 cols
CHUNK_ROWS   = 1024          # row-aligned stripe
CHUNK_SHAPE  = (CHUNK_ROWS, CTX-1)

# ---------------- document order (same as evaluation) ----------------------
doc_seed = 753
val_ds   = load_dataset("pietrolesci/pile-validation",
                        split="validation",
                        verification_mode="no_checks")
doc_order = list(range(len(val_ds)))
random.Random(doc_seed).shuffle(doc_order)

paths = [os.path.join(SRC_DIR, f"doc{doc_id}.h5") for doc_id in doc_order
         if os.path.exists(os.path.join(SRC_DIR, f"doc{doc_id}.h5"))]

if not paths:
    raise RuntimeError("no .h5 files found in expected order")

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
    nll_ds = big.create_dataset("nll",
                                shape=(N_tokens_total, CTX-1),
                                dtype="f2",
                                chunks=CHUNK_SHAPE,
                                compression="gzip",
                                compression_opts=4)
    ptr_ds = big.create_dataset("doc_ptr", (N_docs + 1,), "i8")

    cursor = 0
    for i, fp in enumerate(tqdm(paths, desc="Merging")):
        with h5py.File(fp, "r") as f:
            mat = f["nll"][...]          # dense (L,2047) float16
        L = mat.shape[0]

        nll_ds[cursor:cursor+L, :] = mat  # contiguous stripe write
        ptr_ds[i] = cursor
        cursor += L

    ptr_ds[-1] = cursor                  # sentinel
    print(f"[DONE] wrote {cursor:,} rows → {DST_FILE}")

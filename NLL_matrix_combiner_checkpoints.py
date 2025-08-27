#!/usr/bin/env python3
# Merge per-document NLL matrices into one big (N_tokens × 2047) file
#  – document order follows deterministic shuffle with one or two seeds

import os, h5py, numpy as np
from datasets import load_dataset
from tqdm import tqdm
import random

# ---------------- parameters -----------------------------------------------
MODEL_SIZE = "14M"
REVISIONS   = ["step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64", "step128", "step256", "step512", "step1000", "step2000", "step4000", "step8000", "step16000", "step32000", "step64000", "step128000"]   #"step143000" is the final revision

# Primary (seed A)
DOC_SEED_A = 753

# Optional secondary (seed B)
USE_SECOND_SEED = False              # ← set True to include a second seed
DOC_SEED_B      = 153
SRC_DIR_B       = f"D:/NLL_matrices_olmo2/seed{DOC_SEED_B}/{MODEL_SIZE}_EOD"


CTX          = 2048          # window length → 2047 cols
CHUNK_ROWS   = 1024          # row-aligned stripe
CHUNK_SHAPE  = (CHUNK_ROWS, CTX-1)


for REVISION in REVISIONS:
    SRC_DIR_A  = f"D:/NLL_matrices/revisions/{MODEL_SIZE}_EOD/{REVISION}"
    DST_FILE   = f"D:/NLL_matrices/revisions/{MODEL_SIZE}_EOD/{REVISION}_merged.h5"

    # ---------------- dataset & helper -----------------------------------------
    val_ds = load_dataset(
        "pietrolesci/pile-validation",
        split="validation",
        verification_mode="no_checks"
    )

    def build_paths_for_seed(src_dir: str, seed: int):
        """Return list of file paths present on disk, following doc order for 'seed'."""
        doc_order = list(range(len(val_ds)))
        random.Random(seed).shuffle(doc_order)
        paths = []
        for doc_id in doc_order:
            fp = os.path.join(src_dir, f"doc{doc_id}.h5")
            if os.path.exists(fp):
                paths.append(fp)
        return paths

    # ---------------- collect paths (one or two seeds) -------------------------
    paths_A = build_paths_for_seed(SRC_DIR_A, DOC_SEED_A)
    if not paths_A:
        raise RuntimeError(f"[ERR] no .h5 files found in expected order for seed {DOC_SEED_A} at {SRC_DIR_A}")

    paths_all = list(paths_A)
    seed_tags = ["A"] * len(paths_A)     # for debug-only labels

    if USE_SECOND_SEED:
        paths_B = build_paths_for_seed(SRC_DIR_B, DOC_SEED_B)
        if not paths_B:
            print(f"[WARN] second seed enabled but found no files at {SRC_DIR_B}")
        else:
            paths_all.extend(paths_B)
            seed_tags.extend(["B"] * len(paths_B))

    print(f"[INFO] seed {DOC_SEED_A}: {len(paths_A):,} docs found at {SRC_DIR_A}")
    if USE_SECOND_SEED:
        print(f"[INFO] seed {DOC_SEED_B}: {len(paths_B):,} docs found at {SRC_DIR_B}")
    print(f"[INFO] total docs to merge: {len(paths_all):,}")

    # ---------------- first pass: total token count (per seed & total) ---------
    def count_rows(paths, label):
        token_counts = []
        for fp in tqdm(paths, desc=f"Counting tokens [{label}]"):
            with h5py.File(fp, "r") as f:
                token_counts.append(f["nll"].shape[0])   # rows = tokens
        n_docs = len(paths)
        n_rows = int(np.sum(token_counts)) if token_counts else 0
        print(f"[INFO] [{label}] {n_docs:,} docs → {n_rows:,} rows")
        return n_docs, n_rows, token_counts

    N_docs_A, N_rows_A, counts_A = count_rows(paths_A, f"seed {DOC_SEED_A}")
    N_docs_B = N_rows_B = 0
    counts_B = []
    if USE_SECOND_SEED and len(paths_all) > len(paths_A):
        N_docs_B, N_rows_B, counts_B = count_rows(paths_all[len(paths_A):], f"seed {DOC_SEED_B}")

    N_docs_total   = N_docs_A + N_docs_B
    N_tokens_total = N_rows_A + N_rows_B
    print(f"[INFO] [TOTAL] {N_docs_total:,} docs → {N_tokens_total:,} rows in super-matrix")

    # ---------------- create big file -----------------------------------------
    with h5py.File(DST_FILE, "w") as big:
        nll_ds = big.create_dataset(
            "nll",
            shape=(N_tokens_total, CTX-1),
            dtype="f2",
            chunks=CHUNK_SHAPE,
            compression="gzip",
            compression_opts=4,
        )
        ptr_ds = big.create_dataset("doc_ptr", (N_docs_total + 1,), "i8")

        cursor = 0
        for i, fp in enumerate(tqdm(paths_all, desc="Merging")):
            with h5py.File(fp, "r") as f:
                mat = f["nll"][...]          # dense (L,2047) float16
            L = mat.shape[0]

            nll_ds[cursor:cursor+L, :] = mat  # contiguous stripe write
            ptr_ds[i] = cursor
            cursor += L

            # optional debug: every ~10k docs, print a brief update
            if (i + 1) % 10000 == 0:
                print(f"[DBG] merged {i+1:,} docs; cursor now at {cursor:,} rows")

        ptr_ds[-1] = cursor                  # sentinel
        print(f"[DONE] wrote {cursor:,} rows from {N_docs_total:,} docs → {DST_FILE}")

        # Optional: record simple provenance attributes (non-breaking)
        big.attrs["seeds_included"] = f"{DOC_SEED_A}" + (f",{DOC_SEED_B}" if USE_SECOND_SEED and N_docs_B else "")
        big.attrs["src_dir_A"] = SRC_DIR_A
        if USE_SECOND_SEED:
            big.attrs["src_dir_B"] = SRC_DIR_B

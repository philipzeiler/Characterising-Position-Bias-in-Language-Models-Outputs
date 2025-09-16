#!/usr/bin/env python3
"""
Analyse NLLs for a specific token across the first DOC_LIMIT docs
— using the SAME shuffled order & subset that was used to build the merged H5.

What it does:
  1) Tokenise SEARCH_STR with the model's tokenizer; print token id(s)
  2) Rebuild shuffled doc order with DOC_SEED, take first DOC_LIMIT docs
  3) Scan those docs in that order, find all positions of the (first) token id
  4) Map each (doc_pos_in_subset, row_in_doc) → global H5 row via doc_ptr
  5) Pull those rows from nll_ds in chunks and compute:
        • mean NLL by P_t column (size CTX-1)
        • overall mean NLL across all positions
  6) Print a small summary
"""

import numpy as np, h5py
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------- user knobs ----------------
SEARCH_STR   = "Q"                                  # what to search
TOKENIZER_ID = "EleutherAI/pythia-14M"             # pick the tokenizer that matches your model family
MERGED_H5    = r"D:/NLL_matrices/14M_EOD_merged.h5"
CTX          = 2048
DOC_SEED     = 753                                   # must match the seed used during NLL generation
DOC_LIMIT    = 5000                                  # must match how many docs you merged
CHUNK_ROWS   = 4096                                  # H5 row read chunk (tune for RAM / speed)
# --------------------------------------------

# 1) Tokenise
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
ids = tok(SEARCH_STR, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
print(f'[INFO] “{SEARCH_STR}” tokenises to IDs:', ids)
if not ids:
    raise ValueError("Empty tokenisation; nothing to search.")
target_id = ids[0]  # focus on first sub-token; extend to span if needed

# 2) Rebuild shuffled order & subset
val = load_dataset("pietrolesci/pile-validation", split="validation", verification_mode="no_checks")
doc_order = list(range(len(val)))
import random; rnd = random.Random(DOC_SEED); rnd.shuffle(doc_order)
subset_doc_ids = doc_order[:DOC_LIMIT]                      # dataset ids actually used
# Map dataset id → position in subset (0..DOC_LIMIT-1). These positions index doc_ptr.
pos_in_subset = {ds_id: i for i, ds_id in enumerate(subset_doc_ids)}

# 3) Scan docs in that SAME order and collect global H5 rows for matches
hit_rows = []     # global row indices in nll_ds
hit_meta = []     # (subset_pos, ds_id, row_in_doc)

for i, ds_id in tqdm(list(enumerate(subset_doc_ids)), desc="search docs"):
    text = val[ds_id]["text"]
    t_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

    # find all occurrences of the single token id
    for t_idx, tid in enumerate(t_ids):
        if tid == target_id:
            # we don't yet know global row; we'll read doc_ptr after opening H5
            hit_meta.append((i, ds_id, t_idx))

if not hit_meta:
    print("[INFO] No occurrences found in the selected subset.")
    raise SystemExit(0)

# 4) Resolve to global row indices with doc_ptr (positions in subset!)
with h5py.File(MERGED_H5, "r") as f:
    doc_ptr = f["doc_ptr"][...]     # length DOC_LIMIT + 1
    if len(doc_ptr) != (DOC_LIMIT + 1):
        raise ValueError(
            f"doc_ptr length ({len(doc_ptr)}) doesn't match DOC_LIMIT+1 ({DOC_LIMIT+1}). "
            "Make sure DOC_LIMIT matches the number of docs you merged."
        )
    for subset_pos, ds_id, row_in_doc in hit_meta:
        base = int(doc_ptr[subset_pos])         # starting row for that doc in merged matrix
        hit_rows.append(base + row_in_doc)

hit_rows = np.array(hit_rows, dtype=np.int64)

# 5) Pull rows from nll_ds in CHUNK_ROWS and aggregate stats
sum_cols   = np.zeros(CTX - 1, dtype=np.float64)
count_cols = np.zeros(CTX - 1, dtype=np.int64)

with h5py.File(MERGED_H5, "r") as f:
    nll_ds = f["nll"]
    # sanity
    if nll_ds.shape[1] != CTX - 1:
        raise ValueError(f"Expected nll width {CTX-1}, got {nll_ds.shape[1]}")

    for start in range(0, len(hit_rows), CHUNK_ROWS):
        rows_chunk = hit_rows[start:start + CHUNK_ROWS]
        mat = nll_ds[rows_chunk, :].astype(np.float32)          # (chunk, CTX-1)
        # finite filter (should be all finite if your build was good)
        m = np.isfinite(mat)
        if not m.all():
            # just drop NaNs in aggregation
            mat = np.where(m, mat, np.nan)

        # accumulate columnwise means robustly
        valid = np.isfinite(mat)
        sum_cols  += np.nansum(mat, axis=0)
        count_cols += np.sum(valid, axis=0)

mean_by_Pt = np.divide(sum_cols, count_cols, out=np.full_like(sum_cols, np.nan), where=count_cols > 0)
overall_mean = np.nanmean(mean_by_Pt)

# 6) Report
print(f"\n[RESULT] Occurrences of token id {target_id} (from “{SEARCH_STR}”): {len(hit_rows):,}")
print(f"[RESULT] Overall mean NLL across all context positions: {overall_mean:.4f}")
print(f"[RESULT] First 10 positions (P_t=1..10): {np.array2string(mean_by_Pt[:10], precision=4)}")
print(f"[RESULT] Last 10 positions (near P_t={CTX-1}): {np.array2string(mean_by_Pt[-10:], precision=4)}")

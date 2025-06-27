#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Stream a merged HDF5 super-matrix → one Parquet file without RAM blow-ups.
#   HDF5 layout must contain:
#       /nll      (N_tokens_total , 2047)  float16
#       /doc_ptr  (N_docs+1,)              int64
# ---------------------------------------------------------------------------

import h5py, pyarrow as pa, pyarrow.parquet as pq
import numpy as np, tqdm

MERGED_H5   = "D:/NLL_matrices/1.4B_merged.h5"
OUT_PARQUET = "D:/NLL_matrices/1.4B_supermatrix.parquet"

schema = pa.schema([("t", pa.int32()),        # token index in doc
                    ("k", pa.int16()),        # window column (0-2046)
                    ("nll", pa.float16())])   # loss value

with h5py.File(MERGED_H5, "r") as f, \
     pq.ParquetWriter(OUT_PARQUET, schema,
                      compression="zstd", compression_level=3) as writer:

    nll_ds  = f["/nll"]                       # dense float16 matrix
    ptr_ds  = f["/doc_ptr"]                   # prefix-sum offsets

    starts = ptr_ds[:-1]                      # for vectorised token→doc
    n_docs = len(starts)

    for d in tqdm.tqdm(range(n_docs), desc="Docs"):
        s, e   = ptr_ds[d], ptr_ds[d+1]
        mat    = nll_ds[s:e, :]               # (L, 2047)  float16
        rows, cols = np.nonzero(mat)          # all filled cells

        tbl = pa.Table.from_pydict({
            "t":   rows.astype(np.int32),
            "k":   cols.astype(np.int16),
            "nll": mat[rows, cols]            # already float16
        })
        writer.write_table(tbl)

print(f"[DONE] Parquet saved → {OUT_PARQUET}")

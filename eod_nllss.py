import h5py, numpy as np, pandas as pd

MERGED = r"D:/NLL_matrices/1.4B_EOS_merged.h5"   # <-- adjust if needed

with h5py.File(MERGED, "r") as f:
    nll_ds = f["nll"]          # (N_tokens_total , 2047)
    ptr    = f["doc_ptr"][...] # (N_docs+1,)

    # rows that correspond to the *last* token of every doc
    eod_rows = ptr[1:] - 1                     # shape (N_docs,)

    # slice all those rows at once; keep as float32 for stats
    eod_mat  = nll_ds[eod_rows, :].astype(np.float32)   # (N_docs, 2047)

# drop NaNs (appear when the EOD was at the extreme right of the window)
valid_vals = eod_mat[np.isfinite(eod_mat)]

# summary statistics
series = pd.Series(valid_vals)
print(series.describe(percentiles=[.25, .5, .75]).to_string())

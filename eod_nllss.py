import h5py, numpy as np, pandas as pd
from pathlib import Path

MERGED = Path(r"D:/NLL_matrices/1.4B_deduped_EOS_merged_2600_docs.h5")  # adjust if needed

with h5py.File(MERGED, "r") as f:
    nll_ds = f["nll"]                 # shape: (N_tokens_total, 2047)
    ptr    = f["doc_ptr"][...]        # shape: (N_docs + 1,)

    # --------‑‑‑ sanity checks on pointers ---------- #
    if ptr[0] != 0 or not np.all(ptr[1:] > ptr[:-1]):
        raise ValueError("doc_ptr must start with 0 and be strictly increasing")
    if ptr[-1] != nll_ds.shape[0]:
        raise ValueError("doc_ptr[-1] must equal the total token count")

    # indices of the last token per document
    eod_rows = ptr[1:] - 1            # shape: (N_docs,)

    # pull all those rows at once
    eod_mat = nll_ds[eod_rows, :].astype(np.float32)   # (N_docs, 2047)

# --------‑‑‑ finite‑value check ---------- #
finite_mask = np.isfinite(eod_mat)
if not finite_mask.all():
    # locate the first offending element for a clear error message
    bad_idx_flat  = np.where(~finite_mask.ravel())[0][0]      # first bad element (flat index)
    n_docs, width = eod_mat.shape
    row, col      = divmod(bad_idx_flat, width)
    doc_id        = row                       # row i corresponds to document i
    tok_row_idx   = eod_rows[row]             # original row in nll_ds

    bad_val = eod_mat[row, col]
    raise ValueError(
        f"Non‑finite value detected!\n"
        f"  • document index : {doc_id}\n"
        f"  • nll_ds row     : {tok_row_idx}\n"
        f"  • column (ctx L) : {col}\n"
        f"  • value          : {bad_val!r}"
    )

# --------‑‑‑ compute stats ---------- #
valid_vals = eod_mat.ravel()                  # all are finite at this point
series     = pd.Series(valid_vals)

pd.options.display.float_format = lambda x: f"{x:,.4f}"
print(series.describe(percentiles=[.25, .5, .75]).to_string())

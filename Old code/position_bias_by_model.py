#!/usr/bin/env python3
"""
Mean NLL vs *context-window* position (P_t) for several models
-------------------------------------------------------------
• x-axis : P_t = 1 … 2048  (token’s offset inside the 2048-token window)
• optional filters
      – FILTER_PD_NONNEG  → keep only windows whose left context is complete (P_d ≥ 0)
      – MAX_DOC_LEN       → ignore documents longer than N tokens
"""

import numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODELS = [
    ("Pythia 14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    ("Pythia 31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
    ("Pythia 70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
    ("Pythia 160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
    #("Pythia 410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
    ("Pythia 1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
    ("Pythia 1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
    #("Pythia 2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
    ("Pythia 6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
    ("Pythia 12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),
]

CTX              = 2048      # length of sliding evaluation window
FILE_LIM         = 5000      # max docs per model (None → all)
Y_MAX            = 10        # y-axis upper limit

FILTER_PD_NONNEG = False      # True → require P_d ≥ 0, all docs have full left context
MAX_DOC_LEN      = 500       # None → keep every doc, else length cut-off
# ────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 10))

for name, h5_path in MODELS:
    sum_p   = np.zeros(CTX,  dtype=np.float64)   # accumulate NLL per P_t
    count_p = np.zeros_like(sum_p, dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        # number of documents to scan for this model
        N_docs = (len(ptr_ds) - 1 if FILE_LIM is None
                                   else min(FILE_LIM, len(ptr_ds) - 1))

        for d in tqdm(range(N_docs), desc=f"{name:>12}"):
            s, e = ptr_ds[d], ptr_ds[d + 1]          # HDF5 slice for doc d
            L    = e - s                             # document length in tokens
            if MAX_DOC_LEN is not None and L > MAX_DOC_LEN:
                continue                            # skip overly long doc

            mat  = nll_ds[s:e, :].astype(np.float32) # NLL matrix (L × 2047)
            # -----------------------------------------------------------------
            # Build *all* (row, col) coordinates that occur in evaluation:
            #   rows = token index within document  (t  = 0 … L-1)
            #   cols = window offset k-1           (0 … CTX-2)   → P_t = k
            # -----------------------------------------------------------------
            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)
            cols = np.tile( np.arange(CTX - 1, dtype=np.int32), L)

            P_t  = cols + 1                          # 1 … 2048 (position in window)
            P_d  = P_t - rows                        # distance from doc start

            # keep every coordinate that stays inside the 2048-token window
            keep = (P_t <= CTX)
            # optional: demand full left context  → P_d must be non-negative
            if FILTER_PD_NONNEG:
                keep &= (P_d >= 0)

            if not keep.any():
                continue

            p_idx   = P_t[keep] - 1                  # 0-based index for sum/count
            nll_val = mat.ravel()[keep]              # matching NLL values

            # accumulate per P_t
            np.add.at(sum_p,   p_idx, nll_val)
            np.add.at(count_p, p_idx, 1)

    mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)
    # draw one line per model
    ax.plot(np.arange(1, CTX + 1), mean_p, label=name, linewidth=1.2)

# ─────────────────────────────── axis & legend ──────────────────────────────
ax.set_xlim(1, CTX)
ax.set_ylim(0, Y_MAX)
ax.set_xticks([1, 256, 512, 768, 1024, 1280, 1536, 1792, 2048])
ax.set_yticks(np.arange(0, Y_MAX + 1, 1))
ax.set_xlabel("Token position within 2048-token context")
ax.set_ylabel("Negative Log Likelihood")

legend_title = "Model"
if FILTER_PD_NONNEG:
    legend_title += "  (windows with P_d < 0 excluded)"
ax.legend(title=legend_title, loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_by_model.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

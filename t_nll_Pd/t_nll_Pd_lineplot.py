#!/usr/bin/env python3
"""
Memory-frugal **bin-at-a-time** plotter
--------------------------------------
Plots smoothed NLL vs. token index (t) for every *P_d* bin while keeping only
one bin in RAM at any moment.  The visual output (colour gradient, labels,
Seaborn theme, rolling mean window, axis limits) is identical to the previous
script, but the peak memory footprint drops from multiple GB to a few MB.

Strategy
~~~~~~~~
1.  For each P_d bin **b** (outer loop):
    • allocate two 1-D NumPy buffers - `sum_t` (float64) and `count_t`
      (int64) - of length `T_MAX+1`.
    • stream through (a slice of) every document; for rows that belong to bin
      **b** and meet the usual filters, accumulate `NLL` into `sum_t[t]` and
      ++`count_t[t]` via `np.add.at`.
    • after scanning all docs, form the mean   μ[t] = sum_t[t]/count_t[t]
      (where count > 0).
    • smooth μ[t] with the same centred rolling window (width `SMOOTH_W`).
    • call `ax.plot(range(T_MAX+1), μ_smooth, colour, …)`.

2.  Proceed to the next bin, reusing the two small buffers.

This keeps RAM ≈ 2·(T_MAX+1)·8 ≈ 0.1 MB, independent of corpus size.
"""

# ────────────────────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────────────────────
CTX       = 2048          # context window → 2047 lags
T_MAX     = 6000          # crop documents at this token index
PD_MIN    = -3952         # left‑context limit (inclusive)
PD_MAX    =  2047         # right‑context limit (inclusive)
PD_BIN    = 50            # width of a P_d bin
SMOOTH_W  = 51           # 50 left + current + 50 right (must be odd)
Y_MAX     = 20            # NLL cut‑off on y‑axis (nats)
FILE_LIM  = 5000            # read this many docs
MERGED_H5 = "D:/NLL_matrices/1.4B_merged.h5"
# ────────────────────────────────────────────────────────────────────────────

import h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib as mpl, seaborn as sns
from tqdm import tqdm

# pre‑compute bin edges/centres & colour mapper --------------------------------
bins        = np.arange(PD_MIN, PD_MAX + PD_BIN, PD_BIN)
labels      = np.arange(len(bins) - 1)
bin_centres = (bins[:-1] + bins[1:]) / 2
norm  = mpl.colors.Normalize(vmin=PD_MIN, vmax=PD_MAX)
cmap  = plt.cm.Spectral_r

# canvas ----------------------------------------------------------------------
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 5))

# main loop – one P_d bin at a time -------------------------------------------
for b in labels:
    colour   = cmap(norm(bin_centres[int(b)]))
    sum_t    = np.zeros(T_MAX + 1, dtype=np.float64)   # ∑ NLL per t
    count_t  = np.zeros_like(sum_t, dtype=np.int64)    # number of values per t

    # scan documents -----------------------------------------------------------
    with h5py.File(MERGED_H5, "r") as f:
        nll_ds = f["nll"]
        ptr_ds = f["doc_ptr"]
        N_docs = len(ptr_ds) - 1

        for d in tqdm(range(min(N_docs, FILE_LIM)), desc=f"bin {b} … docs"):
            start, end = ptr_ds[d], ptr_ds[d+1]
            mat = nll_ds[start:end, :].astype(np.float32)
            if np.isnan(mat).any():
                raise ValueError(f"NaN detected in document {d}")

            L, C   = mat.shape
            rows   = np.repeat(np.arange(L, dtype=np.int64), C)
            cols   = np.tile(np.arange(C, dtype=np.int64), L)
            nll_v  = mat.ravel()
            P_t    = cols + 1
            P_d    = P_t - rows

            keep = (
                (rows <= T_MAX) &
                (P_d  >= PD_MIN) & (P_d < PD_MAX) &
                (nll_v <= Y_MAX) &
                ((P_d - PD_MIN) // PD_BIN == b)
            )
            if not keep.any():
                continue

            t_sel  = rows[keep]
            nll_sel = nll_v[keep].astype(np.float64, copy=False)
            np.add.at(sum_t,   t_sel, nll_sel)
            np.add.at(count_t, t_sel, 1)

    # derive mean & smooth -----------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_t = np.where(count_t > 0, sum_t / count_t, np.nan)

    mu_smooth = (
        pd.Series(mean_t)
        .rolling(window=SMOOTH_W, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    ax.plot(np.arange(T_MAX + 1), mu_smooth, color=colour,
            linewidth=1.0, alpha=0.9)

# aesthetics ------------------------------------------------------------------
ax.set_xlim(0, T_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_xlabel("t  (token index in document)")
ax.set_ylabel("NLL (nats)")
ax.set_title(
    f"Smoothed NLL vs t   coloured by signed P_d\n"
    f"(P_d bin = {PD_BIN}, smoothing ±{SMOOTH_W//2})"
)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="P_d  (−3952 left … +2047 right)")

plt.tight_layout()
fig_path = "t_nll_Pd_lineplot_binwise_5000_docs.png"
plt.savefig(fig_path, dpi=1200)
print(f"[INFO] saved to {fig_path}")
plt.show()

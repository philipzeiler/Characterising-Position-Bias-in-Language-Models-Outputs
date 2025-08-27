#!/usr/bin/env python3
"""
Memory-frugal **bin-at-a-time** plotter
--------------------------------------
Plots smoothed NLL vs. token index (t) for every *P_d* bin while keeping only
one bin in RAM at any moment.

New in this revision
~~~~~~~~~~~~~~~~~~~~
* Log-spaced P_d buckets:  0, ±1, ±2, ±4, … up to +2047 / –4096.
* Stratified sampling:      each bucket contributes the same number of rows.
* **Uniform colour spacing**: every bucket gets an evenly spaced colour from
  *Spectral_r* (index-based, not value-based) so narrow central bins are no
  longer crammed into one hue.
"""
import h5py, numpy as np, pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.ticker as mtick
# ────────────────────────────────────────────────────────────────────────────
# Parameters
# ────────────────────────────────────────────────────────────────────────────
CTX       = 2048
T_MAX     = 6000
PD_MIN    = -4096                       # <-- CHANGED (keep left tail)
PD_MAX    =  2047
SMOOTH_W  = 1
Y_MAX     = 10
FILE_LIM  = 5000
MODEL_NAME = "Pythia 160M"
MERGED_H5 = "D:/NLL_matrices/160M_EOD_merged.h5"

# ────────────────────────────────────────────────────────────────────────────
# Log-spaced P_d bin edges: 0, ±1, ±2, ±4 …         (powers of two)
# ────────────────────────────────────────────────────────────────────────────
pow2   = {2**k for k in range(1, 12)}                       # 2 … 2048  # <-- CHANGED
signed = {0, 1, -1} | pow2 | {-x for x in pow2}             # add both signs  # <-- CHANGED
pd_edges = np.array(sorted(signed | {PD_MIN, PD_MAX + 1}))  # final edges  # <-- CHANGED

pd_labels  = np.arange(len(pd_edges) - 1)
pd_centres = (pd_edges[:-1] + pd_edges[1:]) / 2

# colour map: **evenly spaced by bin index** --------------------------------- 
import matplotlib as mpl, matplotlib.pyplot as plt           # imports early
n_bins    = len(pd_labels)                                   # <-- CHANGED
cmap      = plt.cm.Spectral_r
idx_norm  = mpl.colors.Normalize(vmin=0, vmax=n_bins - 1)    # <-- CHANGED

# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(20, 10))
sns.set_context("paper", font_scale=2.1)

# ────────────────────────────── Pass 0: count rows per bin ──────────────────
bin_counts = np.zeros(n_bins, dtype=np.int64)
with h5py.File(MERGED_H5, "r") as f:
    nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
    for d in tqdm(range(min(FILE_LIM, len(ptr_ds) - 1)), desc="count pass"):
        s, e = ptr_ds[d], ptr_ds[d + 1]
        mat  = nll_ds[s:e, :]
        L, C = mat.shape
        rows = np.repeat(np.arange(L, dtype=np.int32), C)
        cols = np.tile(np.arange(C, dtype=np.int32), L)
        P_d  = cols + 1 - rows
        keep = (rows <= T_MAX) & (P_d >= PD_MIN) & (P_d <= PD_MAX)
        idx  = np.searchsorted(pd_edges, P_d[keep], side="right") - 1
        np.add.at(bin_counts, idx, 1)

N_target = bin_counts.min()
print("[INFO] per-bin populations:", bin_counts)
print("[INFO] using N_target =", N_target)

# ────────────────────────────── Main loop: one bin at a time ────────────────
rng = np.random.default_rng(seed=0)
for b in pd_labels:
    colour   = cmap(idx_norm(b))                            # <-- CHANGED
    sum_t    = np.zeros(T_MAX + 1, dtype=np.float64)
    count_t  = np.zeros_like(sum_t, dtype=np.int64)
    res_t, res_nll = [], []                                 # reservoir buf
    seen = 0

    with h5py.File(MERGED_H5, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        for d in tqdm(range(min(FILE_LIM, len(ptr_ds) - 1)),
                      desc=f"bin {b}"):
            s, e = ptr_ds[d], ptr_ds[d + 1]
            mat  = nll_ds[s:e, :].astype(np.float32)
            L, C = mat.shape
            rows = np.repeat(np.arange(L, dtype=np.int32), C)
            cols = np.tile(np.arange(C, dtype=np.int32), L)
            nll  = mat.ravel()
            P_d  = cols + 1 - rows

            keep = (
                (rows <= T_MAX) &
                (P_d >= PD_MIN) & (P_d <= PD_MAX) 
            #&    (nll <= Y_MAX)
            )
            if not keep.any():
                continue
            rows, P_d, nll = rows[keep], P_d[keep], nll[keep]

            in_bin = (P_d >= pd_edges[b]) & (P_d < pd_edges[b + 1])
            if not in_bin.any():
                continue
            t_bin, nll_bin = rows[in_bin], nll[in_bin]

            for i in range(len(t_bin)):
                seen += 1
                if len(res_t) < N_target:
                    res_t.append(t_bin[i]); res_nll.append(nll_bin[i])
                else:
                    j = rng.integers(0, seen)
                    if j < N_target:
                        res_t[j]   = t_bin[i]
                        res_nll[j] = nll_bin[i]

    # accumulate & smooth -----------------------------------------------------
    np.add.at(sum_t,   res_t,   res_nll)
    np.add.at(count_t, res_t,   1)

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

# ────────────────────────────── Axis & colour-bar ───────────────────────────
ax.set_xscale('symlog', base=2, linthresh=2)   # keeps t = 0 visible
ax.set_xlim(0, T_MAX)        
powers = [0] + [2**k for k in range(int(np.log2(T_MAX)) + 1)]
ax.set_xticks(powers)
ax.get_xaxis().set_major_formatter(mtick.FormatStrFormatter('%d'))        
ax.set_ylim(0, Y_MAX)
ax.set_yticks(np.arange(0, Y_MAX + 1, 1))
ax.set_xlabel("Token position within document")
ax.set_ylabel("Negative Log Likelihood")
ax.set_title(
    f"{MODEL_NAME}: {N_target:,} tokens randomly sampled per document position bin"
)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=idx_norm)        # <-- CHANGED
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, label="Document start position relative to context window")     # <-- CHANGED
cb.set_ticks(pd_labels)                                     # <-- CHANGED
cb.set_ticklabels([
    str(pd_edges[i])
    if pd_edges[i] + 1 == pd_edges[i + 1]
    else f"{pd_edges[i]}…{pd_edges[i + 1] - 1}"
    for i in pd_labels
])
plt.tight_layout()
plt.savefig(
    f"D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/{MODEL_NAME}_t_nll_Pd.pdf",
    format="pdf",                     # ← explicit, but not strictly required
    bbox_inches="tight",              # trims white margins
)
plt.show()

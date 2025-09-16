#!/usr/bin/env python3
"""
NLL vs P_d  -  hue = t-bin, **stratified and size-matched across bins**

Two-pass algorithm
------------------
1. First pass: count how many rows each t-bin contains (no storage).
2. Pick N_target = min(counts)   # or any value ≤ min(counts)
3. Second pass: keep exactly N_target rows per bin via reservoir sampling.

Revision  ▸  log₂ x-axis  ▸  extended t-bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* x-axis now shown in log₂ scale, so P_d = 1,2,4,8,…,2048 are evenly spaced.
* `T_EDGES` automatically spans powers-of-two up to 2048 and then ∞.
"""


"""
The shaded band around each line defaults to a 95% bootstrap confidence interval for the mean. 
Seaborn aggregates y at each x using the mean and then draws the 95% CI around that estimate by bootstrapping; 
you can turn it off or switch to other intervals with 
errorbar=... (e.g., None, "sd", "se", or ("ci", 99) for a 99% CI).
seaborn.pydata.org

How to interpret it: the band shows uncertainty in the estimated mean at each x, 
not the spread of the raw data and not a prediction interval. 
If you repeatedly resampled your dataset and re-estimated the mean, 
~95% of those CIs would cover the true mean (frequentist interpretation). 
Its not “±1 std dev,” and it doesnt say where new observations will fall. 
"""

import h5py, numpy as np, pandas as pd, math
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

# ---------------------------------------------------------------- parameters
MODEL_NAME  = "Pythia 1.4B"
MERGED_H5   = "D:/NLL_matrices/1.4B_EOD_merged.h5"   #size_EOD_merged.h5
FILE_LIMIT  = 5000         # how many docs to scan
#Y_MAX       = 10           # (disabled in filters below)
PD_MIN, PD_MAX = -2_047, 2_047
#-1023, 1023

# NEW: toggle to place the legend outside (right side)
LEGEND_OUTSIDE = True

# dynamic powers-of-two bucket edges ------------------------------------------------
T_EDGES = [0, 1] + [2**k for k in range(1, 12)] + [math.inf]#12 for pythia, 11 for gpt2
T_LABELS = [
    f"{T_EDGES[i]}"                                   # show just “0”, “1”
    if T_EDGES[i] + 1 == T_EDGES[i + 1]               # width-1 bin?
    else f"{T_EDGES[i]}–{T_EDGES[i + 1] - 1}"         # closed range
    if math.isfinite(T_EDGES[i + 1])                  # finite upper edge
    else f"{T_EDGES[i]}+"                             # open-ended last bin
    for i in range(len(T_EDGES) - 1)
]

rng = np.random.default_rng(seed=0)

# ───────────────────────────── helpers ──────────────────────────────────────
def t_bin_labels(t_arr):
    """Vectorised mapping of t values to label strings."""
    idx = np.searchsorted(T_EDGES, t_arr, side="right") - 1
    return np.array(T_LABELS, dtype=object)[idx]

# ───────────────────────────── 1st pass – counts ────────────────────────────
bin_counts = dict.fromkeys(T_LABELS, 0)

with h5py.File(MERGED_H5, "r") as f:
    nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
    N_docs = min(FILE_LIMIT, len(ptr_ds)-1)

    for d in tqdm(range(N_docs), desc="pass 1: counting"):
        start, end = ptr_ds[d], ptr_ds[d+1]
        mat = nll_ds[start:end, :]

        L, C = mat.shape
        rows = np.repeat(np.arange(L, dtype=np.int32), C)
        cols = np.tile(  np.arange(C, dtype=np.int32), L)
        nll  = mat.ravel()

        P_t = cols + 1
        P_d = P_t - rows
        ok  = (P_d >= PD_MIN) & (P_d <= PD_MAX)  # P_d>=1  <-- CHANGED (min)
        if not ok.any():
            continue

        labels = t_bin_labels(rows[ok])
        for lbl in np.unique(labels):
            bin_counts[lbl] += (labels == lbl).sum()

print("[INFO] population sizes:")
for k, v in bin_counts.items():
    print(f"  {k:>7}: {v:,}")

N_target = min(bin_counts.values())
print(f"[INFO] using N_target = {N_target:,} rows per bin")

# ───────────────────────────── 2nd pass – sampling ──────────────────────────
reservoir = {lbl: [] for lbl in T_LABELS}
seen_cnt  = dict.fromkeys(T_LABELS, 0)

with h5py.File(MERGED_H5, "r") as f:
    nll_ds, ptr_ds = f["nll"], f["doc_ptr"]

    for d in tqdm(range(N_docs), desc="pass 2: sampling"):
        start, end = ptr_ds[d], ptr_ds[d+1]
        mat = nll_ds[start:end, :].astype(np.float32)

        L, C = mat.shape
        rows = np.repeat(np.arange(L, dtype=np.int32), C)
        cols = np.tile(  np.arange(C, dtype=np.int32), L)
        nll  = mat.ravel()

        P_t = cols + 1
        P_d = P_t - rows
        ok  = (P_d >= PD_MIN) & (P_d <= PD_MAX)
        if not ok.any():
            continue

        labels = t_bin_labels(rows[ok])
        for i in range(len(labels)):
            lbl = labels[i]
            seen_cnt[lbl] += 1
            buf = reservoir[lbl]
            if len(buf) < N_target:
                buf.append((P_d[i], nll[i], lbl))
            else:
                j = rng.integers(0, seen_cnt[lbl])
                if j < N_target:
                    buf[j] = (P_d[i], nll[i], lbl)

print("\n[INFO] final buffer sizes:")
for lbl in T_LABELS:
    print(f"  {lbl:>7}: {len(reservoir[lbl]):,}")

# ───────────────────────────── DataFrame & plot ─────────────────────────────
flat = [row for buf in reservoir.values() for row in buf]

# Aggregate means to reduce Seaborn bootstrapping time (optional)
df_plot = pd.DataFrame(flat, columns=["P_d", "NLL", "t_bin"])

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)

plt.figure(figsize=(20, 10))

# --- NEW: red→yellow→blue palette to match other plots ----------------------
USE_REVERSED_COLORS = False  # set True if you want blue→red instead
n_bins   = len(T_LABELS)
cmap     = mpl.colormaps.get_cmap("RdYlBu")  # modern API
if USE_REVERSED_COLORS:
    palette_list = [cmap(1 - (i / max(n_bins - 1, 1))) for i in range(n_bins)]
else:
    palette_list = [cmap(i / max(n_bins - 1, 1)) for i in range(n_bins)]
PALETTE_MAP = {lbl: palette_list[i] for i, lbl in enumerate(T_LABELS)}

ax = sns.lineplot(
    data=df_plot, x="P_d", y="NLL", hue="t_bin",
    hue_order=T_LABELS,               # ensure stable label→color mapping
    palette=PALETTE_MAP               # <-- was palette="viridis"
)


# log₂ x-axis ---------------------------------------------------------------
ax.set_xscale('symlog', base=2, linthresh=2)
ax.set_xlim(PD_MIN, PD_MAX)
ax.set_ylim(0, 10)

# ───── legend placement (toggle) ────────────────────────────────────────────
if LEGEND_OUTSIDE:
    # Reserve a right margin so the legend doesn't overlap the plot
    plt.subplots_adjust(right=0.78)
    leg = ax.legend(
        title="Relative token position",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),    # just outside the right edge
        frameon=True,
        handlelength=4,
        borderaxespad=0.0,
    )
else:
    leg = ax.legend(title="Token position within document",
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    borderaxespad=0)

# make legend line samples thicker, either way
for line in leg.get_lines():
    line.set_linewidth(6)

ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticks([-2048,-1024,-512,-256,-128,-64,-32,-16,-8,-4,-2,-1, 0, 1,2,4,8,16,32,64,128,256,512,1024,2048])

plt.xlabel("Document start position offset to left context window edge")
plt.ylabel("Negative Log Likelihood")
#plt.title(f"{MODEL_NAME}: {N_target:,} tokens randomly sampled per token position bin")
plt.tight_layout()
plt.savefig(
    f"D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/{MODEL_NAME}_Pd_nll_t.pdf",
    format="pdf",                     # ← explicit, but not strictly required
    bbox_inches="tight",              # trims white margins
)

plt.show()

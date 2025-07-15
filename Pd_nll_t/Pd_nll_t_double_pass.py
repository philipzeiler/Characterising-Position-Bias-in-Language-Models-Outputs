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

import h5py, numpy as np, pandas as pd, math
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------- parameters
MERGED_H5   = "D:/NLL_matrices/1.4B_EOS_merged.h5"
FILE_LIMIT  = 2740          # how many docs to scan
#Y_MAX       = 10           # (disabled in filters below)
PD_MIN, PD_MAX = -2_047, 2_047

# dynamic powers‑of‑two bucket edges ------------------------------------------------
T_EDGES = [0, 1] + [2**k for k in range(1, 12)] + [math.inf]  # <-- CHANGED
T_LABELS = [f"{T_EDGES[i]}–{T_EDGES[i+1]-1}" if math.isfinite(T_EDGES[i+1])
            else f"{T_EDGES[i]}+" for i in range(len(T_EDGES)-1)]

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
plt.figure(figsize=(12, 4))
ax = sns.lineplot(data=df_plot, x="P_d", y="NLL", hue="t_bin",
                  palette="viridis")

# log₂ x‑axis ---------------------------------------------------------------
ax.set_xscale('symlog', base=2, linthresh=2)
ax.set_xlim(PD_MIN, PD_MAX)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xticks([-2048,-1024,-512,-256,-128,-64,-32,-16,-8,-4,-2,-1, 0, 1,2,4,8,16,32,64,128,256,512,1024,2048])

plt.xlabel("P_d  (log2 scale - doc-start position in context window)")
plt.ylabel("NLL (nats)")
plt.title(
    f"NLL vs P_d  ({N_target:,} rows per t-bin, stratified, log2 x-axis)")
plt.tight_layout()
plt.show()

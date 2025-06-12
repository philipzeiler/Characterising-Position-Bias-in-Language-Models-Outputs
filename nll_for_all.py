#!/usr/bin/env python3
# Exact per-position mean / median / IQR with ±1 SE ribbon
# Uses a tidy pandas DataFrame for the line plot.

import glob, os, h5py, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

FOLDER = "D:/NLL_matrices/1_4B"
paths  = glob.glob(os.path.join(FOLDER, "*.h5"))
if not paths:
    raise RuntimeError("no .h5 files found")

with h5py.File(paths[0]) as f:
    CONTEXT = f["nll"].shape[1]

# ── running sums for mean and variance ─────────────────────────────
sum_vec   = np.zeros(CONTEXT, dtype=np.float64)
sumsq_vec = np.zeros(CONTEXT, dtype=np.float64)
count_vec = np.zeros(CONTEXT, dtype=np.int64)

for p in paths:
    with h5py.File(p) as f:
        mat = f["nll"][...].astype(np.float64)
    mask         = np.isfinite(mat)
    sum_vec   += np.nan_to_num(mat, nan=0.0).sum(axis=0)
    sumsq_vec += np.nan_to_num(mat**2, nan=0.0).sum(axis=0)
    count_vec  += mask.sum(axis=0)

mean_vec = sum_vec / count_vec
var_vec  = (sumsq_vec / count_vec) - mean_vec**2
std_vec  = np.sqrt(np.maximum(var_vec, 0.0))
se_vec   = std_vec / np.sqrt(count_vec)

# ── exact median & IQR (25 – 75 %) per column ─────────────────────
median_vec = np.empty(CONTEXT, dtype=np.float32)
q1_vec     = np.empty(CONTEXT, dtype=np.float32)
q3_vec     = np.empty(CONTEXT, dtype=np.float32)

print("[INFO] computing exact quartiles …")
for col in tqdm(range(CONTEXT)):
    col_vals = []
    for p in paths:
        with h5py.File(p) as f:
            data = f["nll"][:, col][:]
        col_vals.append(data[np.isfinite(data)])
    col_all = np.concatenate(col_vals).astype(np.float32)
    median_vec[col] = np.median(col_all)
    q1_vec[col]     = np.percentile(col_all, 25)
    q3_vec[col]     = np.percentile(col_all, 75)

# ── build tidy DataFrame for lineplot ─────────────────────────────
pos = np.arange(CONTEXT)
df_lines = pd.DataFrame({
    "pos": np.tile(pos, 2),
    "value": np.concatenate([mean_vec, median_vec]),
    "stat": np.repeat(["mean", "median"], CONTEXT)
})

# ── plot ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
pos = np.arange(CONTEXT)

plt.figure(figsize=(12, 3))
ax = plt.gca()

# IQR (25–75 %) band — retains original teal colour
ax.fill_between(pos, q1_vec, q3_vec,
                alpha=.25, color=sns.color_palette("Set1")[0],
                label="IQR (25–75 %)")

# ±1 SD band — light orange
ax.fill_between(pos, mean_vec - std_vec, mean_vec + std_vec,
                alpha=.15, color=sns.color_palette("Set2")[1],
                label="±1 SD")

# ±1 SE ribbon — thin grey
ax.fill_between(pos, mean_vec - se_vec, mean_vec + se_vec,
                alpha=.12, color="grey", label="±1 SE")

# mean & median lines
sns.lineplot(x=pos, y=mean_vec,   linewidth=1.8, ax=ax,
             label="mean",   color=sns.color_palette("Set1")[0])
sns.lineplot(x=pos, y=median_vec, linewidth=1.2, ax=ax,
             label="median", color=sns.color_palette("Set1")[1], linestyle="--")

# axis limits (log-scale)
finite = np.concatenate([mean_vec, q1_vec, q3_vec])
finite = finite[np.isfinite(finite)]
ymin   = max(1e-4, 10 ** np.floor(np.log10(finite.min())))

ax.set(xlim=(0, CONTEXT-1), ylim=(ymin, None),
       xlabel="context position (0–2046)", ylabel="NLL (log scale)")
ax.set_yscale("log")
ax.set_title("Per-position NLL across all documents\n"
             "IQR band · ±1 SD band · ±1 SE ribbon")
ax.legend()
plt.tight_layout()
plt.show()

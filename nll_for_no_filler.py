#!/usr/bin/env python3
# Exact per-position mean / median / IQR for NLL matrices,
# considering ONLY tokens with index ≥ 2048 (full-context tokens).

import glob, os, h5py, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

FOLDER   = "nll_matrices"
H5_GLOB  = os.path.join(FOLDER, "*.h5")
CTX      = 2048                         # context length
TAIL_MIN = CTX                          # first row to include (0-based)

# ── gather paths of *long* documents ───────────────────────────────────────
paths_long = []
for p in glob.glob(H5_GLOB):
    with h5py.File(p, "r") as f:
        rows, cols = f["nll"].shape
    if rows > TAIL_MIN:                 # need at least 2049 tokens
        paths_long.append((p, rows))

if not paths_long:
    raise RuntimeError("no document longer than 2048 tokens found")

print(f"[INFO] {len(paths_long)} matrices have > {TAIL_MIN} tokens")

CONTEXT = cols                          # normally 2047

# ── mean (exact) over the tails ────────────────────────────────────────────
sum_vec   = np.zeros(CONTEXT, dtype=np.float64)
count_vec = np.zeros(CONTEXT, dtype=np.int64)
nan_found = False

for p, rows in paths_long:
    with h5py.File(p, "r") as f:
        tail = f["nll"][TAIL_MIN:, :].astype(np.float64)   # slice rows 2048+
    if np.isnan(tail).any():
        print(f"\033[91m[WARNING] NaNs detected in {p}\033[0m")
        nan_found = True
    mask        = ~np.isnan(tail)
    sum_vec   += np.nan_to_num(tail, nan=0.0).sum(axis=0)
    count_vec += mask.sum(axis=0)

mean_vec = sum_vec / count_vec

# ── exact per-column median & IQR on tails ─────────────────────────────────
median_vec = np.empty(CONTEXT, dtype=np.float32)
q1_vec     = np.empty(CONTEXT, dtype=np.float32)
q3_vec     = np.empty(CONTEXT, dtype=np.float32)

print("[INFO] computing exact quartiles over token-tail subset…")
for col in tqdm(range(CONTEXT)):
    col_vals = []
    for p, rows in paths_long:
        with h5py.File(p, "r") as f:
            data = f["nll"][TAIL_MIN:, col][:]             # 1-D slice
        data = data[~np.isnan(data)]
        if data.size:
            col_vals.append(data.astype(np.float32))
    col_all = np.concatenate(col_vals)
    median_vec[col] = np.median(col_all)
    q1_vec[col]     = np.percentile(col_all, 25)
    q3_vec[col]     = np.percentile(col_all, 75)

msg = "[INFO] no NaNs encountered" if not nan_found else \
      "\033[91m[WARNING] NaNs found - inspect data\033[0m"
print(msg)

# ── plotting (unchanged styling) ───────────────────────────────────────────
sns.set_theme(style="whitegrid")
pos = np.arange(CONTEXT)

plt.figure(figsize=(12, 3))
ax = plt.gca()
ax.fill_between(pos, q1_vec, q3_vec, alpha=.25,
                color=sns.color_palette()[0], label="IQR (25-75 %)")
sns.lineplot(x=pos, y=mean_vec,   linewidth=1.8, ax=ax,
             label="mean",   color=sns.color_palette()[0])
sns.lineplot(x=pos, y=median_vec, linewidth=1.2, ax=ax,
             label="median", color=sns.color_palette()[2], linestyle="--")

finite = np.concatenate([mean_vec, median_vec, q1_vec, q3_vec])
finite = finite[np.isfinite(finite)]
ymin   = max(10 ** np.floor(np.log10(finite.min())), 1e-4)

ax.set(xlim=(0, CONTEXT-1), ylim=(ymin, None),
       xlabel="context position (0-2046)",
       ylabel="NLL (log scale)")
ax.set_yscale("log")
ax.set_title(f"NLL statistics for document tails (tokens ≥ {TAIL_MIN}) "
             f"across {len(paths_long)} matrices")
ax.legend()
plt.tight_layout()
plt.show()

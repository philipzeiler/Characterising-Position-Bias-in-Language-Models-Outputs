#!/usr/bin/env python3
# Exact per-position mean/median/IQR for all NLL matrices
# with NaN‐detection warnings.

import glob, os, h5py, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

FOLDER = "nll_matrices"
paths  = glob.glob(os.path.join(FOLDER, "*.h5"))
if not paths:
    raise RuntimeError("no .h5 files found")

with h5py.File(paths[0]) as f:                                   # hyperslab docs
    CONTEXT = f["nll"].shape[1]

# ── mean (exact) ───────────────────────────────────────────────
sum_vec   = np.zeros(CONTEXT, dtype=np.float64)
count_vec = np.zeros(CONTEXT, dtype=np.int64)
nan_found = False

for p in paths:
    with h5py.File(p) as f:
        mat = f["nll"][...].astype(np.float64)                   # promote precision
    if np.isnan(mat).any():                                      # isnan docs
        print(f"\033[91m[WARNING] NaNs detected in {p}\033[0m")
        nan_found = True
    mask = ~np.isnan(mat)
    sum_vec   += np.nan_to_num(mat, nan=0.).sum(axis=0)
    count_vec += mask.sum(axis=0)

mean_vec = sum_vec / count_vec                                   # exact nanmean

# ── exact median & IQR (column-wise concat) ───────────────────
median_vec = np.empty(CONTEXT, dtype=np.float32)
q1_vec     = np.empty(CONTEXT, dtype=np.float32)
q3_vec     = np.empty(CONTEXT, dtype=np.float32)

print("[INFO] computing exact quartiles (this may take a while)…")
for col in tqdm(range(CONTEXT)):
    col_vals = []
    for p in paths:
        with h5py.File(p) as f:
            data = f["nll"][:, col][:]                           # hyperslab slice
        data = data[~np.isnan(data)]
        if data.size:
            col_vals.append(data.astype(np.float32))
    col_all = np.concatenate(col_vals)                           # concatenate docs
    median_vec[col] = np.median(col_all)                         # exact median
    q1_vec[col]     = np.percentile(col_all, 25)                 # exact pct 25
    q3_vec[col]     = np.percentile(col_all, 75)

if nan_found:
    print("\033[91m[WARNING] One or more files contained NaNs; "
          "verify data integrity before publication.\033[0m")
else:
    print("[INFO] no NaNs encountered - matrices are fully populated.")

# ── plot ───────────────────────────────────────────────────────
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

finite_vals = np.concatenate([mean_vec, median_vec, q1_vec, q3_vec])
finite_vals = finite_vals[np.isfinite(finite_vals)]
ymin_auto   = 10 ** (np.floor(np.log10(finite_vals.min())))
ymin_final  = max(ymin_auto, 1e-4)

ax.set(xlim=(0, CONTEXT-1),
       ylim=(ymin_final, None),
       xlabel="context position (0-2046)",
       ylabel="NLL (log scale)")
ax.set_yscale("log")
ax.set_title("Exact per-position NLL across all documents")
ax.legend()
plt.tight_layout()
plt.show()

#!/usr/bin/env python3
# Exact per-position mean / median / IQR for *short-document* NLL matrices
# (only matrices whose first dimension < 500 are included)

import glob, os, h5py, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

FOLDER   = "nll_matrices"
H5_GLOB  = os.path.join(FOLDER, "*.h5")
MAX_LEN  = 500                     # keep docs shorter than this many tokens

# ── discover & filter paths ────────────────────────────────────────────────
all_paths = glob.glob(H5_GLOB)
if not all_paths:
    raise RuntimeError("no .h5 files found")

paths = []
for p in all_paths:
    with h5py.File(p, "r") as f:
        rows, cols = f["nll"].shape
    if rows < MAX_LEN:
        paths.append(p)

if not paths:
    raise RuntimeError(f"no matrices shorter than {MAX_LEN} tokens found")

print(f"[INFO] {len(paths)} of {len(all_paths)} matrices "
      f"passed the <{MAX_LEN}-token filter")

# ── infer CONTEXT length from first file ───────────────────────────────────
with h5py.File(paths[0], "r") as f:
    CONTEXT = f["nll"].shape[1]          # == 2047 for 2 048-token context

# ── mean (exact) ───────────────────────────────────────────────────────────
sum_vec   = np.zeros(CONTEXT, dtype=np.float64)
count_vec = np.zeros(CONTEXT, dtype=np.int64)
nan_found = False

for p in paths:
    with h5py.File(p, "r") as f:
        mat = f["nll"][...].astype(np.float64)
    if np.isnan(mat).any():
        print(f"\033[91m[WARNING] NaNs detected in {p}\033[0m")
        nan_found = True
    mask        = ~np.isnan(mat)
    sum_vec   += np.nan_to_num(mat, nan=0.0).sum(axis=0)
    count_vec += mask.sum(axis=0)

mean_vec = sum_vec / count_vec                               # exact nanmean

# ── exact median & IQR (column-wise concat) ───────────────────────────────
median_vec = np.empty(CONTEXT, dtype=np.float32)
q1_vec     = np.empty(CONTEXT, dtype=np.float32)
q3_vec     = np.empty(CONTEXT, dtype=np.float32)

print("[INFO] computing exact quartiles (this may take a while)…")
for col in tqdm(range(CONTEXT)):
    col_vals = []
    for p in paths:
        with h5py.File(p, "r") as f:
            data = f["nll"][:, col][:]
        data = data[~np.isnan(data)]
        if data.size:
            col_vals.append(data.astype(np.float32))
    col_all = np.concatenate(col_vals)      # all values for this position
    median_vec[col] = np.median(col_all)
    q1_vec[col]     = np.percentile(col_all, 25)
    q3_vec[col]     = np.percentile(col_all, 75)

if nan_found:
    print("\033[91m[WARNING] NaNs were found; verify data integrity\033[0m")
else:
    print("[INFO] no NaNs encountered – matrices are fully populated")

# ── plot (unchanged) ───────────────────────────────────────────────────────
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
       xlabel="context position (0-2046)", ylabel="NLL (log scale)")
ax.set_yscale("log")
ax.set_title(f"Per-position NLL for docs < {MAX_LEN} tokens "
             f"({len(paths)} matrices)")
ax.legend()
plt.tight_layout()
plt.show()

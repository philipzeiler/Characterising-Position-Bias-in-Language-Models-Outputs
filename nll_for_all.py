import glob, h5py, numpy as np
import seaborn as sns, matplotlib.pyplot as plt, os

# --- parameters -------------------------------------------------------------
FOLDER = "nll_matrices"          # folder containing per-doc *.h5 saved earlier
CONTEXT = 2047                   # number of context positions in each matrix

# --- pass 1: aggregate ------------------------------------------------------
sum_vec   = np.zeros(CONTEXT, dtype=np.float64)   # running sum of NLLs
count_vec = np.zeros(CONTEXT, dtype=np.int64)     # running count (non-NaN)

paths = glob.glob(os.path.join(FOLDER, "*.h5"))
print(f"[INFO] found {len(paths)} files in {FOLDER!r}")

for p in paths:
    with h5py.File(p, "r") as f:
        mat = f["nll"][...].astype(np.float32)    # (L, 2047)
    # mask non-NaNs
    good = ~np.isnan(mat)
    # accumulate
    sum_vec   += np.nan_to_num(mat, nan=0.).sum(axis=0)
    count_vec += good.sum(axis=0)

# avoid division by zero for empty columns (should not happen)
mean_vec = np.divide(sum_vec, count_vec, out=np.zeros_like(sum_vec), where=count_vec>0)

# --- aggregate stats per column for variability (median/IQR) ---------------
# collect per-file stats to avoid stacking all matrices in RAM
median_list, q1_list, q3_list = [], [], []

for p in paths:
    with h5py.File(p, "r") as f:
        mat = f["nll"][...].astype(np.float32)
    median_list.append(np.nanmedian(mat, axis=0))
    q1_list.append(np.nanpercentile(mat, 25, axis=0))
    q3_list.append(np.nanpercentile(mat, 75, axis=0))

median_vec = np.nanmean(np.stack(median_list), axis=0)
q1_vec     = np.nanmean(np.stack(q1_list), axis=0)
q3_vec     = np.nanmean(np.stack(q3_list), axis=0)

# --- plot -------------------------------------------------------------------
sns.set_theme(style="whitegrid")
pos = np.arange(CONTEXT)

plt.figure(figsize=(12, 3))
ax = plt.gca()
ax.fill_between(pos, q1_vec, q3_vec, alpha=.25,
                color=sns.color_palette()[0], label="IQR (25–75 pct)")
sns.lineplot(x=pos, y=mean_vec,   linewidth=1.8, ax=ax,
             label="mean",   color=sns.color_palette()[0])
sns.lineplot(x=pos, y=median_vec, linewidth=1.2, ax=ax,
             label="median", color=sns.color_palette()[2], linestyle="--")

EPS = 1e-6
ax.set(xlim=(0, CONTEXT-1), ylim=(EPS, None),
       xlabel="context position (0–2046)",
       ylabel="NLL (log scale)")
ax.set_yscale("log")
ax.set_title("Average per-position NLL across all documents")
ax.legend()
plt.tight_layout()
plt.show()

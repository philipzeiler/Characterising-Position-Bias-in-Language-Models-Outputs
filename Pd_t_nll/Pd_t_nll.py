#!/usr/bin/env python3
# 6 000 × 6 000 heat-map of mean NLL(p_d, t)

import glob, os, h5py, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# ---------------- parameters ---------------------------
CTX          = 2048
PD_MIN       = -3952               # left bound on x-axis
PD_MAX       = 2048                # right bound
GRID_X       = PD_MAX - PD_MIN + 1 # = 8 049   (but we’ll clip to 6 000)
GRID_Y       = 6000

# restrict to exactly 6 000 columns so the figure is square
X_OFFSET     = PD_MIN              # shift P_d so min maps to 0
def pd_to_col(p): return p - X_OFFSET    # vectorised later
def clip_cols(c): return np.clip(c, 0, GRID_Y-1)

# -------------- allocate running stats -----------------
sum_mat  = np.zeros((GRID_Y, GRID_Y), dtype=np.float64)  # sum of NLLs
cnt_mat  = np.zeros((GRID_Y, GRID_Y), dtype=np.int32)    # how many token hits

# -------------- iterate over HDF5 matrices -------------
FOLDER = "D:/NLL_matrices/2.8B"
paths  = glob.glob(os.path.join(FOLDER, "*.h5"))
if not paths:
    raise RuntimeError("No .h5 files found")

for fp in tqdm(paths, desc="Accumulating"):
    with h5py.File(fp, "r") as f:
        mat = f["nll"][...]            # (L, 2047) float16 + NaNs

    mask          = np.isfinite(mat)
    if not mask.any():
        continue

    rows, cols    = np.nonzero(mask)         # rows = t, cols = k-1
    nll_vals      = mat[rows, cols].astype(np.float64)

    P_t           = cols + 1
    P_d           = P_t - rows

    # keep only desired x-range and t-range
    keep_mask = (P_d >= PD_MIN) & (P_d <= PD_MAX) & (rows < GRID_Y)
    if not keep_mask.any():
        continue

    t_idx   = rows[keep_mask]                       # y coordinate
    x_idx   = clip_cols(pd_to_col(P_d[keep_mask]))  # x coordinate (0-5999)
    vals    = nll_vals[keep_mask]

    # accumulate
    np.add.at(sum_mat, (t_idx, x_idx), vals)
    np.add.at(cnt_mat, (t_idx, x_idx), 1)

# -------------- final mean -----------------------------
mean_mat = np.divide(sum_mat, cnt_mat,
                     out=np.full_like(sum_mat, np.nan),
                     where=cnt_mat > 0)

LEFT  = CTX - 6000                # 2048 - 6000 = -3952
RIGHT = 2048

# -------------- plot -----------------------------------
sns.set_theme(style="white")
plt.figure(figsize=(8, 8))

vmax_auto = np.nanmax(mean_mat)    #dynamic top of colour map
vmax_auto = 3 #I hardcoded it for testing
cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

im = plt.imshow(mean_mat,
                origin="lower",
                extent=[LEFT, RIGHT,
                        0, GRID_Y-1],
                aspect="auto",
                cmap=cmap,
                vmin=0, vmax=vmax_auto,      #dynamic vmax
                rasterized=True)

plt.colorbar(im, label="mean NLL (nats)")
plt.xlabel(f"P_d  (document start, {LEFT} … {RIGHT})")
plt.ylabel("t  (token index inside document)")

# optional: nicer tick marks
plt.xticks([LEFT, LEFT//2, 0, RIGHT//2, RIGHT])

plt.tight_layout()
plt.savefig("nll_pd_t_heatmap_test.png", dpi=1200)
plt.show()
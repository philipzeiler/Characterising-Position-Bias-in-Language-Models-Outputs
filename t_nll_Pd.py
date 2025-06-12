#!/usr/bin/env python3
# Scatter:  t  vs  NLL  with hue = P_d   (green→red)

import glob, os, h5py, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- parameters ------------------------------------------------------
CTX        = 2048
T_MAX      = 6000                     # x-axis limit
PD_MAX     = 6000                     # colour scale upper bound
Y_MAX      = 10                       # y-axis limit
TARGET_PTS = 20_000_000                # down-sample for fast plotting
FILE_LIMIT = 500                      # how many files to look at
FOLDER     = "D:/NLL_matrices/1_4B"

# ---------- gather data -----------------------------------------------------
paths = glob.glob(os.path.join(FOLDER, "*.h5"))
if not paths:
    raise RuntimeError("no .h5 files found")

counter = 0
frames = []
for fp in tqdm(paths, desc="loading"):
    with h5py.File(fp, "r") as f:
        mat = f["nll"][...].astype(np.float32)

    mask = np.isfinite(mat)
    if not mask.any():
        continue

    rows, cols  = np.nonzero(mask)          # rows = t, cols = k-1
    nll_vals    = mat[rows, cols]
    P_t         = cols + 1
    P_d = np.abs(P_t - rows)          # distance from doc start, always ≥0


    # clip to target ranges
    keep = (rows <= T_MAX) & (P_d >= 0) & (P_d <= PD_MAX) & (nll_vals <= Y_MAX)
    if keep.any():
        frames.append(pd.DataFrame({
            "t":   rows[keep],
            "NLL": nll_vals[keep],
            "P_d": P_d[keep]
        }))
    if counter > FILE_LIMIT:
        break
    counter = counter+1

df = pd.concat(frames, ignore_index=True)
print(f"[INFO] collected {len(df):,} rows")

# ---------- down-sample -----------------------------------------------------
frac = min(1.0, TARGET_PTS / len(df))
df_plot = df.sample(frac=frac, random_state=0, ignore_index=True)
print(f"[INFO] plotting {len(df_plot):,} points (frac={frac:.3f})")

# ---------- plot ------------------------------------------------------------
sns.set_theme(style="white")
plt.figure(figsize=(10, 5))

scatter = plt.scatter(df_plot["t"], df_plot["NLL"],
                      c=df_plot["P_d"],
                      cmap=plt.cm.Spectral_r,
                      vmin=0, vmax=PD_MAX,
                      s=3, alpha=.3, rasterized=True)

cbar = plt.colorbar(scatter, label="P_d (green 0 … red 6 000)")
plt.xlim(0, T_MAX)
plt.ylim(0, Y_MAX)
plt.xlabel("t  (token index in document)")
plt.ylabel("NLL (nats)")
plt.title("NLL vs t   coloured by P_d")
plt.tight_layout()
plt.savefig("t_nll_Pd2.png", dpi=600)
print("[INFO] saved to nll_t_nll_pd_hue.png")
plt.show()

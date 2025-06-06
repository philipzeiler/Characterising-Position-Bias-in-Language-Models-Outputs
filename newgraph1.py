#!/usr/bin/env python3
# NLL vs P_d with hue = t  (fast vectorised build)

import glob, os, h5py, numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
import math


FOLDER = "nll_matrices_timetest"
paths  = glob.glob(os.path.join(FOLDER, "*.h5"))
if not paths:
    raise RuntimeError("no .h5 files found")

frames = []

for fp in tqdm(paths, desc="Loading matrices"):
    with h5py.File(fp, "r") as f:
        mat = f["nll"][...].astype(np.float32)        # (L, 2047)

    mask         = np.isfinite(mat)
    if not mask.any():
        continue

    rows, cols   = np.nonzero(mask)
    nll_vals     = mat[rows, cols]

    P_t          = cols + 1 
    P_d          = P_t - rows
    t_vals       = rows

    frames.append(pd.DataFrame(
        {"P_d": P_d, "NLL": nll_vals, "t": t_vals}
    ))

df = pd.concat(frames, ignore_index=True)
df = df[df.P_d>=0]
print(f"[INFO] collected {len(df):,} points")


print("[INFO] down-sampling for plotting …")
target_points = 2_000_000
frac = min(1.0, target_points / len(df))
df_plot = df.sample(frac=frac,
                    random_state=0,
                    axis=0,
                    ignore_index=True,
                   )

# --- OPTIONAL: bin t to ~10 colours instead of 30 000 --------
bins  = [0, 1, 2, 4, 8, 16, 32, 64, 128, math.inf]
labels = [f"{bins[i]}–{bins[i+1]-1}" if math.isfinite(bins[i+1])
          else f"{bins[i]}+" for i in range(len(bins)-1)]
df_plot["t_bin"] = pd.cut(df_plot["t"], bins=bins, labels=labels)

# --- plotting with scatter + rasterized ----------------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 4))

print("Plotting:")
sns.lineplot(data=df_plot,
                x="P_d", y="NLL",
                hue="t_bin",
                palette="viridis",
                #scatter=False,
                #s=3,
                #alpha=.3,
                #rasterized=True
                )         # Matplotlib draws as one image

plt.xlabel("P_d  (doc-start position in context window)")
plt.ylabel("NLL (nats)")
plt.title("NLL vs P_d (2 M sampled points, hue = binned token offset t)")
plt.tight_layout()
plt.show()


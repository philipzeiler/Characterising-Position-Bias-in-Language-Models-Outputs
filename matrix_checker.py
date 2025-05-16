# ─── Deep sanity checker 2.0  (seaborn edition) ─────────────────────────────
import h5py, numpy as np, random, pandas as pd
import seaborn as sns                      # <─ NEW
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")           # cosmetic default

# --------------------------- load ------------------------------------------------
PATH = "nll_matrices/doc63579_fp16.h5"     # adjust as needed
with h5py.File(PATH) as f:
    mat = f["nll"][...]                    # (L, 2047)  float16 + NaN
mat = mat.astype(np.float32)               # prevent float16 overflow

L, C = mat.shape
print(f"Matrix shape      : {L} tokens  ×  {C} positions")

# ----------------------- basic stats --------------------------------------------
total_nans = np.isnan(mat).sum()
print(f"Total NaNs        : {total_nans:,}  ({100*total_nans/(L*C):.2f} %)")

filled_per_token = np.sum(~np.isnan(mat), axis=1)
print("\nFilled-positions per token:")
print(pd.Series(filled_per_token).describe().to_string())      # pandas describe

for row in (0, L//2, L-1):
    cols = np.where(~np.isnan(mat[row]))[0]
    print(f"\nToken {row:4d}: first col = {cols[0]}, "
          f"last col = {cols[-1]}, count = {len(cols)}")

print("\nRandom spot-checks (token, position) → NLL")
rng = np.random.default_rng(0)
for _ in range(8):
    t  = rng.integers(L)
    p  = rng.choice(np.where(~np.isnan(mat[t]))[0])
    print(f"({t:4d},{p:4d}) → {mat[t, p]:.5f}")

# ---------------------- heat-map -------------------------------------------------
vmax = np.nanpercentile(mat, 95)
plt.figure(figsize=(12, 4))
sns.heatmap(mat,
            vmin=0, vmax=vmax,             # nice dynamic range
            cmap="viridis", cbar_kws={"label": "NLL"},
            xticklabels=256, yticklabels=250)          # thin ticks for clarity
plt.xlabel("context position")
plt.ylabel("token index in document")
plt.title("NLL heat-map (white = NaN)")
plt.tight_layout()

# ---------------- aggregate curve (mean / median, IQR) – log-scale ----------

# per-column stats
mean_vals   = np.nanmean(mat, axis=0)            # (2047,)
median_vals = np.nanmedian(mat, axis=0)
q1_vals     = np.nanpercentile(mat, 25, axis=0)
q3_vals     = np.nanpercentile(mat, 75, axis=0)

# --- avoid log(0) by clipping very small numbers ---------------------------
EPS = 1e-6
mean_vals   = np.clip(mean_vals,   EPS, None)
median_vals = np.clip(median_vals, EPS, None)
q1_vals     = np.clip(q1_vals,     EPS, None)
q3_vals     = np.clip(q3_vals,     EPS, None)

pos = np.arange(C)                # 0 … 2046

plt.figure(figsize=(12, 3))
ax = plt.gca()

# IQR band
ax.fill_between(pos, q1_vals, q3_vals,
                alpha=.25, color=sns.color_palette()[0],
                label="IQR (25 – 75 pct)")

# mean & median lines
sns.lineplot(x=pos, y=mean_vals,
             label="mean", linewidth=1.8,
             color=sns.color_palette()[0], ax=ax)
sns.lineplot(x=pos, y=median_vals,
             label="median", linewidth=1.2, linestyle="--",
             color=sns.color_palette()[2], ax=ax)

# --- axis formatting --------------------------------------------------------
ax.set(xlim=(0, C-1),
       ylim=(EPS, None),           # bottom at ε so log scale works
       xlabel="context position (0-2046)",
       ylabel="NLL (log scale)")
ax.set_title("Per-position NLL distribution (log-scale)\n"
             "band = IQR, lines = mean & median")

ax.set_yscale("log")
ax.legend()
plt.tight_layout()

plt.show()
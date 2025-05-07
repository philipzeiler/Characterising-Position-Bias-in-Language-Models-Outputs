# ─── Deep sanity checker ────────────────────────────────────────────────────
import h5py, numpy as np, random, pandas as pd

with h5py.File("nll_matrices_snake/doc000000_ctx2048.h5") as f:
    mat = f["nll"][...]              # (L, 2048)  FP32 + NaN

L, C = mat.shape
print(f"Matrix shape      : {L} tokens  ×  {C} positions")

total_nans = np.isnan(mat).sum()
print(f"Total NaNs        : {total_nans:,}  ({100*total_nans/(L*C):.2f} %)")

filled_per_token = np.sum(~np.isnan(mat), axis=1)
summary = pd.Series(filled_per_token).describe()
print("\nFilled-positions per token:")
print(summary.to_string())

# Sentinel rows --------------------------------------------------------------
for row in (0, L//2, L-1):
    cols = np.where(~np.isnan(mat[row]))[0]
    print(f"\nToken {row:3d}: first filled col = {cols[0]}, "
          f"last filled col = {cols[-1]},  count = {len(cols)}")

# Random spot-checks ---------------------------------------------------------
print("\nRandom spot-checks (token, position) → NLL")
for _ in range(8):
    t = random.randrange(L)
    p_choices = np.where(~np.isnan(mat[t]))[0]
    p = random.choice(p_choices)
    print(f"({t:3d}, {p:4d}) → {mat[t, p]:.5f}")


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.imshow(np.nan_to_num(mat, nan=np.nanmax(mat)), aspect="auto",
           vmax=np.nanpercentile(mat, 95))
plt.colorbar(label="NLL")
plt.xlabel("context position")
plt.ylabel("token index in doc-0")
plt.title("NLL heat-map (white = no evaluation)")
plt.tight_layout()

# --- aggregate curve: only positions with full left context -----------------
FULL_START = L              # 1 + max token index that still lacks context

avg_curve = np.nanmean(mat[:, FULL_START:], axis=0)     # shape (2048-338,)

plt.figure(figsize=(10, 2.2))
plt.plot(np.arange(FULL_START, C), avg_curve)
plt.xlabel("context position")
plt.ylabel("mean NLL")
plt.title(f"Mean NLL vs. context position  (columns {FULL_START}-{C-1})")
plt.axhline(avg_curve.mean(), color="k", lw=0.5, alpha=.5)
plt.tight_layout()

plt.show()
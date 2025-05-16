import h5py, numpy as np, matplotlib.pyplot as plt, os

H5_PATH = "nll_matrices/doc1821_fp16.h5"   # adjust if you saved elsewhere
CTX     = 2047

with h5py.File(H5_PATH) as f:
    mat = f["nll"][:]                      # shape (L, 2048)

L, C = mat.shape
assert C == CTX, "matrix width mismatch"

norm_mat = np.full_like(mat, np.nan, dtype=np.float32)

for r in range(L):
    start = r                              # first column with full context
    seg   = mat[r, start:]
    mu, sd = seg.mean(), seg.std()
    if sd == 0:  sd = 1.                   # safety
    norm_mat[r, start:] = (seg - mu) / sd  # keep only full-context part

# ── plot ──────────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 4))
plt.imshow(norm_mat, aspect="auto",
           cmap="RdBu_r", vmin=-1, vmax=1)  # focus on ±1 σ
plt.colorbar(label="row-wise z-score of NLL")
plt.xlabel("context position")
plt.ylabel("token index in doc-0")
plt.title("Token-wise NLL deviations (only after full left context is present)")
plt.tight_layout()
plt.show()

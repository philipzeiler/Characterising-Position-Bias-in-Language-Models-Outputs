#!/usr/bin/env python3
# Position-bias over training checkpoints (equidistant x; start at 0; add 143000)
import os, h5py, numpy as np
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODEL_SIZES = [
    "14M",
    "31M",
    "70M",
    "160M",
    "410M",
    "1B",
    "1.4B",
    "2.8B",
]

# Canonical step list used for tick labels and ordering (equidistant spacing):
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000,
         8000, 16000, 32000, 64000, 128000]
FINAL_STEP = 143000  # extra point on the right, different path template

BASE_DIR_TEMPLATE   = r"D:/NLL_matrices/revisions/{MODEL_SIZE}_EOD/step{STEP}_merged.h5"
FINAL_PATH_TEMPLATE = r"D:/NLL_matrices/{MODEL_SIZE}_EOD_merged.h5"   # path for step 143000

CTX        = 2048
FILE_LIM   = 5000
MAX_DOC_LEN = 500
P_START    = 500  # baseline alignment point

# ───────────────────────────── helpers ──────────────────────────────────────
def posbias_for_h5(h5_path: str) -> float | None:
    """Compute the ΔNLL range (max-min) over P_t ∈ [P_START, CTX] for one file."""
    try:
        with h5py.File(h5_path, "r") as f:
            nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
            sum_p   = np.zeros(CTX, dtype=np.float64)
            count_p = np.zeros(CTX, dtype=np.int64)
            N_docs  = min(FILE_LIM, len(ptr_ds) - 1)

            for d in range(N_docs):
                s, e = int(ptr_ds[d]), int(ptr_ds[d+1])
                L = e - s
                if MAX_DOC_LEN and L > MAX_DOC_LEN:
                    continue
                mat = nll_ds[s:e, :].astype(np.float32)  # (L, CTX-1)

                rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)
                cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)
                P_t  = cols + 2                           # labels 2..CTX
                P_d  = P_t - rows

                keep = (P_t <= CTX)  # (optionally: P_d >= 0 if you want full left context)
                if not np.any(keep): 
                    continue

                p_idx   = (P_t[keep] - 1).astype(np.int64)   # 1..2047 → 0..2046
                nll_val = mat.ravel()[keep]
                m = np.isfinite(nll_val)
                if np.any(m):
                    np.add.at(sum_p,   p_idx[m], nll_val[m])
                    np.add.at(count_p, p_idx[m], 1)

        mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)
        base   = mean_p[P_START - 1]
        if not np.isfinite(base):
            return None
        shifted = mean_p - base
        shifted[P_START - 1] = 0.0

        sl = shifted[P_START - 1:]                 # plotted range
        if not np.isfinite(sl).any():
            return None
        return float(np.nanmax(sl) - np.nanmin(sl))
    except OSError:
        return None

# ───────────────────────────── plotting ─────────────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

cmap    = mpl.cm.get_cmap("RdYlBu", len(MODEL_SIZES) * 2)
colors  = [cmap(i * 2) for i in range(len(MODEL_SIZES))]

max_len_over_models = 0
all_step_labels     = STEPS + [FINAL_STEP]   # canonical labels (for ticks)
series_data         = []                     # list of (model_name, y_values)

for mi, msize in enumerate(MODEL_SIZES):
    print("Starting another model!")
    y_vals = []

    # regular checkpoints
    for step in STEPS:
        path = BASE_DIR_TEMPLATE.format(MODEL_SIZE=msize, STEP=step)
        pb = posbias_for_h5(path)
        if pb is not None:
            y_vals.append(pb)

    # final checkpoint 143000 from different location
    final_path = FINAL_PATH_TEMPLATE.format(MODEL_SIZE=msize)
    pb_final = posbias_for_h5(final_path)
    if pb_final is not None:
        y_vals.append(pb_final)

    if not y_vals:
        continue

    series_data.append((msize, y_vals))
    max_len_over_models = max(max_len_over_models, len(y_vals))

# plot each model on equidistant integer x = 0..n-1
for (msize, y_vals), col in zip(series_data, colors):
    x_idx = np.arange(len(y_vals), dtype=int)
    ax.plot(x_idx, y_vals, color=col, linewidth=2.2, label=msize)

# ───────────────────── ticks & labels (start at 0; include 143000) ─────────
# Left edge pinned to exactly 0; right edge to last index available globally
ax.set_xlim(0, max(0, max_len_over_models - 1))  # start exactly at 0

# Use the first max_len_over_models canonical labels as x tick labels
tick_positions = np.arange(max_len_over_models, dtype=int)
tick_labels    = [str(s) for s in all_step_labels[:max_len_over_models]]
ax.xaxis.set_major_locator(FixedLocator(tick_positions))
ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))

ax.set_xlabel("Training checkpoint")

#ax.set_ylabel("Token output position bias in NLL")
ax.set_ylabel(fr"Token OP Bias ($\mathrm{{OPB}}^{{\mathrm{{tok}}}}$)")

ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
#ax.set_title("Pythia standard models: 264 docs used per checkpoint per model")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_pythia_checkpoints.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

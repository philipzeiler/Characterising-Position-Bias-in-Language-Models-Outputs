#!/usr/bin/env python3
# EOD-only position-bias over training checkpoints (equidistant x; start at 0; add 143000)
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

CTX         = 2048
FILE_LIM    = 5000
MAX_DOC_LEN = 500
P_START     = 500  # baseline alignment point (absolute P_t)

# ───────────────────────────── helpers ──────────────────────────────────────
def posbias_for_h5(h5_path: str) -> float | None:
    """
    EOD-only version:
      • For each document, take ONLY the last token row (the appended EOD).
      • That row has length CTX-1 and corresponds to absolute positions P_t = 2..CTX.
      • Average EOD NLL across docs per absolute position, align at P_START,
        and return the ΔNLL range (max−min) over P_t ∈ [P_START, CTX].
    """
    try:
        N_POS = CTX - 1  # columns for P_t = 2..CTX
        sum_p   = np.zeros(N_POS, dtype=np.float64)
        count_p = np.zeros(N_POS, dtype=np.int64)

        with h5py.File(h5_path, "r") as f:
            nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
            N_docs  = min(FILE_LIM, len(ptr_ds) - 1)

            for d in range(N_docs):
                s, e = int(ptr_ds[d]), int(ptr_ds[d+1])
                L = e - s
                if MAX_DOC_LEN and L > MAX_DOC_LEN:
                    continue
                if L <= 0:
                    continue

                # --- EOD row: last token of the document (since EOD was appended) ---
                eod_row_idx = e - 1
                vec = nll_ds[eod_row_idx, :].astype(np.float32)  # shape (CTX-1,)

                # (optional) enforce full-left-context for EOD: keep first min(N_POS, L) entries
                # Here we follow your prior EOD graphs: no additional filter by default.
                vals = vec

                # Guard against NaNs
                m = np.isfinite(vals)
                sum_p += np.where(m, vals, 0.0)
                count_p += m.astype(np.int64)

        # Mean EOD NLL per k (k = 1..CTX-1 → P_t = 2..CTX)
        mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)

        # Build full array indexed by absolute P_t = 1..CTX so we can align at P_START
        mean_full = np.full(CTX + 1, np.nan, dtype=np.float64)  # index 0 unused
        mean_full[2:CTX + 1] = mean_p                           # map k→P_t

        base = mean_full[P_START]
        if not np.isfinite(base):
            return None

        shifted = mean_full - base
        shifted[P_START] = 0.0

        sl = shifted[P_START:CTX + 1]  # P_t ∈ [P_START, CTX]
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
#ax.set_ylabel(fr"Token output position bias $\mathrm{{OPB}}^{{\mathrm{{tok}}}}$(EOD)")
ax.set_ylabel(fr"Token OP Bias ($\mathrm{{OPB}}^{{\mathrm{{tok}}}}$(EOD))")


ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
#ax.set_title("Pythia standard models: 264 docs used per checkpoint per model")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_pythia_checkpoints_EOD_only.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

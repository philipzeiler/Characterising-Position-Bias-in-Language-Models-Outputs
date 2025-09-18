#!/usr/bin/env python3
"""
Document-level position bias across checkpoints
-----------------------------------------------
For each model & checkpoint:
  • Build mean NLL vs document position P_d (1..P_END), ignoring NaNs
  • Align at P_d = 1 (ΔNLL)
  • Position bias = max(ΔNLL) − min(ΔNLL) over plotted P_d range
Plot one line per model:
  • x-axis: equidistant checkpoints (labels show step numbers)
  • y-axis: position-bias magnitude (absolute Δ range)
"""

import os, numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm

# ───────────────────────── user settings ─────────────────────────
# Choose which Pythia models to include. Add others to this list when needed.
MODEL_SIZES = ["14M",
               "31M",
               "70M", 
               "160M", 
               "410M", 
               "1B", 
               "1.4B", 
               "2.8B"
               ]

# The canonical set of checkpoints you’ve evaluated (equidistant on x).
# The last "143000" is the final merged-at-root file, handled specially below.
CHECKPOINT_STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000
]

# Base path templates:
#  - For every step != 143000, files live in revisions/{MODEL}_EOD/step{step}_merged.h5
#  - For 143000, file is at the model root: {MODEL}_EOD_merged.h5
BASE_REV_DIR = r"D:/NLL_matrices/revisions/{model}_EOD/step{step}_merged.h5"
BASE_FINAL   = r"D:/NLL_matrices/{model}_EOD_merged.h5"

# Curve construction parameters (same logic as your Pd graph)
CTX                      = 2048
FILE_LIM                 = 500000  # no harm; we’ll stop at available docs
USE_INTERESTING_PD       = False   # set True if you want the sparse P_d set
FILTER_FULL_LEFT_CONTEXT = False   # if True, keep only P_d <= 1 (with P_d>=1 base keep → only P_d==1)
MAX_DOC_LEN              = 500     # keep consistent with your checkpoint graphs

P_ALIGN = 1
P_START = 1
P_END   = 1548                # use 1536 for Pythia, 2048 for OLMo2 if you port this there

# X ticks = show all steps as labels on an equidistant axis
X_LABEL_STEPS = CHECKPOINT_STEPS  # override to subset if desired

# Optional: hardcode Y ticks (or set to None for auto)
Y_TICKS = None

# ───────────────────────── helpers ───────────────────────────────
def interesting_mask(Pd: np.ndarray, ctx: int) -> np.ndarray:
    Pd_i  = Pd.astype(np.int64)
    absPd = np.abs(Pd_i)
    in_small = (Pd_i >= -128) & (Pd_i <= 128)
    is_pow2  = (absPd > 0) & ((absPd & (absPd - 1)) == 0)
    within_4096 = absPd <= 4096
    return in_small | (is_pow2 & within_4096)

def path_for(model: str, step: int) -> str:
    """Return the merged .h5 path for a given model size and checkpoint step."""
    if step == 143000:
        return BASE_FINAL.format(model=model)
    return BASE_REV_DIR.format(model=model, step=step)

def compute_doc_level_posbias(h5_path: str) -> float | None:
    """
    Compute document-level position bias for a single merged .h5:
      1) mean NLL vs P_d in [1..P_END], ignoring NaNs
      2) align at P_d=1 (ΔNLL)
      3) return max(ΔNLL) − min(ΔNLL) over plotted range
    Returns None if baseline or the slice is not finite after filtering.
    """
    if not os.path.exists(h5_path):
        return None

    sum_pd   = np.zeros(P_END, dtype=np.float64)     # bins 0..P_END-1 → P_d = i+1
    count_pd = np.zeros(P_END, dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs_all     = len(ptr_ds) - 1
        N_docs         = N_docs_all if FILE_LIM is None else min(FILE_LIM, N_docs_all)

        for d in range(N_docs):
            s, e = int(ptr_ds[d]), int(ptr_ds[d+1])
            L    = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            mat = nll_ds[s:e, :].astype(np.float32)    # (L, CTX-1)

            # Build coordinates
            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)  # token idx in doc
            cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)    # column idx in window
            P_t  = cols + 2                                          # label positions 2..CTX
            P_d  = P_t - rows                                        # doc start position within window

            # Base keep: P_d within [1..P_END]
            keep = (P_d >= 1) & (P_d <= P_END)

            # Optional filters
            if FILTER_FULL_LEFT_CONTEXT:
                keep &= (P_d <= 1)
            if USE_INTERESTING_PD:
                keep &= interesting_mask(P_d, CTX)

            if not np.any(keep):
                continue

            d_idx   = (P_d[keep] - 1).astype(np.int64)    # 0..P_END-1
            nll_val = mat.ravel()[keep]
            m = np.isfinite(nll_val)
            if np.any(m):
                np.add.at(sum_pd,   d_idx[m], nll_val[m])
                np.add.at(count_pd, d_idx[m], 1)

    mean_pd  = np.where(count_pd > 0, sum_pd / count_pd, np.nan)
    baseline = mean_pd[P_ALIGN - 1]
    if not np.isfinite(baseline):
        return None

    shifted = mean_pd - baseline
    shifted[P_ALIGN - 1] = 0.0

    # Extract bias range over plotted slice, ignoring NaNs
    slice_vals = shifted[P_START - 1 : P_END]
    finite_any = np.isfinite(slice_vals).any()
    if not finite_any:
        return None

    # Use NaN-ignoring min/max semantics
    y_min = np.nanmin(slice_vals)
    y_max = np.nanmax(slice_vals)
    return float(y_max - y_min)

# ───────────────────────── plotting ──────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

cmap    = mpl.cm.get_cmap("RdYlBu", len(MODEL_SIZES) * 2)
colours = [cmap(i * 2) for i in range(len(MODEL_SIZES))]

# X positions: equidistant indices for the checkpoints
x_pos  = np.arange(len(CHECKPOINT_STEPS), dtype=float)
x_left = 0.0               # start exactly at 0
x_right = len(CHECKPOINT_STEPS) - 1

# (Optional) label indices where you want tick labels
x_tick_idx   = np.arange(len(X_LABEL_STEPS))
x_tick_lbls  = [str(s) for s in X_LABEL_STEPS]

# Keep track of global y-range for axis padding/ticks
global_min = +np.inf
global_max = -np.inf

for midx, model in enumerate(MODEL_SIZES):
    y_bias = []
    steps_found = []

    print(f"\n[INFO] Computing doc-level position bias for model {model} …")
    for i, step in enumerate(CHECKPOINT_STEPS):
        path = path_for(model, step)
        bias = compute_doc_level_posbias(path)
        if bias is None:
            print(f"  [SKIP] {model} step {step}: missing or no finite data → {path}")
            y_bias.append(np.nan)
        else:
            print(f"  [OK]   {model} step {step}: bias = {bias:.6f}")
            y_bias.append(bias)
            steps_found.append(step)

    y_arr = np.array(y_bias, dtype=np.float64)
    if np.isfinite(y_arr).any():
        global_min = min(global_min, float(np.nanmin(y_arr)))
        global_max = max(global_max, float(np.nanmax(y_arr)))

    # Plot line for the model (equidistant x)
    ax.plot(
        x_pos, y_arr,
        color=colours[midx],
        linewidth=2.5,
        linestyle="-",
        alpha=1.0,
        label=model
    )

# Axes cosmetics
ax.set_xlim(x_left, x_right)
ax.set_xlabel("Training checkpoint")
#ax.set_ylabel("Document start output position bias in NLL")
#ax.set_ylabel(fr"Document output position bias $\mathrm{{OPB}}^{{\mathrm{{doc}}}}$")
ax.set_ylabel(fr"Document OP Bias ($\mathrm{{OPB}}^{{\mathrm{{doc}}}}$)")


# Custom x ticks at the equidistant positions, labeled by step
ax.xaxis.set_major_locator(FixedLocator(x_tick_idx))            # custom positions
ax.xaxis.set_major_formatter(FixedFormatter(x_tick_lbls))       # custom labels

# Y ticks and gridlines: either fixed or auto with headroom
ax.grid(False)
if Y_TICKS is not None and len(Y_TICKS) > 1:
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{y:g}" for y in Y_TICKS]))
    ax.set_ylim(min(Y_TICKS), max(Y_TICKS))
else:
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 0.0, 1.0
    span = max(global_max - global_min, 1e-9)
    ax.set_ylim(global_min - 0.05*span, global_max + 0.20*span)

# Gridlines only at major y-ticks
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
ax.yaxis.grid(True, which="major", linestyle="-", alpha=0.25)

leg = ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
for line in leg.get_lines():
    line.set_linewidth(6)

#ax.set_title("Pythia standard models: 264 docs used per checkpoint per model")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/doc_level_posbias_vs_checkpoints_pythia.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

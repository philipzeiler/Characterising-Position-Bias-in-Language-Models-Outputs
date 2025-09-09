#!/usr/bin/env python3
"""
Mean *ΔNLL* vs document position P_d  (aligned at P_d = 1)
• x-axis : P_d = 1 … 2048 (log2)
• ΔNLL aligned at P_d = 1
• Only “interesting” P_d kept; docs with L ≤ 2000
• NaNs ignored
• Solid lines; gridlines only at your major y-ticks
"""

import numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODELS = [
    #("1B",  r"D:/NLL_matrices/0425-1B_EOD_merged.h5"),
    #("7B",  r"D:/NLL_matrices/1124-7B_EOD_merged.h5"),
    #("13B", r"D:/NLL_matrices/1124-13B_EOD_merged.h5"),
    ("14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    ("31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
    ("70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
    ("160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
    ("410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
    ("1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
    ("1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
    ("2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
    ("6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
    ("12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),
    #("70M no EOD",  r"D:/NLL_matrices/70M_merged.h5"),
    #("160M no EOD", r"D:/NLL_matrices/160M_merged.h5"),
    #("410M no EOD", r"D:/NLL_matrices/410M_merged.h5"),
    #("1.4B no EOD", r"D:/NLL_matrices/1.4B_merged.h5"),
    #("2.8B no EOD", r"D:/NLL_matrices/2.8B_merged.h5"),
    # ("70M",  r"D:/NLL_matrices/70M_deduped_merged.h5"),
    # ("160M", r"D:/NLL_matrices/160M_deduped_merged.h5"),
    # ("410M", r"D:/NLL_matrices/410M_deduped_merged.h5"),
    # ("1B",   r"D:/NLL_matrices/1B_deduped_merged.h5"),
    # ("1.4B", r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
    # ("2.8B", r"D:/NLL_matrices/2.8B_deduped_merged.h5"),
    # ("6.9B", r"D:/NLL_matrices/6.9B_deduped_merged.h5"),
    # ("12B",  r"D:/NLL_matrices/12B_deduped_merged.h5"),
    # ("Step 0",  r"D:/NLL_matrices/revisions/70M_EOD/step0_merged.h5"),
    # ("Step 1",  r"D:/NLL_matrices/revisions/70M_EOD/step1_merged.h5"),
    # ("Step 2",  r"D:/NLL_matrices/revisions/70M_EOD/step2_merged.h5"),
    # ("Step 4",  r"D:/NLL_matrices/revisions/70M_EOD/step4_merged.h5"),
    # ("Step 8",  r"D:/NLL_matrices/revisions/70M_EOD/step8_merged.h5"),
    # ("Step 16",  r"D:/NLL_matrices/revisions/70M_EOD/step16_merged.h5"),
    # ("Step 32",  r"D:/NLL_matrices/revisions/70M_EOD/step32_merged.h5"),
    # ("Step 64",  r"D:/NLL_matrices/revisions/70M_EOD/step64_merged.h5"),
    # ("Step 128",  r"D:/NLL_matrices/revisions/70M_EOD/step128_merged.h5"),
    # ("Step 256",  r"D:/NLL_matrices/revisions/70M_EOD/step256_merged.h5"),
    # ("Step 512",  r"D:/NLL_matrices/revisions/70M_EOD/step512_merged.h5"),
    # ("Step 1000",  r"D:/NLL_matrices/revisions/70M_EOD/step1000_merged.h5"),
    # ("Step 2000",  r"D:/NLL_matrices/revisions/70M_EOD/step2000_merged.h5"),
    # ("Step 4000",  r"D:/NLL_matrices/revisions/70M_EOD/step4000_merged.h5"),
    # ("Step 8000",  r"D:/NLL_matrices/revisions/70M_EOD/step8000_merged.h5"),
    # ("Step 16000",  r"D:/NLL_matrices/revisions/70M_EOD/step16000_merged.h5"),
    # ("Step 32000",  r"D:/NLL_matrices/revisions/70M_EOD/step32000_merged.h5"),
    # ("Step 64000",  r"D:/NLL_matrices/revisions/70M_EOD/step64000_merged.h5"),
    # ("Step 128000",  r"D:/NLL_matrices/revisions/70M_EOD/step128000_merged.h5"),
    # ("Step 143000",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
]

CTX                      = 2048
FILE_LIM                 = 500000
USE_INTERESTING_PD = False #Put true for sparse OLMO2 evaluation, otherwise false
FILTER_FULL_LEFT_CONTEXT = False
MAX_DOC_LEN              = 500 #512 or 2048 for olmo
LEGEND_OUTSIDE = False   # ← set False to keep legend inside the axes


P_ALIGN = 4
P_START = 4
P_END   = 1548  # cut x at 2048 for olmo or 1548 for pythia

# >>> Your custom tick lists (edit these) <<<
X_TICKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
           #1024, 
           1548, 2048]
Y_TICKS = None#[-0.004, -0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016]; #set to a list to hardcode or None

# ───────────────────────────── helpers ──────────────────────────────────────
def interesting_mask(Pd: np.ndarray, ctx: int) -> np.ndarray:
    Pd_i  = Pd.astype(np.int64)
    absPd = np.abs(Pd_i)
    in_small = (Pd_i >= -128) & (Pd_i <= 128)
    is_pow2  = (absPd > 0) & ((absPd & (absPd - 1)) == 0)
    within_4096 = absPd <= 4096
    return in_small | (is_pow2 & within_4096)

# ────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(10, 10)) #set to 10, 10 for compact, 20,10 for standard.

# palette
cmap     = mpl.cm.get_cmap("RdYlBu", len(MODELS) * 2)
colours  = [cmap(i * 2) for i in range(len(MODELS))]

global_min = +np.inf
global_max = -np.inf
model_docs_used = {}

for idx, (name, h5_path) in enumerate(MODELS):
    sum_pd   = np.zeros(P_END,  dtype=np.float64)   # bins 0..P_END-1 → P_d = i
    count_pd = np.zeros(P_END,  dtype=np.int64)
    docs_used = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs_all = len(ptr_ds) - 1
        N_docs     = N_docs_all if FILE_LIM is None else min(FILE_LIM, N_docs_all)

        for d in tqdm(range(N_docs), desc=f"{name:>12}"):
            s, e = int(ptr_ds[d]), int(ptr_ds[d + 1])
            L    = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            mat = nll_ds[s:e, :].astype(np.float32)  # (L, CTX-1)

            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)   # 0..L-1
            cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)     # 0..CTX-2
            P_t  = cols + 2                                           # 2..CTX
            P_d  = P_t - rows

            keep = (P_d >= 1) & (P_d <= P_END)
            if FILTER_FULL_LEFT_CONTEXT:
                keep &= (P_d <= 1)
            if USE_INTERESTING_PD:
                keep &= interesting_mask(P_d, CTX)

            if not np.any(keep):
                continue
            docs_used += 1

            d_idx   = (P_d[keep] - 1).astype(np.int64)                 # 0..P_END-1
            nll_val = mat.ravel()[keep]
            m = np.isfinite(nll_val)
            if np.any(m):
                np.add.at(sum_pd,   d_idx[m], nll_val[m])
                np.add.at(count_pd, d_idx[m], 1)

    mean_pd = np.where(count_pd > 0, sum_pd / count_pd, np.nan)

    baseline = mean_pd[P_ALIGN - 1]
    if not np.isfinite(baseline):
        print(f"[WARN] {name}: baseline at P_d={P_ALIGN} is NaN after filtering; skipping.")
        continue
    shifted = mean_pd - baseline
    shifted[P_ALIGN - 1] = 0.0

    valid = shifted[P_START - 1 : P_END]
    valid = valid[np.isfinite(valid)]
    if valid.size:
        global_min = min(global_min, float(np.min(valid)))
        global_max = max(global_max, float(np.max(valid)))

    model_docs_used[name] = docs_used

    x_all = np.arange(P_START, P_END + 1)      # 1..2048
    y_all = shifted[P_START - 1 : P_END]
    have  = np.isfinite(y_all)

    label = rf"{name}  ($\mathrm{{NLL}}_{{\mathrm{{base}}}} = {baseline:.2f}$)"
    ax.plot(
        x_all[have], y_all[have],
        color=colours[idx],
        linewidth=2.5,
        linestyle="-",
        alpha=1.0,
        label=label
    )

# ─────────────────────── axis formatting & legend ───────────────────────────
# Log2 x-axis, with YOUR custom ticks
ax.set_xscale("log", base=2)                                         # log scale
ax.set_xlim(P_START, P_END)
ax.xaxis.set_major_locator(FixedLocator(X_TICKS))                    # your ticks
ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in X_TICKS]))  # your labels

ax.set_xlabel("Document start position within 2048-token context window")
ax.set_ylabel("ΔNLL relative to value at beginning of context window")

# Y ticks and gridlines: either your hardcoded list or auto + headroom
import math
ax.grid(False)  # clear any style-sheet grid first to avoid mixed offsets     # pyplot.grid docs
if Y_TICKS is not None and len(Y_TICKS) > 1:
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{y:g}" for y in Y_TICKS]))
    ax.set_ylim(min(Y_TICKS), max(Y_TICKS))
else:
    # auto ticks + 50% headroom
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 0.0, 1.0
    span = max(global_max - global_min, 1e-6)
    ax.set_ylim(global_min - 0.02*span, global_max + 0.50*span)
# draw gridlines only at major y-ticks (regular spacing you defined)
ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())                 # no minor ticks
ax.yaxis.grid(True, which="major", linestyle="-", alpha=0.25)        # grid only on majors

if LEGEND_OUTSIDE:
    # Reserve some room on the right for the legend
    plt.subplots_adjust(right=0.78)  # tweak (0.75–0.85) as needed

    # Place legend outside, vertically centered, aligned to the axes’ right edge
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),   # just outside the axes, mid-height
        frameon=True,
        handlelength=4,
        borderaxespad=0.0,
    )
else:
    leg = ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=4,
        borderaxespad=0.4,
    )

for line in leg.get_lines():
    line.set_linewidth(6)

print("\n[DEBUG] documents actually used per model:")
for mdl, n in model_docs_used.items():
    print(f"  {mdl:>20}: {n} docs")

ax.set_title("Pythia 6.9B: 1516 docs, Pythia 12B: 1245 docs, all others: 2813 docs")
#("Pythia 70M: 264 docs used per checkpoint")
#("12B: 608 docs, 6.9B: 712 docs, rest > 2300 docs")
#("2.8B: 2262 docs, all others: 2849 docs")
#("OLMo 2 1B: 1612 docs, 7B: 10874 docs, 13B: 1966 docs")


plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/delta_nll_vs_Pd_log2_pythia_standard_start4.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

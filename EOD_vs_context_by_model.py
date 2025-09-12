#!/usr/bin/env python3
"""
EOD-only mean *ΔNLL* vs context-window position
-----------------------------------------------
• We take ONLY the last token of each document (EOD) and compute its NLL at
  every absolute position k ∈ {1..2047} within a 2048-token window.
• Each curve is vertically shifted so that NLL(P_t = P_ALIGN) ≡ 0.
• The visible x-range starts at P_START (no empty left margin).

Optional filters
~~~~~~~~~~~~~~~~
  - `FILTER_PD_FULL_LEFT` Keep only positions where the EOD has full left
                          context from within its own doc (k ≤ L).
  - `MAX_DOC_LEN`         Skip documents longer than N tokens.
"""

import numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODELS = [
        #   ("Pythia 14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
        #   ("Pythia 31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
        #   ("Pythia 70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
        #   ("Pythia 160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
        #   ("Pythia 410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
        #   ("Pythia 1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
        #   ("Pythia 1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
        #   ("Pythia 2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
        #   ("Pythia 2.8B deduped EOD", r"D:/NLL_matrices/2.8B_deduped_EOD_merged.h5"),
        #   ("Pythia 6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
        #   ("Pythia 12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),

          ("Small", r"D:/NLL_matrices/gpt2_merged.h5"),
          ("Medium", r"D:/NLL_matrices/gpt2-medium_merged.h5"),
          ("Large", r"D:/NLL_matrices/gpt2-large_merged.h5"),
          ("XL", r"D:/NLL_matrices/gpt2-xl_merged.h5"),
          ]


CTX                 = 1024
N_POS               = CTX - 1            # positions 1..2047
FILE_LIM            = 5000               # docs per model (None → all)
FILTER_PD_FULL_LEFT = False              # if True, require k ≤ L for EOD
MAX_DOC_LEN         = 300                # None → keep all
LEGEND_OUTSIDE      = False               # (kept for compatibility, ignored if SPLIT_LEGEND=True)
SPLIT_LEGEND        = False               # ← NEW: split legend into upper-left & lower-left

P_ALIGN             = 300                # align baseline at this position
P_START             = 300                # show from this position (no left gap)
P_END               = CTX                # ← show through *2048* on x-axis

# ────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# colour palette: neighbours get similar hues
N_models = len(MODELS)
cmap     = mpl.cm.get_cmap("RdYlBu", N_models * 2)
colours  = [cmap(i * 2) for i in range(N_models)]

global_min = +np.inf
global_max = -np.inf
model_docs_used = {}

for idx, (name, h5_path) in enumerate(MODELS):
    # accumulate ΣNLL_EOD and counts for every position (1..2047)
    sum_p   = np.zeros(N_POS, dtype=np.float64)
    count_p = np.zeros(N_POS, dtype=np.int64)
    docs_used = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs_total = len(ptr_ds) - 1
        N_docs = N_docs_total if FILE_LIM is None else min(FILE_LIM, N_docs_total)

        for d in tqdm(range(N_docs), desc=f"{name:>12}"):
            s, e = int(ptr_ds[d]), int(ptr_ds[d + 1])
            L    = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            # EOD row = last token of the document (since EOD was appended)
            eod_row = e - 1
            vec = nll_ds[eod_row, :].astype(np.float32)   # shape (2047,)

            # full-left-context for EOD is simply k ≤ L
            valid_len = min(N_POS, L) if FILTER_PD_FULL_LEFT else N_POS

            vals = vec[:valid_len]
            # guard against NaN/Inf
            m = np.isfinite(vals)
            sum_p[:valid_len]   += np.where(m, vals, 0.0)
            count_p[:valid_len] += m.astype(np.int64)

            # count doc usage
            if not FILTER_PD_FULL_LEFT:
                docs_used += 1
            else:
                docs_used += int(valid_len > 0 and np.any(m))

    # mean EOD NLL per (k = 1..2047)
    mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)

    # ── FIX: build a “full” array indexed by *absolute context position* P_t = 1..2048
    # mean_p holds values for P_t = 2..2048. We place them at indices 2..2048 so
    # that index == P_t; index 1 (P_t=1) stays NaN (no prediction at the very first position).
    mean_full = np.full(CTX + 1, np.nan, dtype=np.float64)  # indices 0..2048; ignore index 0
    mean_full[2:CTX + 1] = mean_p  # map k=1..2047 → P_t=2..2048

    # align at P_ALIGN (which is expressed in absolute P_t coordinates)
    baseline = mean_full[P_ALIGN] if 1 <= P_ALIGN <= CTX else np.nan
    shifted  = mean_full - baseline
    if np.isfinite(baseline):
        shifted[P_ALIGN] = 0.0  # numeric hygiene

    # restrict to visible range and update global min/max
    x = np.arange(P_START, P_END + 1)         # e.g., 500..2048  (length matches y)
    y = shifted[P_START : P_END + 1]
    vfin = y[np.isfinite(y)]
    if vfin.size:
        global_min = min(global_min, float(np.min(vfin)))
        global_max = max(global_max, float(np.max(vfin)))
    label = rf"{name}  ($\mathrm{{NLL}}_{{\mathrm{{base}}}} = {baseline:.2f}$)"
    # plot curve
    ax.plot(x, y, color=colours[idx], linewidth=2.0, label=label)
    model_docs_used[name] = docs_used

# ─────────────────────── axis formatting & legend ───────────────────────────
ax.set_xlim(P_START, P_END)  # now truly ends at 2048

# show ticks inside the visible range, including 2048
ticks_all = [300, 500, 768, 1024, 1280, 1536, 1792, 2048]
ax.set_xticks([t for t in ticks_all if P_START <= t <= P_END])
ax.set_xlabel("Token position within 2048-token context window")

# y-axis ticks & limits (include 0)
if not np.isfinite(global_min) or not np.isfinite(global_max):
    global_min, global_max = 0.0, 1.0
span   = max(global_max - global_min, 1e-6)
yticks = np.round(np.linspace(global_min+0.001, global_max, 12), 3)
if 0.0 not in yticks:
    yticks = np.sort(np.append(yticks, 0.0))
ax.set_ylim(global_min - 0.02 * span, global_max + 0.02 * span)
ax.set_yticks(yticks)
ax.set_ylabel(f"NLL relative to value at context position {P_ALIGN}")

# ── SPLIT LEGEND (top-left & bottom-left) ───────────────────────────────────
if SPLIT_LEGEND:
    handles, labels = ax.get_legend_handles_labels()
    n = len(labels)
    mid = (n + 1) // 2  # first half gets the extra if odd

    leg1 = ax.legend(handles[:mid], labels[:mid],
                     loc="upper left", frameon=True,
                     handlelength=4, borderaxespad=0.4)
    for line in leg1.get_lines():
        line.set_linewidth(6)
    ax.add_artist(leg1)  # keep first legend when drawing the second

    leg2 = ax.legend(handles[mid:], labels[mid:],
                     loc="lower left", frameon=True,
                     handlelength=4, borderaxespad=0.4)
    for line in leg2.get_lines():
        line.set_linewidth(6)
else:
    if LEGEND_OUTSIDE:
        plt.subplots_adjust(right=0.78)
        leg = ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            handlelength=4,
            borderaxespad=0.0,
        )
        for line in leg.get_lines():
            line.set_linewidth(6)
    else:
        leg = ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
        for line in leg.get_lines():
            line.set_linewidth(6)

print("\n[DEBUG] documents actually used per model:")
for mdl, n in model_docs_used.items():
    print(f"  {mdl:>12}: {n} docs")

ax.set_title("GPT-2 Models: 1484 docs used per model")
#(f"Pythia 6.9B: 1516 docs, Pythia 12B: 1245 docs, Pythia 2.8B deduped: 641 docs, all others: 2813 docs")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_EOD_gpt2.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

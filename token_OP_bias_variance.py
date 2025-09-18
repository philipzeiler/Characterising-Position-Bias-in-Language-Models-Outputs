#!/usr/bin/env python3
"""
ΔVariance (of NLL) vs context-window position  (aligned at P_t = 500)
---------------------------------------------------------------------
• x-axis :  P_t = 500 … 2048
• each curve shows the *change in variance* relative to the model's
  own variance at P_t = 500 (unbiased sample variance, ddof=1)
• legend shows the model name *and* its raw variance at P_t=500
• y-axis auto-zooms to min/max of the shifted curves and includes 0

Optional filters
~~~~~~~~~~~~~~~~
  - `FILTER_PD_NONNEG`  keep only windows with full left context (P_d ≥ 0)
  - `MAX_DOC_LEN`       skip documents longer than N tokens
"""

import numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODELS = [
    ("Pythia 14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    ("Pythia 31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
    ("Pythia 70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
    #("Pythia 70M (no EOD)",  r"D:/NLL_matrices/70M_merged.h5"),
    #("Pythia 70M deduped",  r"D:/NLL_matrices/70M_deduped_merged.h5"),
    ("Pythia 160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
    #("Pythia 160M (no EOD)", r"D:/NLL_matrices/160M_merged.h5"),
    ("Pythia 410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
    #("Pythia 410M (no EOD)", r"D:/NLL_matrices/410M_merged.h5"),
    ("Pythia 1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
    ("Pythia 1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
    #("Pythia 1.4B (no EOD)", r"D:/NLL_matrices/1.4B_merged.h5"),
    #("Pythia 1.4B deduped", r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
    ("Pythia 2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
    ("Pythia 6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
    ("Pythia 12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),
]

CTX              = 2048      # sliding-window length
FILE_LIM         = 5000      # docs per model (None → all)
FILTER_PD_NONNEG = False     # require full left context?
MAX_DOC_LEN      = 500       # skip docs longer than this (None → keep all)

P_START          = 500       # leftmost position to display & alignment point
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# --- colour palette: neighbours get similar hues ---------------------------
N_models  = len(MODELS)
cmap      = mpl.cm.get_cmap("RdYlBu", N_models * 2)
colours   = [cmap(i * 2) for i in range(N_models)]     # take every 2nd colour

global_min = +np.inf
global_max = -np.inf
model_docs_used = {}

for idx, (name, h5_path) in enumerate(MODELS):
    # accumulate sums needed for variance per P_t
    sum_p    = np.zeros(CTX,  dtype=np.float64)
    sumsq_p  = np.zeros(CTX,  dtype=np.float64)
    count_p  = np.zeros(CTX,  dtype=np.int64)
    docs_used = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs = len(ptr_ds) - 1 if FILE_LIM is None else min(FILE_LIM, len(ptr_ds) - 1)

        for d in tqdm(range(N_docs), desc=f"{name:>12}"):
            s, e = ptr_ds[d], ptr_ds[d + 1]
            L    = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            mat = nll_ds[s:e, :].astype(np.float32)   # (L, 2047)

            # --- build coordinates for every evaluated window ----------------
            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)  # token index t
            cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)    # offset k-1
            P_t  = cols + 1                                          # 1 … 2048
            P_d  = P_t - rows                                        # dist. from doc start

            keep = P_t <= CTX
            if FILTER_PD_NONNEG:
                keep &= P_d >= 0

            if not keep.any():
                continue
            docs_used += 1

            p_idx   = P_t[keep] - 1
            nll_val = mat.ravel()[keep]
            m       = np.isfinite(nll_val)              # True for finite entries only
            np.add.at(sum_p,   p_idx[m], nll_val[m])
            np.add.at(sumsq_p, p_idx[m], nll_val[m] * nll_val[m])
            np.add.at(count_p, p_idx[m], 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p   = np.where(count_p > 0, sum_p / count_p, np.nan)
        var_pop  = np.where(count_p > 0, sumsq_p / count_p - mean_p * mean_p, np.nan)
        var_samp = np.where(count_p > 1, var_pop * (count_p / (count_p - 1)), np.nan)
        var_samp = np.where(np.isfinite(var_samp), np.maximum(var_samp, 0.0), np.nan)  # clamp tiny negatives

    # [CHANGED] baseline alignment at P_START (like your ΔNLL plot)
    # Explanation: subtract each model's variance at P_t = 500, so curves start at 0.
    baseline = var_samp[P_START - 1]
    if not np.isfinite(baseline):
        print(f"[WARN] {name}: variance baseline at P_t={P_START} is NaN; skipping plot.")
        model_docs_used[name] = docs_used
        continue
    delta_var = var_samp - baseline
    delta_var[P_START - 1] = 0.0  # exact zero at the alignment point

    # [CHANGED] track global min/max from the *shifted* values (P_t ≥ P_START)
    valid_vals = delta_var[P_START - 1:][np.isfinite(delta_var[P_START - 1:])]
    if valid_vals.size:
        global_min = min(global_min, float(np.min(valid_vals)))
        global_max = max(global_max, float(np.max(valid_vals)))

    model_docs_used[name] = docs_used

    # [CHANGED] plot Δvariance and show baseline value in legend
    ax.plot(
        np.arange(P_START, CTX + 1),
        delta_var[P_START - 1:],
        color=colours[idx],
        linewidth=2.0,
        label=f"{name}  (variance baseline = {baseline:.3f})",
    )

# ─────────────────────── axis formatting & legend ───────────────────────────
# x-axis (unchanged)
ax.set_xlim(P_START, CTX)
ax.set_xticks([500, 768, 1024, 1280, 1536, 1792, 2048])
ax.set_xlabel("Token position within 2048-token context window")

# [CHANGED] y-axis: identical logic to ΔNLL figure, but for Δvariance
# Explanation: allow negatives/positives and ensure 0 is included in ticks.
span      = global_max - global_min if global_max > global_min else 1.0
yticks    = np.round(np.linspace(global_min, global_max, 10), 3)
if 0.0 not in yticks:
    yticks = np.sort(np.append(yticks, 0.0))
margin    = 0.02 * span
ax.set_ylim(global_min - margin, global_max + margin)
ax.set_yticks(yticks)
ax.set_ylabel(f"ΔVariance of NLL relative to value at context position {P_START}")

# legend with thicker colour strips (unchanged)
leg = ax.legend(
    loc="upper left",
    frameon=True,
    handlelength=4,
    borderaxespad=0.4,
)
for line in leg.get_lines():
    line.set_linewidth(6)

print("\n[DEBUG] documents actually used per model:")
for mdl, n in model_docs_used.items():
    print(f"  {mdl:>12}: {n} docs")

# [CHANGED] title to reflect Δvariance + alignment
ax.set_title(f"Pythia 6.9B: 1516 docs, Pythia 12B: 1245 docs, all others: 2813 docs")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_variance_delta.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

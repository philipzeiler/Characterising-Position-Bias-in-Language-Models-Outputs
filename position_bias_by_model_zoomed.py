#!/usr/bin/env python3
"""
Mean *ΔNLL* vs context-window position  (aligned at P_t = 500)
--------------------------------------------------------------
• x-axis :  P_t = 500 … 2048  
• every curve is vertically shifted so that   NLL(P_t=500) ≡ 0  
• legend shows the model name *and* its raw NLL at P_t=500  
• y-axis automatically zooms to the min/max of the shifted curves  

Optional filters
~~~~~~~~~~~~~~~~
  - `FILTER_PD_NONNEG`  keep only windows with full left context (P_d ≥ 0)  
  - `MAX_DOC_LEN`       skip documents longer than N tokens  
"""

import numpy as np, h5py, matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
MODELS = [
    # ("Pythia 14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    # ("Pythia 31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
    # ("Pythia 70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
    # ("Pythia 160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
    # ("Pythia 410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
    # ("Pythia 1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
    # ("Pythia 1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
    # ("Pythia 2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
    # ("Pythia 6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
    # ("Pythia 12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),
    # ("Step 0",  r"D:/NLL_matrices/revisions/14M_EOD/step0_merged.h5"),
    # ("Step 1",  r"D:/NLL_matrices/revisions/14M_EOD/step1_merged.h5"),
    # ("Step 2",  r"D:/NLL_matrices/revisions/14M_EOD/step2_merged.h5"),
    # ("Step 4",  r"D:/NLL_matrices/revisions/14M_EOD/step4_merged.h5"),
    # ("Step 8",  r"D:/NLL_matrices/revisions/14M_EOD/step8_merged.h5"),
    # ("Step 16",  r"D:/NLL_matrices/revisions/14M_EOD/step16_merged.h5"),
    # ("Step 32",  r"D:/NLL_matrices/revisions/14M_EOD/step32_merged.h5"),
    # ("Step 64",  r"D:/NLL_matrices/revisions/14M_EOD/step64_merged.h5"),
    # ("Step 128",  r"D:/NLL_matrices/revisions/14M_EOD/step128_merged.h5"),
    # ("Step 256",  r"D:/NLL_matrices/revisions/14M_EOD/step256_merged.h5"),
    # ("Step 512",  r"D:/NLL_matrices/revisions/14M_EOD/step512_merged.h5"),
    # ("Step 1000",  r"D:/NLL_matrices/revisions/14M_EOD/step1000_merged.h5"),
    # ("Step 2000",  r"D:/NLL_matrices/revisions/14M_EOD/step2000_merged.h5"),
    # ("Step 4000",  r"D:/NLL_matrices/revisions/14M_EOD/step4000_merged.h5"),
    # ("Step 8000",  r"D:/NLL_matrices/revisions/14M_EOD/step8000_merged.h5"),
    # ("Step 16000",  r"D:/NLL_matrices/revisions/14M_EOD/step16000_merged.h5"),
    # ("Step 32000",  r"D:/NLL_matrices/revisions/14M_EOD/step32000_merged.h5"),
    # ("Step 64000",  r"D:/NLL_matrices/revisions/14M_EOD/step64000_merged.h5"),
    # ("Step 128000",  r"D:/NLL_matrices/revisions/14M_EOD/step128000_merged.h5"),
    # ("Step 143000",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    #("OLMo 2 1B", r"D:/NLL_matrices/0425-1B_EOD_merged.h5"),
    #("OLMo 2 1B (full eval)", r"D:/NLL_matrices/0425-1B_FULL_EVAL_EOD_merged.h5"),
    #("OLMo 2 7B", r"D:/NLL_matrices/1124-7B_EOD_merged.h5"),
    #("OLMo 2 13B", r"D:/NLL_matrices/1124-13B_EOD_merged.h5"),
    #("gpt2-small", r"D:/NLL_matrices/gpt2_merged.h5"),
    #("gpt2-medium", r"D:/NLL_matrices/gpt2-medium_merged.h5"),
    ("70M",  r"D:/NLL_matrices/70M_deduped_merged.h5"),
    ("160M", r"D:/NLL_matrices/160M_deduped_merged.h5"),
    ("410M", r"D:/NLL_matrices/410M_deduped_merged.h5"),
    ("1B",   r"D:/NLL_matrices/1B_deduped_merged.h5"),
    ("1.4B", r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
    #("2.8B", r"D:/NLL_matrices/2.8B_deduped_merged.h5"),
    ("2.8B with EOD", r"D:/NLL_matrices/2.8B_deduped_EOD_merged.h5"),
    ("6.9B", r"D:/NLL_matrices/6.9B_deduped_merged.h5"),
    ("12B",  r"D:/NLL_matrices/12B_deduped_merged.h5"),
]

CTX              = 2048      # sliding-window length
FILE_LIM         = 5000       # docs per model (None → all)
FILTER_PD_NONNEG = False     # require full left context?
MAX_DOC_LEN      = 500       # skip docs longer than this (None → keep all)

P_START          = 500       # reference position for alignment (1-based)
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# --- colour palette: neighbours get similar hues ---------------------------
N_models  = len(MODELS)
cmap      = mpl.cm.get_cmap("RdYlBu", N_models * 2)
colours   = [cmap(i * 2) for i in range(N_models)]     # take every 2nd colour

global_min = +np.inf          # track overall min / max (shifted)
global_max = -np.inf
model_docs_used = {}

for idx, (name, h5_path) in enumerate(MODELS):
    # accumulate ΣNLL and counts for every P_t (1 … 2048)
    sum_p   = np.zeros(CTX,  dtype=np.float64)
    count_p = np.zeros_like(sum_p, dtype=np.int64)
    docs_used = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs = len(ptr_ds) - 1 if FILE_LIM is None else min(FILE_LIM, len(ptr_ds) - 1)

        for d in tqdm(range(N_docs), desc=f"{name:>12}"):
            s, e = ptr_ds[d], ptr_ds[d + 1]        # HDF5 slice for document d
            L    = e - s                           # document length
            if MAX_DOC_LEN and L > MAX_DOC_LEN:    # skip long docs
                continue

            # --- NLL matrix for this document --------------------------------
            mat = nll_ds[s:e, :].astype(np.float32)             # shape (L, 2047)

            # --- build coordinates for every evaluated window ----------------
            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)  # token index t
            cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)    # offset k-1
            P_t  = cols + 2                                          # 1 … 2048
            P_d  = P_t - rows                                        # dist. from doc start

            keep = P_t <= CTX                                        # stays inside window
            if FILTER_PD_NONNEG:                                     # optional filter
                keep &= P_d >= 0

            if not keep.any():
                continue
            docs_used += 1

            p_idx   = P_t[keep] - 1                 # 1..2047
            nll_val = mat.ravel()[keep]
            m       = np.isfinite(nll_val)          # mask finite values only

            np.add.at(sum_p,   p_idx[m], nll_val[m])
            np.add.at(count_p, p_idx[m], 1)

    # --- mean NLL per P_t for this model ------------------------------------
    mean_p   = np.where(count_p > 0, sum_p / count_p, np.nan)
    baseline = mean_p[P_START - 1]            # raw NLL at alignment point
    shifted  = mean_p - baseline              # ΔNLL so that start == 0
    shifted[P_START - 1] = 0.0                # force exact zero (numeric hygiene)

    # --- NEW: position-bias metric (max Δ − min Δ over plotted range) -------
    # Uses NaN-ignoring reductions to avoid sparse/NaN columns skewing results.
    slice_vals = shifted[P_START - 1:]  # plotted range: P_t ∈ [P_START, CTX]
    if np.isfinite(slice_vals).any():
        y_min = np.nanmin(slice_vals)                      # ignore NaNs
        y_max = np.nanmax(slice_vals)
        i_min = int(np.nanargmin(slice_vals))              # index within slice
        i_max = int(np.nanargmax(slice_vals))
        pt_min = P_START + i_min
        pt_max = P_START + i_max
        pos_bias = y_max - y_min                           # absolute range
        print(f"[POSBIAS] {name}: min={y_min:.6f} @ P_t={pt_min}, "
              f"max={y_max:.6f} @ P_t={pt_max}, Δ={pos_bias:.6f}")
    else:
        print(f"[POSBIAS] {name}: no finite values in plotted range; skipping.")

    # --- update global min / max --------------------------------------------
    valid_vals = shifted[P_START - 1:][np.isfinite(shifted[P_START - 1:])]
    if valid_vals.size:
        global_min = min(global_min, np.min(valid_vals))
        global_max = max(global_max, np.max(valid_vals))

    model_docs_used[name] = docs_used
    # --- plot curve (P_t ≥ P_START) -----------------------------------------
    label = rf"{name}  ($\mathrm{{NLL}}_{{\mathrm{{base}}}} = {baseline:.2f}$)"

    ax.plot(
        np.arange(P_START, CTX + 1),
        shifted[P_START - 1:],
        color=colours[idx],
        linewidth=2.0,
        label=label,
    )

# ─────────────────────── axis formatting & legend ───────────────────────────
# x-axis
ax.set_xlim(P_START, CTX)
ax.set_xticks([500, 768, 1024, 1280, 1536, 1792, 2048
               #, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096
               ])
ax.set_xlabel("Token position within 2048-token context window")

# y-axis : 10 evenly-spaced ticks, guaranteed to include 0
span      = global_max - global_min if global_max > global_min else 1.0
tick_step = span / 9
yticks    = np.round(np.linspace(global_min, global_max, 10), 3)
if 0.0 not in yticks:
    yticks = np.sort(np.append(yticks, 0.0))
margin    = 0.02 * span
ax.set_ylim(global_min - margin, global_max + margin)
ax.set_yticks(yticks)
ax.set_ylabel(f"ΔNLL relative to value at context position {P_START}")

# legend with thicker colour strips
leg = ax.legend(
    #title="Model  (raw NLL at Pₜ = 500)",
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

avg_docs = (sum(model_docs_used.values()) /
            max(len(model_docs_used), 1))

# title & save
#ax.set_title(f"Average of {round(avg_docs)} documents used per model")
ax.set_title("Pythia deduped models - 12B: 602 docs, 6.9B: 704 docs, 2.8B: 641 docs, rest: 2814 docs")
#("Pythia 6.9B: 1516 docs, Pythia 12B: 1245 docs, all others: 2813 docs")

#(f"Pythia 14M: 264 docs used per checkpoint")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/position_bias_by_model_pythia_deduped.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

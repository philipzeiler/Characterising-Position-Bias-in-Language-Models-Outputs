#!/usr/bin/env python3
# OPBtok (ΔNLL range) vs model size, grouped by family; supports per-family CTX, P_START, MAX_DOC_LEN

import os, h5py, numpy as np
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, FixedLocator, FixedFormatter
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
# Optional per-family keys:
#   - "CTX":         context window length (e.g., 2048 for Pythia/OLMo2, 1024 for GPT-2)
#   - "P_START":     alignment position (absolute token index) used for ΔNLL baseline
#   - "MAX_DOC_LEN": max document length to include for this family (None → keep all)
FAMILIES = [
    {
        "family": "Pythia standard",
        "CTX": 2048,
        "P_START": 500,
        "MAX_DOC_LEN": 500,  # optional; if omitted → DEFAULT_MAX_DOC_LEN
        "models": [
            ("14M",   r"D:/NLL_matrices/14M_EOD_merged.h5"),
            ("31M",   r"D:/NLL_matrices/31M_EOD_merged.h5"),
            ("70M",   r"D:/NLL_matrices/70M_EOD_merged.h5"),
            ("160M",  r"D:/NLL_matrices/160M_EOD_merged.h5"),
            ("410M",  r"D:/NLL_matrices/410M_EOD_merged.h5"),
            ("1B",    r"D:/NLL_matrices/1B_EOD_merged.h5"),
            ("1.4B",  r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
            ("2.8B",  r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
            ("6.9B",  r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
            ("12B",   r"D:/NLL_matrices/12B_EOD_merged.h5"),
        ],
    },
    {
        "family": "Pythia deduped",
        "CTX": 2048,
        "P_START": 500,
        "MAX_DOC_LEN": 500,
        "models": [
            ("70M",   r"D:/NLL_matrices/70M_deduped_merged.h5"),
            ("160M",  r"D:/NLL_matrices/160M_deduped_merged.h5"),
            ("410M",  r"D:/NLL_matrices/410M_deduped_merged.h5"),
            ("1B",    r"D:/NLL_matrices/1B_deduped_merged.h5"),
            ("1.4B",  r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
            ("2.8B",  r"D:/NLL_matrices/2.8B_deduped_EOD_merged.h5"),
            ("6.9B",  r"D:/NLL_matrices/6.9B_deduped_merged.h5"),
            ("12B",   r"D:/NLL_matrices/12B_deduped_merged.h5"),
        ],
    },
    {
        "family": "GPT-2",
        "CTX": 1024,
        "P_START": 300,
        "MAX_DOC_LEN": 300,
        "models": [
            ("117M",  r"D:/NLL_matrices/gpt2_merged.h5"),
            ("345M",  r"D:/NLL_matrices/gpt2-medium_merged.h5"),
            ("762M",  r"D:/NLL_matrices/gpt2-large_merged.h5"),
            ("1.5B",  r"D:/NLL_matrices/gpt2-xl_merged.h5"),
        ],
    },
]

# Global defaults (used if a family omits CTX/P_START/MAX_DOC_LEN)
DEFAULT_CTX         = 2048
DEFAULT_P_START     = 500
DEFAULT_MAX_DOC_LEN = 500     # None → keep all

FILE_LIM    = 50000     # docs per model (None → all)

# ───────────────────────────── helpers ──────────────────────────────────────
def _parse_size(size_str: str) -> float:
    """Convert '70M' → 70e6, '1.4B' → 1.4e9 for x-axis sorting/scale."""
    s = size_str.strip().upper().replace(" ", "")
    mult = 1.0
    if s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("B"):
        mult, s = 1e9, s[:-1]
    return float(s) * mult

def posbias_for_h5(h5_path: str, CTX: int, P_START: int, MAX_DOC_LEN: int | None) -> float | None:
    """
    Compute the ΔNLL range (max-min) over P_t ∈ [P_START, CTX] for one merged file.
    Uses per-family CTX, P_START, and MAX_DOC_LEN passed in.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
            sum_p   = np.zeros(CTX, dtype=np.float64)
            count_p = np.zeros(CTX, dtype=np.int64)
            N_docs  = len(ptr_ds) - 1
            if FILE_LIM is not None:
                N_docs = min(FILE_LIM, N_docs)

            for d in range(N_docs):
                s, e = int(ptr_ds[d]), int(ptr_ds[d + 1])
                L = e - s
                if MAX_DOC_LEN and L > MAX_DOC_LEN:
                    continue

                # (L, CTX-1) matrix; columns correspond to absolute P_t = 2..CTX
                mat = nll_ds[s:e, :].astype(np.float32)

                # Build coordinates for windows in this doc
                rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)
                cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)
                P_t  = cols + 2     # absolute positions 2..CTX
                # P_d = P_t - rows   # (not used, but kept for clarity)

                keep = (P_t <= CTX)
                if not np.any(keep):
                    continue

                p_idx   = (P_t[keep] - 1).astype(np.int64)  # map 1..CTX → 0..CTX-1
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
        sl = shifted[P_START - 1:CTX]          # plotted range
        if not np.isfinite(sl).any():
            return None
        return float(np.nanmax(sl) - np.nanmin(sl))
    except OSError:
        return None

# ───────────────────────────── compute series ───────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# Fixed colors by family (per request)
FAMILY_COLORS = {
    "Pythia standard": "red",
    "Pythia deduped":  "blue",
    "GPT-2":           "green",
}

series = []   # list of dicts: {"family", "sizes", "sizes_numeric", "bias_vals"}
x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf

for fi, fam in enumerate(FAMILIES):
    fam_name = fam["family"]
    fam_ctx  = int(fam.get("CTX", DEFAULT_CTX))
    fam_ps   = int(fam.get("P_START", DEFAULT_P_START))
    fam_maxL = fam.get("MAX_DOC_LEN", DEFAULT_MAX_DOC_LEN)
    models   = fam["models"]

    print(f"\n[START] Family: {fam_name}  |  CTX={fam_ctx}  |  P_START={fam_ps}  |  MAX_DOC_LEN={fam_maxL}")
    sizes_str, sizes_num, biases = [], [], []

    for size_str, path in models:
        print(f"  ├─ Computing OPBtok for {fam_name} / {size_str} …")
        pb = posbias_for_h5(path, fam_ctx, fam_ps, fam_maxL)
        if pb is None:
            print(f"  │   [WARN] Skipping (missing baseline or file): {path}")
            continue

        s_num = _parse_size(size_str)
        sizes_str.append(size_str)
        sizes_num.append(s_num)
        biases.append(pb)

        x_min = min(x_min, s_num); x_max = max(x_max, s_num)
        y_min = min(y_min, pb);    y_max = max(y_max, pb)
        print(f"  │   ΔNLL range (OPBtok) = {pb:.6f}")

    # sort within family by numeric size so lines connect left→right
    order = np.argsort(sizes_num)
    sizes_str  = [sizes_str[i]  for i in order]
    sizes_num  = [sizes_num[i]  for i in order]
    biases     = [biases[i]     for i in order]

    if sizes_num:
        series.append({
            "family": fam_name,
            "sizes_str": sizes_str,
            "sizes_num": np.array(sizes_num, dtype=float),
            "bias": np.array(biases, dtype=float),
            "color": FAMILY_COLORS.get(fam_name, "black"),  # ← fixed color mapping
        })

# ───────────────────────────── plot ─────────────────────────────────────────
for s in series:
    ax.plot(
        s["sizes_num"], s["bias"],
        marker="o",
        markersize=10,      # bigger dots
        linestyle=":",      # dotted line
        linewidth=2.0,
        color=s["color"],
        label=s["family"],
    )

# x-axis: model size in log2 scale, ticks at Pythia sizes with literal labels like "14M", "31M", etc.
ax.set_xscale("log", base=2)

# Build Pythia tick set (union of standard + deduped), sorted by numeric size
pythia_tick_pairs = []
for fam in FAMILIES:
    if fam["family"] in ("Pythia standard", "Pythia deduped"):
        for size_str, _ in fam["models"]:
            pythia_tick_pairs.append((_parse_size(size_str), size_str))
# unique by numeric value, then sort
tmp = {}
for num, lab in pythia_tick_pairs:
    tmp[num] = lab
pythia_tick_nums = sorted(tmp.keys())
pythia_tick_labs = [tmp[num] for num in pythia_tick_nums]

# Use FixedLocator/FixedFormatter so our custom labels are not overridden
ax.xaxis.set_major_locator(FixedLocator(pythia_tick_nums))
ax.xaxis.set_major_formatter(FixedFormatter(pythia_tick_labs))

ax.set_xlim(x_min * 0.95, x_max * 1.05)

ax.set_xlabel("Model size in number of parameters (log scale)")
#ax.set_ylabel("Token output position bias in NLL")
ax.set_ylabel(fr"Token output position bias $\mathrm{{OPB}}^{{\mathrm{{tok}}}}$")

ax.set_ylim(bottom=0)  # start y-axis at 0

ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
#ax.set_title("Token output position bias by model size and family")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/opbtok_all_models.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

#!/usr/bin/env python3
# OPBtok-EOD (ΔNLL range using only EOD rows) vs model size, grouped by family

import os, h5py, numpy as np
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter   # ← changed: use Fixed* for literal labels
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
        "MAX_DOC_LEN": 500,
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
            #("70M",   r"D:/NLL_matrices/70M_deduped_merged.h5"),
            #("160M",  r"D:/NLL_matrices/160M_deduped_merged.h5"),
            #("410M",  r"D:/NLL_matrices/410M_deduped_merged.h5"),
            #("1B",    r"D:/NLL_matrices/1B_deduped_merged.h5"),
            #("1.4B",  r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
            ("2.8B",  r"D:/NLL_matrices/2.8B_deduped_EOD_merged.h5"),
            #("6.9B",  r"D:/NLL_matrices/6.9B_deduped_merged.h5"),
            #("12B",   r"D:/NLL_matrices/12B_deduped_merged.h5"),
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
FILE_LIM            = 50000   # docs per model (None → all)

# ───────────────────────────── helpers ──────────────────────────────────────
def _parse_size(size_str: str) -> float:
    """Convert '70M' → 70e6, '1.4B' → 1.4e9 for x-axis positioning."""
    s = size_str.strip().upper().replace(" ", "")
    mult = 1.0
    if s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("B"):
        mult, s = 1e9, s[:-1]
    return float(s) * mult

def posbias_eod_for_h5(h5_path: str, CTX: int, P_START: int, MAX_DOC_LEN: int | None) -> float | None:
    """
    EOD-only OPBtok:
      • For each document, take ONLY the last row (EOD token) of its NLL matrix.
      • Average EOD NLL at absolute positions P_t ∈ {2..CTX}.
      • Align at P_START; return max(Δ) - min(Δ) over P_t ∈ [P_START..CTX].
    """
    try:
        with h5py.File(h5_path, "r") as f:
            nll_ds, ptr_ds = f["nll"], f["doc_ptr"]

            sum_k   = np.zeros(CTX - 1, dtype=np.float64)  # k=1..CTX-1 → P_t=2..CTX
            count_k = np.zeros(CTX - 1, dtype=np.int64)

            N_docs = len(ptr_ds) - 1
            if FILE_LIM is not None:
                N_docs = min(FILE_LIM, N_docs)

            for d in range(N_docs):
                s, e = int(ptr_ds[d]), int(ptr_ds[d + 1])
                L = e - s
                if MAX_DOC_LEN and L > MAX_DOC_LEN:
                    continue
                if L <= 0:
                    continue

                eod_row = e - 1
                vec = nll_ds[eod_row, :].astype(np.float32)   # (CTX-1,)
                m = np.isfinite(vec)
                if not np.any(m):
                    continue

                sum_k   += np.where(m, vec, 0.0)
                count_k += m.astype(np.int64)

        mean_k = np.where(count_k > 0, sum_k / count_k, np.nan)

        # Map to absolute P_t where index == P_t (1..CTX), P_t=1 is NaN
        mean_full = np.full(CTX + 1, np.nan, dtype=np.float64)
        mean_full[2:CTX + 1] = mean_k

        base = mean_full[P_START] if 1 <= P_START <= CTX else np.nan
        if not np.isfinite(base):
            return None

        shifted = mean_full - base
        shifted[P_START] = 0.0

        sl = shifted[P_START:CTX + 1]
        if not np.isfinite(sl).any():
            return None

        return float(np.nanmax(sl) - np.nanmin(sl))
    except OSError:
        return None

# ───────────────────────────── compute series ───────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# Fixed colors by family
FAMILY_COLORS = {
    "Pythia standard": "red",
    "Pythia deduped":  "blue",
    "GPT-2":           "green",
}

series = []   # list of dicts: {"family", "sizes_str", "sizes_num", "bias", "color"}
x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf

for fam in FAMILIES:
    fam_name = fam["family"]
    fam_ctx  = int(fam.get("CTX", DEFAULT_CTX))
    fam_ps   = int(fam.get("P_START", DEFAULT_P_START))
    fam_maxL = fam.get("MAX_DOC_LEN", DEFAULT_MAX_DOC_LEN)
    models   = fam["models"]

    print(f"\n[START] Family: {fam_name}  |  CTX={fam_ctx}  |  P_START={fam_ps}  |  MAX_DOC_LEN={fam_maxL}")
    sizes_str, sizes_num, biases = [], [], []

    for size_str, path in models:
        print(f"  ├─ Computing OPBtok-EOD for {fam_name} / {size_str} …")
        pb = posbias_eod_for_h5(path, fam_ctx, fam_ps, fam_maxL)
        if pb is None:
            print(f"  │   [WARN] Skipping (missing baseline or file): {path}")
            continue

        s_num = _parse_size(size_str)
        sizes_str.append(size_str)
        sizes_num.append(s_num)
        biases.append(pb)

        x_min = min(x_min, s_num); x_max = max(x_max, s_num)
        y_min = min(y_min, pb);    y_max = max(y_max, pb)
        print(f"  │   ΔNLL range (OPBtok-EOD) = {pb:.6f}")

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
            "color": FAMILY_COLORS.get(fam_name, "black"),
        })

# ───────────────────────────── plot ─────────────────────────────────────────
for s in series:
    ax.plot(
    s["sizes_num"], s["bias"],
    marker="o",
    markersize=10,     # ← bigger dots (try 10–14)
    linestyle=":",     # ← dotted line
    linewidth=2.0,
    color=s["color"],
    label=s["family"],
)

# x-axis: log2 scale; ticks placed at Pythia sizes with literal labels ("14M", "31M", …)
ax.set_xscale("log", base=2)

# gather unique Pythia sizes → numeric positions + literal labels
pythia_tick_pairs = []
for fam in FAMILIES:
    if fam["family"] in ("Pythia standard", "Pythia deduped"):
        for size_str, _ in fam["models"]:
            pythia_tick_pairs.append((_parse_size(size_str), size_str))
# dedupe by numeric value, keep last label, then sort by numeric
tmp = {}
for num, lab in pythia_tick_pairs:
    tmp[num] = lab
pythia_tick_nums = sorted(tmp.keys())
pythia_tick_labs = [tmp[num] for num in pythia_tick_nums]

# ← key change: lock ticks & labels so they display literally as "14M", "31M", etc.
ax.xaxis.set_major_locator(FixedLocator(pythia_tick_nums))
ax.xaxis.set_major_formatter(FixedFormatter(pythia_tick_labs))

ax.set_xlim(x_min * 0.95, x_max * 1.05)

ax.set_xlabel("Model Number of Parameters (log scale)")
ax.set_ylabel(fr"Token OP Bias ($\mathrm{{OPB}}^{{\mathrm{{tok}}}}$(EOD))")
#ax.set_ylabel("Token output position bias in NLL (EOD only)")
ax.set_ylim(bottom=0)

ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
#ax.set_title("EOD-only token output position bias by model size and family")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/opbtok_eod_all_models.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

#!/usr/bin/env python3
# Document-level OPB (ΔNLL over P_d) vs model size, grouped by family.
# Carries over aesthetics: log2 x with literal Pythia labels, dotted lines, big dots.
# Change: For OLMo 2 models ONLY, compute bias using *only* the “interesting” P_d positions.

import os, h5py, numpy as np
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl
from matplotlib.ticker import FixedLocator, FixedFormatter  # fixed ticks/labels
from tqdm import tqdm

# ───────────────────────────── user settings ────────────────────────────────
# Per-family options:
#   - "CTX":         context window (e.g., 2048 for Pythia/OLMo2, 1024 for GPT-2)
#   - "P_ALIGN":     P_d to align at (baseline), usually 1
#   - "P_START":     first P_d to include in the plotted slice
#   - "P_END":       last P_d to include (≤ CTX)
#   - "MAX_DOC_LEN": skip docs longer than this (None → keep all)
FAMILIES = [
    {
        "family": "Pythia standard",
        "CTX": 2048, "P_ALIGN": 1, "P_START": 1, "P_END": 1548, "MAX_DOC_LEN": 500,
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
        "CTX": 2048, "P_ALIGN": 1, "P_START": 1, "P_END": 1548, "MAX_DOC_LEN": 500,
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
        "CTX": 1024, "P_ALIGN": 1, "P_START": 1, "P_END": 724, "MAX_DOC_LEN": 300,
        "models": [
            ("117M",  r"D:/NLL_matrices/gpt2_merged.h5"),
            ("345M",  r"D:/NLL_matrices/gpt2-medium_merged.h5"),
            ("762M",  r"D:/NLL_matrices/gpt2-large_merged.h5"),
            ("1.5B",  r"D:/NLL_matrices/gpt2-xl_merged.h5"),
        ],
    },
    {
        "family": "OLMo 2",
        "CTX": 4096, "P_ALIGN": 1, "P_START": 1, "P_END": 2048, "MAX_DOC_LEN": 2048,
        "models": [
            ("1B",   r"D:/NLL_matrices/0425-1B_EOD_merged.h5"),
            ("7B",   r"D:/NLL_matrices/1124-7B_EOD_merged.h5"),
            ("13B",  r"D:/NLL_matrices/1124-13B_EOD_merged.h5"),
        ],
    },
]

# Global defaults if a family omits a key
DEFAULT_CTX         = 2048
DEFAULT_P_ALIGN     = 1
DEFAULT_P_START     = 1
DEFAULT_P_END       = 1536
DEFAULT_MAX_DOC_LEN = 512
FILE_LIM            = 50000   # None → use all

# Apply “interesting Pd only” filter to these families:
INTERESTING_PD_FAMILIES = {"OLMo 2"}

# ───────────────────────────── helpers ──────────────────────────────────────
def _parse_size(size_str: str) -> float:
    """Convert '70M' → 70e6, '1.4B' → 1.4e9 (for numeric x positions)."""
    s = size_str.strip().upper().replace(" ", "")
    if s.endswith("M"): return float(s[:-1]) * 1e6
    if s.endswith("B"): return float(s[:-1]) * 1e9
    return float(s)  # fallback if a plain number slips in

def interesting_mask(Pd: np.ndarray, ctx: int) -> np.ndarray:
    """Keep only “interesting” P_d: within ±128 OR power-of-two up to 4096."""
    Pd_i  = Pd.astype(np.int64)
    absPd = np.abs(Pd_i)
    in_small = (Pd_i >= -128) & (Pd_i <= 128)
    is_pow2  = (absPd > 0) & ((absPd & (absPd - 1)) == 0)
    within_4096 = absPd <= 4096
    return in_small | (is_pow2 & within_4096)

def doc_posbias_for_h5(h5_path: str, CTX: int, P_ALIGN: int, P_START: int, P_END: int,
                       MAX_DOC_LEN: int | None, use_interesting_pd: bool = False) -> float | None:
    """
    Document-level position bias for a merged .h5:
      1) mean NLL vs P_d in [1..P_END], ignoring NaNs
      2) align at P_d=P_ALIGN (ΔNLL)
      3) return max(ΔNLL) − min(ΔNLL) over P_d ∈ [P_START..P_END]
      4) If use_interesting_pd=True, restrict to “interesting” P_d positions
    """
    if not os.path.exists(h5_path):
        return None

    sum_pd   = np.zeros(P_END, dtype=np.float64)   # bins 0..P_END-1 → P_d = i+1
    count_pd = np.zeros(P_END, dtype=np.int64)

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        N_docs = len(ptr_ds) - 1
        if FILE_LIM is not None:
            N_docs = min(FILE_LIM, N_docs)

        for d in range(N_docs):
            s, e = int(ptr_ds[d]), int(ptr_ds[d + 1])
            L = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            mat = nll_ds[s:e, :].astype(np.float32)  # (L, CTX-1) for P_t = 2..CTX

            rows = np.repeat(np.arange(L, dtype=np.int32), CTX - 1)   # token index
            cols = np.tile(np.arange(CTX - 1, dtype=np.int32), L)     # column index
            P_t  = cols + 2                                           # absolute 2..CTX
            P_d  = P_t - rows                                         # doc-start position

            keep = (P_d >= 1) & (P_d <= P_END) & (P_t <= CTX)
            if use_interesting_pd:
                keep &= interesting_mask(P_d, CTX)

            if not np.any(keep):
                continue

            d_idx   = (P_d[keep] - 1).astype(np.int64)               # 0..P_END-1
            nll_val = mat.ravel()[keep]
            m = np.isfinite(nll_val)
            if np.any(m):
                np.add.at(sum_pd,   d_idx[m], nll_val[m])
                np.add.at(count_pd, d_idx[m], 1)

    mean_pd  = np.where(count_pd > 0, sum_pd / count_pd, np.nan)
    base     = mean_pd[P_ALIGN - 1] if 1 <= P_ALIGN <= P_END else np.nan
    if not np.isfinite(base):
        return None

    shifted  = mean_pd - base
    shifted[P_ALIGN - 1] = 0.0

    sl = shifted[P_START - 1 : P_END]     # plotted slice
    if not np.isfinite(sl).any():
        return None

    y_min = np.nanmin(sl)
    y_max = np.nanmax(sl)
    return float(y_max - y_min)

# ───────────────────────────── compute series ───────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

# Fixed colors by family
FAMILY_COLORS = {
    "Pythia standard": "red",
    "Pythia deduped":  "blue",
    "GPT-2":           "green",
    "OLMo 2":          "purple",
}

series = []   # dicts: {"family","sizes_str","sizes_num","bias","color"}
x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf

for fam in FAMILIES:
    fam_name = fam["family"]
    CTX      = int(fam.get("CTX",      DEFAULT_CTX))
    P_ALIGN  = int(fam.get("P_ALIGN",  DEFAULT_P_ALIGN))
    P_START  = int(fam.get("P_START",  DEFAULT_P_START))
    P_END    = int(fam.get("P_END",    min(DEFAULT_P_END, CTX)))
    MAXL     =      fam.get("MAX_DOC_LEN", DEFAULT_MAX_DOC_LEN)

    use_interesting = fam_name in INTERESTING_PD_FAMILIES  # ← ONLY OLMo 2

    print(f"\n[START] {fam_name}: CTX={CTX} | P_ALIGN={P_ALIGN} | P_START={P_START} | P_END={P_END} | MAX_DOC_LEN={MAXL} | interesting={use_interesting}")
    sizes_str, sizes_num, biases = [], [], []

    for size_str, path in fam["models"]:
        print(f"  ├─ {size_str}: {path}")
        pb = doc_posbias_for_h5(path, CTX, P_ALIGN, P_START, P_END, MAXL, use_interesting_pd=use_interesting)
        if pb is None:
            print(f"  │   [WARN] skipped (missing file or baseline)")
            continue
        print(f"  │   ΔNLL range (doc-level OPB) = {pb:.6f}")
        s_num = _parse_size(size_str)
        sizes_str.append(size_str); sizes_num.append(s_num); biases.append(pb)

        x_min = min(x_min, s_num); x_max = max(x_max, s_num)
        y_min = min(y_min, pb);    y_max = max(y_max, pb)

    # sort by numeric size for left→right lines
    if sizes_num:
        order = np.argsort(sizes_num)
        series.append({
            "family": fam_name,
            "sizes_str": [sizes_str[i] for i in order],
            "sizes_num": np.array([sizes_num[i] for i in order], dtype=float),
            "bias":      np.array([biases[i]    for i in order], dtype=float),
            "color": FAMILY_COLORS.get(fam_name, "black"),
        })

# ───────────────────────────── plot ─────────────────────────────────────────
for s in series:
    ax.plot(
        s["sizes_num"], s["bias"],
        marker="o", markersize=10, linestyle=":", linewidth=2.2,
        color=s["color"], label=s["family"]
    )

# x-axis: log2; ticks at *Pythia* sizes with literal labels "14M", "31M", etc.
ax.set_xscale("log", base=2)
pythia_tick_pairs = []
for fam in FAMILIES:
    if fam["family"] in ("Pythia standard", "Pythia deduped"):
        for size_str, _ in fam["models"]:
            pythia_tick_pairs.append((_parse_size(size_str), size_str))
# dedupe by numeric, then sort
tmp = {}
for num, lab in pythia_tick_pairs:
    tmp[num] = lab
tick_nums = sorted(tmp.keys())
tick_labs = [tmp[n] for n in tick_nums]

if tick_nums:
    ax.xaxis.set_major_locator(FixedLocator(tick_nums))        # fixed positions
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labs))     # literal labels

# padding & axis labels
if np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max:
    ax.set_xlim(x_min * 0.95, x_max * 1.05)

ax.set_xlabel("Model size in parameters (log scale)")
#ax.set_ylabel("Document output position bias")
ax.set_ylabel(fr"Document output position bias $\mathrm{{OPB}}^{{\mathrm{{doc}}}}$")
ax.set_ylim(bottom=0)  # start y-axis at 0

ax.legend(loc="upper right", frameon=True, handlelength=4, borderaxespad=0.4)
#ax.set_title("Document-level output position bias vs model size (by family)")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/docopb_all_models.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

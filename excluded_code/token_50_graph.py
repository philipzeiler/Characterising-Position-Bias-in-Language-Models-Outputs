#!/usr/bin/env python3
"""
Token-50 mean *ΔNLL* vs context-window position, with optional "left neighbor ≤ 500" filter
-------------------------------------------------------------------------------------------
Step 1: Tokenize the first N_DOCS shuffled Pile-validation docs (seed=DOC_SEED),
        find occurrences of token ID 50 (with an occurrence-mode filter), and print a summary.
Step 2: For each merged model HDF5, plot the mean NLL of ONLY the rows where
        token ID == 50, as a function of absolute position P_t ∈ {2..2048}.
        Curves are vertically aligned at P_ALIGN; the x-range starts at P_START.
"""

# ───────────────────────────── user settings ────────────────────────────────
TOKEN_ID           = 50
N_DOCS             = 5_000
DOC_SEED           = 753

TOKENIZER_MODEL_ID = "EleutherAI/pythia-70M"
TOKENIZER_REVISION = "step143000"
INCLUDE_SPECIAL    = False

OCCURRENCE_FILTER          = "first_only"    # {"all","first_only","nonfirst_only"}

REQUIRE_LEFT_NEIGHBOR_SHORT = True
LEFT_NEIGHBOR_MAXLEN        = 500

MODELS = [
    ("70M deduped",  r"D:/NLL_matrices/70M_deduped_merged.h5"),
    ("160M deduped", r"D:/NLL_matrices/160M_deduped_merged.h5"),
    ("410M deduped", r"D:/NLL_matrices/410M_deduped_merged.h5"),
    ("1B deduped",   r"D:/NLL_matrices/1B_deduped_merged.h5"),
    ("1.4B deduped", r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
    ("6.9B deduped", r"D:/NLL_matrices/6.9B_deduped_merged.h5"),
    ("12B deduped",  r"D:/NLL_matrices/12B_deduped_merged.h5"),
]

# Plot controls
CTX         = 2048
N_POS       = CTX - 1      # columns correspond to P_t = 2..CTX
MAX_DOC_LEN = 500
P_ALIGN     = 500          # absolute P_t for baseline
P_START     = 500
P_END       = CTX          # through absolute position 2048

# --- smoothing (NEW) ---
SMOOTH = True          # set False to disable smoothing
SMOOTH_WINDOW = 5      # odd integer >=3; 3 is mild & safe

# ── imports ─────────────────────────────────────────────────────────────────
import random, numpy as np, h5py, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl

# ───────────────────────────── helpers ──────────────────────────────────────
def _occ_filter_mask(pos_indices: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "all":
        return torch.ones_like(pos_indices, dtype=torch.bool)
    elif mode == "first_only":
        return (pos_indices == 0)
    elif mode == "nonfirst_only":
        return (pos_indices > 0)
    else:
        raise ValueError("OCCURRENCE_FILTER must be one of {'all','first_only','nonfirst_only'}")

# NEW: NaN-aware moving average smoothing
def _smooth_nanaware(y: np.ndarray, w: int) -> np.ndarray:
    """
    Centered moving average with window w, ignoring NaNs.
    Keeps length; returns NaN where the window has no finite values.
    """
    if w is None or w <= 1:
        return y
    kernel = np.ones(int(w), dtype=float)
    y0     = np.nan_to_num(y, nan=0.0)
    mask   = np.isfinite(y).astype(float)

    num = np.convolve(y0,   kernel, mode="same")
    den = np.convolve(mask, kernel, mode="same")

    out = np.full_like(y, np.nan, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out

# ─────────────────────── Step 1: find occurrences & lengths ────────────────
ds  = load_dataset("pietrolesci/pile-validation", split="validation",
                   verification_mode="no_checks")
tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_REVISION)

doc_ids = list(range(len(ds)))
random.Random(DOC_SEED).shuffle(doc_ids)
doc_ids = doc_ids[:N_DOCS]

occ_positions_per_doc = []
occ_counts_per_doc    = []
length_per_doc        = []

total_tokens_scanned  = 0
total_occurrences     = 0
docs_with_occurrence  = 0

decoded_vis = tok.decode([TOKEN_ID], clean_up_tokenization_spaces=False, skip_special_tokens=False)
decoded_vis = decoded_vis.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

for doc_id in tqdm(doc_ids, desc=f"Scanning for ID={TOKEN_ID}"):
    enc = tok(
        ds[doc_id]["text"],
        return_tensors="pt",
        add_special_tokens=INCLUDE_SPECIAL
    )
    ids = enc.input_ids[0]
    L   = int(ids.numel())
    length_per_doc.append(L)
    total_tokens_scanned += L

    occ_mask_all = (ids == TOKEN_ID)
    if occ_mask_all.any():
        pos_indices = torch.nonzero(occ_mask_all, as_tuple=False).view(-1)
        keep_mask = _occ_filter_mask(pos_indices, OCCURRENCE_FILTER)
        pos_kept  = pos_indices[keep_mask]
        if pos_kept.numel() > 0:
            arr = pos_kept.cpu().numpy()
            occ_positions_per_doc.append(arr)
            occ_counts_per_doc.append(int(arr.size))
            total_occurrences += int(arr.size)
            docs_with_occurrence += 1
        else:
            occ_positions_per_doc.append(np.empty((0,), dtype=np.int64))
            occ_counts_per_doc.append(0)
    else:
        occ_positions_per_doc.append(np.empty((0,), dtype=np.int64))
        occ_counts_per_doc.append(0)

occurrences_left_ok = 0
if REQUIRE_LEFT_NEIGHBOR_SHORT:
    for d in range(0, len(doc_ids) - 1):
        if length_per_doc[d + 1] <= LEFT_NEIGHBOR_MAXLEN:
            occurrences_left_ok += occ_counts_per_doc[d]

print("\n─ Occurrence summary (Step 1) ─────────────────────────")
print(f"Token ID: {TOKEN_ID}  decoded: '{decoded_vis}'")
print(f"Docs scanned: {len(doc_ids):,}  (seed={DOC_SEED})")
print(f"Total tokens scanned: {total_tokens_scanned:,}")
print(f"OCCURRENCE_FILTER = '{OCCURRENCE_FILTER}'")
print(f"Docs with ≥1 matching occurrence (after OCCURRENCE_FILTER): {docs_with_occurrence:,}")
print(f"Total matching occurrences (after OCCURRENCE_FILTER): {total_occurrences:,}")
if REQUIRE_LEFT_NEIGHBOR_SHORT:
    print(f"Occurrences remaining if 'left neighbor (d+1) ≤ {LEFT_NEIGHBOR_MAXLEN}' enforced: {occurrences_left_ok:,}")
print("──────────────────────────────────────────────────────\n")

# ─────────────────────── Step 2: build the plot ────────────────────────────
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=2.1)
fig, ax = plt.subplots(figsize=(20, 10))

N_models = len(MODELS)
cmap     = mpl.cm.get_cmap("RdYlBu", N_models * 2)
colours  = [cmap(i * 2) for i in range(N_models)]

global_min = +np.inf
global_max = -np.inf

# NEW: track occurrences used per model for the title summary
occ_used_by_model = {}

for idx, (name, h5_path) in enumerate(MODELS):
    sum_p   = np.zeros(N_POS, dtype=np.float64)   # Σ NLL at each column (P_t = 2..CTX)
    count_p = np.zeros(N_POS, dtype=np.int64)     # # of occurrences contributing
    occ_used_total = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        M_docs = len(ptr_ds) - 1
        D_use = min(M_docs, len(occ_positions_per_doc))

        for d in tqdm(range(D_use), desc=f"{name:>12}"):
            if REQUIRE_LEFT_NEIGHBOR_SHORT:
                if (d + 1) >= len(length_per_doc) or length_per_doc[d + 1] > LEFT_NEIGHBOR_MAXLEN:
                    continue

            s = int(ptr_ds[d]); e = int(ptr_ds[d + 1]); L = e - s
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            occs = occ_positions_per_doc[d]
            if occs.size == 0:
                continue
            occs = occs[occs < L]
            if occs.size == 0:
                continue

            nll_doc = nll_ds[s:e, :].astype(np.float32)   # (L, 2047) for P_t = 2..2048
            rows    = nll_doc[occs, :]                    # (n_occ, 2047)

            m = np.isfinite(rows)
            sum_p   += np.where(m, rows, 0.0).sum(axis=0)
            count_p += m.sum(axis=0).astype(np.int64)
            occ_used_total += int(occs.size)

    # mean NLL of token-50 per absolute position
    mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)

    # Align at P_ALIGN (absolute P_t) → column index (P_ALIGN - 2)
    baseline = mean_p[P_ALIGN - 2] if 2 <= P_ALIGN <= CTX else np.nan
    shifted  = mean_p - baseline
    if np.isfinite(baseline):
        shifted[P_ALIGN - 2] = 0.0

    x = np.arange(max(2, P_START), P_END + 1)              # absolute positions 2..2048
    y = shifted[(x[0] - 2):(x[-1] - 2) + 1]

    # --- NEW: optional smoothing before stats/plot ---
    y_plot = _smooth_nanaware(y, SMOOTH_WINDOW) if SMOOTH else y

    vfin = y_plot[np.isfinite(y_plot)]
    if vfin.size:
        global_min = min(global_min, float(np.min(vfin)))
        global_max = max(global_max, float(np.max(vfin)))

    # remember occurrences used for this model (for the title)
    occ_used_by_model[name] = occ_used_total

    label = rf"{name}  ($\mathrm{{NLL}}_{{\mathrm{{base}}}} = {baseline:.2f}$)"
    ax.plot(x, y_plot, color=colours[idx], linewidth=2.0, label=label)

# ─────────────── axis formatting & legend (no left gap; stop at 2048) ──────
ax.set_xlim(max(2, P_START), P_END)
ticks_all = [2, 256, 500, 768, 1024, 1280, 1536, 1792, 2048]
ax.set_xticks([t for t in ticks_all if max(2, P_START) <= t <= P_END])
ax.set_xlabel("Token position within 2048-token context window")

if not np.isfinite(global_min) or not np.isfinite(global_max):
    global_min, global_max = 0.0, 1.0
span   = max(global_max - global_min, 1e-6)
yticks = np.round(np.linspace(global_min, global_max, 10), 3)
ax.set_ylim(global_min - 0.02 * span, global_max + 0.02 * span)
ax.set_yticks(yticks)
ax.set_ylabel(f"ΔNLL for token ID {TOKEN_ID} relative to value at context position {P_ALIGN}")

leg = ax.legend(loc="lower left", frameon=True, handlelength=4, borderaxespad=0.4)
for line in leg.get_lines():
    line.set_linewidth(6)

# ─────────────── Title summary (unchanged except uses occ_used_by_model) ───
def _find_first_key_containing(sub):
    for k in occ_used_by_model:
        if sub in k:
            return k
    return None

k_69  = _find_first_key_containing("6.9B")
k_12  = _find_first_key_containing("12B")
others = [k for k in occ_used_by_model.keys() if k not in {k_69, k_12}]

c_69 = occ_used_by_model.get(k_69, 0)
c_12 = occ_used_by_model.get(k_12, 0)
c_others_set = {occ_used_by_model[k] for k in others} if others else set()

if len(c_others_set) == 1:
    c_others = c_others_set.pop() if c_others_set else 0
    title = f"Pythia 12B: {c_12:,} occurrences, Pythia 6.9B: {c_69:,} occurrences, all others: {c_others:} occurrences"
else:
    if c_others_set:
        title = (f"Pythia 6.9B: {c_69:,} occurrences, Pythia 12B: {c_12:,} occurrences, "
                 f"all others: {min(c_others_set):,}–{max(c_others_set):,} occurrences")
    else:
        title = f"Pythia 6.9B: {c_69:,} occurrences, Pythia 12B: {c_12:,} occurrences"

ax.set_title(title)

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/token50_delta_nll_vs_position_first_only_test.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

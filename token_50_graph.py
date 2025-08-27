#!/usr/bin/env python3
"""
Token-50 mean *ΔNLL* vs context-window position, with optional "left neighbor ≤ 500" filter
-------------------------------------------------------------------------------------------
Step 1: Tokenize the first N_DOCS shuffled Pile-validation docs (seed=DOC_SEED),
        find occurrences of token ID 50 (with an occurrence-mode filter), and print a summary.
Step 2: For each merged model HDF5, plot the mean NLL of ONLY the rows where
        token ID == 50, as a function of absolute position k ∈ {1..2047}.
        Curves are vertically aligned at P_ALIGN; the x-range starts at P_START.

Filters
~~~~~~~
- OCCURRENCE_FILTER ∈ {"all","first_only","nonfirst_only"}
- REQUIRE_LEFT_NEIGHBOR_SHORT: if True, for an occurrence in doc d, only include it if
  the *left neighbor* doc in the snake (i.e., doc d+1 in shuffled order) has length
  ≤ LEFT_NEIGHBOR_MAXLEN (500). If d is the last usable doc (no neighbor), exclude.
- MAX_DOC_LEN applies to the *current* doc (as in earlier figures).

Assumptions
~~~~~~~~~~~
- The merged HDF5 files were built in the same deterministic shuffled doc order.
- Tokenizer/vocab matches the evaluation runs (EleutherAI/pythia-*).
"""

# ───────────────────────────── user settings ────────────────────────────────
TOKEN_ID           = 50
N_DOCS             = 5_000
DOC_SEED           = 753

# Tokenizer used to discover positions of TOKEN_ID (same family/vocab as eval)
TOKENIZER_MODEL_ID = "EleutherAI/pythia-70M"
TOKENIZER_REVISION = "step143000"
INCLUDE_SPECIAL    = False   # keep False to mirror earlier runs

# Which occurrences to include in the plot:
#   "all"           → every occurrence of ID 50
#   "first_only"    → only if it is the first token in the document (t == 0)
#   "nonfirst_only" → only if it is not the first token (t > 0)
OCCURRENCE_FILTER          = "nonfirst_only"    # {"all","first_only","nonfirst_only"}

# NEW: require the *left neighbor* doc (doc d+1) to be short (≤ 500) to include an occurrence
REQUIRE_LEFT_NEIGHBOR_SHORT = True
LEFT_NEIGHBOR_MAXLEN        = 500

# Models (merged HDF5 with datasets: "nll" float16 [N_tokens_total, 2047],
# and "doc_ptr" int64 [N_docs+1])
MODELS = [
    # Example: checkpoints of the same size
    #("Pythia 14M",  r"D:/NLL_matrices/14M_EOD_merged.h5"),
    #("Pythia 31M",  r"D:/NLL_matrices/31M_EOD_merged.h5"),
    #("Pythia 70M",  r"D:/NLL_matrices/70M_EOD_merged.h5"),
    ("Pythia 70M deduped",  r"D:/NLL_matrices/70M_deduped_merged.h5"),
    #("Pythia 160M", r"D:/NLL_matrices/160M_EOD_merged.h5"),
    #("Pythia 160M (no EOD)", r"D:/NLL_matrices/160M_merged.h5"),
    #("Pythia 410M", r"D:/NLL_matrices/410M_EOD_merged.h5"),
    #("Pythia 410M (no EOD)", r"D:/NLL_matrices/410M_merged.h5"),
    #("Pythia 1B",   r"D:/NLL_matrices/1B_EOD_merged.h5"),
    #("Pythia 1.4B", r"D:/NLL_matrices/1.4B_EOD_merged.h5"),
    #("Pythia 1.4B (no EOD)", r"D:/NLL_matrices/1.4B_merged.h5"),
    ("Pythia 1.4B deduped", r"D:/NLL_matrices/1.4B_deduped_merged.h5"),
    #("Pythia 2.8B", r"D:/NLL_matrices/2.8B_EOD_merged.h5"),
    #("Pythia 6.9B", r"D:/NLL_matrices/6.9B_EOD_merged.h5"),
    #("Pythia 12B",  r"D:/NLL_matrices/12B_EOD_merged.h5"),
]


# Plot controls
CTX         = 2048
N_POS       = CTX - 1      # 1..2047
MAX_DOC_LEN = 500          # None → keep all (applies to *current* doc)
P_ALIGN     = 500          # baseline alignment position
P_START     = 500          # crop left margin (start plotting at 500)
P_END       = N_POS        # 2047 (no data at 2048)

# ── imports ─────────────────────────────────────────────────────────────────
import random, numpy as np, h5py, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt, seaborn as sns, matplotlib as mpl

# ───────────────────────────── helpers ──────────────────────────────────────
def _occ_filter_mask(pos_indices: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a boolean mask over occurrences based on OCCURRENCE_FILTER."""
    if mode == "all":
        return torch.ones_like(pos_indices, dtype=torch.bool)
    elif mode == "first_only":
        return (pos_indices == 0)
    elif mode == "nonfirst_only":
        return (pos_indices > 0)
    else:
        raise ValueError("OCCURRENCE_FILTER must be one of {'all','first_only','nonfirst_only'}")

# ─────────────────────── Step 1: find occurrences & lengths ────────────────
ds  = load_dataset("pietrolesci/pile-validation", split="validation",
                   verification_mode="no_checks")
tok = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID, revision=TOKENIZER_REVISION)

# deterministically shuffle docs
doc_ids = list(range(len(ds)))
random.Random(DOC_SEED).shuffle(doc_ids)
doc_ids = doc_ids[:N_DOCS]

occ_positions_per_doc = []   # numpy arrays of t-positions per doc (after OCCURRENCE_FILTER)
occ_counts_per_doc    = []   # counts per doc (after OCCURRENCE_FILTER)
length_per_doc        = []   # tokenizer tokenized lengths (per doc)

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
    ids = enc.input_ids[0]              # 1D LongTensor (length L)
    L   = int(ids.numel())
    length_per_doc.append(L)
    total_tokens_scanned += L

    # all positions where token == TOKEN_ID
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

# Optional left-neighbor (doc d+1) length-based pruning — summary only
occurrences_left_ok = 0
if REQUIRE_LEFT_NEIGHBOR_SHORT:
    # d ranges up to len(doc_ids) - 2; neighbor is d+1
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

# palette: neighbours get similar hues
N_models = len(MODELS)
cmap     = mpl.cm.get_cmap("RdYlBu", N_models * 2)
colours  = [cmap(i * 2) for i in range(N_models)]

global_min = +np.inf
global_max = -np.inf

for idx, (name, h5_path) in enumerate(MODELS):
    sum_p   = np.zeros(N_POS, dtype=np.float64)   # Σ NLL at each position
    count_p = np.zeros(N_POS, dtype=np.int64)     # # of occurrences contributing
    occ_used_total = 0

    with h5py.File(h5_path, "r") as f:
        nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
        M_docs = len(ptr_ds) - 1

        # cap to docs we actually scanned (keeps index alignment)
        D_use = min(M_docs, len(occ_positions_per_doc))

        for d in tqdm(range(D_use), desc=f"{name:>12}"):
            # left-neighbor requirement (based on tokenizer lengths)
            if REQUIRE_LEFT_NEIGHBOR_SHORT:
                # must have a neighbor at d+1 and its length must be ≤ threshold
                if (d + 1) >= len(length_per_doc) or length_per_doc[d + 1] > LEFT_NEIGHBOR_MAXLEN:
                    continue

            s = int(ptr_ds[d])
            e = int(ptr_ds[d + 1])
            L = e - s

            # mirror earlier figures if desired (applies to current doc)
            if MAX_DOC_LEN and L > MAX_DOC_LEN:
                continue

            occs = occ_positions_per_doc[d]
            if occs.size == 0:
                continue

            # guard in case tokenizer length != merged length (shouldn’t happen, but safe)
            occs = occs[occs < L]
            if occs.size == 0:
                continue

            # read this doc’s NLL block once; then take rows for our occurrences
            # nll_doc: (L, 2047) float16 → cast to float32 for accumulation
            nll_doc = nll_ds[s:e, :].astype(np.float32)   # load contiguous stripe
            rows    = nll_doc[occs, :]                    # (n_occ, 2047)

            # mask finite
            m = np.isfinite(rows)
            sum_p   += np.where(m, rows, 0.0).sum(axis=0)
            count_p += m.sum(axis=0).astype(np.int64)
            occ_used_total += int(occs.size)

    # mean NLL of token-50 per position (over the chosen occurrences)
    mean_p = np.where(count_p > 0, sum_p / count_p, np.nan)

    # align at P_ALIGN
    baseline = mean_p[P_ALIGN - 1] if 1 <= P_ALIGN <= N_POS else np.nan
    shifted  = mean_p - baseline
    if np.isfinite(baseline):
        shifted[P_ALIGN - 1] = 0.0

    # restrict to visible range and update global min/max
    x = np.arange(P_START, P_END + 1)                # e.g., 500..2047
    y = shifted[P_START - 1 : P_END]                 # same length as x
    vfin = y[np.isfinite(y)]
    if vfin.size:
        global_min = min(global_min, float(np.min(vfin)))
        global_max = max(global_max, float(np.max(vfin)))

    # plot curve
    left_tag = f", left-neigh≤{LEFT_NEIGHBOR_MAXLEN}" if REQUIRE_LEFT_NEIGHBOR_SHORT else ""
    lbl = (f"{name}  (baseline={baseline:.3f}, occ_used={occ_used_total:,}{left_tag})"
           if np.isfinite(baseline) else f"{name}")
    ax.plot(x, y, color=colours[idx], linewidth=2.0, label=lbl)

# ─────────────── axis formatting & legend (no left gap; stop at 2047) ──────
ax.set_xlim(P_START, P_END)
ticks_all = [1, 256, 500, 768, 1024, 1280, 1536, 1792, 2047]
ax.set_xticks([t for t in ticks_all if P_START <= t <= P_END])
ax.set_xlabel("Token position within 2048-token context window")

# y-axis: include 0 and auto span
if not np.isfinite(global_min) or not np.isfinite(global_max):
    global_min, global_max = 0.0, 1.0
span   = max(global_max - global_min, 1e-6)
yticks = np.round(np.linspace(global_min, global_max, 10), 3)
if 0.0 not in yticks:
    yticks = np.sort(np.append(yticks, 0.0))
ax.set_ylim(global_min - 0.02 * span, global_max + 0.02 * span)
ax.set_yticks(yticks)
ax.set_ylabel(f"ΔNLL for token ID {TOKEN_ID} (relative to position {P_ALIGN})")

# legend with thicker colour strips
leg = ax.legend(loc="upper left", frameon=True, handlelength=4, borderaxespad=0.4)
for line in leg.get_lines():
    line.set_linewidth(6)

filt_text = {"all": "all occurrences", "first_only": "first-token only", "nonfirst_only": "non-first only"}[OCCURRENCE_FILTER]
left_text = " | left neighbor ≤ 500" if REQUIRE_LEFT_NEIGHBOR_SHORT else ""
ax.set_title(f"Token {TOKEN_ID} ('{decoded_vis}') ΔNLL vs position — {filt_text}{left_text}, aligned at {P_ALIGN}, shown from {P_START}")

plt.tight_layout()
plt.savefig(
    "D:/Sync/Sync/ETH Stuff/Bachelor Thesis/Code/graphs/token50_delta_nll_vs_position_with_left_neighbor_filter_nonfirst.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.show()

#!/usr/bin/env python3
"""
Mean NLL of the end-of-document (EOD) token versus its position P_t
inside the 2048-token context window.

Assumption: every document received exactly one EOD token which is the
*last* token in that document during the “snake” evaluation run.  Hence
row L-1 of each per-doc NLL matrix corresponds to the EOD token.
"""

import h5py, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------- parameters
MERGED_H5  = "D:/NLL_matrices/1.4B_deduped_EOS_merged_2600_docs.h5"   # merged file with EOD rows
FILE_LIMIT = 5_000                                  # how many docs to scan
T_MAX      = 2047                                   # context length − 1

# ---------------------------------------------------------------- buffers
sum_t   = np.zeros(T_MAX, dtype=np.float64)         # accumulate ∑NLL per P_t
count_t = np.zeros_like(sum_t, dtype=np.int64)      # how many docs per P_t

# ---------------------------------------------------------------- pass through docs
with h5py.File(MERGED_H5, "r") as f:
    nll_ds, ptr_ds = f["nll"], f["doc_ptr"]
    N_docs = min(FILE_LIMIT, len(ptr_ds) - 1)

    for d in tqdm(range(N_docs), desc="scanning docs"):
        s, e      = ptr_ds[d], ptr_ds[d + 1]
        mat_eod   = nll_ds[e - 1, :]                 # row L-1  (shape: 2047,)

        # mat_eod already float16; convert once to float64 for safe accumulation
        nll_row   = mat_eod.astype(np.float64, copy=False)

        # accumulate across docs; no need to check NaN if matrices are dense
        sum_t   += nll_row
        count_t += 1

# ---------------------------------------------------------------- mean NLL per P_t
mean_t = sum_t / count_t          # count_t is scalar = #docs, so no div-by-zero

# ---------------------------------------------------------------- plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 4))
plt.plot(np.arange(1, T_MAX + 1), mean_t, linewidth=1.2)

plt.xlim(1, T_MAX)
plt.ylim(0, 24)
plt.xlabel("P_t  (token position in 2048-token context window)")
plt.ylabel("Mean NLL for EOD token (nats)")
plt.title(f"EOD-token NLL vs P_t   •   {N_docs:,} documents")
plt.tight_layout()
plt.show()

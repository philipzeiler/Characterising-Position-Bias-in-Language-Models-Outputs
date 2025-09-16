#!/usr/bin/env python3
"""
Count total occurrences of one or more token IDs across the first N_DOCS
shuffled Pile-validation documents (shuffle seed = DOC_SEED).

Hardcode TOKEN_IDS below, e.g. [50] or [50, 11, 198].
"""

# ── settings (same defaults you used before) ────────────────────────────────
TOKEN_IDS      = [50, 27, 187, 187]                       # ← put your token IDs here
N_DOCS         = 5_00000
DOC_SEED       = 753
MODEL_ID       = "EleutherAI/pythia-1.4B"
REVISION       = "step143000"
INCLUDE_SPECIAL = False                     # keep False to match earlier runs

# ── imports ─────────────────────────────────────────────────────────────────
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch

# ── dataset & tokenizer ─────────────────────────────────────────────────────
ds  = load_dataset("pietrolesci/pile-validation", split="validation",
                   verification_mode="no_checks")
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

# (Optional) sanity check for vocab size
vocab_size = getattr(tok, "vocab_size", None)
if vocab_size is not None:
    bad = [tid for tid in TOKEN_IDS if tid < 0 or tid >= vocab_size]
    if bad:
        raise ValueError(f"Token IDs out of range for this tokenizer: {bad} (vocab_size={vocab_size})")

# deterministically shuffle docs
doc_ids = list(range(len(ds)))
random.Random(DOC_SEED).shuffle(doc_ids)
doc_ids = doc_ids[:N_DOCS]

# ── counting ────────────────────────────────────────────────────────────────
counts = {tid: 0 for tid in TOKEN_IDS}
total_tokens = 0

# tensor of target IDs for vectorized comparison
target_ids_tensor = torch.tensor(TOKEN_IDS)

for doc_id in tqdm(doc_ids, desc="Counting"):
    enc = tok(
        ds[doc_id]["text"],
        return_tensors="pt",
        add_special_tokens=INCLUDE_SPECIAL
    )
    ids = enc.input_ids[0]                        # 1D tensor of token IDs
    total_tokens += int(ids.numel())

    # vectorized: compare all tokens against all target IDs at once
    # shape: (len(ids), len(TOKEN_IDS)) → sum over rows to get per-ID counts
    matches_per_id = ids.view(-1, 1).eq(target_ids_tensor).sum(dim=0).tolist()
    for tid, add in zip(TOKEN_IDS, matches_per_id):
        counts[tid] += int(add)

# ── pretty print ────────────────────────────────────────────────────────────
print("\n─ Results ───────────────────────────────────────────")
print(f"Model / rev: {MODEL_ID} @ {REVISION}")
print(f"Docs scanned: {len(doc_ids):,} (seed={DOC_SEED})")
print(f"Total tokens scanned: {total_tokens:,}")
print("------------------------------------------------------")
print(f"{'token_id':>9}   {'decoded':<20}   {'count':>12}   {'freq':>12}")
print("-" * 60)

for tid in TOKEN_IDS:
    decoded = tok.decode([tid], clean_up_tokenization_spaces=False, skip_special_tokens=False)
    # make control chars visible
    decoded_vis = (decoded.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
    cnt = counts[tid]
    freq = (cnt / total_tokens) if total_tokens else 0.0
    print(f"{tid:9d}   {decoded_vis:<20}   {cnt:12,d}   {freq:12.8f}")

print("──────────────────────────────────────────────────────\n")

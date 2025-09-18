#!/usr/bin/env python3
"""
Count the 100 most common *n-token* prefixes (n-grams) at the start of the
first 5 000 shuffled Pile-validation documents (shuffle seed = 753).

Set N_GRAM below to 1, 2, 3 … to switch between unigrams, bigrams, trigrams, …
"""

# ── settings ────────────────────────────────────────────────────────────────
N_GRAM   = 4               # ← change me: 1=single tokens, 2=bigrams, 3=trigrams …
N_DOCS   = 5_00000
DOC_SEED = 753
MODEL_ID = "EleutherAI/pythia-1.4B"
REVISION = "step143000"

# ── imports ────────────────────────────────────────────────────────────────
from collections import Counter
import random
from datasets import load_dataset
from transformers import AutoTokenizer

# ── dataset & tokenizer (same as NLL run) ───────────────────────────────────
ds   = load_dataset("pietrolesci/pile-validation", split="validation",
                    verification_mode="no_checks")
tok  = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

doc_ids = list(range(len(ds)))
random.Random(DOC_SEED).shuffle(doc_ids)

# ── count n-token prefixes ─────────────────────────────────────────────────
counter = Counter()

for doc_id in doc_ids[:N_DOCS]:
    ids = tok(ds[doc_id]["text"], return_tensors="pt").input_ids[0]
    if len(ids) < N_GRAM:        # skip ultra-short docs
        continue
    prefix = tuple(ids[:N_GRAM].tolist())   # immutable & hashable
    counter[prefix] += 1

# ── print top-100 table ────────────────────────────────────────────────────
top = counter.most_common(100)  # [CHANGED] compute once so we can size the new column

# [CHANGED] build pretty token-ids strings and compute a good column width
ids_strs = {
    ids_tpl: "[" + ", ".join(map(str, ids_tpl)) + "]"
    for ids_tpl, _ in top
}
ids_width = max((len(s) for s in ids_strs.values()), default=8)  # robust fallback

print(f"\nTop-100 {N_GRAM}-token prefixes in {N_DOCS:,} docs:\n")
# [CHANGED] header now includes a token_ids column with dynamic width
print(f"{'rank':>4} {'count':>7}   {'token_ids':<{ids_width}}   decoded sequence")
print("-" * (4 + 1 + 7 + 3 + ids_width + 3 + 40))  # rough separator length

for rank, (ids_tpl, cnt) in enumerate(top, 1):
    decoded = tok.decode(list(ids_tpl),
                         clean_up_tokenization_spaces=False,
                         skip_special_tokens=False)  # matches your original intent
    visible = (decoded
               .replace("\n", "\\n")
               .replace("\r", "\\r")
               .replace("\t", "\\t"))
    ids_str = ids_strs[ids_tpl]  # [CHANGED] reuse the precomputed string
    # [CHANGED] print the aligned token_ids column before the decoded text
    print(f"{rank:4d} {cnt:7d}   {ids_str:<{ids_width}}   {visible}")

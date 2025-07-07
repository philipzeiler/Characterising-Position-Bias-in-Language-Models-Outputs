#!/usr/bin/env python3
"""
Count the 100 most common *n-token* prefixes (n-grams) at the start of the
first 5 000 shuffled Pile-validation documents (shuffle seed = 753).

Set N_GRAM below to 1, 2, 3 … to switch between unigrams, bigrams, trigrams, …
"""

# ── settings ────────────────────────────────────────────────────────────────
N_GRAM   = 2               # ← change me: 1=single tokens, 2=bigrams, 3=trigrams …
N_DOCS   = 5_000
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
print(f"\nTop-100 {N_GRAM}-token prefixes in {N_DOCS:,} docs:\n")
print(f"{'rank':>4} {'count':>7}   decoded sequence")
print("-"*60)

for rank, (ids_tpl, cnt) in enumerate(counter.most_common(100), 1):
    decoded = tok.decode(list(ids_tpl),
                         clean_up_tokenization_spaces=False,
                         skip_special_tokens=False)
    visible = (decoded
               .replace("\n", "\\n")
               .replace("\r", "\\r")
               .replace("\t", "\\t"))
    print(f"{rank:4d} {cnt:7d}   {visible}")

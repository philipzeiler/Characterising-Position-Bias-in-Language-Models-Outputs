#!/usr/bin/env python3
# ---------------------------------------------------------------
# Stats for the Pile-validation set (PyTorch tokenizer version)
# • Computes min / max / mean / median document length
# • Counts docs longer than the 2 048-token context window
# • NEW: counts how many *tokens* live in docs shorter than 2 048
# ---------------------------------------------------------------
import numpy as np, torch, datasets, tqdm
from transformers import AutoTokenizer

VAL_DS   = "pietrolesci/pile-validation"
MODEL_ID = "EleutherAI/pythia-1.4b"

tok = AutoTokenizer.from_pretrained(MODEL_ID)                 # HF tokenizer :contentReference[oaicite:1]{index=1}
val = datasets.load_dataset(VAL_DS, split="validation",
                            verification_mode="no_checks")    # HF datasets :contentReference[oaicite:2]{index=2}

lengths = []
for doc in tqdm.tqdm(val, desc="tokenising"):
    n_tokens = len(tok(doc["text"]).input_ids)  # length of each document
    lengths.append(n_tokens)

lengths = np.asarray(lengths)
above_ctx     = np.sum(lengths > 2048)                       # doc count
total_tokens  = lengths.sum(dtype=np.int64)                  # all tokens   :contentReference[oaicite:3]{index=3}
short_tokens  = lengths[lengths < 2048].sum(dtype=np.int64)  # tokens in short docs

print(f"Documents analysed           : {len(lengths):,}")
print(f"Min tokens per doc           : {lengths.min():,}")
print(f"Max tokens per doc           : {lengths.max():,}")
print(f"Mean tokens per doc          : {lengths.mean():,.2f}")
print(f"Median tokens per doc        : {np.median(lengths):,}")
print(f"Docs > 2048 tokens           : {above_ctx:,}")
print(f"Total tokens in split        : {total_tokens:,}")
print(f"Tokens in docs < 2048 tokens : {short_tokens:,} "
      f"({100*short_tokens/total_tokens:.2f} % of total)")

#!/usr/bin/env python3
# ------------------------------------------------------------
# What % of the Pile (validation) has length ≤ 500 tokens
# (tokenized with a Pythia tokenizer; no special tokens)
# ------------------------------------------------------------
DS_ID        = "pietrolesci/pile-validation"
MODEL_ID     = "EleutherAI/pythia-1.4B"   # any Pythia tokenizer is fine
MAX_DOCS     = None                       # set e.g. 50_000 to sample
THRESHOLD    = 500                        # tokens
BATCH_SIZE   = 1024                       # tokenizer batch size
USE_FAST     = True                       # HF fast tokenizer
N_PROCS_WIN  = 1                          # HF recommends 1 on Windows

import os, sys
from multiprocessing import freeze_support
import numpy as np
import datasets
from transformers import AutoTokenizer

def collect_lengths(ds, tok, batch_size=BATCH_SIZE):
    """Tokenize texts and attach a 'length' column (token count)."""
    def _fn(batch):
        ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
        return {"length": [len(x) for x in ids]}

    # Multiprocessing: on Windows stick to 1 process to avoid fork issues
    num_proc = N_PROCS_WIN if sys.platform.startswith("win") else os.cpu_count()

    return ds.map(
        _fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=[c for c in ds.column_names if c != "text"],
        num_proc=num_proc,
        desc="Tokenizing to measure lengths"
    )

def main():
    print("[INFO] Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=USE_FAST)
    # NOTE: We intentionally do NOT add special tokens for length stats.
    # (Special tokens would inflate counts and not match your evaluation setup.)

    print("[INFO] Loading dataset …")
    ds = datasets.load_dataset(DS_ID, split="validation", verification_mode="no_checks")
    if MAX_DOCS:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))

    ds = collect_lengths(ds, tok)
    lengths = np.asarray(ds["length"], dtype=np.int32)

    total = lengths.size
    le_thr = int((lengths <= THRESHOLD).sum())
    pct    = 100.0 * le_thr / total if total > 0 else 0.0

    print("\n── Results ─────────────────────────────────────────────")
    print(f"Tokenizer: {MODEL_ID}")
    print(f"Dataset  : {DS_ID} (validation)")
    print(f"Docs     : {total:,}")
    print(f"≤ {THRESHOLD:} tokens: {le_thr:,}  ({pct:.4f} %)")
    print("────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    freeze_support()  # required for Windows multiprocessing
    main()

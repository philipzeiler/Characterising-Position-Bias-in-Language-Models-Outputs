#!/usr/bin/env python3
# GPT-2 WebText test set length summary
# - Loads local webtext.test.jsonl
# - Tokenizes with GPT-2 tokenizer (no extra specials)
# - Prints mean/median and % <= {200,250,300,350,400,450,500}

import os, sys
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
WEBTEXT_TEST = r"D:/datasets/webtext.test.jsonl"   # adjust if needed
MODEL_ID     = "gpt2"                               # tokenizer only
THRESHOLDS   = [200, 250, 300, 350, 400, 450, 500]
MAX_DOCS     = None                                 # None → all

# ── Load data & tokenizer ───────────────────────────────────────────────────
if not os.path.exists(WEBTEXT_TEST):
    raise FileNotFoundError(f"Missing WebText test set at: {WEBTEXT_TEST}")

print("[INFO] loading GPT-2 tokenizer …")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
# We want raw lengths of the documents themselves, so do NOT add special tokens.
# (GPT-2 used <|endoftext|> as a delimiter; we’re measuring pre-delimiter length.)

print("[INFO] loading WebText test JSONL …")
ds = load_dataset("json", data_files={"test": WEBTEXT_TEST}, split="test")
if MAX_DOCS:
    ds = ds.select(range(MAX_DOCS))

# ── Compute tokenized lengths ────────────────────────────────────────────────
def _encode_len(batch):
    ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
    return {"length": [len(x) for x in ids]}

# On Windows, prefer single process to avoid spawn issues; elsewhere you can bump this.
num_proc = 1

ds = ds.map(
    _encode_len, batched=True, batch_size=1024, num_proc=num_proc,
    remove_columns=[c for c in ds.column_names if c != "text"],
    desc="Tokenizing (no specials)"
)

lengths = np.asarray(ds["length"], dtype=np.int32)
N = int(lengths.size)

# ── Stats ───────────────────────────────────────────────────────────────────
mean_len   = float(lengths.mean()) if N else float("nan")
median_len = float(np.median(lengths)) if N else float("nan")

print("\n── WebText test set length summary (GPT-2 tokenizer, no specials) ──")
print(f"Documents: {N:,}")
print(f"Mean length   : {mean_len:,.2f} tokens")
print(f"Median length : {median_len:,.0f} tokens")

for thr in THRESHOLDS:
    pct = (lengths <= thr).mean() * 100.0 if N else 0.0
    cnt = int((lengths <= thr).sum())
    print(f"≤ {thr:>4} tokens : {cnt:>7,} docs  ({pct:6.2f} %)")

# Optional: also show max/min to sanity-check distribution tails
print(f"Min / Max     : {lengths.min():,} / {lengths.max():,}")

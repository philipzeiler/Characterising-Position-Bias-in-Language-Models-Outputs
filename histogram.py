#!/usr/bin/env python3
# ------------------------------------------------------------
# Windows-safe histogram of doc lengths for *The Pile*
#   – 1 500 bins
#   – log₂ x-axis (1 … longest document) with decimal tick labels
#   – linear y-axis (0 … tallest bucket)
#   – thin red guide at 2 048 tokens
#   – thin guides at Q1, median, mean, Q3
#   – small, angled x-tick labels to avoid overlap
# ------------------------------------------------------------
MAX_DOCS   = None          # None → full set
N_BINS     = 1_500
SAVE_PATH  = "pile_val_hist.png"
OPEN_AFTER = True

import os, sys, subprocess
from pathlib import Path
import numpy as np
import datasets
from transformers import AutoTokenizer
import seaborn as sns, matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from multiprocessing import freeze_support

DS_ID, MODEL_ID = "pietrolesci/pile-validation", "EleutherAI/pythia-1.4b"

# ─────────────────────────────────────────────────────────────
def collect_lengths(ds, tok):
    """Tokenise docs and attach a 'length' column (token count)."""
    def _fn(batch):
        ids = tok(batch["text"], add_special_tokens=False)["input_ids"]
        return {"length": [len(i) for i in ids]}

    num_proc = 1 if sys.platform.startswith("win") else os.cpu_count()
    return ds.map(
        _fn, batched=True, batch_size=1_024,
        remove_columns=["text"], num_proc=num_proc,
        desc="tokenising"
    )

# ─────────────────────────────────────────────────────────────
def main():
    print("[INFO] loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    print("[INFO] loading dataset …")
    ds = datasets.load_dataset(
        DS_ID, split="validation", verification_mode="no_checks"
    )
    if MAX_DOCS:
        ds = ds.select(range(MAX_DOCS))

    ds = collect_lengths(ds, tok)
    lengths = np.asarray(ds["length"], dtype=np.int32)

    # ── statistics ──────────────────────────────────────────
    q1     = np.percentile(lengths, 25)
    median = np.median(lengths)
    mean   = lengths.mean()
    q3     = np.percentile(lengths, 75)

    print(f"\nDocs analysed : {len(lengths):,}")
    print(
        f"Min / Q1 / Median / Mean / Q3 / Max : "
        f"{lengths.min():,} / {q1:,.0f} / {median:,.0f} / "
        f"{mean:,.2f} / {q3:,.0f} / {lengths.max():,}"
    )
    print(f"Docs > 2 048 tokens : {(lengths > 2_048).sum():,}")

    # ── histogram ───────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    ax = sns.histplot(
        lengths,
        bins=N_BINS,
        log_scale=(2, False),          # base-2 x, linear y
    )

    # x-axis: log₂ scale, decimal tick labels
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, lengths.max())

    powers    = np.arange(0, int(np.log2(lengths.max())) + 1)
    tick_vals = 2 ** powers
    ax.set_xticks(tick_vals)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Make tick labels small & angled
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # y-axis: 0 … tallest bar
    tallest = max(p.get_height() for p in ax.patches)
    ax.set_ylim(0, tallest)

    # ── vertical guides ────────────────────────────────────
    ax.axvline(2_048, color="red",   linewidth=1, label="2048 tokens")

    ax.axvline(q1,     color="grey", linestyle="--", linewidth=1,
               label="25 % quartile")
    ax.axvline(median, color="green", linestyle="-.", linewidth=1,
               label="Median")
    ax.axvline(mean,   color="blue", linestyle="--", linewidth=1,
               label="Mean")
    ax.axvline(q3,     color="grey", linestyle="--", linewidth=1,
               label="75 % quartile")

    ax.set(
        xlabel="Document length (tokens, log2 spacing)",
        ylabel="Number of documents",
        title="Document length distribution - The Pile (validation)",
    )
    ax.legend()
    plt.tight_layout()

    # ── save + open ─────────────────────────────────────────
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=300)
    plt.show(block=False); plt.pause(2)

    if OPEN_AFTER:
        if sys.platform.startswith("win"):
            os.startfile(SAVE_PATH)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, SAVE_PATH], check=False)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    freeze_support()     # required for Windows multiprocessing
    main()

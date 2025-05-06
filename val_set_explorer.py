# stats_pile_val_lengths.py
# ---------------------------------------------------------------
# • Computes min / max / mean / median token count per document
# • Tokeniser: EleutherAI/pythia‑1.4b‑deduped (GPT‑NeoX)
# ---------------------------------------------------------------
import numpy as np, torch, datasets, tqdm
from transformers import AutoTokenizer

VAL_DS   = "pietrolesci/pile-validation"
MODEL_ID = "EleutherAI/pythia-1.4b"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
val = datasets.load_dataset(VAL_DS, split="validation",
                            verification_mode="no_checks")

lengths = []
for doc in tqdm.tqdm(val, desc="tokenising"):
    n_tokens = len(tok(doc["text"]).input_ids)
    lengths.append(n_tokens)

lengths = np.asarray(lengths)
above_ctx = np.sum(lengths > 2048) 

print(f"Documents analysed : {len(lengths):,}")
print(f"Min tokens         : {lengths.min():,}")
print(f"Max tokens         : {lengths.max():,}")
print(f"Mean tokens        : {lengths.mean():,.2f}")
print(f"Median tokens      : {np.median(lengths):,}")
print(f"Docs > 2048 tokens : {above_ctx:,}")

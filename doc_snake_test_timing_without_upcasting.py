#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Variable-length “snake” NLL evaluation  +  per-segment timing
# ─────────────────────────────────────────────────────────────────────────────
import os, random, itertools, time, numpy as np, torch, h5py, torch.nn.functional as F
from collections import deque, defaultdict
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer

# ── Timing helper ───────────────────────────────────────────────────────────
class Timer:
    """Context-manager that adds elapsed seconds to timers[name]."""
    def __init__(self, name, timers):
        self.name = name; self.timers = timers
    def __enter__(self):
        self.t0 = time.perf_counter()
    def __exit__(self, *_):
        self.timers[self.name] += time.perf_counter() - self.t0

timers = defaultdict(float)                     # section → cumulative seconds

# ── 0. Determinism flags ────────────────────────────────────────────────────
with Timer("00-setup", timers):
    seed, doc_seed = 42, 420
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# ── 1. Hyper-parameters ─────────────────────────────────────────────────────
CTX   = 2048
BATCH = 7
MODEL_ID = "EleutherAI/pythia-1.4b"
REVISION  = "step143000"
n_docs    = 5

# ── 2. Model & tokenizer ────────────────────────────────────────────────────
with Timer("01-model-load", timers):
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeoXForCausalLM.from_pretrained(
                MODEL_ID, revision=REVISION, torch_dtype=torch.float16
            ).to(dev).eval()
    tok   = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
print(f"[INFO] model on {dev} – dtype {next(model.parameters()).dtype}")

# ── 3. Dataset & shuffled doc order ─────────────────────────────────────────
with Timer("02-dataset-load", timers):
    val_ds = load_dataset("pietrolesci/pile-validation",
                          split="validation",
                          verification_mode="no_checks")
    doc_order = list(range(len(val_ds)))
    random.Random(doc_seed).shuffle(doc_order)
print("[DEBUG] first 100 shuffled doc ids:", doc_order[:100])

# ── 4. Helper: FP16 batched NLL ─────────────────────────────────────────────
def nll_for(batch_ids: torch.Tensor) -> torch.Tensor:
    """Returns [B, CTX-1] FP16 CPU tensor of per-token NLLs."""
    with Timer("04-model-forward", timers):
        with torch.inference_mode():
            logits = model(batch_ids[:, :-1]).logits       # [B,T,V] FP16
            lp = torch.log_softmax(logits, -1).permute(0, 2, 1)
            tgt = batch_ids[:, 1:]
            nll = F.nll_loss(lp, tgt, reduction="none").cpu()
    return nll

# ── 5. Snake initialisation ─────────────────────────────────────────────────
with Timer("03-snake-init", timers):
    eos_id      = tok.eos_token_id
    snake_ids   = deque([eos_id] * (CTX - 1))
    snake_meta  = deque([None]  * (CTX - 1))
    active_docs = {}
    docs_entered = docs_done = shift = 0
    os.makedirs("nll_matrices_timetest", exist_ok=True)

# ── 6. Main loop ────────────────────────────────────────────────────────────
while True:

    with Timer("05A-snake-fill", timers):
        while len(snake_ids) < CTX + BATCH - 1:
            if docs_entered < n_docs:
                doc_id = doc_order[docs_entered]; docs_entered += 1
                tokens = tok(val_ds[doc_id]["text"],
                              return_tensors="pt").input_ids[0]
                L = len(tokens)
                active_docs[doc_id] = {
                    "matrix": np.full((L, CTX-1), np.nan, np.float16)
                }
                snake_ids.extendleft(int(t) for t in reversed(tokens))
                snake_meta.extendleft((doc_id, idx)
                                       for idx in reversed(range(L)))
                print(f"[INFO] entered doc {doc_id} (len={L}) "
                      f"docs_entered={docs_entered}")
            else:
                snake_ids.appendleft(eos_id)
                snake_meta.appendleft(None)

    with Timer("05B-build-windows", timers):
        windows_ids, windows_meta = [], []
        for off in range(BATCH):
            left  = len(snake_ids) - CTX - off
            right = len(snake_ids) - off
            windows_ids.append(list(itertools.islice(snake_ids, left, right)))
            windows_meta.append(list(itertools.islice(snake_meta, left, right)))
        batch_tensor = torch.as_tensor(windows_ids,
                                       dtype=torch.long,
                                       device=dev)

    nll_batch = nll_for(batch_tensor).numpy()          # [B,CTX-1]

    with Timer("05C-scatter", timers):
        for b in range(BATCH):
            meta_slice = windows_meta[b]
            for k in range(1, CTX):
                meta = meta_slice[k]
                if meta is None:
                    continue
                doc_id, tok_idx = meta
                active_docs[doc_id]["matrix"][tok_idx, k-1] = nll_batch[b, k-1]

    with Timer("05D-slide", timers):
        for _ in range(BATCH):
            snake_ids.pop(); snake_meta.pop()

    with Timer("05E-save", timers):
        current_ids = {m[0] for m in snake_meta if m is not None}
        finished = [d for d in active_docs if d not in current_ids]
        for d in finished:
            out_path = f"nll_matrices_timetest/doc{d}_fp16.h5"
            with h5py.File(out_path, "w") as f:
                f.create_dataset("nll",
                                 data=active_docs[d]["matrix"],
                                 compression="gzip")
            print(f"[INFO] saved doc {d} → {out_path}")
            del active_docs[d]; docs_done += 1

    # logging
    if shift % 300 == 0:
        print(f"[DBG] shift={shift*BATCH:7d}  snake_len={len(snake_ids):5d}  "
              f"open_docs={len(active_docs)}")

    if docs_done == n_docs and all(m is None for m in snake_meta):
        break

    shift += 1

# ── 7. Report timings ───────────────────────────────────────────────────────
print("\n=== Time spent per segment ===")
width = max(len(k) for k in timers)
for name, t in sorted(timers.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{name.ljust(width)} : {t:10.3f} s")

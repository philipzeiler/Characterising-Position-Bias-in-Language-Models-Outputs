# ─────────────────────────────────────────────────────────────────────────────
#  Pythia-1.4B log-likelihood demo — FP16 weights (as trained) + FP32 logits
# ─────────────────────────────────────────────────────────────────────────────
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch, os, random, numpy as np
import torch.nn.functional as F

# ─── 0. Determinism & seeds ─────────────────────────────────────────────────
seed = 42
torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False      # keep exact FP16 maths
torch.backends.cudnn.allow_tf32       = False

#Define checkpoints
model_name = "EleutherAI/pythia-1.4b"
final_snapshot   = "step143000"
#cache_dir  = "./PLACEHOLDER" #TODO: Define where to load model (for cluster)

# ─── 2. Load *FP16* weights ──────────────────────────────
model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=final_snapshot,
#            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=final_snapshot,
#                cache_dir=cache_dir
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)                                 # move params/buffers to GPU
print("CUDA available? ", torch.cuda.is_available())
print("Model device:  ", next(model.parameters()).device)
if torch.cuda.is_available():
    print("GPU name:    ", torch.cuda.get_device_name(0))

# ─── 3. Tokenise ----------------------------------------------------------------
sentence = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
    "Quisque faucibus ex sapien vitae pellentesque sem placerat. "
    "In id cursus mi pretium tellus duis convallis. "
    "Tempus leo eu aenean sed diam urna tempor. "
    "Pulvinar vivamus fringilla lacus nec metus bibendum egestas. "
    "Iaculis massa nisl malesuada lacinia integer nunc posuere. "
    "Ut hendrerit semper vel class aptent taciti sociosqu. "
    "Ad litora torquent per conubia nostra inceptos himenaeos."
)
inputs = tokenizer(sentence, return_tensors="pt").to(device)

# ─── 4. Forward pass (no gradients) -------------------------------------------
with torch.inference_mode():
    logits = model(**inputs).logits.float()

log_probs = F.log_softmax(logits, dim=-1)

# ─── 5. Per-token log-probabilities -------------------------------------------
ids = inputs["input_ids"][0]
for idx, tok_id in enumerate(ids):
    tok_str = tokenizer.decode(tok_id)
    tgt_id  = ids[idx + 1] if idx < len(ids) - 1 else tokenizer.eos_token_id
    lp      = log_probs[0, idx, tgt_id].item()
    print(f"Token: |{tok_str}|  log p_next = {lp:+.7f}")
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch, os, random, numpy as np
import torch.nn.functional as F

#Remove randomness for reproducibility
seed = 42
torch.manual_seed(seed);         random.seed(seed);       np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False

#Define checkpoints
model_name = "EleutherAI/pythia-1.4b"
final_snapshot   = "step143000"
cache_dir  = "./PLACEHOLDER" #TODO: Define where to load model (for cluster)

#Load weights in FULL FP-32
model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=final_snapshot,
#            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            device_map=None            #run entire model on single device -- maybe disable for cluster??
)
tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=final_snapshot,
                cache_dir=cache_dir
)

#Feedback about GPU being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("CUDA available?   ", torch.cuda.is_available())
print("Model device:     ", next(model.parameters()).device)
if torch.cuda.is_available():
    print("GPU name:        ", torch.cuda.get_device_name(0))

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

#Forward pass (no gradients)
with torch.inference_mode():
    logits = model(**inputs).logits.float()        #cast activations to FP-32

log_probs = F.log_softmax(logits, dim=-1)

#Per-token log-likelihood printout
input_ids = inputs["input_ids"]
for i in range(input_ids.size(1)):
    tok_id      = input_ids[0, i]
    tok_str     = tokenizer.decode(tok_id)
    next_tok_id = input_ids[0, i + 1] if i < input_ids.size(1) - 1 else tokenizer.eos_token_id
    lp          = log_probs[0, i, next_tok_id].item()
    print(f"Token: |{tok_str}|  log p_next = {lp:+.6f}")
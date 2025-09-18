import os
from datasets import load_dataset

os.environ["DATA_DIR"] = "D:/dolma"
print(dataset = load_dataset("allenai/dolma", split=None,trust_remote_code=True))

#dataset = load_dataset("allenai/dolma", split="poopie",trust_remote_code=True)
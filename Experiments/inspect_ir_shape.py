import numpy as np
import torch
from tabpfn import TabPFNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

X_pool = np.random.randn(500, 2).astype(np.float32)
y_dummy = np.zeros(500)
y_dummy[:250] = 1

clf = TabPFNClassifier(device=device)
clf.fit(X_pool, y_dummy)

transformer = clf.model[2]

captured = {}
def hook(module, input):
    captured['ir'] = input[0]

handle = transformer.decoder.register_forward_pre_hook(hook)

X_query = np.random.randn(20, 2).astype(np.float32)
probs = clf.predict_proba(X_query)

handle.remove()

print("captured IR shape:", captured['ir'].shape)
print("captured IR dtype:", captured['ir'].dtype)
print("probs shape:", probs.shape)
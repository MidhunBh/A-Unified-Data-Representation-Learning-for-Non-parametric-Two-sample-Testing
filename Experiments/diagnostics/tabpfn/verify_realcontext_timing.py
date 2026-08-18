import numpy as np
import torch
import time
from tabpfn import TabPFNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

CONTEXT_SIZE = 500   # safely under TabPFN v1's 1024 cap

np.random.seed(1102)
X_pool = np.random.randn(10000, 2).astype(np.float32)   # worst case: N=10000 pool

# real subsample for context, NOT the full pool
ctx_idx = np.random.choice(len(X_pool), CONTEXT_SIZE, replace=False)
X_context = X_pool[ctx_idx]
y_placeholder = np.arange(CONTEXT_SIZE) % 2   # parity, safe from leakage

clf = TabPFNClassifier(device=device)

t0 = time.time()
clf.fit(X_context, y_placeholder)
print("fit time (context=500):", time.time() - t0, "s")

transformer = clf.model[2]
captured = {}
def hook(module, input):
    captured['ir'] = input[0].detach()
handle = transformer.decoder.register_forward_pre_hook(hook)

t0 = time.time()
with torch.no_grad():
    _ = clf.predict_proba(X_pool)   # query = FULL pool, not capped
elapsed = time.time() - t0
handle.remove()

ir = captured['ir']
print("IR shape (context+query):", ir.shape)
print("expected: context=500, query=10000, total=10500")
print("predict_proba time at N=10000 query:", elapsed, "s")
print("estimated time for 100 trials at N=10000:", elapsed * 100 / 60, "minutes")
print("estimated time for all 6 N (100 trials each, rough):", elapsed * 100 * 6 / 60, "minutes")
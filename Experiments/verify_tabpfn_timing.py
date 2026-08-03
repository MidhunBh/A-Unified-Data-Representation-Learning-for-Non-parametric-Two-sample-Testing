import numpy as np
import torch
import time
from tabpfn import TabPFNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Fixed dummy context — same every trial, keeps representations label-blind
np.random.seed(42)
X_context = np.random.randn(10, 2).astype(np.float32)
y_context = np.array([0]*5 + [1]*5)

clf = TabPFNClassifier(device=device)
clf.fit(X_context, y_context)

transformer = clf.model[2]
captured = {}
def hook(module, input):
    captured['ir'] = input[0]
handle = transformer.decoder.register_forward_pre_hook(hook)

# Realistic worst case: N=10000 pool at d=2 (n_train=n_test=1250, x4 for both distributions' splits)
X_pool = np.random.randn(10000, 2).astype(np.float32)

t0 = time.time()
_ = clf.predict_proba(X_pool)
elapsed = time.time() - t0
handle.remove()

ir = captured['ir']
print("IR shape:", ir.shape)
print("extraction time at N=10000:", elapsed, "s")
print("estimated time for 100 trials:", elapsed * 100 / 60, "minutes (this N value only)")
print("estimated time for all 6 N values (100 trials each):", elapsed * 100 * 6 / 60, "minutes (rough, smaller N will be faster)")
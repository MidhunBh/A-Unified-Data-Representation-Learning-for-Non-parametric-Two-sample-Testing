import numpy as np
import torch
import time
from tabpfn import TabPFNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

CONTEXT_SIZE = 500

np.random.seed(1102)
# Simulating AE's IR output shape (x_out=30) at worst-case sizes
IR_train_P = np.random.randn(2500, 30).astype(np.float32)
IR_train_Q = np.random.randn(2500, 30).astype(np.float32)
IR_test = np.random.randn(5000, 30).astype(np.float32)

idx_p = np.random.choice(len(IR_train_P), CONTEXT_SIZE // 2, replace=False)
idx_q = np.random.choice(len(IR_train_Q), CONTEXT_SIZE // 2, replace=False)
X_context = np.concatenate([IR_train_P[idx_p], IR_train_Q[idx_q]], axis=0)
y_context = np.concatenate([np.zeros(len(idx_p)), np.ones(len(idx_q))])  # real labels

clf = TabPFNClassifier(device=device)

t0 = time.time()
clf.fit(X_context, y_context)
print("fit time (30-dim IR, context=500, real labels):", time.time() - t0, "s")

t0 = time.time()
probs = clf.predict_proba(IR_test)
elapsed = time.time() - t0

print("predict_proba shape:", probs.shape, "(expect (5000, 2))")
print("predict_proba time (query=5000):", elapsed, "s")
print("estimated time for 100 trials at largest N:", elapsed * 100 / 60, "minutes")
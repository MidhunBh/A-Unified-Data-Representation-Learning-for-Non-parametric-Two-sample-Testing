import numpy as np
import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

from tabpfn import TabPFNClassifier

X_pool = np.random.randn(500, 2).astype(np.float32)
y_dummy = np.zeros(500)
y_dummy[:250] = 1   # two classes required, content doesn't matter for embedding extraction

t0 = time.time()
clf = TabPFNClassifier(device=device)
clf.fit(X_pool, y_dummy)
print("fit time:", time.time() - t0, "s")

print("\nclassifier attributes:", [a for a in dir(clf) if not a.startswith('_')])

# v1 loads a plain nn.Module checkpoint — find the actual model object
for attr in ["model", "network"]:
    if hasattr(clf, attr):
        obj = getattr(clf, attr)
        print(f"\n{attr}: {type(obj)}")
        if isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                print(f"  [{i}]: {type(item)}")
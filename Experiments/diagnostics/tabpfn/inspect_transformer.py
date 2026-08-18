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
print("=== TransformerModel structure ===")
print(transformer)

print("\n=== Named children (top-level modules) ===")
for name, module in transformer.named_children():
    print(f"  {name}: {type(module).__name__}")

print("\n=== Parameter count ===")
total = sum(p.numel() for p in transformer.parameters())
print(f"  {total:,} parameters")
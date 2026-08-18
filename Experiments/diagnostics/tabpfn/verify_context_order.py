import numpy as np
import torch
from tabpfn import TabPFNClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

X_context = np.random.randn(10, 2).astype(np.float32)
y_context = np.array([0]*5 + [1]*5)

clf = TabPFNClassifier(device=device)
clf.fit(X_context, y_context)

transformer = clf.model[2]
captured = {}
def hook(module, input):
    captured['ir'] = input[0].detach().clone()
handle = transformer.decoder.register_forward_pre_hook(hook)

# Two totally different queries, same fixed context
X_query_A = np.zeros((5, 2), dtype=np.float32)
_ = clf.predict_proba(X_query_A)
ir_A = captured['ir']

X_query_B = np.ones((5, 2), dtype=np.float32) * 100
_ = clf.predict_proba(X_query_B)
ir_B = captured['ir']

handle.remove()

print("ir_A shape:", ir_A.shape)
first10_diff = (ir_A[:10] - ir_B[:10]).abs().mean().item()
last5_diff   = (ir_A[10:] - ir_B[10:]).abs().mean().item()
print(f"mean abs diff, first 10 rows (should be ~0 if these are context): {first10_diff:.6f}")
print(f"mean abs diff, last 5 rows  (should be large if these are query): {last5_diff:.6f}")

if first10_diff < 1e-4 and last5_diff > 1e-2:
    print("\nCONFIRMED: context first, query last")
elif last5_diff < 1e-4 and first10_diff > 1e-2:
    print("\nCONFIRMED: query first, context last")
else:
    print("\nUNCLEAR — paste this output, don't proceed until resolved")
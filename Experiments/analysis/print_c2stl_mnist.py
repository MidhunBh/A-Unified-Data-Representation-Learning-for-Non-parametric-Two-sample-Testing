import json, numpy as np
d = json.load(open("result/c2st_mnist_linear_power.json"))
print("C2ST-L (linear kernel) | MNIST vs Fake MNIST | Table 3")
for m, r in zip(d["M"], d["result"]):
    print(f"M={m}: {r:.3f}")

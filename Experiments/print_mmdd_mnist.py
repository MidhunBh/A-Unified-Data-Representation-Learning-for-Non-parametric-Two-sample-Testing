import json
d = json.load(open("result/mmd_d_mnist_power.json"))
ref = {"200": 0.996, "400": 1.000, "600": 1.000, "800": 1.000, "1000": 1.000}
print("MMD-D | MNIST vs Fake MNIST | Table 3")
for m, r in zip(d["M"], d["result"]):
    print(f"M={m}: {r:.3f}")

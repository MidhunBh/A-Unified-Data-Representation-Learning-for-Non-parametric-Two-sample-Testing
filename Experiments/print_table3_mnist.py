import json, os, pickle, numpy as np

results = {}

# C2ST poly kernel
with open("result/c2st_mnist_baseline_mmd.pkl", "rb") as f:
    data = pickle.load(f)
results["C2ST (poly)"] = [float(np.mean(data[i])) for i in range(5)]

# C2ST-L linear kernel
d = json.load(open("result/c2st_mnist_linear_power.json"))
results["C2ST-L (linear)"] = [float(x) for x in d["result"]]

# RL-C2ST
d = json.load(open("result/c2st_semi_mnist.json"))
results["RL-C2ST-S"] = [float(x["C2ST-S"]) for x in d["result"]]
results["RL-C2ST-L"] = [float(x["C2ST-L"]) for x in d["result"]]

# MMD-D
d = json.load(open("result/mmd_d_mnist_power.json"))
results["MMD-D"] = [float(x) for x in d["result"]]

# RL-MMD-D
d = json.load(open("result/rl_mmd_d_mnist_power.json"))
results["RL-MMD-D"] = [float(x) for x in d["result"]]

M_list = [100, 200, 300, 400, 500]
print("Table 3 MNIST — your results")
print(f"{'Method':<20} {'M=100':>6} {'M=200':>6} {'M=300':>6} {'M=400':>6} {'M=500':>6}")
print("-" * 56)
for method, vals in results.items():
    row = f"{method:<20}"
    for v in vals:
        row += f" {v:>6.3f}"
    print(row)
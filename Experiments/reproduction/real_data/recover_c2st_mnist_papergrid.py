import pickle
import json
import numpy as np
import time
import subprocess

with open("result/c2st_mnist_linear_power_papergrid.pkl", "rb") as f:
    summary = pickle.load(f)

n_list = [200, 400, 600, 800, 1000]

print("Recovered results:")
for n, s in zip(n_list, summary):
    print(f"M={n}: {np.mean(s):.3f}  (trials: {len(s)})")

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
    ).decode().strip()
except Exception as e:
    print(f"\ncould not get git commit hash: {e}")
    commit = "unknown"

meta = {
    "commit": commit,
    "method": "C2ST-L (linear kernel)",
    "dataset": "MNIST vs Fake MNIST",
    "metric": "test power",
    "panel": "Table 3 — paper grid rerun",
    "M": n_list,
    "N_TRAIL": 100,
    "note": "Rerun at paper's actual Table 3 grid [200,400,600,800,1000], vs repo default [100,200,300,400,500]. Recovered from pkl after original run's metadata-write step crashed on a git subprocess PATH issue (not a data issue).",
    "result": [float(np.mean(s)) for s in summary],
    "ts": time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/c2st_mnist_linear_power_papergrid.json", "w") as f:
    json.dump(meta, f, indent=2)
print("\nsaved to result/c2st_mnist_linear_power_papergrid.json")
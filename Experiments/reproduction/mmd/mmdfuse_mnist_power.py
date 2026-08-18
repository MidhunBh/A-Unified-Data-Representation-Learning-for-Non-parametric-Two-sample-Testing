import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm
import json
import subprocess
import time
from mmd_fuse import *
from utils import sample_mnist_semi

alpha = 0.05
N_TRAIL = 10
n_list = [100, 200, 300, 400, 500]

os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs

data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

results = []
key = random.PRNGKey(42)

for M in n_list:
    outputs = []
    for kk in tqdm.trange(N_TRAIL, desc=f"M={M}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_fake_all, M, M, kk=kk)
        S1 = torch.cat([s1_tr, s1_te], dim=0).numpy().reshape(2*M, -1).astype(np.float32)
        S2 = torch.cat([s2_tr, s2_te], dim=0).numpy().reshape(2*M, -1).astype(np.float32)

        key, subkey = random.split(key)
        out = mmdfuse(S1, S2, subkey)
        outputs.append(int(out))

    power = float(np.mean(outputs))
    results.append({"M": M, "power": power})
    print(f"M={M}: MMD-FUSE power={power:.3f}", flush=True)

    with open("result/mmdfuse_mnist_power.pkl", "wb") as f:
        pickle.dump(results, f)

    meta = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip(),
        "method": "MMD-FUSE",
        "dataset": "MNIST vs Fake MNIST", "metric": "test power",
        "panel": "Table 3 — last missing MNIST cell",
        "M_planned": n_list, "M_completed": [r["M"] for r in results],
        "N_TRAIL": N_TRAIL,
        "note": "Raw flattened pixels (1024-dim), no encoder — matches MMD-FUSE's no-representation-learning baseline definition (Biggs et al. 2023). Input is train+test COMBINED per class (2M real, 2M fake) — no data splitting, matching how MMD-FUSE was already run on HDGM in this project. Labeled under the same M as other MNIST methods for table alignment, though MMD-FUSE technically sees 2M per class where split-based methods use M for train + M for test.",
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/mmdfuse_mnist_power.json", "w") as f:
        json.dump(meta, f, indent=2)

print("\n=== DONE ===", flush=True)
for r in results:
    print(r, flush=True)
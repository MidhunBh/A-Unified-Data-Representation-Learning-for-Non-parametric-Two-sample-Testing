import os
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import tqdm
import json
import subprocess
import time
from utils import *

# Device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
batch_size = 100
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
channels = 1
img_size = 32

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            *discriminator_block(channels, 8, bn=False),
            *discriminator_block(8, 16),
            *discriminator_block(16, 32),
        )
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(
            nn.Linear(32 * ds_size ** 2, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
            nn.Softmax(dim=1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)

# Load real MNIST only — no fake data needed for this check
os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)

for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs

n_list = [100, 200, 300, 400, 500]

all_results_s = []
all_results_l = []

for n in n_list:
    n_train = n
    n_test = n
    trials_s = []
    trials_l = []

    for kk in tqdm.trange(N_TRAIL, desc=f"M={n} (Type-I)"):
        # KEY CHANGE: real MNIST vs real MNIST — H0 is true, P=Q
        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(
            data_real_all, data_real_all, n_train, n_test, kk=kk)

        S_tr = torch.cat([s1_tr, s2_tr], dim=0).to(device, dtype)
        y_tr = torch.cat([
            torch.zeros(n_train), torch.ones(n_train)
        ]).to(device, dtype).long()
        S_te = torch.cat([s1_te, s2_te], dim=0).to(device, dtype)

        torch.random.manual_seed(kk * 1102 + 819)
        discriminator = Discriminator().to(device, dtype)

        optimizer = torch.optim.Adam(
            discriminator.parameters(), lr=0.0005, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss().to(device)
        dataset = torch.utils.data.TensorDataset(S_tr, y_tr)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size * 2, shuffle=True)

        for epoch in range(2 * n):
            for S_b, y_b in loader:
                loss = criterion(discriminator(S_b), y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)

        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST_D(
                S_te, n_test, N_PER, alpha, discriminator, device, dtype)
            H_L[k], _, _ = TST_LCE_D(
                S_te, n_test, N_PER, alpha, discriminator, device, dtype)

        trials_s.append(H_S.mean())
        trials_l.append(H_L.mean())

        if (kk + 1) % 10 == 0:
            print(f"  M={n} trial {kk+1}/100 | "
                  f"Type-I S: {np.mean(trials_s):.3f} "
                  f"Type-I L: {np.mean(trials_l):.3f}")

    all_results_s.append(trials_s)
    all_results_l.append(trials_l)

    se_s = np.std(trials_s) / np.sqrt(N_TRAIL)
    se_l = np.std(trials_l) / np.sqrt(N_TRAIL)
    print(f"\nM={n} Type-I FINAL (should be near 0.05):")
    print(f"  C2ST-S: {np.mean(trials_s):.3f} +/- {se_s:.3f} (SE)")
    print(f"  C2ST-L: {np.mean(trials_l):.3f} +/- {se_l:.3f} (SE)\n")

with open("result/c2st_mnist_100trials_type1.pkl", "wb") as f:
    pickle.dump({"S": all_results_s, "L": all_results_l, "n_list": n_list}, f)

summary_rows = []
for i, n in enumerate(n_list):
    se_s = float(np.std(all_results_s[i]) / np.sqrt(N_TRAIL))
    se_l = float(np.std(all_results_l[i]) / np.sqrt(N_TRAIL))
    summary_rows.append({
        "M"        : n,
        "C2ST-S"   : float(np.mean(all_results_s[i])),
        "C2ST-S_se": se_s,
        "C2ST-L"   : float(np.mean(all_results_l[i])),
        "C2ST-L_se": se_l,
    })

meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"   : "C2ST + C2ST-L",
    "dataset"  : "MNIST vs MNIST (real vs real, H0 true)",
    "metric"   : "type-I error",
    "panel"    : "Table 3 validation check — not in paper",
    "N_TRAIL"  : N_TRAIL,
    "N_TEST"   : N_TEST,
    "N_PER"    : N_PER,
    "alpha"    : alpha,
    "note"     : "checks whether inflated power vs paper (Table 3) is genuine or overfitting artifact. If Type-I stays near 0.05, power result is trustworthy.",
    "result"   : summary_rows,
    "ts"       : time.strftime("%Y-%m-%d %H:%M"),
}

with open("result/c2st_mnist_100trials_type1.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n=== FINAL TYPE-I TABLE (target: ~0.05 everywhere) ===")
print(f"{'M':<6} {'C2ST-S':>16} {'C2ST-L':>16}")
for r in summary_rows:
    flag_s = " <-- INFLATED" if r["C2ST-S"] > 0.10 else ""
    flag_l = " <-- INFLATED" if r["C2ST-L"] > 0.10 else ""
    print(f"M={r['M']:<4} "
          f"{r['C2ST-S']:.3f}+/-{r['C2ST-S_se']:.3f}{flag_s}  "
          f"{r['C2ST-L']:.3f}+/-{r['C2ST-L_se']:.3f}{flag_l}")

print("\nlogged to result/c2st_mnist_100trials_type1.json")
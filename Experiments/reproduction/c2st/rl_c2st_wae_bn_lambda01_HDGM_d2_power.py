import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import json
import subprocess
import time
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("device:", device)

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
x_in = 2
H = 30
x_out = 30
batch_size = 512

AE_EPOCHS = 2000
WARMUP_EPOCHS = 1000
LAMBDA_TARGET = 1.0
N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [125, 250, 500, 750, 1000, 1250]


class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_in, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, x_out),
            nn.BatchNorm1d(x_out),
        )
        self.decoder = nn.Sequential(
            nn.Linear(x_out, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, x_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_wae(S, device, dtype, x_in=x_in, H=H, x_out=x_out):
    model = WAE_BN(x_in, H, x_out).to(device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    dataset = torch.utils.data.TensorDataset(S)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(AE_EPOCHS):
        lam = 0.0 if epoch < WARMUP_EPOCHS else LAMBDA_TARGET * min(1.0, (epoch - WARMUP_EPOCHS) / WARMUP_EPOCHS)
        for (x,) in loader:
            recon, z = model(x)
            recon_loss = F.mse_loss(recon, x)
            z_prior = torch.randn_like(z)
            combined = torch.cat([z, z_prior], dim=0)
            mmd_reg, _, _ = MMDl(combined, len_s=z.shape[0], kernel="rbf")
            loss = recon_loss + lam * mmd_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch+1}: lambda={lam:.2f} recon={recon_loss.item():.4f} mmd={mmd_reg.item():.4f}")
    model.eval()  # BatchNorm must use running stats for Phase 2/3, not batch stats
    return model


wae_result = []
for n in n_list:
    n_train = n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=x_in, kk=kk, level="hard")

        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        wae = train_wae(S_encoder, device, dtype)
        for p in wae.parameters():
            p.requires_grad = False

        with torch.no_grad():
            IR_s1_tr = wae.encoder(MatConvert(s1_tr, device, dtype))
            IR_s1_te = wae.encoder(MatConvert(s1_te, device, dtype))
            IR_s2_tr = wae.encoder(MatConvert(s2_tr, device, dtype))
            IR_s2_te = wae.encoder(MatConvert(s2_te, device, dtype))

        S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0)
        y = torch.cat([torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))]).to(device, dtype).long()

        model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_out, H, x_out, N_EPOCH, batch_size, device, dtype, model=None, lr_c2st=0.002)

        S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0)
        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)

        summary_s.append(H_S.sum() / N_TEST_F)
        summary_l.append(H_L.sum() / N_TEST_F)

    power_s, power_l = float(np.mean(summary_s)), float(np.mean(summary_l))
    wae_result.append({"N": 8*n, "RL-C2ST-S": power_s, "RL-C2ST-L": power_l})
    print(f"N={8*n}: RL-S(WAE)={power_s:.3f}  RL-L(WAE)={power_l:.3f}", flush=True)

    with open("result/rl_c2st_wae_bn_lambda01_HDGM_d2_power.pkl", "wb") as f:
        pickle.dump(wae_result, f)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing").decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "commit": commit, "method": "RL-C2ST (WAE + BatchNorm)", "dataset": "HDGM-D", "level": "hard",
        "metric": "test power", "panel": "Fig 3b - 4th bar (final)", "d": x_in,
        "N_planned": [8*n for n in n_list], "N_completed": [r["N"] for r in wae_result],
        "N_TRAIL": N_TRAIL, "lambda_mmd_final": LAMBDA_TARGET, "warmup_epochs": WARMUP_EPOCHS,
        "note": "Two prior attempts (lambda=10, lambda=1, no BatchNorm) both showed genuine encoder collapse -- confirmed via direct full-pipeline test, not just covariance proxies, several of which gave misleading results. Fix: nn.BatchNorm1d after the encoder's final layer, prevents collapse to a degenerate solution. Verified at N=10000 (single trial, power=1.000/1.000, matching AE/C2ST) before this full run.",
        "result": wae_result, "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/rl_c2st_wae_bn_lambda01_HDGM_d2_power.json", "w") as f:
        json.dump(meta, f, indent=2)

print("=== DONE ===")
for r in wae_result:
    print(r)

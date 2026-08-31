import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
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

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
H = 30
final_dr_dim = 30
channels, img_size, z_size = 1, 32, 100
AE_EPOCHS = 1000
WARMUP_EPOCHS = 500
LAMBDA_TARGET = 0.1   # matching WAE-HDGM-d10's best-evidenced value, not d2's lambda=1
N_EPOCH = 1000
N_TRAIL = 10
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [100, 200, 300, 400, 500]


class WAE_Img(nn.Module):
    def __init__(self, channels, img_size, z_size):
        super().__init__()
        self.encoder_trunk = Encoder_Img(channels, img_size, z_size)
        self.enc_bn = nn.BatchNorm1d(z_size)   # same fix that solved both WAE-d2 and VAE-mnist's mu collapse
        self.decoder = Decoder_Img(channels, img_size, z_size)

    def encode(self, x):
        h = self.encoder_trunk(x)
        return self.enc_bn(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def train_wae_img(S, device, dtype):
    model = WAE_Img(channels, img_size, z_size).to(device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S), batch_size=200, shuffle=True)
    for epoch in range(AE_EPOCHS):
        lam = 0.0 if epoch < WARMUP_EPOCHS else LAMBDA_TARGET * min(1.0, (epoch - WARMUP_EPOCHS) / WARMUP_EPOCHS)
        for (x,) in loader:
            recon, z = model(x)
            recon_loss = F.mse_loss(recon, x)
            z_prior = torch.randn_like(z)
            mmd_reg, _, _ = MMDl(torch.cat([z, z_prior], dim=0), len_s=z.shape[0], kernel="rbf")
            loss = recon_loss + lam * mmd_reg
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model


os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs
data_fake_all = pickle.load(open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

wae_result = []
for n in n_list:
    n_train = n_test = n
    summary_s, summary_l = [], []
    for kk in tqdm.trange(N_TRAIL, desc=f"M={n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(data_real_all, data_fake_all, n_train, n_test, kk=kk)
        S_pool = torch.cat([s1_tr, s1_te, s2_tr, s2_te], dim=0).to(device, dtype)
        wae = train_wae_img(S_pool, device, dtype)
        for p in wae.parameters(): p.requires_grad = False

        with torch.no_grad():
            IR_s1_tr = wae.encode(s1_tr.to(device, dtype))
            IR_s1_te = wae.encode(s1_te.to(device, dtype))
            IR_s2_tr = wae.encode(s2_tr.to(device, dtype))
            IR_s2_te = wae.encode(s2_te.to(device, dtype))

        S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0)
        y = torch.cat([torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))]).to(device, dtype).long()
        model_C2ST_L, w, b = C2ST_NN_fit(S, y, z_size, H, final_dr_dim, N_EPOCH, 128, device, dtype, model=None, lr_c2st=0.002)

        S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0)
        H_S = np.zeros(N_TEST); H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
        summary_s.append(H_S.sum() / N_TEST_F); summary_l.append(H_L.sum() / N_TEST_F)

    power_s, power_l = float(np.mean(summary_s)), float(np.mean(summary_l))
    wae_result.append({"M": n, "RL-C2ST-S": power_s, "RL-C2ST-L": power_l})
    print(f"M={n}: RL-S(WAE)={power_s:.3f}  RL-L(WAE)={power_l:.3f}", flush=True)

    with open("result/rl_c2st_wae_mnist_power.pkl", "wb") as f:
        pickle.dump(wae_result, f)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing").decode().strip()
    except Exception:
        commit = "unknown"
    meta = {
        "commit": commit, "method": "RL-C2ST (WAE, conv)", "dataset": "MNIST vs Fake MNIST",
        "metric": "test power", "panel": "Fig 3a - WAE bar", "M_planned": n_list,
        "M_completed": [r["M"] for r in wae_result], "N_TRAIL": N_TRAIL, "lambda_mmd_final": LAMBDA_TARGET,
        "note": "Conv WAE (Encoder_Img/Decoder_Img backbone), single BatchNorm1d on encoder output (same fix pattern as WAE-d2's collapse and VAE-mnist's mu collapse). lambda=0.1, matching WAE-HDGM-d10's best-evidenced value rather than d2's lambda=1, since a single scalar clearly did not transfer across HDGM dimensionality either.",
        "result": wae_result, "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/rl_c2st_wae_mnist_power.json", "w") as f:
        json.dump(meta, f, indent=2)

print("=== DONE ===")
for r in wae_result: print(r)

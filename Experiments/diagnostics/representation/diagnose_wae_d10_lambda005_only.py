import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

x_in, H, x_out = 10, 30, 30
alpha = 0.05
n = 250
N_TRAIL = 15
N_TEST = 30
N_PER = 100
AE_EPOCHS = 2000
WARMUP_EPOCHS = 1000

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out), nn.BatchNorm1d(x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_wae(S, lambda_target):
    model = WAE_BN(x_in, H, x_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S), batch_size=512, shuffle=True)
    for epoch in range(AE_EPOCHS):
        lam = 0.0 if epoch < WARMUP_EPOCHS else lambda_target * min(1.0, (epoch - WARMUP_EPOCHS) / WARMUP_EPOCHS)
        for (x,) in loader:
            recon, z = model(x)
            recon_loss = F.mse_loss(recon, x)
            mmd_reg, _, _ = MMDl(torch.cat([z, torch.randn_like(z)], dim=0), len_s=z.shape[0], kernel="rbf")
            (recon_loss + lam * mmd_reg).backward()
            optimizer.step(); optimizer.zero_grad()
    model.eval()
    return model

lambda_target = 0.05
print(f"\n=== lambda_target={lambda_target}, N=2000, {N_TRAIL} trials ===")
summary_s, summary_l = [], []
for kk in tqdm.trange(N_TRAIL, desc=f"lambda={lambda_target}"):
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")
    S_encoder = MatConvert(np.concatenate([s1_tr, s1_te, s2_tr, s2_te]), device, torch.float)
    wae = train_wae(S_encoder, lambda_target)
    for p in wae.parameters(): p.requires_grad = False

    with torch.no_grad():
        IR_s1_tr = wae.encoder(MatConvert(s1_tr, device, torch.float))
        IR_s1_te = wae.encoder(MatConvert(s1_te, device, torch.float))
        IR_s2_tr = wae.encoder(MatConvert(s2_tr, device, torch.float))
        IR_s2_te = wae.encoder(MatConvert(s2_te, device, torch.float))

    S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0)
    y = torch.cat([torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))]).to(device, torch.float).long()
    model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_out, H, x_out, 1000, 512, device, torch.float, model=None, lr_c2st=0.002)

    S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0)
    H_S = np.zeros(N_TEST); H_L = np.zeros(N_TEST)
    for k in range(N_TEST):
        H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
        H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
    summary_s.append(H_S.mean()); summary_l.append(H_L.mean())

print(f"\nlambda={lambda_target}: power_S={np.mean(summary_s):.3f}  power_L={np.mean(summary_l):.3f}")
print(f"(compare: lambda=0.02 -> 0.133/0.133 | lambda=0.1 (100-trial) -> 0.04/0.06)")

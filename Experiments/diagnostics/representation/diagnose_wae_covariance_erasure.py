import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 2, 30, 30

# Real HDGM-hard data, one trial, matching what every driver actually sees
s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(500, 500, d=x_in, kk=0, level="hard")
S_pool = np.concatenate([s1_tr, s1_te, s2_tr, s2_te], axis=0)
P_raw = np.concatenate([s1_tr, s1_te], axis=0)
Q_raw = np.concatenate([s2_tr, s2_te], axis=0)

print("=== RAW DATA (before any encoder) ===")
print("cov(P):\n", np.cov(P_raw.T))
print("cov(Q):\n", np.cov(Q_raw.T))
print("(expect Q's off-diagonal near +/-0.5, P's near 0 -- this IS the signal)")

# Train plain AE and WAE (lambda=1, the better of your two runs) on this SAME pool
S_t = MatConvert(S_pool, device, torch.float)

print("\n=== Training plain AE ===")
ae_encoder = train_autoencoder(S_t, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
for p in ae_encoder.parameters(): p.requires_grad = False

class WAE(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

print("\n=== Training WAE (lambda_target=1, annealed) ===")
torch.manual_seed(1102)
wae = WAE(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(wae.parameters(), lr=0.002)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S_t), batch_size=512, shuffle=True)
for epoch in range(2000):
    lam = 0.0 if epoch < 1000 else 1.0 * min(1.0, (epoch - 1000) / 1000)
    for (x,) in loader:
        recon, z = wae(x)
        recon_loss = F.mse_loss(recon, x)
        z_prior = torch.randn_like(z)
        mmd_reg, _, _ = MMDl(torch.cat([z, z_prior], dim=0), len_s=z.shape[0], kernel="rbf")
        loss = recon_loss + lam * mmd_reg
        optimizer.zero_grad(); loss.backward(); optimizer.step()
for p in wae.parameters(): p.requires_grad = False

P_t = MatConvert(P_raw, device, torch.float)
Q_t = MatConvert(Q_raw, device, torch.float)

with torch.no_grad():
    P_ae = ae_encoder(P_t).cpu().numpy()
    Q_ae = ae_encoder(Q_t).cpu().numpy()
    P_wae = wae.encoder(P_t).cpu().numpy()
    Q_wae = wae.encoder(Q_t).cpu().numpy()

def offdiag_gap(P_enc, Q_enc, label):
    # only look at first 2 encoded dims for a direct, comparable off-diagonal check
    cov_p = np.cov(P_enc[:, :2].T)
    cov_q = np.cov(Q_enc[:, :2].T)
    print(f"{label}: cov(P)[0,1]={cov_p[0,1]:.4f}  cov(Q)[0,1]={cov_q[0,1]:.4f}  |gap|={abs(cov_p[0,1]-cov_q[0,1]):.4f}")

print("\n=== ENCODED SPACE (first 2 of x_out dims) ===")
raw_cov_p = np.cov(P_raw.T)[0,1]
raw_cov_q = np.cov(Q_raw.T)[0,1]
print(f"RAW:  cov(P)[0,1]={raw_cov_p:.4f}  cov(Q)[0,1]={raw_cov_q:.4f}  |gap|={abs(raw_cov_p-raw_cov_q):.4f}")
offdiag_gap(P_ae, Q_ae, "AE  ")
offdiag_gap(P_wae, Q_wae, "WAE ")

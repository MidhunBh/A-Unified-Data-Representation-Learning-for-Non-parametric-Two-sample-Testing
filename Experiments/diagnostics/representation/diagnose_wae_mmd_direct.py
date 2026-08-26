import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 2, 30, 30

mu_mx, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters=2, d=x_in, cluster_gap=0.5)
np.random.seed(0)
P_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_1, 1000)
P_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_1, 1000)
Q_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_2[0], 1000)
Q_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_2[1], 1000)
P_raw = np.concatenate([P_c0, P_c1]); Q_raw = np.concatenate([Q_c0, Q_c1])
S_pool = MatConvert(np.concatenate([P_raw, Q_raw]), device, torch.float)

def mmd_between(A_np, B_np):
    A = MatConvert(A_np, device, torch.float)
    B = MatConvert(B_np, device, torch.float)
    val, _, _ = MMDl(torch.cat([A, B], dim=0), len_s=len(A_np), kernel="rbf")
    return val.item()

print("=== RAW DATA: MMD(P, Q) ===")
print(f"  {mmd_between(P_raw, Q_raw):.4f}")

print("\n=== Training AE ===")
ae_encoder = train_autoencoder(S_pool, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
for p in ae_encoder.parameters(): p.requires_grad = False

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out), nn.BatchNorm1d(x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

print("\n=== Training WAE-BN ===")
torch.manual_seed(1102)
wae = WAE_BN(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(wae.parameters(), lr=0.002)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S_pool), batch_size=512, shuffle=True)
for epoch in range(2000):
    lam = 0.0 if epoch < 1000 else 1.0 * min(1.0, (epoch - 1000) / 1000)
    for (x,) in loader:
        recon, z = wae(x)
        recon_loss = F.mse_loss(recon, x)
        mmd_reg, _, _ = MMDl(torch.cat([z, torch.randn_like(z)], dim=0), len_s=z.shape[0], kernel="rbf")
        (recon_loss + lam * mmd_reg).backward()
        optimizer.step(); optimizer.zero_grad()
wae.eval()
for p in wae.parameters(): p.requires_grad = False

with torch.no_grad():
    P_ae = ae_encoder(MatConvert(P_raw, device, torch.float)).cpu().numpy()
    Q_ae = ae_encoder(MatConvert(Q_raw, device, torch.float)).cpu().numpy()
    P_wae = wae.encoder(MatConvert(P_raw, device, torch.float)).cpu().numpy()
    Q_wae = wae.encoder(MatConvert(Q_raw, device, torch.float)).cpu().numpy()

print("\n=== ENCODED (full 30-dim): MMD(P, Q) -- the actual signal Phase 3 relies on ===")
print(f"  AE : {mmd_between(P_ae, Q_ae):.4f}")
print(f"  WAE: {mmd_between(P_wae, Q_wae):.4f}")
print("(compare all three -- if WAE's MMD is comparable to AE's, the representation is fine and the earlier 'collapse' reading was a diagnostic artifact, not a real problem)")

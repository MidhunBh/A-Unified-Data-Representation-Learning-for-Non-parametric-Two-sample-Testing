import numpy as np
import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 2, 30, 30

mu_mx, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters=2, d=x_in, cluster_gap=0.5)
np.random.seed(0)
Q_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_2[0], 1000)

# Retrain the exact same WAE-BN from the last run (same seed) and inspect its raw output
import torch.nn as nn
import torch.nn.functional as F

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, x_out), nn.BatchNorm1d(x_out),
        )
        self.decoder = nn.Sequential(
            nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

P_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_1, 1000)
P_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_1, 1000)
Q_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_2[1], 1000)
S_pool = MatConvert(np.concatenate([P_c0, P_c1, Q_c0, Q_c1]), device, torch.float)

torch.manual_seed(1102)
wae = WAE_BN(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(wae.parameters(), lr=0.002)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S_pool), batch_size=512, shuffle=True)
for epoch in range(2000):
    lam = 0.0 if epoch < 1000 else 1.0 * min(1.0, (epoch - 1000) / 1000)
    for (x,) in loader:
        recon, z = wae(x)
        recon_loss = F.mse_loss(recon, x)
        z_prior = torch.randn_like(z)
        mmd_reg, _, _ = MMDl(torch.cat([z, z_prior], dim=0), len_s=z.shape[0], kernel="rbf")
        (recon_loss + lam * mmd_reg).backward()
        optimizer.step(); optimizer.zero_grad()

wae.eval()
with torch.no_grad():
    Z = wae.encoder(MatConvert(Q_c0, device, torch.float)).cpu().numpy()

print("per-dimension variance (first 5 of 30 dims):", Z[:, :5].var(axis=0))
print("full 30-dim covariance matrix rank:", np.linalg.matrix_rank(np.cov(Z.T), tol=1e-6))
print("(expect rank near 30 if healthy; rank 1 or near-1 = collapsed to a line/point despite per-dim variance)")

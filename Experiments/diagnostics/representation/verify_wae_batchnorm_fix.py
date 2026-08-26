import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 2, 30, 30

mu_mx, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters=2, d=x_in, cluster_gap=0.5)
n_per_cluster = 1000

np.random.seed(0)
P_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_1, n_per_cluster)
P_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_1, n_per_cluster)
Q_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_2[0], n_per_cluster)
Q_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_2[1], n_per_cluster)

S_pool_np = np.concatenate([P_c0, P_c1, Q_c0, Q_c1], axis=0)
S_pool = MatConvert(S_pool_np, device, torch.float)

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_in, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, x_out),
            nn.BatchNorm1d(x_out),   # the fix -- structurally prevents collapse to a constant
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

print("=== Training WAE-BN (lambda_target=1, annealed) ===")
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
        loss = recon_loss + lam * mmd_reg
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (epoch + 1) % 400 == 0:
        print(f"epoch {epoch+1}: lambda={lam:.2f} recon={recon_loss.item():.4f} mmd={mmd_reg.item():.4f}")

wae.eval()  # BatchNorm needs eval mode to use running stats, not batch stats, for the diagnostic check below
for p in wae.parameters(): p.requires_grad = False

def enc_cov(fn, X_np):
    with torch.no_grad():
        Z = fn(MatConvert(X_np, device, torch.float)).cpu().numpy()
    return np.cov(Z[:, :2].T)[0,1]

print("\n=== ENCODED, PER-CLUSTER (first 2 of x_out dims) ===")
p0, p1 = enc_cov(wae.encoder, P_c0), enc_cov(wae.encoder, P_c1)
q0, q1 = enc_cov(wae.encoder, Q_c0), enc_cov(wae.encoder, Q_c1)
print(f"WAE-BN: P-c0={p0:.4f}  P-c1={p1:.4f}  Q-c0={q0:.4f}  Q-c1={q1:.4f}")
print(f"        cluster0 |Q-P| gap = {abs(q0-p0):.4f}   cluster1 |Q-P| gap = {abs(q1-p1):.4f}")
print("(compare to prior WAE result: all zeros -- any nonzero gap here confirms BatchNorm broke the collapse)")

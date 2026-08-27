import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 10, 30, 30

mu_mx, sigma_mx_1, sigma_mx_2 = generate_hdgm_cov_matrix(n_clusters=2, d=x_in, cluster_gap=0.5)
n_per_cluster = 1000
np.random.seed(0)
P_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_1, n_per_cluster)
P_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_1, n_per_cluster)
Q_c0 = np.random.multivariate_normal(mu_mx[0], sigma_mx_2[0], n_per_cluster)
Q_c1 = np.random.multivariate_normal(mu_mx[1], sigma_mx_2[1], n_per_cluster)
S_pool = MatConvert(np.concatenate([P_c0, P_c1, Q_c0, Q_c1]), device, torch.float)

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out), nn.BatchNorm1d(x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

torch.manual_seed(1102)
wae = WAE_BN(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(wae.parameters(), lr=0.002)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(S_pool), batch_size=512, shuffle=True)
print("Training WAE-BN at d=10, lambda_target=1 (same as d=2)...")
for epoch in range(2000):
    lam = 0.0 if epoch < 1000 else 0.3 * min(1.0, (epoch - 1000) / 1000)
    for (x,) in loader:
        recon, z = wae(x)
        recon_loss = F.mse_loss(recon, x)
        mmd_reg, _, _ = MMDl(torch.cat([z, torch.randn_like(z)], dim=0), len_s=z.shape[0], kernel="rbf")
        (recon_loss + lam * mmd_reg).backward()
        optimizer.step(); optimizer.zero_grad()
wae.eval()
for p in wae.parameters(): p.requires_grad = False

# Run through the FULL pipeline at the hardest, most easily-broken point:
# large N, where collapse would show up as failure to reach the ceiling C2ST/AE hit
s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(1250, 1250, d=x_in, kk=0, level="hard")
with torch.no_grad():
    IR_s1_tr = wae.encoder(MatConvert(s1_tr, device, torch.float))
    IR_s1_te = wae.encoder(MatConvert(s1_te, device, torch.float))
    IR_s2_tr = wae.encoder(MatConvert(s2_tr, device, torch.float))
    IR_s2_te = wae.encoder(MatConvert(s2_te, device, torch.float))

S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0)
y = torch.cat([torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))]).to(device, torch.float).long()
model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_out, H, x_out, 1000, 512, device, torch.float, model=None, lr_c2st=0.002)

S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0)
H_S = np.zeros(100); H_L = np.zeros(100)
for k in range(100):
    H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), 100, 0.05, model_C2ST_L, w, b)
    H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), 100, 0.05, model_C2ST_L, w, b)
print(f"WAE-BN d=10, N=10000, single trial: power_S={H_S.mean():.3f}  power_L={H_L.mean():.3f}")
print("(compare: your saved AE-d10 at N=10000 -- check this against the full curve)")

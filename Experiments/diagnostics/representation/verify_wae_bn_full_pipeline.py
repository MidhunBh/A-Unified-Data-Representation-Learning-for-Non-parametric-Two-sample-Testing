import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
x_in, H, x_out = 2, 30, 30
alpha = 0.05

s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(1250, 1250, d=x_in, kk=0, level="hard")
S_pool = MatConvert(np.concatenate([s1_tr, s1_te, s2_tr, s2_te]), device, torch.float)

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out), nn.BatchNorm1d(x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

print("Training WAE-BN at N=10000 (largest, easiest N -- where AE/C2ST both hit ~1.000)...")
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
    H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), 100, alpha, model_C2ST_L, w, b)
    H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), 100, alpha, model_C2ST_L, w, b)
print(f"\nWAE-BN at N=10000, single trial: power_S={H_S.mean():.3f}  power_L={H_L.mean():.3f}")
print("(compare: your saved C2ST/AE at N=10000 both hit 1.000/1.000 -- if this is also near 1.0, WAE-BN is fine and every diagnostic above was measuring the wrong thing)")

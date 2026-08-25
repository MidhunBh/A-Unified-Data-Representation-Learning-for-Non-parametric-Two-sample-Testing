import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MMDl

x_in, H, x_out = 2, 30, 30
device = "cuda" if torch.cuda.is_available() else "cpu"

class WAE(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

dummy = torch.randn(512, x_in).to(device)
N_STEPS, WARMUP = 400, 200

for lam_target in [1.0, 2.0]:
    torch.manual_seed(0)
    model = WAE(x_in, H, x_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    print(f"\n=== lambda_target={lam_target} ===")
    for step in range(N_STEPS):
        lam = 0.0 if step < WARMUP else lam_target * min(1.0, (step - WARMUP) / WARMUP)
        recon, z = model(dummy)
        recon_loss = F.mse_loss(recon, dummy)
        z_prior = torch.randn_like(z)
        combined = torch.cat([z, z_prior], dim=0)
        mmd_reg, _, _ = MMDl(combined, len_s=z.shape[0], kernel="rbf")
        loss = recon_loss + lam * mmd_reg
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}: lambda={lam:.2f}  recon={recon_loss.item():.4f}  mmd={mmd_reg.item():.4f}")

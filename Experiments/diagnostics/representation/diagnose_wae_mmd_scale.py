import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MMDl

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Check 1: MMDl between two independent N(0,I) batches, no network ===")
for _ in range(5):
    a = torch.randn(256, 30).to(device)
    b = torch.randn(256, 30).to(device)
    combined = torch.cat([a, b], dim=0)
    val, _, _ = MMDl(combined, len_s=256, kernel="rbf")
    print(f"  MMDl(N(0,I), N(0,I)): {val.item():.4f}")

x_in, H, x_out = 2, 30, 30

class WAE(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

dummy = torch.randn(512, x_in).to(device)

for lam in [10, 1, 0.1]:
    torch.manual_seed(0)
    model = WAE(x_in, H, x_out).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    print(f"\n=== Check 2: lambda_mmd={lam} ===")
    for step in range(300):
        recon, z = model(dummy)
        recon_loss = F.mse_loss(recon, dummy)
        z_prior = torch.randn_like(z)
        combined = torch.cat([z, z_prior], dim=0)
        mmd_reg, _, _ = MMDl(combined, len_s=z.shape[0], kernel="rbf")
        loss = recon_loss + lam * mmd_reg
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"  step {step+1}: recon={recon_loss.item():.4f}  mmd={mmd_reg.item():.4f}")

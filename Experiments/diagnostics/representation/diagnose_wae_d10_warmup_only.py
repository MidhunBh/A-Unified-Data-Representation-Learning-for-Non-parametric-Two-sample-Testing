import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MMDl

x_in, H, x_out = 10, 30, 30
device = "cuda" if torch.cuda.is_available() else "cpu"

class WAE_BN(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(x_in, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_out), nn.BatchNorm1d(x_out))
        self.decoder = nn.Sequential(nn.Linear(x_out, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, H), nn.Softplus(), nn.Linear(H, x_in))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = WAE_BN(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
dummy = torch.randn(512, x_in).to(device)

print("Pure warmup, beta=0 throughout, watching ONLY reconstruction convergence at d=10 (no lambda involved at all)...")
for step in range(2000):
    recon, z = model(dummy)
    loss = F.mse_loss(recon, dummy)  # reconstruction only, no MMD term
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step + 1) % 200 == 0:
        print(f"step {step+1}: recon={loss.item():.4f}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
channels, img_size, z_size = 1, 32, 100

class VAE_Img(nn.Module):
    def __init__(self, channels, img_size, z_size):
        super().__init__()
        self.encoder_trunk = Encoder_Img(channels, img_size, z_size)
        self.fc_mu = nn.Sequential(nn.Linear(z_size, z_size), nn.BatchNorm1d(z_size, track_running_stats=False))
        self.fc_logvar = nn.Linear(z_size, z_size)
        self.decoder = Decoder_Img(channels, img_size, z_size)
    def encode(self, x):
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

model = VAE_Img(channels, img_size, z_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
dummy = torch.randn(200, channels, img_size, img_size).to(device)

for step in range(1000):
    beta = 0.0 if step < 500 else min(1.0, (step - 500) / 500)
    recon, mu, logvar = model(dummy)
    loss = F.mse_loss(recon, dummy) + beta * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
    optimizer.zero_grad(); loss.backward(); optimizer.step()

print("=== TRAIN mode mu stats ===")
model.train()
with torch.no_grad():
    mu_t, _ = model.encode(dummy)
print("mean:", mu_t.mean().item(), "std:", mu_t.std().item())

print("=== EVAL mode mu stats (what Phase 2/3 actually sees) ===")
model.eval()
with torch.no_grad():
    mu_e, _ = model.encode(dummy)
print("mean:", mu_e.mean().item(), "std:", mu_e.std().item())
print("(if eval stats collapse toward 0 while train stats look fine, this confirms the running-stats hypothesis)")

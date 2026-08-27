import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
channels, img_size, z_size = 1, 32, 100

class VAE_Img(nn.Module):
    def __init__(self, channels, img_size, z_size):
        super().__init__()
        self.encoder_trunk = Encoder_Img(channels, img_size, z_size)  # reuses existing conv trunk
        self.fc_mu = nn.Linear(z_size, z_size)
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
dummy = torch.randn(64, channels, img_size, img_size).to(device)

N_STEPS, WARMUP = 400, 200
print("Conv VAE, dummy MNIST-shaped data, KL-annealed (mean-based, the fixed formula)...")
for step in range(N_STEPS):
    beta = 0.0 if step < WARMUP else min(1.0, (step - WARMUP) / WARMUP)
    recon, mu, logvar = model(dummy)
    recon_loss = F.mse_loss(recon, dummy)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + beta * kl
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step + 1) % 50 == 0:
        print(f"step {step+1}: beta={beta:.2f} recon={recon_loss.item():.4f} kl={kl.item():.4f}")

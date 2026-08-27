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

print("Instrumented -- every term of KL printed separately, no clamp, beta=0 throughout...")
for step in range(200):
    recon, mu, logvar = model(dummy)
    recon_loss = F.mse_loss(recon, dummy)

    term_1 = 1.0
    term_logvar = logvar.mean().item()
    term_mu2 = mu.pow(2).mean().item()
    term_explogvar = logvar.exp().mean().item()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss  # beta=0, KL not in the graph, isolates what drives recon-only training
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if (step + 1) % 25 == 0:
        print(f"step {step+1}: mean(logvar)={term_logvar:.4f}  mean(mu^2)={term_mu2:.4f}  mean(exp(logvar))={term_explogvar:.4f}  -> KL={kl.item():.4f}")

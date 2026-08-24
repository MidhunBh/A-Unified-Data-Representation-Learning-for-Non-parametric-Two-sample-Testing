import torch
import torch.nn as nn
import torch.nn.functional as F

x_in, H, x_out = 2, 30, 30
device = "cuda" if torch.cuda.is_available() else "cpu"

class VAE(nn.Module):
    def __init__(self, x_in, H, x_out):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(x_in, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
        )
        self.fc_mu = nn.Linear(H, x_out)
        self.fc_logvar = nn.Linear(H, x_out)
        self.decoder = nn.Sequential(
            nn.Linear(x_out, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, H), nn.Softplus(),
            nn.Linear(H, x_in),
        )

    def encode(self, x):
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

model = VAE(x_in, H, x_out).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
dummy = torch.randn(512, x_in).to(device)

beta = 1.0
print("Training 200 steps on dummy HDGM-shaped data, watching recon and KL separately...")
for step in range(200):
    recon, mu, logvar = model(dummy)
    recon_loss = F.mse_loss(recon, dummy)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / dummy.shape[0]
    loss = recon_loss + beta * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 40 == 0:
        print(f"step {step+1}: recon={recon_loss.item():.4f}  kl={kl.item():.4f}  total={loss.item():.4f}")

print("\nIf recon trends down and kl stays bounded (not exploding), mechanics are sound.")

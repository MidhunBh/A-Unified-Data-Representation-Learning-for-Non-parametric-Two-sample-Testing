import torch
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
H = 30
final_dr_dim = 30

class ExtendedModelFixed(torch.nn.Module):
    def __init__(self, encoder, ae_latent_dim, H, final_out):
        super().__init__()
        self.encoder = encoder
        self.additional_layers = torch.nn.Sequential(
            torch.nn.Linear(ae_latent_dim, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, final_out, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.additional_layers(x)

for ae_latent_dim in [5, 10, 30, 50, 100]:
    dummy_ae = AutoEncoder(10, H, ae_latent_dim).to(device)
    encoder = dummy_ae.encoder
    for p in encoder.parameters():
        p.requires_grad = False
    model = ExtendedModelFixed(encoder, ae_latent_dim, H, final_dr_dim).to(device)
    x = torch.randn(5, 10).to(device)
    out = model(x)
    print(f"ae_latent_dim={ae_latent_dim}: output shape={out.shape} (expect (5, {final_dr_dim}))")
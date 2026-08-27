import torch
import torch.nn as nn
from utils import Encoder_Img

device = "cuda" if torch.cuda.is_available() else "cpu"
channels, img_size, z_size = 1, 32, 100

torch.manual_seed(0)
enc = Encoder_Img(channels, img_size, z_size).to(device)
fc_mu = nn.Linear(z_size, z_size).to(device)
fc_logvar = nn.Linear(z_size, z_size).to(device)

dummy = torch.randn(64, channels, img_size, img_size).to(device)

with torch.no_grad():
    h = enc(dummy)
    mu = fc_mu(h)
    logvar = fc_logvar(h)

print("mu     -- mean:", mu.mean().item(), " std:", mu.std().item())
print("logvar -- mean:", logvar.mean().item(), " std:", logvar.std().item(), " max:", logvar.max().item())
print("exp(logvar) -- mean:", logvar.exp().mean().item(), " max:", logvar.exp().max().item())
print("(if exp(logvar) max is large, e.g. >10, that single term is what is driving KL through the roof)")

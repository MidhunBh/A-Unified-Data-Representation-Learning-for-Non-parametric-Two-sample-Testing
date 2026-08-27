import torch
from utils import Encoder_Img

device = "cuda" if torch.cuda.is_available() else "cpu"
channels, img_size, z_size = 1, 32, 100

torch.manual_seed(0)
enc = Encoder_Img(channels, img_size, z_size).to(device)
dummy = torch.randn(64, channels, img_size, img_size).to(device)

with torch.no_grad():
    h = enc(dummy)

print("Encoder_Img raw output stats (before any mu/logvar heads):")
print(f"  mean: {h.mean().item():.4f}")
print(f"  std:  {h.std().item():.4f}")
print(f"  min:  {h.min().item():.4f}")
print(f"  max:  {h.max().item():.4f}")
print("(compare: HDGM's flat trunk output typically stayed in a small, bounded range -- if this is much larger, that explains the unbounded KL)")

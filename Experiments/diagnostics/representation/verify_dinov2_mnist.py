import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
import time
from utils import *

import numpy as np
import torch
import torch.nn.functional as F
import time
from utils import *


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

print("\nLoading DINOv2 (dinov2_vits14) from torch.hub...")
t0 = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
print(f"loaded in {time.time()-t0:.1f}s, no gating encountered")

dummy = torch.randn(50, 1, 32, 32).to(device)   # matches this repo's MNIST shape

def prep_for_dinov2(x):
    x = x.repeat(1, 3, 1, 1)                                   # 1ch -> 3ch
    x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
    return x

t0 = time.time()
with torch.no_grad():
    out = dinov2(prep_for_dinov2(dummy))
print(f"output shape: {out.shape} (expect (50, 384) for vits14)")
print(f"forward time for 50 images: {time.time()-t0:.3f}s")

t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = dinov2(prep_for_dinov2(torch.randn(256, 1, 32, 32).to(device)))
elapsed = time.time() - t0
per_image = elapsed / (10 * 256)
print(f"\nper-image cost: {per_image*1000:.3f} ms")
print(f"estimated one-time cost, full 70,000-image MNIST pool: {per_image*70000/60:.1f} minutes")

# Confirm the key design trick: sample_mnist_semi is pure index-based
# selection, so it works identically on precomputed EMBEDDINGS as on
# raw images — this is what lets us embed the whole pool ONCE.
fake_real_emb = torch.randn(1000, 384)
fake_fake_emb = torch.randn(500, 384)
s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(fake_real_emb, fake_fake_emb, 100, 100, kk=0)
print(f"\nsample_mnist_semi on embeddings: s1_tr={s1_tr.shape}, s2_tr={s2_tr.shape}")
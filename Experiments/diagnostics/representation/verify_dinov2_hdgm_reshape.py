import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
import time
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False

x_in = 2
side = int(np.ceil(np.sqrt(x_in)))
print(f"grid side for d={x_in}: {side} ({side*side} cells, {x_in} real values, {side*side-x_in} zero-padded)", flush=True)

s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(100, 100, d=x_in, kk=0, level="hard")
S_pool = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
print(f"pool shape: {S_pool.shape}", flush=True)

n = S_pool.shape[0]
padded = np.zeros((n, side * side), dtype=np.float32)
padded[:, :x_in] = S_pool
grid = torch.from_numpy(padded).view(n, 1, side, side)
grid = grid.to(device).repeat(1, 3, 1, 1)
grid = F.interpolate(grid, size=224, mode='bilinear', align_corners=False)
print(f"reshaped grid shape: {grid.shape}", flush=True)

t0 = time.time()
with torch.no_grad():
    emb = dinov2(grid)
print(f"embedding shape: {emb.shape}, time for {n} samples: {time.time()-t0:.2f}s", flush=True)
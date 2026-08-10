import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
import time

print("step 1: imports done", flush=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"step 2: device={device}", flush=True)

print("step 3: about to torch.hub.load...", flush=True)
t0 = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print(f"step 4: hub.load returned in {time.time()-t0:.1f}s", flush=True)

print("step 5: moving to device...", flush=True)
dinov2 = dinov2.to(device)
print("step 6: moved to device", flush=True)

print("step 7: setting eval mode...", flush=True)
dinov2 = dinov2.eval()
print("step 8: eval mode set", flush=True)

for p in dinov2.parameters():
    p.requires_grad = False
print("step 9: params frozen", flush=True)

print("step 10: building dummy input...", flush=True)
dummy = torch.randn(2, 1, 32, 32).to(device)   # tiny batch, just 2 images
dummy = dummy.repeat(1, 3, 1, 1)
dummy = F.interpolate(dummy, size=224, mode='bilinear', align_corners=False)
print(f"step 11: dummy shape={dummy.shape}", flush=True)

print("step 12: about to run forward pass...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = dinov2(dummy)
print(f"step 13: forward pass done in {time.time()-t0:.1f}s, output shape={out.shape}", flush=True)

print("DONE", flush=True)
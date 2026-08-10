import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device, flush=True)

np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
H = 30
final_dr_dim = 30
N_TRAIL = 10
N_TEST = 100
N_PER = 100
n = 100
img_size = 32
batch_embed = 256

print("Loading DINOv2...", flush=True)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
DINOV2_DIM = 384

def embed_all(images):
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_embed):
            batch = images[i:i+batch_embed].to(device)
            batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
            emb = dinov2(batch)
            out.append(emb.cpu())
    return torch.cat(out, dim=0)

os.makedirs("./data/mnist", exist_ok=True)
dataloader_real_all = torch.utils.data.DataLoader(
    datasets.MNIST("./data/mnist", train=True, download=False,
        transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])),
    batch_size=60000, shuffle=True)
for i, (imgs, labels) in enumerate(dataloader_real_all):
    data_real_all = imgs

print("Embedding 2000 real images (enough for M=100 null test)...", flush=True)
real_embeddings = embed_all(data_real_all[:2000])
print(f"done: {real_embeddings.shape}", flush=True)

print("\n--- NULL TEST: real MNIST vs real MNIST, M=100 ---", flush=True)
summary_s, summary_l = [], []
for kk in tqdm.trange(N_TRAIL, desc="M=100 NULL"):
    IR_s1_tr, IR_s1_te, IR_s2_tr, IR_s2_te = sample_mnist_semi(
        real_embeddings, real_embeddings, n, n, kk=kk)   # SAME pool, both "sides"

    S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0).to(device, dtype)
    y = torch.cat([torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))]).to(device, dtype).long()

    model_C2ST_L, w, b = C2ST_NN_fit(
        S, y, DINOV2_DIM, H, final_dr_dim, N_epoch=1000, batch_size=128,
        device=device, dtype=dtype, model=None, lr_c2st=0.002)

    S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0).to(device, dtype)
    H_S, H_L = np.zeros(N_TEST), np.zeros(N_TEST)
    for k in range(N_TEST):
        H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
        H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)

    summary_s.append(H_S.sum() / 100.0)
    summary_l.append(H_L.sum() / 100.0)

print(f"\nType-I (should be ~0.05 if calibrated): S={np.mean(summary_s):.3f}, L={np.mean(summary_l):.3f}", flush=True)
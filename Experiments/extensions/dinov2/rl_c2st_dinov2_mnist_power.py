import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm
import json
import subprocess
import time
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
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
N_TEST_F = 100.0
N_PER = 100
n_list = [100, 200, 300, 400, 500]
img_size = 32
batch_embed = 256

print("Loading DINOv2 (dinov2_vits14)...", flush=True)
t0 = time.time()
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
DINOV2_DIM = 384
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

def embed_all(images, label):
    out = []
    n_batches = (len(images) + batch_embed - 1) // batch_embed
    with torch.no_grad():
        for i in range(0, len(images), batch_embed):
            batch_num = i // batch_embed + 1
            batch = images[i:i+batch_embed].to(device)
            batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
            emb = dinov2(batch)
            out.append(emb.cpu())
            if batch_num % 20 == 0 or batch_num == n_batches:
                print(f"  {label}: batch {batch_num}/{n_batches}", flush=True)
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

data_fake_all = pickle.load(
    open('./data/Fake_MNIST_data_EP100_N10000.pckl', 'rb'))[0]
data_fake_all = torch.from_numpy(data_fake_all).float()

print(f"\nEmbedding {len(data_real_all)} real + {len(data_fake_all)} fake images (one-time)...", flush=True)
t0 = time.time()
real_embeddings = embed_all(data_real_all, "real")
fake_embeddings = embed_all(data_fake_all, "fake")
print(f"embedding done in {(time.time()-t0)/60:.1f} minutes", flush=True)
print(f"real_embeddings: {real_embeddings.shape}, fake_embeddings: {fake_embeddings.shape}", flush=True)

results = []
for n in n_list:
    n_train = n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"M={n}"):
        IR_s1_tr, IR_s1_te, IR_s2_tr, IR_s2_te = sample_mnist_semi(
            real_embeddings, fake_embeddings, n_train, n_test, kk=kk)

        S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0).to(device, dtype)
        y = torch.cat([
            torch.zeros(len(IR_s1_tr)), torch.ones(len(IR_s2_tr))
        ]).to(device, dtype).long()

        model_C2ST_L, w, b = C2ST_NN_fit(
            S, y, DINOV2_DIM, H, final_dr_dim, N_epoch=1000, batch_size=128,
            device=device, dtype=dtype, model=None, lr_c2st=0.002)

        S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0).to(device, dtype)

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)

        summary_s.append(H_S.sum() / N_TEST_F)
        summary_l.append(H_L.sum() / N_TEST_F)

    power_s, power_l = float(np.mean(summary_s)), float(np.mean(summary_l))
    results.append({"M": n, "RL-C2ST-S": power_s, "RL-C2ST-L": power_l})
    print(f"M={n}: RL-S(DINOv2)={power_s:.3f}  RL-L(DINOv2)={power_l:.3f}", flush=True)

    # save after every M level
    with open("result/rl_c2st_dinov2_mnist_power.pkl", "wb") as f:
        pickle.dump(results, f)

    meta = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip(),
        "method": "RL-C2ST (frozen DINOv2 vits14 encoder, Option A)",
        "dataset": "MNIST vs Fake MNIST", "metric": "test power",
        "panel": "Fig 3a — 4th bar, pretrained-encoder Phase1",
        "M_planned": n_list, "M_completed": [r["M"] for r in results],
        "N_TRAIL": N_TRAIL, "dinov2_dim": DINOV2_DIM,
        "note": "Phase 1 = frozen DINOv2 vits14, self-supervised pretrained on LVD-142M, embedded ONCE for the entire pool before the trial loop. Phase 2 = unchanged C2ST_NN_fit, x_in=384. N_TRAIL=10 matches validated MNIST convention.",
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/rl_c2st_dinov2_mnist_power.json", "w") as f:
        json.dump(meta, f, indent=2)

print("\n=== DONE ===", flush=True)
for r in results:
    print(r, flush=True)
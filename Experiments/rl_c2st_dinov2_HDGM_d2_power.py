import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pickle
import torch
import torch.nn.functional as F
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

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
x_in = 2
H = 30
final_dr_dim = 30
N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [125, 250, 500, 750, 1000, 1250]
batch_embed = 512

print("Loading DINOv2 (dinov2_vits14)...", flush=True)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2 = dinov2.to(device).eval()
for p in dinov2.parameters():
    p.requires_grad = False
DINOV2_DIM = 384

side = int(np.ceil(np.sqrt(x_in)))

def vec_to_dinov2_embedding(x_np):
    n = x_np.shape[0]
    padded = np.zeros((n, side * side), dtype=np.float32)
    padded[:, :x_in] = x_np
    grid = torch.from_numpy(padded).view(n, 1, side, side)

    out = []
    with torch.no_grad():
        for i in range(0, n, batch_embed):
            batch = grid[i:i+batch_embed].to(device)
            batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=224, mode='bilinear', align_corners=False)
            emb = dinov2(batch)
            out.append(emb.cpu())
    return torch.cat(out, dim=0)

dinov2_result = []
for n in n_list:
    n_train = n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train, n_test, d=x_in, kk=kk, level="hard")

        n1_tr, n1_te, n2_tr = len(s1_tr), len(s1_te), len(s2_tr)
        S_pool = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        IR_pool = vec_to_dinov2_embedding(S_pool)

        IR_s1_tr = IR_pool[:n1_tr]
        IR_s1_te = IR_pool[n1_tr:n1_tr+n1_te]
        IR_s2_tr = IR_pool[n1_tr+n1_te:n1_tr+n1_te+n2_tr]
        IR_s2_te = IR_pool[n1_tr+n1_te+n2_tr:]

        S = torch.cat([IR_s1_tr, IR_s2_tr], dim=0).to(device, dtype)
        y = torch.cat([
            torch.zeros(n1_tr), torch.ones(n2_tr)
        ]).to(device, dtype).long()

        model_C2ST_L, w, b = C2ST_NN_fit(
            S, y, DINOV2_DIM, H, final_dr_dim, N_EPOCH, batch_size=1024,
            device=device, dtype=dtype, model=None, lr_c2st=0.002)

        S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0).to(device, dtype)

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, n1_te, N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, n1_te, N_PER, alpha, model_C2ST_L, w, b)

        summary_s.append(H_S.sum() / N_TEST_F)
        summary_l.append(H_L.sum() / N_TEST_F)

    power_s, power_l = float(np.mean(summary_s)), float(np.mean(summary_l))
    dinov2_result.append({"N": 8*n, "RL-C2ST-S": power_s, "RL-C2ST-L": power_l})
    print(f"N={8*n}: RL-S(DINOv2)={power_s:.3f}  RL-L(DINOv2)={power_l:.3f}", flush=True)

    with open("result/rl_c2st_dinov2_HDGM_d2_power.pkl", "wb") as f:
        pickle.dump(dinov2_result, f)

    meta = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip(),
        "method": "RL-C2ST (frozen DINOv2 vits14, vector-reshaped-to-image encoding)",
        "dataset": "HDGM-D", "level": "hard", "metric": "test power",
        "panel": "not in paper — exploratory cross-modal encoder test",
        "d": x_in, "N_planned": [8*n for n in n_list],
        "N_completed": [r["N"] for r in dinov2_result],
        "N_TRAIL": N_TRAIL, "dinov2_dim": DINOV2_DIM, "grid_side": side,
        "note": f"HDGM vectors reshaped into a zero-padded {side}x{side} grid, upsampled to 224x224, repeated to 3ch, embedded via frozen DINOv2. This vector-to-image encoding is one specific, arbitrary choice — the spatial structure DINOv2 attends over is an artifact of this reshape, not real structure in the data (unlike MNIST). Phase 1 runs per-trial since HDGM is freshly generated each trial. Phase 2 unchanged C2ST_NN_fit.",
        "results": dinov2_result,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/rl_c2st_dinov2_HDGM_d2_power.json", "w") as f:
        json.dump(meta, f, indent=2)

print("\n=== DONE ===", flush=True)
for r in dinov2_result:
    print(r, flush=True)
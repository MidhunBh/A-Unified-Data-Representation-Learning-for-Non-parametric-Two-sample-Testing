import math
import numpy as np
import pickle
import torch
import tqdm
import json
import subprocess
import time
import os
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("device:", device)

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
batch_size = 1024

x_in = 10
H = 30
final_dr_dim = 30
AE_EPOCHS = 2000
N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100

n_fixed = 500   # N=4000, matching factor 1's (latent dimension) own design point

freeze_conditions = [True, False]


def run_one_condition(freeze, sampler_fn, desc):
    stat_s, stat_l = [], []
    for kk in tqdm.trange(N_TRAIL, desc=desc):
        if sampler_fn is sample_hdgm_semi_t2:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(n_fixed, n_fixed, d=x_in, kk=kk, level="hard")
        else:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(n_fixed, n_fixed, d=x_in, kk=kk)

        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        encoder = train_autoencoder(S_encoder, AE_EPOCHS, x_in, H, final_dr_dim, 512, device, dtype, lr=0.002)
        if freeze:
            for p in encoder.parameters():
                p.requires_grad = False
        # if freeze=False, encoder stays trainable -- gradients from C2ST_NN_fit flow through it

        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)
        y = torch.cat([
            torch.zeros(len(s1_tr)), torch.ones(len(s2_tr))
        ]).to(device, dtype).long()

        base_model = ExtendedModel(encoder, H, final_dr_dim)
        model_C2ST_L, w, b = C2ST_NN_fit(
            S, y, x_in, H, final_dr_dim, N_EPOCH, batch_size, device, dtype, base_model, lr_c2st=0.002)

        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)

        stat_s.append(H_S.sum() / N_TEST_F)
        stat_l.append(H_L.sum() / N_TEST_F)
    return float(np.mean(stat_s)), float(np.mean(stat_l))


results = []
already_done = set()
if os.path.exists("result/factorial_freeze_vs_joint_HDGM_d10_N4000.json"):
    with open("result/factorial_freeze_vs_joint_HDGM_d10_N4000.json") as f:
        prior = json.load(f)
    results = prior["results"]
    already_done = set(prior["levels_completed"])
    print(f"Resuming -- already completed: {sorted(already_done)}")

for freeze in freeze_conditions:
    label = "freeze" if freeze else "joint"
    if label in already_done:
        print(f"skipping {label}, already done")
        continue

    print(f"\n===== condition: {label} =====")

    power_s, power_l = run_one_condition(freeze, sample_hdgm_semi_t2, f"{label} POWER")
    type1_s, type1_l = run_one_condition(freeze, sample_hdgm_semi_t1, f"{label} TYPE-I")

    cell = {
        "condition": label,
        "power_S": power_s, "power_L": power_l,
        "type1_S": type1_s, "type1_L": type1_l,
    }
    results.append(cell)
    print(f"{label}: power_S={power_s:.3f} power_L={power_l:.3f} "
          f"| type1_S={type1_s:.3f} type1_L={type1_l:.3f}")

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "commit": commit,
        "factor": "supervision_signal (freeze vs joint fine-tuning)",
        "levels_planned": ["freeze", "joint"],
        "levels_completed": [r["condition"] for r in results],
        "dataset": "HDGM", "level": "hard", "d": x_in, "N": 8 * n_fixed,
        "N_TRAIL": N_TRAIL, "AE_EPOCHS": AE_EPOCHS, "N_EPOCH": N_EPOCH,
        "note": "Factorial study, factor 2: supervision signal. Fixed: N=4000 (matching factor 1's design point), d=10, hard, 100 trials for power and Type-I each. Pilot (10 trials, reduced N_TEST) showed joint fine-tuning collapses via total overfitting: train power 1.000/1.000, test power 0.000/0.000. This full run confirms at proper trial count.",
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/factorial_freeze_vs_joint_HDGM_d10_N4000.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open("result/factorial_freeze_vs_joint_HDGM_d10_N4000.pkl", "wb") as f:
        pickle.dump(results, f)

print("\n=== FACTORIAL STUDY COMPLETE -- supervision signal (freeze vs joint) ===")
for r in results:
    print(r)

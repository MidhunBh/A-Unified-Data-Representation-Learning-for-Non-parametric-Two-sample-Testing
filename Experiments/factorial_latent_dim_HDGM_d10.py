import math
import numpy as np
import pickle
import torch
import tqdm
import json
import subprocess
import time
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

n_fixed = 500   # n_train=n_test=500 -> N=4000

x_out_list = [5, 10, 30, 50, 100]


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


def run_one_condition(x_out, sampler_fn, desc):
    stat_s, stat_l = [], []
    for kk in tqdm.trange(N_TRAIL, desc=desc):
        if sampler_fn is sample_hdgm_semi_t2:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(
                n_fixed, n_fixed, d=x_in, kk=kk, level="hard")
        else:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(
                n_fixed, n_fixed, d=x_in, kk=kk)

        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        encoder = train_autoencoder(S_encoder, AE_EPOCHS, x_in, H, x_out,
                                     512, device, dtype, lr=0.002)
        for p in encoder.parameters():
            p.requires_grad = False

        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)
        y = torch.cat([
            torch.zeros(len(s1_tr)), torch.ones(len(s2_tr))
        ]).to(device, dtype).long()

        base_model = ExtendedModelFixed(encoder, x_out, H, final_dr_dim)
        model_C2ST_L, w, b = C2ST_NN_fit(
            S, y, x_in, H, final_dr_dim, N_EPOCH, batch_size, device, dtype,
            base_model, lr_c2st=0.002)

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
for x_out in x_out_list:
    print(f"\n===== x_out (AE latent dim) = {x_out} =====")

    power_s, power_l = run_one_condition(
        x_out, sample_hdgm_semi_t2, f"x_out={x_out} POWER")
    type1_s, type1_l = run_one_condition(
        x_out, sample_hdgm_semi_t1, f"x_out={x_out} TYPE-I")

    cell = {
        "x_out": x_out,
        "power_S": power_s, "power_L": power_l,
        "type1_S": type1_s, "type1_L": type1_l,
    }
    results.append(cell)
    print(f"x_out={x_out}: power_S={power_s:.3f} power_L={power_l:.3f} "
          f"| type1_S={type1_s:.3f} type1_L={type1_l:.3f}")

    # save after every level — protects against overnight interruption
    meta = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip(),
        "factor": "latent_dimension (AE x_out)",
        "levels_planned": x_out_list,
        "levels_completed": [r["x_out"] for r in results],
        "dataset": "HDGM", "level": "hard", "d": x_in, "N": 8 * n_fixed,
        "N_TRAIL": N_TRAIL, "AE_EPOCHS": AE_EPOCHS, "N_EPOCH": N_EPOCH,
        "H_fixed": H, "final_dr_dim_fixed": final_dr_dim,
        "note": "Factorial study, factor 1: AE latent dimension. Fixed: frozen AE encoder, N=4000 (largest AE-vs-C2ST gap point from Fig3c), 100 trials for power and Type-I each. Classifier head width (H) and final DR dim held constant at 30 across all levels via ExtendedModelFixed, decoupling the varied factor from downstream architecture.",
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/factorial_latent_dim_HDGM_d10_N4000.json", "w") as f:
        json.dump(meta, f, indent=2)

print("\n=== FACTORIAL STUDY COMPLETE — latent dimension ===")
for r in results:
    print(r)
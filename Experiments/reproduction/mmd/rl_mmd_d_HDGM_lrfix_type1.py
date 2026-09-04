import math
import numpy as np
import pickle
import torch
import tqdm
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

x_in = 2
H = 30
x_out = 30

N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100

n_list = [125, 250, 500, 750, 1000, 1250]
LR_MMD = 0.000005   # corrected value, matches the power-side fix

mmd_baseline_result = []
for n in n_list:
    n_train = n
    n_test = n
    summary = []
    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n} TYPE-I"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t1(
            n_train, n_test, d=x_in, kk=kk)

        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        encoder = train_autoencoder(S_encoder, epoch=2000, x_in=x_in, H=H, x_out=x_out,
                            batch_size=512, device=device, dtype=dtype, lr=0.002)
        for param in encoder.parameters():
            param.requires_grad = False

        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)
        S_ir = encoder(S)
        model_mmd, sigma, sigma0, ep = MMD_D_fit(S_ir, x_out, H, x_out, N_EPOCH, device, dtype, lr_mmd=LR_MMD)

        H_MMD = np.zeros(N_TEST)
        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)
        S_test_ir = encoder(S_test)
        for k in range(N_TEST):
            H_MMD[k], _, _ = TST_MMD_u(
                model_mmd(S_test_ir), n_test*2, N_PER, S_test_ir, sigma, sigma0, ep, alpha, k * kk + 2024)

        summary.append(H_MMD.sum() / N_TEST_F)
    mmd_baseline_result.append(summary)
    print(f"N={8*n}: Type-I = {np.mean(summary):.3f}", flush=True)

    with open("result/rl_mmd_d_HDGM_lrfix_d2_type1.pkl", "wb") as file:
        pickle.dump(mmd_baseline_result, file)

    import json, time
    meta = {
        "method": "RL-MMD-D (lr_mmd corrected)",
        "dataset": "HDGM-S", "level": "hard", "metric": "type-I error", "panel": "Fig 4b",
        "d": x_in, "N": [8*n for n in n_list[:len(mmd_baseline_result)]],
        "N_TRAIL": N_TRAIL, "lr_mmd": LR_MMD,
        "note": "Type-I companion to the lr_mmd=5e-6 power fix. Original lr_mmd=5e-5 version was inflated above 0.05 at every N (0.059-0.083). Checking whether the retuned rate also resolves calibration, not just power.",
        "result": [float(np.mean(r)) for r in mmd_baseline_result],
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open("result/rl_mmd_d_HDGM_lrfix_d2_type1.json", "w") as f:
        json.dump(meta, f, indent=2)

print("=== DONE ===")

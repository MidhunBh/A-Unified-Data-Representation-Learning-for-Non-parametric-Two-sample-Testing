import numpy as np
import torch
import tqdm
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

x_in, H, x_out = 2, 30, 30
alpha = 0.05
N_PER, N_TEST = 100, 100
N_TRAIL = 30   # full rigor this time, not a scan

for n in [125, 250, 500, 750]:
    print(f"\n=== n={n} (N={8*n}), d=2 ===")

    # C2ST baseline -- no encoder at all
    print("-- C2ST (no encoder) --")
    summary_s_c2st, summary_l_c2st = [], []
    for kk in tqdm.trange(N_TRAIL, desc=f"C2ST n={n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")
        S = MatConvert(np.concatenate([s1_tr, s2_tr]), device, torch.float)
        y = torch.cat([torch.zeros(len(s1_tr)), torch.ones(len(s2_tr))]).to(device, torch.float).long()
        model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_in, H, x_out, 1000, 512, device, torch.float, model=None, lr_c2st=0.002)
        S_test = MatConvert(np.concatenate([s1_te, s2_te]), device, torch.float)
        H_S = np.zeros(N_TEST); H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
        summary_s_c2st.append(H_S.mean()); summary_l_c2st.append(H_L.mean())

    # RL-C2ST -- AE Phase1, frozen, same conditions
    print("-- RL-C2ST (AE, frozen) --")
    summary_s_rl, summary_l_rl = [], []
    for kk in tqdm.trange(N_TRAIL, desc=f"RL-C2ST n={n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")
        S_encoder = MatConvert(np.concatenate([s1_tr, s1_te, s2_tr, s2_te]), device, torch.float)
        encoder = train_autoencoder(S_encoder, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
        for p in encoder.parameters(): p.requires_grad = False
        S = MatConvert(np.concatenate([s1_tr, s2_tr]), device, torch.float)
        y = torch.cat([torch.zeros(len(s1_tr)), torch.ones(len(s2_tr))]).to(device, torch.float).long()
        base_model = ExtendedModel(encoder, H, x_out)
        model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_in, H, x_out, 1000, 512, device, torch.float, base_model, lr_c2st=0.002)
        S_test = MatConvert(np.concatenate([s1_te, s2_te]), device, torch.float)
        H_S = np.zeros(N_TEST); H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
        summary_s_rl.append(H_S.mean()); summary_l_rl.append(H_L.mean())

    print(f"\nn={n}: C2ST S={np.mean(summary_s_c2st):.3f} L={np.mean(summary_l_c2st):.3f}  |  RL-C2ST S={np.mean(summary_s_rl):.3f} L={np.mean(summary_l_rl):.3f}")

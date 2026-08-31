import numpy as np
import torch
import tqdm
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

x_in, H, x_out = 10, 30, 30
alpha = 0.05
n = 500  # N=4000
N_PER, N_TEST = 100, 20  # reduced for pilot speed

for freeze in [True, False]:
    print(f"\n=== freeze={freeze} ===")
    summary_s, summary_l = [], []
    for kk in tqdm.trange(10, desc=f"freeze={freeze}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")
        S_encoder = MatConvert(np.concatenate([s1_tr, s1_te, s2_tr, s2_te]), device, torch.float)
        encoder = train_autoencoder(S_encoder, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
        if freeze:
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
        summary_s.append(H_S.mean()); summary_l.append(H_L.mean())
    print(f"freeze={freeze}: power_S={np.mean(summary_s):.3f}  power_L={np.mean(summary_l):.3f}")

import numpy as np
import torch
import tqdm
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

x_in, H, x_out = 10, 30, 30
alpha = 0.05
n = 500
N_PER, N_TEST = 100, 20

print("=== freeze=False, checking TRAIN vs TEST power ===")
summary_train_s, summary_train_l = [], []
summary_test_s, summary_test_l = [], []

for kk in tqdm.trange(10, desc="freeze=False"):
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")
    S_encoder = MatConvert(np.concatenate([s1_tr, s1_te, s2_tr, s2_te]), device, torch.float)
    encoder = train_autoencoder(S_encoder, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
    # freeze=False: encoder stays trainable

    S = MatConvert(np.concatenate([s1_tr, s2_tr]), device, torch.float)
    y = torch.cat([torch.zeros(len(s1_tr)), torch.ones(len(s2_tr))]).to(device, torch.float).long()
    base_model = ExtendedModel(encoder, H, x_out)
    model_C2ST_L, w, b = C2ST_NN_fit(S, y, x_in, H, x_out, 1000, 512, device, torch.float, base_model, lr_c2st=0.002)

    # TRAIN power -- same S used to fit the classifier
    H_S_tr = np.zeros(N_TEST); H_L_tr = np.zeros(N_TEST)
    for k in range(N_TEST):
        H_S_tr[k], _, _ = TST_C2ST(S, len(s1_tr), N_PER, alpha, model_C2ST_L, w, b)
        H_L_tr[k], _, _ = TST_LCE(S, len(s1_tr), N_PER, alpha, model_C2ST_L, w, b)
    summary_train_s.append(H_S_tr.mean()); summary_train_l.append(H_L_tr.mean())

    # TEST power -- held out, matching the original pilot
    S_test = MatConvert(np.concatenate([s1_te, s2_te]), device, torch.float)
    H_S_te = np.zeros(N_TEST); H_L_te = np.zeros(N_TEST)
    for k in range(N_TEST):
        H_S_te[k], _, _ = TST_C2ST(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
        H_L_te[k], _, _ = TST_LCE(S_test, len(s1_te), N_PER, alpha, model_C2ST_L, w, b)
    summary_test_s.append(H_S_te.mean()); summary_test_l.append(H_L_te.mean())

print(f"\nTRAIN power: S={np.mean(summary_train_s):.3f}  L={np.mean(summary_train_l):.3f}")
print(f"TEST  power: S={np.mean(summary_test_s):.3f}  L={np.mean(summary_test_l):.3f}")
print("(train high + test ~0 = overfitting; both ~0 = training itself never worked)")

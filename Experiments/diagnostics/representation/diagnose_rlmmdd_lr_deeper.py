import numpy as np
import torch
import tqdm
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

x_in, H, x_out = 2, 30, 30
alpha = 0.05
n = 250
N_TRAIL = 15
N_TEST = 30
N_PER = 100

lr_mmd = 0.0000005
print(f"=== lr_mmd={lr_mmd}, N=2000, {N_TRAIL} trials ===")
outcomes = []
for kk in tqdm.trange(N_TRAIL, desc=f"lr_mmd={lr_mmd}"):
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=x_in, kk=kk, level="hard")

    S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
    S_encoder = MatConvert(S_encoder, device, torch.float)
    encoder = train_autoencoder(S_encoder, epoch=2000, x_in=x_in, H=H, x_out=x_out, batch_size=512, device=device, dtype=torch.float, lr=0.002)
    for p in encoder.parameters():
        p.requires_grad = False

    S = np.concatenate((s1_tr, s2_tr), axis=0)
    S = MatConvert(S, device, torch.float)
    S_ir = encoder(S)
    model_mmd, sigma, sigma0, ep = MMD_D_fit(S_ir, x_out, H, x_out, 1000, device, torch.float, lr_mmd=lr_mmd)

    S_test = np.concatenate((s1_te, s2_te), axis=0)
    S_test = MatConvert(S_test, device, torch.float)
    S_test_ir = encoder(S_test)

    H_MMD = np.zeros(N_TEST)
    for k in range(N_TEST):
        H_MMD[k], _, _ = TST_MMD_u(model_mmd(S_test_ir), n*2, N_PER, S_test_ir, sigma, sigma0, ep, alpha, k * kk + 2024)
    outcomes.append(H_MMD.mean())

print(f"lr_mmd={lr_mmd}: power={np.mean(outcomes):.3f}")
print("(compare: 5e-6 -> 0.462, 5e-5 -> 0.269, 5e-4 -> 0.229)")

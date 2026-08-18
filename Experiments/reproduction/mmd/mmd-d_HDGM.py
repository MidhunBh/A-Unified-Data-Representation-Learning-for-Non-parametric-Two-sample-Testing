import math
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import tqdm
import copy
from utils import *

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

# Setup seeds
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True

# Setup for experiments
dtype = torch.float
alpha = 0.05  # test threshold
batch_size = 1024  # batch size

x_in = 2  # number of neurons in the input layer, i.e., dimension of data
H = 30  # number of neurons in the hidden layer
x_out = 30  # number of neurons in the output layer

# learning_rate = 0.00005
# learning_rate = 0.00001

N_EPOCH = 1000  # number of training epochs
N_TRAIL = 100  # number of trails
N_TEST = 100  # number of test sets
N_TEST_F = 100.0  # number of test sets (float)0
N_PER = 100  # permutation times

n_list = [125, 250, 500, 750, 1000, 1250]
lr_c2st_list = [0.005, 0.002, 0.001]

mmd_baseline_result = []
for n in n_list:
    n_train = n
    n_test = n

    summary = []
    for kk in tqdm.trange(100):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train, n_test,d=2, kk=kk)
        
        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)

        model_mmd, sigma, sigma0, ep = MMD_D_fit(S, x_in, H, x_out, N_EPOCH, device, dtype, lr_mmd=0.00005)

        h, _, _ = TST_MMD_u(model_mmd(S), n_train*2, N_PER, S, sigma, sigma0, ep, alpha, kk * 4202)
        print("Training set power: ", h)
        

        # perform test
        H_MMD = np.zeros(N_TEST)
        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)
        for k in range(N_TEST):
            H_MMD[k], _, _ = TST_MMD_u(
                model_mmd(S_test), n_test*2, N_PER, S_test, sigma, sigma0, ep, alpha, k * kk + 2024) 

        print("Test Power of MMD-D: ", H_MMD.sum() / N_TEST_F)
        summary.append(H_MMD.sum() / N_TEST_F)

        # h1, _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
        # summary.append(h1)
    mmd_baseline_result.append(summary)
    print("Test power at training size = ", n, " : ", np.mean(summary))
    # break       
        
with open("result/mmd-d_HDGM_baseline_0.00005_d2.pkl", "wb") as file:
    pickle.dump(mmd_baseline_result, file)

import json, subprocess, time
meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"  : "MMD-D",
    "dataset" : "HDGM-D",
    "d"       : x_in,
    "n_list"  : n_list,
    "N_TRAIL" : N_TRAIL,
    "N_EPOCH" : N_EPOCH,
    "lr_mmd"  : 0.00005,
    "result"  : [float(np.mean(r)) for r in mmd_baseline_result],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/mmd-d_HDGM_baseline_0.00005_d2.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/mmd-d_HDGM_baseline_0.00005_d2.json")
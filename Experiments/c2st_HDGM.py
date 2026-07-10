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
batch_size = 2048  # batch size

x_in = 2  # number of neurons in the input layer, i.e., dimension of data
H = 30  # number of neurons in the hidden layer
x_out = 30  # number of neurons in the output layer

# learning_rate = 0.00005
learning_rate = 0.00001

N_EPOCH = 1000  # number of training epochs
N_TRAIL = 100  # number of trails
N_TEST = 100  # number of test sets
N_TEST_F = 100.0  # number of test sets (float)0
N_PER = 100  # permutation times

n_list = [125, 250, 500, 750, 1000, 1250]
lr_c2st_list = [0.005, 0.002, 0.001]

# for lr_c2st in lr_c2st_list:
c2st_baseline_result_t2 = []
for n in n_list:
    n_train = n
    n_test = n

    summary_s = []
    summary_l = []
    for kk in range(N_TRAIL):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=x_in, kk=kk, level="hard")
        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)

        y = torch.concat(
            (torch.zeros(n_train*2), torch.ones(n_train*2))).to(device, dtype).long()

        model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(
            S, y, x_in, H, x_out, N_EPOCH, batch_size, device, dtype, lr_c2st=0.005)

        # perform test
        H_C2ST_S = np.zeros([N_TEST])
        H_C2ST_L = np.zeros([N_TEST])
        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)

        for k in range(N_TEST):
            H_C2ST_S[k], _, _ = TST_C2ST(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            H_C2ST_L[k], _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            # H_C2ST_L[k], _, _ = TST_MMD_l(model_C2ST_L(S_test), n_train*2, N_PER, alpha, k)
        # print("Test Power of C2ST-L: ", H_C2ST_L.sum() / N_TEST_F)
        print(f"Test Power of C2ST-S at n = {n}: ", H_C2ST_S.sum() / N_TEST_F)
        print(f"Test Power of C2ST-L at n = {n}: ", H_C2ST_L.sum() / N_TEST_F)
        summary_s.append(H_C2ST_S.sum() / N_TEST_F)
        summary_l.append(H_C2ST_L.sum() / N_TEST_F)

        # h1, _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
        # summary.append(h1)
    c2st_baseline_result_t2.append((summary_s, summary_l))
    print("\n\n=====================================================")
    print("Average Test Power of C2ST-S: ", np.mean(summary_s))
    print("Average Test Power of C2ST-L: ", np.mean(summary_l))
    print("=====================================================\n\n")
    # break

with open('result/c2st_HDGM_d2_power.pkl', 'wb') as file:    # Use pickle.dump() to write the list to the file
    pickle.dump(c2st_baseline_result_t2, file)


import json, subprocess, time
meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"  : "C2ST + C2ST-L",
    "dataset" : "HDGM-D",
    "level"   : "hard",
    "metric"  : "test power",
    "panel"   : "Fig 3b + Fig 4a",
    "d"       : x_in,
    "N"       : [8*n for n in n_list],
    "N_TRAIL" : N_TRAIL,
    "N_EPOCH" : N_EPOCH,
    "result"  : [
        {"N": 8*n, "C2ST-S": float(np.mean(s)), "C2ST-L": float(np.mean(l))}
        for n, (s, l) in zip(n_list, c2st_baseline_result_t2)
    ],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/c2st_HDGM_d2_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/c2st_HDGM_d2_power.json")
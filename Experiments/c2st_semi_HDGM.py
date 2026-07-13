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
# N_EPOCH = 600  # number of training epochs
N_TRAIL = 100  # number of trails
N_TEST = 100  # number of test sets
N_TEST_F = 100.0  # number of test sets (float)0
N_PER = 100  # permutation times

n_list = [125, 250, 500, 750, 1000, 1250]
#n_list = [250, 375, 500]
# n_list = [7.5, 10, 12.5]
lr_c2st_list = [0.005, 0.002, 0.001]

# for lr_c2st in lr_c2st_list:
c2st_semi_baseline_result_t2 = []
for n in n_list:
    n_train = n
    n_test = n

    summary_l = []
    summary_s = []
    # summary_h = []
    for kk in range(N_TRAIL):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=x_in, kk=kk, level="hard")

        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        # epoch = np.int64(400 * 2000 / (n_train + n_test))
        epoch = 2000

        encoder = train_autoencoder(S_encoder, epoch, x_in, H, x_out, 512, device, dtype, lr=0.002)
        for param in encoder.parameters():
            param.requires_grad = False

        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)

        y = torch.concat(
            (torch.zeros(np.int64(n_train*2)), torch.ones(np.int64(n_train*2)))).to(device, dtype).long()

        base_model = ExtendedModel(encoder, H, x_out)
        model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(
            S, y, x_in, H, x_out, N_EPOCH, batch_size, device, dtype, base_model, lr_c2st=0.002)

        # perform test
        H_C2ST_S = np.zeros([N_TEST])
        H_C2ST_L = np.zeros([N_TEST])
        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)

        for k in range(N_TEST):
            H_C2ST_S[k], _, _ = TST_C2ST(S_test, np.int64(n_test*2), N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            H_C2ST_L[k], _, _ = TST_LCE(S_test, np.int64(n_test*2), N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            # H_C2ST_S[k], _, _ = TST_C2ST(S_test, 25, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            # H_C2ST_L[k], _, _ = TST_LCE(S_test, 25, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
            # H_C2ST_L[k], _, _ = TST_MMD_l(model_C2ST_L(S_test), n_train*2, N_PER, alpha, k)
        # print("Test Power of C2ST-L: ", H_C2ST_L.sum() / N_TEST_F)
        print(f"Test Power of C2ST-S at n = {n_train}: ", H_C2ST_S.sum() / N_TEST_F)
        print(f"Test Power of C2ST-L at n = {n_train}: ", H_C2ST_L.sum() / N_TEST_F)
        summary_s.append(H_C2ST_S.sum() / N_TEST_F)
        summary_l.append(H_C2ST_L.sum() / N_TEST_F)

        # h1, _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
        # summary.append(h1)
        print("\n\n=====================================================")
        print("Average Test Power of C2ST-S: ", np.mean(summary_s))
        print("Average Test Power of C2ST-L: ", np.mean(summary_l))
        print("=====================================================\n\n")

    c2st_semi_baseline_result_t2.append((np.mean(summary_s), np.mean(summary_l)))

print(c2st_semi_baseline_result_t2)
        
with open('result/c2st_semi_HDGM_d2_power.pkl', 'wb') as file:
    pickle.dump(c2st_semi_baseline_result_t2, file)

import json, subprocess, time
meta = {
    "commit"  : subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
                ).decode().strip(),
    "method"  : "RL-C2ST (AE)",
    "dataset" : "HDGM-D",
    "level"   : "hard",
    "metric"  : "test power",
    "panel"   : "Fig 3b",
    "d"       : x_in,
    "N"       : [8*n for n in n_list],
    "N_TRAIL" : N_TRAIL,
    "N_EPOCH" : N_EPOCH,
    "epochs_ae" : 2000,
    "result"  : [
        {"N": 8*n, "RL-C2ST-S": float(s), "RL-C2ST-L": float(l)}
        for n, (s, l) in zip(n_list, c2st_semi_baseline_result_t2)
    ],
    "ts"      : time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/c2st_semi_HDGM_d2_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/c2st_semi_HDGM_d2_power.json")

# print("\n\n=====================================================")
# print("Change model to just representation learning")
# print("=====================================================\n\n")

# c2st_semi_baseline_result_t2 = []
# for n in n_list:
#     n_train = n
#     n_test = n

#     summary = []
#     for kk in range(N_TRAIL):
#         s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=2, kk=kk)

#         S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
#         S_encoder = MatConvert(S_encoder, device, dtype)
#         epoch = np.int64(400 * 2000 / (n_train + n_test))

#         encoder = train_autoencoder(S_encoder, epoch, x_in, H, x_out, batch_size, device, dtype)
#         # for param in encoder.parameters():
#         #     param.requires_grad = False

#         S = np.concatenate((s1_tr, s2_tr), axis=0)
#         S = MatConvert(S, device, dtype)

#         y = torch.concat(
#             (torch.zeros(n_train*2), torch.ones(n_train*2))).to(device, dtype).long()

#         # base_model = ExtendedModel(encoder, H, x_out)
#         model_C2ST_L, w_C2ST_L, b_C2ST_L = C2ST_NN_fit(
#             S, y, x_in, H, x_out, N_EPOCH, batch_size, device, dtype, encoder, lr_c2st=0.005)

#         # perform test
#         H_C2ST_L = np.zeros(N_TEST)
#         S_test = np.concatenate((s1_te, s2_te), axis=0)
#         S_test = MatConvert(S_test, device, dtype)

#         for k in range(N_TEST):
#             H_C2ST_L[k], _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
#         print("Test Power of C2ST-L: ", H_C2ST_L.sum() / N_TEST_F)
#         summary.append(H_C2ST_L.sum() / N_TEST_F)

#         # h1, _, _ = TST_LCE(S_test, n_test*2, N_PER, alpha, model_C2ST_L, w_C2ST_L, b_C2ST_L)
#         # summary.append(h1)
#     c2st_semi_baseline_result_t2.append(summary)
#     print("\n\n=====================================================")
#     print("Average test power at training size = ", n_train, " : ", np.mean(summary))
#     print("=====================================================\n\n")
#     # break

# with open('result/unfreeze_c2st_HDGM_semi_1024_0.005_2000_unschedule_d2.pkl', 'wb') as file:
#     # Use pickle.dump() to write the list to the file
#     pickle.dump(c2st_semi_baseline_result_t2, file)
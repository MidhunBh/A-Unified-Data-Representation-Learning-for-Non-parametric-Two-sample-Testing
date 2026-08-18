import numpy as np
import subprocess
from mmd_fuse import *
from utils import sample_hdgm_semi_t1

alpha = 0.05
n_list = [125, 250]   # two smallest only
d = 2
N_TRAIL = 10           # tiny, hypothesis test not estimation

key = random.PRNGKey(42)

for n in n_list:
    n_train = n_test = n
    outputs = []
    for i in range(N_TRAIL):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t1(n_train, n_test, d=d, kk=i)
        S1 = np.concatenate((s1_tr, s1_te), axis=0)
        S2 = np.concatenate((s2_tr, s2_te), axis=0)

        key, subkey = random.split(key)
        out = mmdfuse(S1, S2, subkey)
        outputs.append(int(out))

    print(f"N={8*n}: raw trial outcomes = {outputs}")
    print(f"N={8*n}: mean = {np.mean(outputs):.3f}\n")
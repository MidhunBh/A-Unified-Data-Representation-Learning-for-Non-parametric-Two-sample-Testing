import numpy as np
from mmd_fuse import *
from utils import sample_hdgm_semi_t1

d = 2
n_train = n_test = 125

s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t1(n_train, n_test, d=d, kk=0)
S1 = np.concatenate((s1_tr, s1_te), axis=0)
S2 = np.concatenate((s2_tr, s2_te), axis=0)

key42 = random.PRNGKey(42)
key123 = random.PRNGKey(123)
_, subkey42 = random.split(key42)
_, subkey123 = random.split(key123)

print("subkey42:", subkey42)
print("subkey123:", subkey123)
print("subkeys equal?", bool((subkey42 == subkey123).all()))

out42 = mmdfuse(S1, S2, subkey42)
out123 = mmdfuse(S1, S2, subkey123)
print("output with subkey42:", out42)
print("output with subkey123:", out123)

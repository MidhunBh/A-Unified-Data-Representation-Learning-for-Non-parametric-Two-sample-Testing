import numpy as np
from utils import *

for n_train, n_test in [(150,350), (200,300), (250,250), (300,200), (350,150)]:
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=10, kk=0, level="hard")
    total = len(s1_tr)+len(s1_te)+len(s2_tr)+len(s2_te)
    print(f"n_train={n_train}, n_test={n_test}: s1_tr={len(s1_tr)} s1_te={len(s1_te)} TOTAL_N={total}")

import numpy as np
from utils import *

for n_train, n_test in [(300,700), (400,600), (500,500), (600,400), (700,300)]:
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n_train, n_test, d=10, kk=0, level="hard")
    print(f"n_train={n_train}, n_test={n_test}: s1_tr={len(s1_tr)} s1_te={len(s1_te)} total_budget={n_train+n_test}")

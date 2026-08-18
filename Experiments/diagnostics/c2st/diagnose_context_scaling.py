import numpy as np
import torch
from tabpfn import TabPFNClassifier

def TST_LCE_from_probs(probs, N1, N_per, alpha, rd_seed=0):
    N = probs.shape[0]
    STAT = abs(probs[:N1, 0].mean() - probs[N1:, 0].mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        np.random.seed(1102 + r * 3 + rd_seed)
        ind = np.random.choice(N, N, replace=False)
        ind_X, ind_Y = ind[:N1], ind[N1:]
        STAT_vector[r] = abs(probs[ind_X, 0].mean() - probs[ind_Y, 0].mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[int(np.ceil(N_per * (1 - alpha)))]
    return (1 if STAT > threshold else 0), threshold, STAT

device = "cuda" if torch.cuda.is_available() else "cpu"

# Modest, non-obvious signal (unlike the earlier toy diagnostic's huge gap) —
# sized to resemble N=10000's per-class train pool (~2500)
np.random.seed(0)
d = 30
n_pool = 2500
mean_shift = 0.15   # small — this is the point

X_p = np.random.randn(n_pool, d)
X_q = np.random.randn(n_pool, d)
X_q[:, 0] += mean_shift

clf = TabPFNClassifier(device=device)

for ctx_per_class in [250, 500]:   # 500/class = our driver's cap; test if more helps
    idx_p = np.random.choice(n_pool, ctx_per_class, replace=False)
    idx_q = np.random.choice(n_pool, ctx_per_class, replace=False)
    X_context = np.concatenate([X_p[idx_p], X_q[idx_q]]).astype(np.float32)
    y_context = np.concatenate([np.zeros(ctx_per_class), np.ones(ctx_per_class)])

    X_query = np.concatenate([X_p[-500:], X_q[-500:]]).astype(np.float32)

    clf.fit(X_context, y_context)
    probs = clf.predict_proba(X_query)
    sep = abs(probs[:500, 0].mean() - probs[500:, 0].mean())

    h_list = [TST_LCE_from_probs(probs, 500, 100, 0.05, rd_seed=0)[0] for _ in range(20)]
    print(f"context={ctx_per_class}/class: separation={sep:.4f}, mean h over 20 reps={np.mean(h_list):.2f}")
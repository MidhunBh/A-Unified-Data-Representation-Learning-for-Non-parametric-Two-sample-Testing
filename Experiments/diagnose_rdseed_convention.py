import numpy as np

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

np.random.seed(0)
N1, N2 = 100, 100
probs = np.zeros((N1 + N2, 2))
probs[:N1, 0] = np.random.uniform(0.85, 0.99, size=N1)   # strongly P-like
probs[N1:, 0] = np.random.uniform(0.01, 0.15, size=N2)   # strongly Q-like

N_PER, alpha, N_TEST = 100, 0.05, 100

h_const = [TST_LCE_from_probs(probs, N1, N_PER, alpha, rd_seed=0)[0] for _ in range(N_TEST)]
print("Constant rd_seed=0 (original repo convention), mean h:", np.mean(h_const))

h_varying = [TST_LCE_from_probs(probs, N1, N_PER, alpha, rd_seed=k)[0] for k in range(N_TEST)]
print("Varying rd_seed=k (what my driver did), mean h:", np.mean(h_varying))
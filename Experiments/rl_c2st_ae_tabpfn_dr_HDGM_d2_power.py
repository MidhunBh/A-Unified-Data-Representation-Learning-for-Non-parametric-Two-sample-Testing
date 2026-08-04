import math
import numpy as np
import pickle
import torch
import tqdm
import json
import subprocess
import time
from tabpfn import TabPFNClassifier
from utils import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("device:", device)

np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True

dtype = torch.float
alpha = 0.05
x_in = 2
H = 30
x_out = 30

AE_EPOCHS = 2000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [125, 250, 500, 750, 1000, 1250]

CONTEXT_SIZE = 500


def TST_LCE_from_probs(probs, N1, N_per, alpha, rd_seed=0):
    """Mirrors paper Eq.2 / repo's TST_LCE, but consumes precomputed
    probabilities (TabPFN's predict_proba) instead of a model+weights triple."""
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


def TST_C2ST_from_probs(probs, N1, N_per, alpha, rd_seed=0):
    """Mirrors paper Eq.1 / repo's TST_C2ST."""
    N = probs.shape[0]
    pred = probs.argmax(axis=1)
    STAT = abs(pred[:N1].mean() - pred[N1:].mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        np.random.seed(1102 + r * 3 + rd_seed)
        ind = np.random.choice(N, N, replace=False)
        ind_X, ind_Y = ind[:N1], ind[N1:]
        STAT_vector[r] = abs(pred[ind_X].mean() - pred[ind_Y].mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[int(np.ceil(N_per * (1 - alpha)))]
    return (1 if STAT > threshold else 0), threshold, STAT


tabpfn_clf = TabPFNClassifier(device=device)


def tabpfn_discriminate(IR_s1_tr, IR_s2_tr, IR_s1_te, IR_s2_te, kk):
    """Phase 2 — TabPFN used natively: real labels, stratified capped context,
    full test-split query. This is Phase 2, so labels are legitimate here."""
    n1, n2 = len(IR_s1_tr), len(IR_s2_tr)
    half_cap = CONTEXT_SIZE // 2

    np.random.seed(1102 * kk + 11)
    idx_p = np.random.choice(n1, min(half_cap, n1), replace=False)
    idx_q = np.random.choice(n2, min(half_cap, n2), replace=False)

    X_context = np.concatenate([
        IR_s1_tr[idx_p].cpu().numpy(), IR_s2_tr[idx_q].cpu().numpy()
    ], axis=0).astype(np.float32)
    y_context = np.concatenate([np.zeros(len(idx_p)), np.ones(len(idx_q))])

    tabpfn_clf.fit(X_context, y_context)

    X_query = np.concatenate([
        IR_s1_te.cpu().numpy(), IR_s2_te.cpu().numpy()
    ], axis=0).astype(np.float32)

    probs = tabpfn_clf.predict_proba(X_query)
    return probs, len(IR_s1_te)


tabpfn_dr_result = []
for n in n_list:
    n_train = n
    n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train, n_test, d=x_in, kk=kk, level="hard")

        # Phase 1 — AE, unchanged from validated c2st_semi_HDGM.py run
        S_encoder = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        S_encoder = MatConvert(S_encoder, device, dtype)
        encoder = train_autoencoder(S_encoder, AE_EPOCHS, x_in, H, x_out,
                                     512, device, dtype, lr=0.002)
        for p in encoder.parameters():
            p.requires_grad = False

        with torch.no_grad():
            IR_s1_tr = encoder(MatConvert(s1_tr, device, dtype))
            IR_s1_te = encoder(MatConvert(s1_te, device, dtype))
            IR_s2_tr = encoder(MatConvert(s2_tr, device, dtype))
            IR_s2_te = encoder(MatConvert(s2_te, device, dtype))

        # Phase 2 — TabPFN discriminator on AE's IR, real labels
        probs, N1 = tabpfn_discriminate(IR_s1_tr, IR_s2_tr, IR_s1_te, IR_s2_te, kk)

        # Phase 3 — permutation test on TabPFN's output
        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST_from_probs(probs, N1, N_PER, alpha, rd_seed=k)
            H_L[k], _, _ = TST_LCE_from_probs(probs, N1, N_PER, alpha, rd_seed=k)

        summary_s.append(H_S.sum() / N_TEST_F)
        summary_l.append(H_L.sum() / N_TEST_F)

    tabpfn_dr_result.append((summary_s, summary_l))
    print(f"N={8*n}: RL-S(AE+TabPFN-DR)={np.mean(summary_s):.3f}  "
          f"RL-L(AE+TabPFN-DR)={np.mean(summary_l):.3f}")

with open("result/rl_c2st_ae_tabpfn_dr_HDGM_d2_power.pkl", "wb") as f:
    pickle.dump(tabpfn_dr_result, f)

meta = {
    "commit": subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
    ).decode().strip(),
    "method": "RL-C2ST (AE Phase1 + TabPFN Phase2 discriminator)",
    "dataset": "HDGM-D", "level": "hard", "metric": "test power",
    "panel": "Fig 3b — novel Phase2 pretrained-discriminator design",
    "d": x_in, "N": [8*n for n in n_list], "N_TRAIL": N_TRAIL,
    "context_size": CONTEXT_SIZE,
    "note": "Phase 1 = unchanged from-scratch AE, identical to validated c2st_semi_HDGM_d2_power run. Phase 2 = TabPFN v1 used natively (real labels, stratified 500-pt context under 1024 cap, full test-split query) replacing ModelLatentF/C2ST_NN_fit. Custom TST_*_from_probs functions mirror paper Eq.1/Eq.2 exactly, adapted to consume TabPFN's precomputed probabilities.",
    "result": [{"N": 8*n, "RL-C2ST-S": float(np.mean(s)), "RL-C2ST-L": float(np.mean(l))}
               for n, (s, l) in zip(n_list, tabpfn_dr_result)],
    "ts": time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/rl_c2st_ae_tabpfn_dr_HDGM_d2_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged")
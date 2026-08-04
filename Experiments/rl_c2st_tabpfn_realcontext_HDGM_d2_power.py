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
batch_size = 1024

x_in = 2
H = 30
x_out = 30

N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [125, 250, 500, 750, 1000, 1250]

TABPFN_HIDDEN = 512
CONTEXT_SIZE = 500   # verified safely under TabPFN v1's 1024 cap

tabpfn_clf = TabPFNClassifier(device=device)


def tabpfn_extract_realcontext(S_pool_np, kk):
    """Phase 1 — context = real random subsample of this trial's pool
    (verified: 500 pts, well under the 1024 hard cap). Query = full real
    pool. Placeholder labels are parity-based on the SUBSAMPLE's own
    index order, carrying no information about true P/Q identity."""
    n_pool = len(S_pool_np)
    np.random.seed(1102 * kk + 7)   # reproducible per-trial subsample
    ctx_idx = np.random.choice(n_pool, min(CONTEXT_SIZE, n_pool), replace=False)
    X_context = S_pool_np[ctx_idx].astype(np.float32)
    y_placeholder = np.arange(len(ctx_idx)) % 2

    tabpfn_clf.fit(X_context, y_placeholder)

    captured = {}
    def hook(module, input):
        captured['ir'] = input[0].detach()
    handle = tabpfn_clf.model[2].decoder.register_forward_pre_hook(hook)

    with torch.no_grad():
        _ = tabpfn_clf.predict_proba(S_pool_np.astype(np.float32))
    handle.remove()

    ctx_len = len(ctx_idx)
    ir_full = captured['ir']                    # (ctx_len + n_pool, ensemble, 512)
    ir_query = ir_full[ctx_len:].mean(dim=1)     # query half, avg over ensemble axis
    return ir_query.to(device, dtype)


tabpfn_baseline_result = []
for n in n_list:
    n_train = n
    n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train, n_test, d=x_in, kk=kk, level="hard")

        S_pool = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        IR_pool = tabpfn_extract_realcontext(S_pool, kk)

        n1_tr, n1_te = len(s1_tr), len(s1_te)
        n2_tr = len(s2_tr)
        IR_s1_tr = IR_pool[:n1_tr]
        IR_s1_te = IR_pool[n1_tr:n1_tr + n1_te]
        IR_s2_tr = IR_pool[n1_tr + n1_te: n1_tr + n1_te + n2_tr]
        IR_s2_te = IR_pool[n1_tr + n1_te + n2_tr:]

        S_train = torch.cat([IR_s1_tr, IR_s2_tr], dim=0)
        y = torch.cat([
            torch.zeros(n1_tr), torch.ones(n2_tr)
        ]).to(device, dtype).long()

        model_C2ST_L, w, b = C2ST_NN_fit(
            S_train, y, TABPFN_HIDDEN, H, x_out, N_EPOCH, batch_size,
            device, dtype, model=None, lr_c2st=0.002)

        S_test = torch.cat([IR_s1_te, IR_s2_te], dim=0)

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)
        for k in range(N_TEST):
            H_S[k], _, _ = TST_C2ST(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)
            H_L[k], _, _ = TST_LCE(S_test, len(IR_s1_te), N_PER, alpha, model_C2ST_L, w, b)

        summary_s.append(H_S.sum() / N_TEST_F)
        summary_l.append(H_L.sum() / N_TEST_F)

    tabpfn_baseline_result.append((summary_s, summary_l))
    print(f"N={8*n}: RL-C2ST-S(TabPFN-realctx)={np.mean(summary_s):.3f}  "
          f"RL-C2ST-L(TabPFN-realctx)={np.mean(summary_l):.3f}")

with open("result/rl_c2st_tabpfn_realcontext_HDGM_d2_power.pkl", "wb") as f:
    pickle.dump(tabpfn_baseline_result, f)

meta = {
    "commit": subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
    ).decode().strip(),
    "method": "RL-C2ST (TabPFN v1, real-subsample context)",
    "dataset": "HDGM-D", "level": "hard", "metric": "test power",
    "panel": "Fig 3b — 4th bar, corrected extraction",
    "d": x_in, "N": [8*n for n in n_list], "N_TRAIL": N_TRAIL,
    "context_size": CONTEXT_SIZE,
    "note": "Corrects earlier dummy-context run. Context = random 500-pt subsample of this trial's real pool (parity-labeled placeholders, capped under TabPFN v1's 1024 limit). Query = full real pool. Compare against rl_c2st_tabpfn_HDGM_d2_power.json (dummy context, negative) and c2st_semi_HDGM_d2_power.json (AE).",
    "result": [{"N": 8*n, "RL-C2ST-S": float(np.mean(s)), "RL-C2ST-L": float(np.mean(l))}
               for n, (s, l) in zip(n_list, tabpfn_baseline_result)],
    "ts": time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/rl_c2st_tabpfn_realcontext_HDGM_d2_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged")
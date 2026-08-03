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

x_in = 2          # matching Fig 3b — d=2
H = 30
x_out = 30         # same latent width as your AE/WAE runs, for direct comparability

N_EPOCH = 1000
N_TRAIL = 100
N_TEST = 100
N_TEST_F = 100.0
N_PER = 100
n_list = [125, 250, 500, 750, 1000, 1250]

TABPFN_HIDDEN = 512   # confirmed from inspect_transformer.py

# --- Fixed dummy context, built once, reused for every extraction call.
# Never touches real HDGM data — keeps every extracted IR label-blind and
# free of any leakage from this run's actual P/Q samples.
np.random.seed(42)
X_context = np.random.randn(10, x_in).astype(np.float32)
y_context = np.array([0] * 5 + [1] * 5)

tabpfn_clf = TabPFNClassifier(device=device)
tabpfn_clf.fit(X_context, y_context)
transformer = tabpfn_clf.model[2]

_captured = {}
def _capture_hook(module, input):
    _captured['ir'] = input[0].detach()
_hook_handle = transformer.decoder.register_forward_pre_hook(_capture_hook)
N_CONTEXT = len(X_context)


def tabpfn_extract(X_query_np):
    """Frozen Phase 1 — one forward pass through TabPFN v1, no training.
    Returns per-sample embeddings for X_query_np, shape (n_query, 512).
    Averages over TabPFN's internal ensemble axis (dim 1)."""
    with torch.no_grad():
        _ = tabpfn_clf.predict_proba(X_query_np.astype(np.float32))
    ir_full = _captured['ir']            # (n_context + n_query, ensemble, 512)
    ir_query = ir_full[N_CONTEXT:]       # confirmed: context first, query last
    ir_query = ir_query.mean(dim=1)      # collapse ensemble axis
    return ir_query.to(device, dtype)


tabpfn_baseline_result = []
for n in n_list:
    n_train = n
    n_test = n
    summary_s, summary_l = [], []

    for kk in tqdm.trange(N_TRAIL, desc=f"N={8*n}"):
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train, n_test, d=x_in, kk=kk, level="hard")

        # Phase 1 — frozen TabPFN embeddings, pooled unlabelled data
        # (same pooling the AE sees: train+test, both distributions, no labels)
        S_pool = np.concatenate((s1_tr, s1_te, s2_tr, s2_te), axis=0)
        IR_pool = tabpfn_extract(S_pool)

        n1_tr, n1_te = len(s1_tr), len(s1_te)
        n2_tr = len(s2_tr)
        IR_s1_tr = IR_pool[:n1_tr]
        IR_s1_te = IR_pool[n1_tr:n1_tr + n1_te]
        IR_s2_tr = IR_pool[n1_tr + n1_te: n1_tr + n1_te + n2_tr]
        IR_s2_te = IR_pool[n1_tr + n1_te + n2_tr:]

        # Phase 2 — SAME C2ST_NN_fit as your plain baseline, just x_in=512
        # instead of x_in=2. No custom wrapper needed — TabPFN's frozen
        # embedding IS the input; ModelLatentF(512,H,x_out) is the one
        # trainable layer stack on top of it.
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
    print(f"N={8*n}: RL-C2ST-S(TabPFN)={np.mean(summary_s):.3f}  "
          f"RL-C2ST-L(TabPFN)={np.mean(summary_l):.3f}")

_hook_handle.remove()

with open("result/rl_c2st_tabpfn_HDGM_d2_power.pkl", "wb") as f:
    pickle.dump(tabpfn_baseline_result, f)

meta = {
    "commit": subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
    ).decode().strip(),
    "method": "RL-C2ST (TabPFN v1 encoder)",
    "dataset": "HDGM-D", "level": "hard", "metric": "test power",
    "panel": "Fig 3b — 4th bar (encoder-type factor), not in paper",
    "d": x_in, "N": [8*n for n in n_list], "N_TRAIL": N_TRAIL,
    "note": "Phase 1 = frozen TabPFN v1 in-context transformer embeddings (512-dim, decoder-input tap, fixed dummy context, no training loop). Phase 2 = unmodified C2ST_NN_fit with x_in=512, identical to plain-baseline call. Directly comparable to c2st_HDGM_d2_power.json (no encoder) and c2st_semi_HDGM_d2_power.json (AE encoder).",
    "result": [{"N": 8*n, "RL-C2ST-S": float(np.mean(s)), "RL-C2ST-L": float(np.mean(l))}
               for n, (s, l) in zip(n_list, tabpfn_baseline_result)],
    "ts": time.strftime("%Y-%m-%d %H:%M"),
}
with open("result/rl_c2st_tabpfn_HDGM_d2_power.json", "w") as f:
    json.dump(meta, f, indent=2)
print("logged to result/rl_c2st_tabpfn_HDGM_d2_power.json")
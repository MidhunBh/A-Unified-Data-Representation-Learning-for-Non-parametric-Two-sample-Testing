import numpy as np
import pickle
import json
import subprocess
import time
from mmd_fuse import *
from utils import sample_hdgm_semi_t1

alpha = 0.05
n_list = [125, 250, 500, 750, 1000, 1250]
d = 2
N_TRAIL = 100
MASTER_SEED = 123

results = []
key = random.PRNGKey(MASTER_SEED)

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

    type1 = float(np.mean(outputs))
    results.append({"N": 8*n, "type1": type1})
    print(f"N={8*n}: Type-I={type1:.3f}", flush=True)

    with open(f"result/mmdfuse_HDGM_type1_fixed_seed{MASTER_SEED}.pkl", "wb") as f:
        pickle.dump(results, f)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=r"C:\Users\midhu\Documents\GitHub\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
        ).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "commit": commit, "method": "MMD-FUSE",
        "dataset": "HDGM-S", "level": "hard", "metric": "type-I error",
        "panel": "Fig 4b - corrected seeding",
        "d": d, "master_seed": MASTER_SEED,
        "N_planned": [8*n for n in n_list], "N_completed": [r["N"] for r in results],
        "N_TRAIL": N_TRAIL,
        "note": "Fixes seeding artifact: kk=subkey previously passed a JAX key into np.random.seed (confirmed from sample_hdgm_semi_t1 source: np.random.seed(1102*kk)). Now: plain integer trial index to sampler, separate split JAX key to mmdfuse only. 10-trial diagnostic passed before this run.",
        "results": results, "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(f"result/mmdfuse_HDGM_type1_fixed_seed{MASTER_SEED}.json", "w") as f:
        json.dump(meta, f, indent=2)

print("=== DONE ===", flush=True)
for r in results:
    print(r, flush=True)

import numpy as np
import pickle
import json
import subprocess
import time

from mmd_fuse import *
from tqdm import tqdm
from utils import sample_hdgm_semi_t1


alpha = 0.05
n_list = [125, 250, 500, 750, 1000, 1250]
d = 2
N_TRAIL = 100
MASTER_SEED = 42

results = []

for n in n_list:
    n_train = n_test = n

    # Match the released author notebook:
    # restart the JAX RNG at seed 42 for every sample size.
    key = random.PRNGKey(MASTER_SEED)

    outputs = []

    for i in tqdm(
        range(N_TRAIL),
        desc=f"MMD-FUSE H0 N={8*n}",
        unit="trial"
    ):
        # Match author notebook RNG choreography.
        key, subkey = random.split(key)

        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t1(
            n_train,
            n_test,
            d=d,
            kk=subkey
        )

        S1 = np.concatenate((s1_tr, s1_te), axis=0)
        S2 = np.concatenate((s2_tr, s2_te), axis=0)

        key, subkey = random.split(key)
        out = mmdfuse(S1, S2, subkey)

        outputs.append(int(out))

    type1 = float(np.mean(outputs))

    results.append({
        "N": 8 * n,
        "type1": type1
    })

    print(f"N={8*n}: Type-I={type1:.3f}", flush=True)

    # Save partial progress after every completed sample size.
    with open(
        "result/mmdfuse_HDGM_type1_authorstyle_nfix.pkl",
        "wb"
    ) as f:
        pickle.dump(results, f)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=(
                r"C:\Users\midhu\Documents\GitHub"
                r"\A-Unified-Data-Representation-Learning-for-Non-parametric-Two-sample-Testing"
            )
        ).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "commit": commit,
        "method": "MMD-FUSE",
        "dataset": "HDGM-S",
        "level": "hard",
        "metric": "type-I error",
        "panel": "Fig 4b - author-style RNG, sample-size fix only",
        "d": d,
        "master_seed": MASTER_SEED,
        "N_planned": [8 * n for n in n_list],
        "N_completed": [r["N"] for r in results],
        "N_TRAIL": N_TRAIL,
        "note": (
            "Forensic reconstruction of the released HDGM MMD-FUSE Type-I "
            "notebook. Only the sample-size bug is corrected: "
            "n_train=n_test=n rather than 500. The notebook's original RNG "
            "choreography is preserved: the JAX key is reset to MASTER_SEED "
            "for each N, a split subkey is passed as kk to the HDGM null "
            "sampler, and a second split subkey is passed to mmdfuse."
        ),
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }

    with open(
        "result/mmdfuse_HDGM_type1_authorstyle_nfix.json",
        "w"
    ) as f:
        json.dump(meta, f, indent=2)


print("=== DONE ===", flush=True)

for r in results:
    print(r, flush=True)
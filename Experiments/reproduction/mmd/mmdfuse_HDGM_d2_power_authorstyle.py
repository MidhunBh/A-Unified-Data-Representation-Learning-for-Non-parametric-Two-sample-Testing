import os
import numpy as np
import pickle
import json
import subprocess
import time

from jax import random
from mmd_fuse import mmdfuse
from tqdm import tqdm
from utils import sample_hdgm_semi_t2


# ============================================================
# MMD-FUSE HDGM d=2 power reconstruction
# Figure 4(a)
# ============================================================

alpha = 0.05

# Internal sample sizes.
# Paper Figure 4 uses N = 8 * n for this HDGM construction.
n_list = [125, 250, 500, 750, 1000, 1250]

d = 2
N_TRAIL = 100
MASTER_SEED = 42

results = []

os.makedirs("result", exist_ok=True)


for n in n_list:

    n_train = n
    n_test = n

    # Match the released author MMD-FUSE notebook:
    # restart the JAX RNG at seed 42 for every sample size.
    key = random.PRNGKey(MASTER_SEED)

    outputs = []

    for i in tqdm(
        range(N_TRAIL),
        desc=f"MMD-FUSE H1 d=2 N={8*n}",
        unit="trial"
    ):

        # ----------------------------------------------------
        # Author notebook RNG choreography
        # ----------------------------------------------------
        key, subkey = random.split(key)

        # H1 / HDGM-D:
        # P and Q differ in covariance structure.
        #
        # IMPORTANT FIX:
        # d=2 is passed explicitly.
        # The previous notebook power run omitted this argument,
        # causing the sampler default d=10 to be used.
        s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
            n_train,
            n_test,
            d=d,
            kk=i
        )

        # MMD-FUSE is a non-training two-sample test, so use all
        # generated observations from each distribution.
        S1 = np.concatenate((s1_tr, s1_te), axis=0)
        S2 = np.concatenate((s2_tr, s2_te), axis=0)

        key, subkey = random.split(key)

        out = mmdfuse(
            S1,
            S2,
            subkey,
            alpha=alpha
        )

        outputs.append(int(out))

    power = float(np.mean(outputs))

    results.append({
        "internal_n": n,
        "paper_N": 8 * n,
        "power": power
    })

    print(
        f"N={8*n}: MMD-FUSE power={power:.3f}",
        flush=True
    )

    # --------------------------------------------------------
    # Save partial progress after every completed sample size
    # --------------------------------------------------------
    with open(
        "result/mmdfuse_HDGM_d2_power_authorstyle.pkl",
        "wb"
    ) as f:
        pickle.dump(results, f)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "not recorded: local git CLI unavailable"

    meta = {
        "commit": commit,
        "method": "MMD-FUSE",
        "dataset": "HDGM-D",
        "level": "hard",
        "metric": "test power",
        "panel": "Figure 4(a)",
        "d": d,
        "alpha": alpha,
        "master_seed": MASTER_SEED,
        "internal_n_planned": n_list,
        "paper_N_planned": [8 * x for x in n_list],
        "paper_N_completed": [r["paper_N"] for r in results],
        "N_TRAIL": N_TRAIL,
        "note": (
            "Reconstruction of the HDGM MMD-FUSE power experiment for Figure 4(a). "
            "The released notebook power loop used the correct changing sample sizes "
            "but omitted d=2 when calling sample_hdgm_semi_t2, therefore silently "
            "using the sampler default d=10. This run explicitly sets d=2. "
            "For compatibility with the current NumPy/JAX environment, the HDGM "
            "sampler uses the outer trial index i as its integer seed (kk=i), while "
            "the original two-split JAX RNG sequence is retained and the second "
            "subkey is passed to mmdfuse. No MMD-FUSE hyperparameters are tuned."
        ),
        "results": results,
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }

    with open(
        "result/mmdfuse_HDGM_d2_power_authorstyle.json",
        "w"
    ) as f:
        json.dump(meta, f, indent=2)


print("\n=== DONE ===", flush=True)

for r in results:
    print(r, flush=True)
import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

# Make Experiments/utils.py importable regardless of launch directory.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from utils import *  # noqa: E402,F403


# ---------------------------------------------------------------------------
# Fixed Tian-style configuration
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dtype = torch.float
alpha = 0.05
batch_size = 1024

x_in = 10
H = 30
x_out = 30

AE_EPOCHS = 2000
N_EPOCH = 1000
N_TRAIL = 100
N_PER = 100

np.random.seed(1102)
torch.manual_seed(1102)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1102)
torch.backends.cudnn.deterministic = True


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_DIR),
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_DIR),
        ).decode().strip()
        return bool(out)
    except Exception:
        return None


def run_one_condition(n_train, n_test, sampler_fn, desc):
    """
    One independent two-sample decision per outer trial.

    IMPORTANT:
    The previous factorial_split_ratio_HDGM_d10.py allocated N_TEST=100
    and called TST_C2ST/TST_LCE 100 times on the same model and same
    held-out data.

    Those test functions deterministically reset their permutation RNG,
    so the 100 calls were identical duplicates. Averaging them therefore
    reduced to the same single binary decision.

    This corrected runner makes exactly one permutation-test decision
    per independent outer trial while retaining N_PER=100 permutations.
    """

    stat_s = []
    stat_l = []

    for kk in tqdm.trange(N_TRAIL, desc=desc):

        if sampler_fn is sample_hdgm_semi_t2:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(
                n_train,
                n_test,
                d=x_in,
                kk=kk,
                level="hard",
            )
        else:
            s1_tr, s1_te, s2_tr, s2_te = sampler_fn(
                n_train,
                n_test,
                d=x_in,
                kk=kk,
            )

        # Preserve Tian/reproduction protocol:
        # unsupervised AE sees pooled train + held-out covariates.
        S_encoder = np.concatenate(
            (s1_tr, s1_te, s2_tr, s2_te),
            axis=0,
        )
        S_encoder = MatConvert(S_encoder, device, dtype)

        encoder = train_autoencoder(
            S_encoder,
            AE_EPOCHS,
            x_in,
            H,
            x_out,
            512,
            device,
            dtype,
            lr=0.002,
        )

        # Matched HDGM d=10 RL-C2ST regime: frozen encoder.
        for p in encoder.parameters():
            p.requires_grad = False

        # Supervised P/Q training split.
        S = np.concatenate((s1_tr, s2_tr), axis=0)
        S = MatConvert(S, device, dtype)

        y = torch.cat(
            [
                torch.zeros(len(s1_tr)),
                torch.ones(len(s2_tr)),
            ]
        ).to(device, dtype).long()

        base_model = ExtendedModel(encoder, H, x_out)

        model_C2ST_L, w, b = C2ST_NN_fit(
            S,
            y,
            x_in,
            H,
            x_out,
            N_EPOCH,
            batch_size,
            device,
            dtype,
            base_model,
            lr_c2st=0.002,
        )

        # Independent held-out testing split.
        S_test = np.concatenate((s1_te, s2_te), axis=0)
        S_test = MatConvert(S_test, device, dtype)

        # One test decision per outer trial.
        # rd_seed=0 explicitly preserves the permutation semantics of
        # the original split-ratio implementation.
        h_s, _, _ = TST_C2ST(
            S_test,
            len(s1_te),
            N_PER,
            alpha,
            model_C2ST_L,
            w,
            b,
            rd_seed=0,
        )

        h_l, _, _ = TST_LCE(
            S_test,
            len(s1_te),
            N_PER,
            alpha,
            model_C2ST_L,
            w,
            b,
            rd_seed=0,
        )

        stat_s.append(int(h_s))
        stat_l.append(int(h_l))

    return {
        "S": stat_s,
        "L": stat_l,
        "mean_S": float(np.mean(stat_s)),
        "mean_L": float(np.mean(stat_l)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    args = parser.parse_args()

    n_train = args.n_train
    n_test = args.n_test

    if n_train + n_test != 1000:
        raise ValueError(
            f"Expected n_train+n_test=1000, got {n_train+n_test}"
        )

    label = f"{n_train}_{n_test}"

    print("=" * 72)
    print("HDGM d=10 fixed-budget split-ratio sensitivity")
    print(f"split       : {label}")
    print(f"device      : {device}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"git commit  : {git_commit()}")
    print(f"git dirty   : {git_dirty()}")
    print(f"AE epochs   : {AE_EPOCHS}")
    print(f"C2ST epochs : {N_EPOCH}")
    print(f"outer trials: {N_TRAIL}")
    print(f"permutations: {N_PER}")
    print("=" * 72, flush=True)

    t0 = time.time()

    power = run_one_condition(
        n_train,
        n_test,
        sample_hdgm_semi_t2,
        f"{label} POWER",
    )

    type1 = run_one_condition(
        n_train,
        n_test,
        sample_hdgm_semi_t1,
        f"{label} TYPE-I",
    )

    elapsed = time.time() - t0

    cell = {
        "split": label,
        "n_train": n_train,
        "n_test": n_test,
        "power_S": power["mean_S"],
        "power_L": power["mean_L"],
        "type1_S": type1["mean_S"],
        "type1_L": type1["mean_L"],
    }

    meta = {
        "schema_version": 2,
        "factor": "data_splitting (train/test ratio)",
        "dataset": "HDGM",
        "level": "hard",
        "d": x_in,
        "total_budget": 1000,
        "git_commit": git_commit(),
        "git_dirty_at_start": git_dirty(),
        "N_TRAIL": N_TRAIL,
        "AE_EPOCHS": AE_EPOCHS,
        "N_EPOCH": N_EPOCH,
        "N_PER": N_PER,
        "encoder_phase2": "frozen",
        "c2st_lr": 0.002,
        "permutation_rd_seed": 0,
        "legacy_N_TEST_loop_removed": True,
        "legacy_note": (
            "Old runner repeated the identical deterministic permutation "
            "test 100 times on each outer trial. This runner performs one "
            "test decision per independent outer trial; N_PER=100 is unchanged."
        ),
        "cell": cell,
        "raw_power_S": power["S"],
        "raw_power_L": power["L"],
        "raw_type1_S": type1["S"],
        "raw_type1_L": type1["L"],
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    outdir = EXPERIMENTS_DIR / "result" / "split_ratio_cells"
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / f"split_{label}.json"
    pkl_path = outdir / f"split_{label}.pkl"

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    with open(pkl_path, "wb") as f:
        pickle.dump(meta, f)

    print()
    print("=" * 72)
    print("RESULT")
    print(json.dumps(cell, indent=2))
    print(f"elapsed hours: {elapsed / 3600:.2f}")
    print(f"saved: {json_path}")
    print(f"saved: {pkl_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm

from utils import (
    C2ST_NN_fit,
    MatConvert,
    TST_C2ST,
    TST_LCE,
    sample_hdgm_semi_t2,
)


# ============================================================
# Matched RAW nonlinear-head control
#
# Scientific question:
#
# Is the encouraging frozen-WAE result
#
#     WAE z=2 (frozen)
#          ->
#     trainable 2 -> 30 -> 30 -> 30 head
#
# actually due to the WAE representation?
#
# Control:
#
#     RAW HDGM x in R^2
#          ->
#     SAME trainable 2 -> 30 -> 30 -> 30 head
#
# Everything else is kept at the current N=2000 recipe.
# ============================================================


MASTER_SEED = 1102

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dtype = torch.float

print("device:", device, flush=True)


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

d = 2
H = 30
FEATURE_DIM = 30

internal_n = 250
paper_N = 8 * internal_n

TRIAL_START = 0
TRIAL_END = 30

N_EPOCH = 1000
BATCH_SIZE = 1024
C2ST_LR = 0.005

N_PER = 100
ALPHA = 0.05


# ============================================================
# IMPORTANT
#
# This is intentionally the SAME trainable head architecture
# used in:
#
# pilot_wae_z2_lam001_d2_frozen_mlphead.py
#
# except there is no WAE in front of it.
#
# So we compare:
#
# WAE:
#   x -> frozen WAE encoder -> z(2)
#     -> this trainable MLP -> 30 features -> C2ST
#
# RAW:
#   x(2)
#     -> this trainable MLP -> 30 features -> C2ST
#
# ============================================================

class RawMatchedMLPHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, FEATURE_DIM, bias=True),
        )

    def forward(self, x):
        return self.head(x)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(MASTER_SEED)
    torch.cuda.manual_seed_all(MASTER_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# Run same outer HDGM trials 0..29
# ------------------------------------------------------------

records = []

bar = tqdm.trange(
    TRIAL_START,
    TRIAL_END,
    desc="RAW matched-head d=2 N=2000",
)

for kk in bar:

    # Exact same HDGM alternative and trial index convention.
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
        internal_n,
        internal_n,
        d=d,
        kk=kk,
        level="hard",
    )

    # Training data.
    S_train_np = np.concatenate(
        (s1_tr, s2_tr),
        axis=0,
    )

    S_train = MatConvert(
        S_train_np,
        device,
        dtype,
    )

    y = torch.cat(
        (
            torch.zeros(len(s1_tr)),
            torch.ones(len(s2_tr)),
        )
    ).to(device, dtype).long()

    # --------------------------------------------------------
    # Reset initialization before constructing the matched head.
    #
    # We deliberately use the same initialization seed for each
    # outer trial, matching the deterministic convention used
    # throughout the Tian reconstruction.
    # --------------------------------------------------------

    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(MASTER_SEED)

    model = RawMatchedMLPHead().to(device, dtype)

    model_c2st, w_c2st, b_c2st = C2ST_NN_fit(
        S_train,
        y,
        d,
        H,
        FEATURE_DIM,
        N_EPOCH,
        BATCH_SIZE,
        device,
        dtype,
        model,
        lr_c2st=C2ST_LR,
    )

    # Held-out test data.
    S_test_np = np.concatenate(
        (s1_te, s2_te),
        axis=0,
    )

    S_test = MatConvert(
        S_test_np,
        device,
        dtype,
    )

    n1_test = len(s1_te)

    # --------------------------------------------------------
    # One permutation test per outer dataset is sufficient.
    #
    # In the older scripts the same call was made 100 times
    # with the same rd_seed=0, giving the identical rejection
    # decision 100 times.
    #
    # Therefore this is computationally equivalent for the
    # outer-trial power estimate, but avoids redundant work.
    # --------------------------------------------------------

    h_s, threshold_s, stat_s = TST_C2ST(
        S_test,
        n1_test,
        N_PER,
        ALPHA,
        model_c2st,
        w_c2st,
        b_c2st,
        rd_seed=0,
    )

    h_l, threshold_l, stat_l = TST_LCE(
        S_test,
        n1_test,
        N_PER,
        ALPHA,
        model_c2st,
        w_c2st,
        b_c2st,
        rd_seed=0,
    )

    record = {
        "trial": int(kk),
        "reject_S": int(h_s),
        "reject_L": int(h_l),
        "stat_S": float(stat_s),
        "threshold_S": float(threshold_s),
        "stat_L": float(stat_l),
        "threshold_L": float(threshold_l),
    }

    records.append(record)

    current_s = np.mean(
        [r["reject_S"] for r in records]
    )

    current_l = np.mean(
        [r["reject_L"] for r in records]
    )

    bar.set_postfix(
        S=f"{current_s:.3f}",
        L=f"{current_l:.3f}",
    )


# ============================================================
# Summary
# ============================================================

power_s = float(
    np.mean([r["reject_S"] for r in records])
)

power_l = float(
    np.mean([r["reject_L"] for r in records])
)

summary = {
    "experiment": "RAW matched nonlinear-head control",
    "scientific_control_for": (
        "frozen WAE z=2 lambda=0.01 + nonlinear 30-D head"
    ),
    "dataset": "HDGM",
    "level": "hard",
    "d": d,
    "internal_n": internal_n,
    "paper_N": paper_N,
    "trial_start": TRIAL_START,
    "trial_end": TRIAL_END,
    "n_outer_trials": len(records),
    "phase1": "NONE / RAW INPUT",
    "phase2_head": "2 -> 30 Softplus -> 30 Softplus -> 30",
    "phase2_epochs": N_EPOCH,
    "phase2_lr": C2ST_LR,
    "batch_size": BATCH_SIZE,
    "N_PER": N_PER,
    "RL-C2ST-S": power_s,
    "RL-C2ST-L": power_l,
    "records": records,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}


os.makedirs("result", exist_ok=True)

json_path = (
    "result/"
    "diagnose_raw_matched_mlphead_HDGM_d2_N2000_30trials.json"
)

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)


print()
print("====================================================")
print("RAW MATCHED-HEAD CONTROL COMPLETE")
print("====================================================")
print()
print(f"paper N      : {paper_N}")
print(f"outer trials : {len(records)}")
print()
print(f"RAW-S = {power_s:.3f}")
print(f"RAW-L = {power_l:.3f}")
print()
print("Reference:")
print("  WAE z2 frozen + nonlinear head : S=0.467  L=0.833")
print("  historical vanilla C2ST        : S~0.470 L~0.830")
print("  historical AE                  : S~0.600 L~0.930")
print()
print("Saved:")
print(f"  {json_path}")

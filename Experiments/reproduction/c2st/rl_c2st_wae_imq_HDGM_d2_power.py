import argparse
import json
import os
import pickle
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from utils import (
    C2ST_NN_fit,
    MatConvert,
    TST_C2ST,
    TST_LCE,
    sample_hdgm_semi_t2,
)


# ============================================================
# RL-C2ST(WAE-MMD/IMQ) — HDGM d=2
#
# Reconstruction of the WAE representation-learning branch.
#
# WAE provenance:
#   Tolstikhin et al. WAE-MMD:
#       reconstruction loss
#       + lambda * MMD(Q_Z, N(0,I))
#
#   MMD kernel:
#       multi-scale inverse multiquadratic (IMQ)
#
# Historical Tian HDGM d=2 RL-C2ST recipe retained for:
#   - representation-learning epoch schedule
#   - batch size
#   - representation-learning LR
#   - raw encoder passed into phase 2
#   - encoder remains unfrozen
#   - C2ST LR
#   - C2ST training epochs
#
# IMPORTANT:
# Tian et al. did not release their WAE implementation.
# Therefore the WAE-MMD/IMQ implementation is a reconstruction
# based on the cited WAE reference, not recovered Tian source.
# ============================================================


# ------------------------------------------------------------
# Command-line options
#
# These will later allow us to split:
#   trials 0..49  -> GPU 0
#   trials 50..99 -> GPU 1
#
# without changing the statistical experiment.
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--trial-start",
    type=int,
    default=0,
    help="First outer trial index, inclusive.",
)

parser.add_argument(
    "--trial-end",
    type=int,
    default=100,
    help="Last outer trial index, exclusive.",
)

parser.add_argument(
    "--shard-id",
    type=str,
    default="full",
    help="Unique output suffix, e.g. gpu0 or gpu1.",
)

parser.add_argument(
    "--lambda-mmd",
    type=float,
    default=10.0,
    help=(
        "WAE MMD penalty weight. Default 10 follows the "
        "canonical Tolstikhin WAE-MMD MNIST configuration; "
        "this value is NOT known to be Tian's exact setting."
    ),
)

parser.add_argument(
    "--n-values",
    type=str,
    default="125,250,500,750,1000,1250",
    help="Comma-separated internal n values.",
)

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume completed trials from this shard's checkpoint.",
)

args = parser.parse_args()


# ============================================================
# Global experiment configuration
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("device:", device, flush=True)


MASTER_SEED = 1102

np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(MASTER_SEED)
    torch.cuda.manual_seed_all(MASTER_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dtype = torch.float

alpha = 0.05

# HDGM d=2
x_in = 2

# Match the released AE architecture
H = 30
x_out = 30

# Historical author-style d=2 settings
batch_size = 1024
WAE_LR = 0.002

N_EPOCH = 1000
C2ST_LR = 0.005

N_TEST = 100
N_PER = 100

N_TEST_F = float(N_TEST)

n_list = [
    int(x.strip())
    for x in args.n_values.split(",")
    if x.strip()
]

LAMBDA_MMD = args.lambda_mmd

IMQ_SCALES = (
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)


# ============================================================
# Output paths
# ============================================================

os.makedirs("result", exist_ok=True)

output_stem = (
    f"rl_c2st_wae_imq_HDGM_d2_power_{args.shard_id}"
)

pkl_path = f"result/{output_stem}.pkl"
json_path = f"result/{output_stem}.json"


# ============================================================
# Git provenance
# ============================================================

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "not recorded: git unavailable"


GIT_COMMIT = get_git_commit()


# ============================================================
# Atomic checkpoint helpers
#
# Write to temporary file first so an interrupted job does not
# leave a half-written JSON/PKL.
# ============================================================

def atomic_pickle_dump(obj, path):
    tmp_path = path + ".tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)

    os.replace(tmp_path, path)


def atomic_json_dump(obj, path):
    tmp_path = path + ".tmp"

    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)

    os.replace(tmp_path, path)


# ============================================================
# Architecture-matched Wasserstein Autoencoder
#
# The encoder is intentionally the same shape as Tian's
# released standard AutoEncoder:
#
# input
#   -> H -> Softplus
#   -> H -> Softplus
#   -> H -> Softplus
#   -> latent
#
# No BatchNorm is inserted.
# ============================================================

class WAE(nn.Module):

    def __init__(self, x_in, H, x_out):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_in, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, x_out, bias=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(x_out, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, x_in, bias=True),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, z


# ============================================================
# Correct squared Euclidean distance matrix
# ============================================================

def pairwise_squared_distance(A, B):

    A_norm = torch.sum(A * A, dim=1, keepdim=True)

    B_norm = torch.sum(
        B * B,
        dim=1,
        keepdim=True,
    ).transpose(0, 1)

    D = (
        A_norm
        + B_norm
        - 2.0 * torch.mm(A, B.transpose(0, 1))
    )

    return torch.clamp(D, min=0.0)


# ============================================================
# Canonical multi-scale IMQ MMD
#
# Tolstikhin WAE-MMD:
#
#     k(x,y) = C / (C + ||x-y||^2)
#
# For Gaussian prior N(0,I):
#
#     C_base = 2 * latent_dim
#
# and the reference WAE implementation sums kernels at:
#
#     0.1, 0.2, 0.5, 1, 2, 5, 10
#
# IMPORTANT:
#
# This uses ONE mathematically consistent kernel for:
#
#     Qz-Qz
#     Pz-Pz
#     Qz-Pz
#
# and uses the full cross-sample term.
#
# This replaces the earlier repo MMDl(..., kernel="rbf"),
# whose independently estimated XX/YY/XY bandwidths make it
# unsuitable as the WAE latent matching penalty.
# ============================================================

def imq_mmd(
    sample_qz,
    sample_pz,
    latent_dim,
):

    n_q = sample_qz.shape[0]
    n_p = sample_pz.shape[0]

    if n_q < 2 or n_p < 2:
        raise ValueError(
            "IMQ-MMD requires at least two samples "
            "from both Qz and Pz."
        )

    D_qq = pairwise_squared_distance(
        sample_qz,
        sample_qz,
    )

    D_pp = pairwise_squared_distance(
        sample_pz,
        sample_pz,
    )

    D_qp = pairwise_squared_distance(
        sample_qz,
        sample_pz,
    )

    # N(0, I) prior => prior variance = 1
    C_base = 2.0 * latent_dim

    mmd = sample_qz.new_tensor(0.0)

    for scale in IMQ_SCALES:

        C = C_base * scale

        K_qq = C / (C + D_qq)
        K_pp = C / (C + D_pp)
        K_qp = C / (C + D_qp)

        # Unbiased within-sample terms.
        qq = (
            K_qq.sum()
            - torch.diagonal(K_qq).sum()
        ) / (
            n_q * (n_q - 1)
        )

        pp = (
            K_pp.sum()
            - torch.diagonal(K_pp).sum()
        ) / (
            n_p * (n_p - 1)
        )

        # Full cross-sample term.
        qp = K_qp.mean()

        mmd = mmd + qq + pp - 2.0 * qp

    return mmd


# ============================================================
# Train WAE representation learner
#
# Phase 1 remains completely label-free.
#
# It receives:
#
# P_train + P_test + Q_train + Q_test
#
# exactly as the author's AE representation-learning pipeline.
# ============================================================

def train_wae(
    S,
    epochs,
    device,
    dtype,
    lambda_mmd,
):

    # Match Tian AE initialization behavior.
    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(MASTER_SEED)

    model = WAE(
        x_in,
        H,
        x_out,
    ).to(device, dtype)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=WAE_LR,
    )

    dataset = torch.utils.data.TensorDataset(S)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    final_recon = np.nan
    final_mmd = np.nan
    final_total = np.nan

    for epoch in range(epochs):

        recon_epoch = 0.0
        mmd_epoch = 0.0
        total_epoch = 0.0
        n_batches = 0

        for (x_batch,) in loader:

            reconstruction, z = model(x_batch)

            # Match the ordinary AE reconstruction objective.
            recon_loss = F.mse_loss(
                reconstruction,
                x_batch,
            )

            # Target aggregated latent prior:
            # N(0, I)
            z_prior = torch.randn_like(z)

            mmd_loss = imq_mmd(
                z,
                z_prior,
                latent_dim=x_out,
            )

            loss = (
                recon_loss
                + lambda_mmd * mmd_loss
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite WAE loss detected: "
                    f"epoch={epoch}, "
                    f"recon={recon_loss.item()}, "
                    f"mmd={mmd_loss.item()}"
                )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            recon_epoch += recon_loss.item()
            mmd_epoch += mmd_loss.item()
            total_epoch += loss.item()

            n_batches += 1

        final_recon = recon_epoch / n_batches
        final_mmd = mmd_epoch / n_batches
        final_total = total_epoch / n_batches

    return (
        model,
        final_recon,
        final_mmd,
        final_total,
    )


# ============================================================
# Historical d=2 epoch schedule
#
# Recovered from author logs:
#
# n=125  -> 3200
# n=250  -> 1600
# n=500  -> 800
# n=750  -> 533
# n=1000 -> 400
# n=1250 -> 320
#
# Formula:
#
# 400 * 2000 / (n_train + n_test)
# ============================================================

def representation_epochs(n_train, n_test):

    return int(
        400 * 2000
        / (n_train + n_test)
    )


# ============================================================
# Result summarization
# ============================================================

def summarize_records(records):

    summary = []

    for n in n_list:

        rows = [
            r
            for r in records
            if r["internal_n"] == n
        ]

        if not rows:
            continue

        power_s = float(
            np.mean(
                [r["trial_power_s"] for r in rows]
            )
        )

        power_l = float(
            np.mean(
                [r["trial_power_l"] for r in rows]
            )
        )

        summary.append(
            {
                "internal_n": n,
                "paper_N": 8 * n,
                "n_outer_trials": len(rows),
                "RL-C2ST-S": power_s,
                "RL-C2ST-L": power_l,
            }
        )

    return summary


# ============================================================
# Metadata / checkpoint
# ============================================================

def build_payload(records):

    return {
        "commit": GIT_COMMIT,

        "method": "RL-C2ST (WAE-MMD/IMQ)",

        "dataset": "HDGM-D",

        "level": "hard",

        "metric": "test power",

        "target_panel": "Figure 3(b)",

        "d": x_in,

        "representation": {
            "type": "WAE-MMD",
            "encoder": (
                "architecture matched to released Tian AE"
            ),
            "prior": "N(0,I)",
            "kernel": "multi-scale IMQ",
            "imq_scales": list(IMQ_SCALES),
            "C_base": 2 * x_out,
            "lambda_mmd": LAMBDA_MMD,
            "lambda_provenance": (
                "Default lambda=10 follows the canonical "
                "Tolstikhin WAE-MMD MNIST configuration. "
                "Tian's exact WAE lambda is not available."
            ),
            "reconstruction_loss": "PyTorch MSELoss",
            "learning_rate": WAE_LR,
            "batch_size": batch_size,
            "epoch_schedule": (
                "int(400*2000/(n_train+n_test))"
            ),
            "batchnorm": False,
            "warmup": False,
        },

        "phase2": {
            "model": (
                "raw WAE encoder passed directly to C2ST_NN_fit"
            ),
            "encoder_frozen": False,
            "epochs": N_EPOCH,
            "batch_size": batch_size,
            "learning_rate": C2ST_LR,
        },

        "testing": {
            "alpha": alpha,
            "N_TEST": N_TEST,
            "N_PER": N_PER,
            "statistics": [
                "C2ST-S",
                "C2ST-L",
            ],
        },

        "outer_trials": {
            "requested_start": args.trial_start,
            "requested_end": args.trial_end,
            "shard_id": args.shard_id,
        },

        "internal_n_planned": n_list,

        "paper_N_planned": [
            8 * n
            for n in n_list
        ],

        "note": (
            "Reconstructed WAE branch because Tian et al. "
            "did not release WAE source code or WAE logs. "
            "The WAE formulation follows Tolstikhin et al.'s "
            "WAE-MMD construction with a standard Gaussian "
            "prior and multi-scale IMQ kernel. "
            "The surrounding HDGM d=2 RL-C2ST training recipe "
            "is matched to the recovered historical author "
            "configuration: adaptive representation-training "
            "epochs, batch size 1024, raw encoder passed into "
            "phase 2, encoder unfrozen, and C2ST LR=0.005. "
            "This script intentionally does not use the repo's "
            "MMDl RBF helper because its separately estimated "
            "XX/YY/XY bandwidths do not define a single common "
            "RBF kernel suitable for WAE prior matching."
        ),

        "summary": summarize_records(records),

        "trial_records": records,

        "ts": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


def save_checkpoint(records):

    payload = build_payload(records)

    atomic_pickle_dump(
        payload,
        pkl_path,
    )

    atomic_json_dump(
        payload,
        json_path,
    )


# ============================================================
# Resume support
# ============================================================

trial_records = []

if args.resume and os.path.exists(pkl_path):

    with open(pkl_path, "rb") as f:
        existing = pickle.load(f)

    trial_records = existing.get(
        "trial_records",
        [],
    )

    print(
        f"Resuming {len(trial_records)} "
        f"completed trial records.",
        flush=True,
    )


completed = {
    (
        r["internal_n"],
        r["outer_trial"],
    )
    for r in trial_records
}


# ============================================================
# Main experiment
# ============================================================

for n in n_list:

    n_train = n
    n_test = n

    wae_epochs = representation_epochs(
        n_train,
        n_test,
    )

    trial_ids = [
        kk
        for kk in range(
            args.trial_start,
            args.trial_end,
        )
        if (n, kk) not in completed
    ]

    progress = tqdm.tqdm(
        trial_ids,
        desc=(
            f"WAE-IMQ d=2 "
            f"N={8*n} "
            f"epochs={wae_epochs}"
        ),
        unit="trial",
        dynamic_ncols=True,
    )

    for kk in progress:

        trial_start_time = time.time()

        # ----------------------------------------------------
        # Generate HDGM-D alternative data
        # ----------------------------------------------------

        (
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ) = sample_hdgm_semi_t2(
            n_train,
            n_test,
            d=x_in,
            kk=kk,
            level="hard",
        )

        # ----------------------------------------------------
        # Phase 1:
        # label-free pooled WAE representation learning
        # ----------------------------------------------------

        S_encoder = np.concatenate(
            (
                s1_tr,
                s1_te,
                s2_tr,
                s2_te,
            ),
            axis=0,
        )

        S_encoder = MatConvert(
            S_encoder,
            device,
            dtype,
        )

        (
            wae,
            final_recon,
            final_mmd,
            final_wae_loss,
        ) = train_wae(
            S_encoder,
            epochs=wae_epochs,
            device=device,
            dtype=dtype,
            lambda_mmd=LAMBDA_MMD,
        )

        # ----------------------------------------------------
        # Phase 2:
        #
        # Historical author-style RL-C2ST path.
        #
        # IMPORTANT:
        # Do NOT freeze encoder.
        # Do NOT pre-transform the inputs and train a new NN.
        #
        # Pass the raw encoder directly into C2ST_NN_fit so
        # it is fine-tuned jointly with the classifier.
        # ----------------------------------------------------

        S_train = np.concatenate(
            (
                s1_tr,
                s2_tr,
            ),
            axis=0,
        )

        S_train = MatConvert(
            S_train,
            device,
            dtype,
        )

        n1_train = s1_tr.shape[0]
        n2_train = s2_tr.shape[0]

        y = torch.cat(
            (
                torch.zeros(n1_train),
                torch.ones(n2_train),
            )
        ).to(
            device,
            dtype,
        ).long()

        encoder = wae.encoder

        (
            model_C2ST,
            w_C2ST,
            b_C2ST,
        ) = C2ST_NN_fit(
            S_train,
            y,
            x_in,
            H,
            x_out,
            N_EPOCH,
            batch_size,
            device,
            dtype,
            encoder,
            lr_c2st=C2ST_LR,
        )

        # ----------------------------------------------------
        # Phase 3:
        # held-out two-sample testing
        # ----------------------------------------------------

        S_test = np.concatenate(
            (
                s1_te,
                s2_te,
            ),
            axis=0,
        )

        S_test = MatConvert(
            S_test,
            device,
            dtype,
        )

        N1_test = s1_te.shape[0]

        H_S = np.zeros(N_TEST)
        H_L = np.zeros(N_TEST)

        for k in range(N_TEST):

            H_S[k], _, _ = TST_C2ST(
                S_test,
                N1_test,
                N_PER,
                alpha,
                model_C2ST,
                w_C2ST,
                b_C2ST,
            )

            H_L[k], _, _ = TST_LCE(
                S_test,
                N1_test,
                N_PER,
                alpha,
                model_C2ST,
                w_C2ST,
                b_C2ST,
            )

        trial_power_s = float(
            H_S.sum() / N_TEST_F
        )

        trial_power_l = float(
            H_L.sum() / N_TEST_F
        )

        elapsed = (
            time.time()
            - trial_start_time
        )

        record = {
            "internal_n": n,
            "paper_N": 8 * n,
            "outer_trial": kk,
            "wae_epochs": wae_epochs,
            "final_reconstruction_loss": (
                float(final_recon)
            ),
            "final_imq_mmd": (
                float(final_mmd)
            ),
            "final_wae_loss": (
                float(final_wae_loss)
            ),
            "trial_power_s": trial_power_s,
            "trial_power_l": trial_power_l,
            "elapsed_seconds": float(elapsed),
        }

        trial_records.append(record)

        completed.add(
            (n, kk)
        )

        # Save after EVERY completed outer trial.
        save_checkpoint(
            trial_records
        )

        progress.set_postfix(
            S=f"{trial_power_s:.0f}",
            L=f"{trial_power_l:.0f}",
            recon=f"{final_recon:.3g}",
            mmd=f"{final_mmd:.3g}",
        )

    current_rows = [
        r
        for r in trial_records
        if r["internal_n"] == n
    ]

    if current_rows:

        current_s = np.mean(
            [
                r["trial_power_s"]
                for r in current_rows
            ]
        )

        current_l = np.mean(
            [
                r["trial_power_l"]
                for r in current_rows
            ]
        )

        print(
            f"N={8*n}: "
            f"RL-S(WAE-IMQ)="
            f"{current_s:.3f}  "
            f"RL-L(WAE-IMQ)="
            f"{current_l:.3f}",
            flush=True,
        )


# ============================================================
# Final checkpoint and report
# ============================================================

save_checkpoint(
    trial_records
)

print(
    "\n=== DONE ===",
    flush=True,
)

for row in summarize_records(
    trial_records
):
    print(
        row,
        flush=True,
    )

print(
    f"\nSaved:\n"
    f"  {pkl_path}\n"
    f"  {json_path}",
    flush=True,
)
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MatConvert, sample_hdgm_semi_t2


# ============================================================
# WAE transfer sanity check: HDGM d=10
#
# Question:
# Does the locked d=2 Phase-1 choice lambda=0.01 transfer
# sensibly to d=10 when latent dimension is set to z=d=10?
#
# NO downstream C2ST power is used here.
#
# Compare only:
#   lambda = 0       -> reconstruction-only reference
#   lambda = 0.01    -> transferred WAE setting
#
# Design point:
#   d = 10
#   internal n = 100
#   paper N = 4 * n * d = 4000
# ============================================================


MASTER_SEED = 1102

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dtype = torch.float

print("device:", device)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

D = 10
LATENT_DIM = 10
H = 30

N = 100
PAPER_N = 4 * N * D

BATCH_SIZE = 1024
LR = 0.002

# Historical d=10 AE representation-training duration.
# This check asks whether the transferred z=10, lambda=.01 WAE
# already achieves good prior matching after 2000 epochs.
EPOCHS = 2000

LAMBDAS = [0.01]
DATA_SEEDS = [0, 1, 2]

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
# WAE
# ============================================================

class WAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(D, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, LATENT_DIM, bias=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, H, bias=True),
            nn.Softplus(),

            nn.Linear(H, D, bias=True),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ============================================================
# IMQ-MMD
# ============================================================

def pairwise_squared_distance(A, B):

    A_norm = (A * A).sum(
        dim=1,
        keepdim=True,
    )

    B_norm = (B * B).sum(
        dim=1,
        keepdim=True,
    ).t()

    D2 = (
        A_norm
        + B_norm
        - 2.0 * A.mm(B.t())
    )

    return torch.clamp(D2, min=0.0)


def imq_mmd(qz, pz):

    nq = qz.shape[0]
    np_ = pz.shape[0]

    Dqq = pairwise_squared_distance(qz, qz)
    Dpp = pairwise_squared_distance(pz, pz)
    Dqp = pairwise_squared_distance(qz, pz)

    C_base = 2.0 * LATENT_DIM

    total = qz.new_tensor(0.0)

    for scale in IMQ_SCALES:

        C = C_base * scale

        Kqq = C / (C + Dqq)
        Kpp = C / (C + Dpp)
        Kqp = C / (C + Dqp)

        qq = (
            Kqq.sum()
            - torch.diagonal(Kqq).sum()
        ) / (
            nq * (nq - 1)
        )

        pp = (
            Kpp.sum()
            - torch.diagonal(Kpp).sum()
        ) / (
            np_ * (np_ - 1)
        )

        qp = Kqp.mean()

        total = total + qq + pp - 2.0 * qp

    return total


# ============================================================
# Effective covariance rank
# ============================================================

def effective_rank(z_np):

    cov = np.cov(
        z_np,
        rowvar=False,
    )

    eigvals = np.linalg.eigvalsh(cov)

    eigvals = np.clip(
        eigvals,
        0.0,
        None,
    )

    total = eigvals.sum()

    if total <= 0:
        return 0.0

    p = eigvals / total

    p = p[p > 1e-12]

    entropy = -np.sum(
        p * np.log(p)
    )

    return float(
        np.exp(entropy)
    )


# ============================================================
# Training
# ============================================================

def train_one(S, lambda_mmd):

    # Important:
    # same WAE initialization for lambda=0 and lambda=.01
    # and same convention as the locked d=2 pipeline.
    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(MASTER_SEED)
        torch.cuda.manual_seed_all(MASTER_SEED)

    model = WAE().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    dataset = torch.utils.data.TensorDataset(S)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    final_recon = None
    final_train_mmd = None

    for epoch in range(EPOCHS):

        for (xb,) in loader:

            x_hat, z = model(xb)

            recon = F.mse_loss(
                x_hat,
                xb,
            )

            prior = torch.randn_like(z)

            mmd = imq_mmd(
                z,
                prior,
            )

            loss = (
                recon
                + lambda_mmd * mmd
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_recon = float(
                recon.detach().cpu()
            )

            final_train_mmd = float(
                mmd.detach().cpu()
            )

    return (
        model,
        final_recon,
        final_train_mmd,
    )


# ============================================================
# Run
# ============================================================

results = []

print()
print("====================================================")
print("HDGM d=10 WAE 2000-EPOCH TRANSFER CHECK")
print("====================================================")
print()
print(f"d           = {D}")
print(f"latent z    = {LATENT_DIM}")
print(f"internal n  = {N}")
print(f"paper N     = {PAPER_N}")
print(f"epochs      = {EPOCHS}")
print(f"lambdas     = {LAMBDAS}")
print()


for kk in DATA_SEEDS:

    print(
        f"----- dataset seed {kk} -----"
    )

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_hdgm_semi_t2(
        N,
        N,
        d=D,
        kk=kk,
        level="hard",
    )

    pooled_np = np.concatenate(
        (
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ),
        axis=0,
    )

    pooled = MatConvert(
        pooled_np,
        device,
        dtype,
    )

    for lam in LAMBDAS:

        start = time.time()

        (
            model,
            recon,
            train_mmd,
        ) = train_one(
            pooled,
            lam,
        )

        model.eval()

        with torch.no_grad():

            z = model.encoder(
                pooled
            )

            # Fresh evaluation prior.
            torch.manual_seed(
                9000 + kk
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed(
                    9000 + kk
                )

            prior = torch.randn_like(z)

            eval_mmd = float(
                imq_mmd(
                    z,
                    prior,
                ).cpu()
            )

            z_np = (
                z.detach()
                .cpu()
                .numpy()
            )

            mean_norm = float(
                np.linalg.norm(
                    z_np.mean(axis=0)
                )
            )

            mean_latent_var = float(
                np.var(
                    z_np,
                    axis=0,
                    ddof=1,
                ).mean()
            )

            eff_rank = effective_rank(
                z_np
            )

        row = {
            "dataset_seed": kk,
            "lambda_mmd": lam,
            "reconstruction_mse": recon,
            "train_imq_mmd": train_mmd,
            "eval_imq_mmd": eval_mmd,
            "latent_mean_norm": mean_norm,
            "mean_latent_variance": mean_latent_var,
            "effective_covariance_rank": eff_rank,
            "elapsed_seconds": (
                time.time() - start
            ),
        }

        results.append(row)

        print(
            f"lambda={lam:<5g} "
            f"recon={recon:.6g}  "
            f"eval_MMD={eval_mmd:.6g}  "
            f"|mean(z)|={mean_norm:.4f}  "
            f"mean_var={mean_latent_var:.4f}  "
            f"eff_rank={eff_rank:.3f}"
        )


# ============================================================
# Aggregate
# ============================================================

print()
print("====================================================")
print("MEAN OVER 3 DATASETS")
print("====================================================")
print()

summary = []

for lam in LAMBDAS:

    rows = [
        r for r in results
        if r["lambda_mmd"] == lam
    ]

    item = {
        "lambda_mmd": lam,

        "reconstruction_mse": float(
            np.mean(
                [r["reconstruction_mse"] for r in rows]
            )
        ),

        "eval_imq_mmd": float(
            np.mean(
                [r["eval_imq_mmd"] for r in rows]
            )
        ),

        "latent_mean_norm": float(
            np.mean(
                [r["latent_mean_norm"] for r in rows]
            )
        ),

        "mean_latent_variance": float(
            np.mean(
                [r["mean_latent_variance"] for r in rows]
            )
        ),

        "effective_covariance_rank": float(
            np.mean(
                [r["effective_covariance_rank"] for r in rows]
            )
        ),
    }

    summary.append(item)

    print(
        f"lambda={lam:<5g}  "
        f"recon={item['reconstruction_mse']:.6g}  "
        f"prior_MMD={item['eval_imq_mmd']:.6g}  "
        f"|mean(z)|={item['latent_mean_norm']:.4f}  "
        f"var={item['mean_latent_variance']:.4f}  "
        f"eff_rank={item['effective_covariance_rank']:.3f}"
    )


# ============================================================
# Save
# ============================================================

payload = {
    "experiment": (
        "HDGM d=10 WAE Phase-1 transfer sanity check"
    ),

    "purpose": (
        "Check transfer of the locked d=2 lambda=0.01 "
        "to z=d=10 without using downstream test power."
    ),

    "d": D,
    "latent_dim": LATENT_DIM,
    "internal_n": N,
    "paper_N": PAPER_N,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "lambdas": LAMBDAS,
    "dataset_seeds": DATA_SEEDS,

    "selection_policy": (
        "No downstream power used. lambda=0 is only a "
        "reconstruction-only reference; lambda=0.01 is the "
        "pre-existing locked d=2 setting being checked for transfer."
    ),

    "results": results,
    "summary": summary,
    "timestamp": time.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
}

os.makedirs(
    "result",
    exist_ok=True,
)

out = (
    "result/"
    "diagnose_wae_d10_z10_lambda001_epochs2000.json"
)

with open(out, "w") as f:
    json.dump(
        payload,
        f,
        indent=2,
    )

print()
print("Saved:")
print(f"  {out}")


import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import MatConvert, sample_hdgm_semi_t2


# ============================================================
# WAE-MMD / IMQ lambda multi-seed diagnostic
#
# HDGM d=2 only.
#
# Purpose:
#   Check whether the promising lambda range from the
#   single-seed diagnostic is stable across different HDGM
#   realizations WITHOUT using downstream C2ST power.
#
# Candidates:
#   lambda = 0, 0.1, 0.3, 1, 10
#
# HDGM outer seeds:
#   kk = 0, 1, 2
#
# For a given HDGM seed, every lambda receives:
#   - exactly the same data
#   - exactly the same network initialization
#   - exactly the same DataLoader shuffle sequence
#   - matching random-number initialization
#
# Metrics:
#   - reconstruction loss
#   - IMQ-MMD to N(0,I)
#   - latent mean
#   - latent variance
#   - number of active latent dimensions
#
# No downstream test power is used for lambda selection.
# ============================================================


# ------------------------------------------------------------
# Device / seeds
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("device:", device, flush=True)


MASTER_SEED = 1102

np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MASTER_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dtype = torch.float


# ------------------------------------------------------------
# HDGM / architecture configuration
# ------------------------------------------------------------

x_in = 2
H = 30
x_out = 30

batch_size = 1024
lr = 0.002

n = 250
n_train = n
n_test = n

paper_N = 8 * n

epochs = int(
    400 * 2000 / (n_train + n_test)
)

hdgm_seeds = [
    0,
    1,
    2,
]

lambda_values = [
    0.0,
    0.1,
    0.3,
    1.0,
    10.0,
]

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
# Architecture matched to released Tian AE
# ============================================================

class WAE(nn.Module):

    def __init__(self):
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
        recon = self.decoder(z)
        return recon, z


# ============================================================
# Canonical multi-scale IMQ MMD
# ============================================================

def pairwise_squared_distance(A, B):

    A_norm = torch.sum(
        A * A,
        dim=1,
        keepdim=True,
    )

    B_norm = torch.sum(
        B * B,
        dim=1,
        keepdim=True,
    ).transpose(0, 1)

    D = (
        A_norm
        + B_norm
        - 2.0 * torch.mm(
            A,
            B.transpose(0, 1),
        )
    )

    return torch.clamp(
        D,
        min=0.0,
    )


def imq_mmd(qz, pz):

    nq = qz.shape[0]
    np_ = pz.shape[0]

    Dqq = pairwise_squared_distance(
        qz,
        qz,
    )

    Dpp = pairwise_squared_distance(
        pz,
        pz,
    )

    Dqp = pairwise_squared_distance(
        qz,
        pz,
    )

    # Standard normal prior:
    # C_base = 2 * latent_dim
    C_base = 2.0 * x_out

    result = qz.new_tensor(0.0)

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

        result = (
            result
            + qq
            + pp
            - 2.0 * qp
        )

    return result


# ============================================================
# Storage
# ============================================================

all_results = []


print(
    "\n"
    "============================================================"
)

print(
    "WAE-MMD / IMQ lambda multi-seed diagnostic"
)

print(
    f"HDGM d=2 | internal n={n} | paper N={paper_N}"
)

print(
    f"representation epochs={epochs}"
)

print(
    f"HDGM seeds={hdgm_seeds}"
)

print(
    f"lambda values={lambda_values}"
)

print(
    "============================================================\n"
)


# ============================================================
# Overall progress:
# 3 HDGM seeds x 5 lambda values = 15 WAE trainings
# ============================================================

total_runs = (
    len(hdgm_seeds)
    * len(lambda_values)
)

overall = tqdm(
    total=total_runs,
    desc=(
        f"WAE-IMQ d=2 "
        f"N={paper_N} multiseed"
    ),
    unit="model",
    dynamic_ncols=True,
)


# ============================================================
# Loop over independent HDGM data realizations
# ============================================================

for kk in hdgm_seeds:

    # --------------------------------------------------------
    # Fixed data for this HDGM seed.
    #
    # Every lambda sees exactly this same realization.
    # --------------------------------------------------------

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_hdgm_semi_t2(
        n_train,
        n_test,
        d=2,
        kk=kk,
        level="hard",
    )

    S_np = np.concatenate(
        (
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ),
        axis=0,
    )

    S = MatConvert(
        S_np,
        device,
        dtype,
    )


    # ========================================================
    # Lambda loop
    # ========================================================

    for lam in lambda_values:

        # ----------------------------------------------------
        # Fair lambda comparison.
        #
        # For this HDGM seed, reset the network/random state
        # before every lambda.
        # ----------------------------------------------------

        run_seed = (
            MASTER_SEED
            + kk
        )

        np.random.seed(run_seed)
        torch.manual_seed(run_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                run_seed
            )

        model = WAE().to(
            device,
            dtype,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        # Same shuffle sequence for all lambda values
        # within this HDGM realization.
        generator = torch.Generator()
        generator.manual_seed(
            run_seed
        )

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                S
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )


        # ----------------------------------------------------
        # WAE training
        # ----------------------------------------------------

        epoch_bar = tqdm(
            range(epochs),
            desc=(
                f"kk={kk} "
                f"lambda={lam}"
            ),
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
        )

        last_recon = np.nan
        last_mmd = np.nan

        for epoch in epoch_bar:

            recon_sum = 0.0
            mmd_sum = 0.0
            n_batches = 0

            for (x_batch,) in loader:

                recon, z = model(
                    x_batch
                )

                recon_loss = (
                    F.mse_loss(
                        recon,
                        x_batch,
                    )
                )

                z_prior = (
                    torch.randn_like(z)
                )

                mmd_loss = imq_mmd(
                    z,
                    z_prior,
                )

                loss = (
                    recon_loss
                    + lam * mmd_loss
                )

                if not torch.isfinite(
                    loss
                ):
                    raise RuntimeError(
                        "Non-finite WAE loss: "
                        f"kk={kk}, "
                        f"lambda={lam}, "
                        f"epoch={epoch}, "
                        f"recon="
                        f"{recon_loss.item()}, "
                        f"mmd="
                        f"{mmd_loss.item()}"
                    )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                recon_sum += (
                    recon_loss.item()
                )

                mmd_sum += (
                    mmd_loss.item()
                )

                n_batches += 1


            last_recon = (
                recon_sum
                / n_batches
            )

            last_mmd = (
                mmd_sum
                / n_batches
            )

            epoch_bar.set_postfix(
                recon=f"{last_recon:.3g}",
                mmd=f"{last_mmd:.3g}",
            )


        # ====================================================
        # Final full-pool diagnostics
        # ====================================================

        model.eval()

        with torch.no_grad():

            recon_all, z_all = model(S)

            recon_final = (
                F.mse_loss(
                    recon_all,
                    S,
                ).item()
            )

            # Fixed evaluation prior per HDGM seed.
            evaluation_seed = (
                MASTER_SEED
                + 10000
                + kk
            )

            torch.manual_seed(
                evaluation_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    evaluation_seed
                )

            prior_all = (
                torch.randn_like(
                    z_all
                )
            )

            mmd_final = (
                imq_mmd(
                    z_all,
                    prior_all,
                ).item()
            )

            latent_mean = (
                z_all.mean(
                    dim=0
                )
            )

            latent_var = (
                z_all.var(
                    dim=0,
                    unbiased=False,
                )
            )

            mean_abs = (
                latent_mean
                .abs()
                .mean()
                .item()
            )

            mean_l2 = (
                torch.linalg.vector_norm(
                    latent_mean
                ).item()
            )

            var_mean = (
                latent_var
                .mean()
                .item()
            )

            var_min = (
                latent_var
                .min()
                .item()
            )

            var_max = (
                latent_var
                .max()
                .item()
            )

            active_dims = int(
                (
                    latent_var > 1e-3
                )
                .sum()
                .item()
            )


        record = {
            "hdgm_seed": kk,

            "lambda": lam,

            "reconstruction_loss":
                recon_final,

            "imq_mmd":
                mmd_final,

            "weighted_mmd":
                lam * mmd_final,

            "latent_mean_abs_average":
                mean_abs,

            "latent_mean_l2":
                mean_l2,

            "latent_variance_mean":
                var_mean,

            "latent_variance_min":
                var_min,

            "latent_variance_max":
                var_max,

            "active_variance_dims_gt_1e-3":
                active_dims,
        }

        all_results.append(
            record
        )


        overall.set_postfix(
            seed=kk,
            lam=lam,
            recon=f"{recon_final:.3g}",
            mmd=f"{mmd_final:.3g}",
            active=f"{active_dims}/30",
        )

        overall.update(1)


overall.close()


# ============================================================
# Aggregate by lambda across HDGM seeds
# ============================================================

aggregate = []

for lam in lambda_values:

    rows = [
        r
        for r in all_results
        if r["lambda"] == lam
    ]

    recon_values = np.array(
        [
            r["reconstruction_loss"]
            for r in rows
        ],
        dtype=float,
    )

    mmd_values = np.array(
        [
            r["imq_mmd"]
            for r in rows
        ],
        dtype=float,
    )

    var_values = np.array(
        [
            r["latent_variance_mean"]
            for r in rows
        ],
        dtype=float,
    )

    mean_values = np.array(
        [
            r["latent_mean_abs_average"]
            for r in rows
        ],
        dtype=float,
    )

    active_values = np.array(
        [
            r[
                "active_variance_dims_gt_1e-3"
            ]
            for r in rows
        ],
        dtype=float,
    )


    aggregate.append(
        {
            "lambda": lam,

            "n_seeds":
                len(rows),

            "reconstruction_mean":
                float(
                    recon_values.mean()
                ),

            "reconstruction_std":
                float(
                    recon_values.std(
                        ddof=1
                    )
                ),

            "imq_mmd_mean":
                float(
                    mmd_values.mean()
                ),

            "imq_mmd_std":
                float(
                    mmd_values.std(
                        ddof=1
                    )
                ),

            "latent_variance_mean":
                float(
                    var_values.mean()
                ),

            "latent_variance_std":
                float(
                    var_values.std(
                        ddof=1
                    )
                ),

            "latent_mean_abs_mean":
                float(
                    mean_values.mean()
                ),

            "active_dims_mean":
                float(
                    active_values.mean()
                ),

            "active_dims_min":
                int(
                    active_values.min()
                ),
        }
    )


# ============================================================
# Save
# ============================================================

os.makedirs(
    "result",
    exist_ok=True,
)

payload = {
    "experiment":
        (
            "WAE-MMD/IMQ lambda "
            "multi-seed diagnostic"
        ),

    "purpose":
        (
            "Check stability of WAE "
            "regularization scale across "
            "multiple HDGM realizations "
            "without using C2ST power."
        ),

    "dataset":
        "HDGM-D",

    "d":
        2,

    "internal_n":
        n,

    "paper_N":
        paper_N,

    "epochs":
        epochs,

    "batch_size":
        batch_size,

    "learning_rate":
        lr,

    "hdgm_seeds":
        hdgm_seeds,

    "lambda_values":
        lambda_values,

    "IMQ_scales":
        list(IMQ_SCALES),

    "latent_dim":
        x_out,

    "C_base":
        2 * x_out,

    "per_seed_results":
        all_results,

    "aggregate":
        aggregate,

    "ts":
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
}


output_path = (
    "result/"
    "diagnose_wae_imq_lambda_multiseed.json"
)

with open(
    output_path,
    "w",
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


# ============================================================
# Print compact summary
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "MULTISEED SUMMARY"
)

print(
    "=" * 78
)

for r in aggregate:

    print(
        f"lambda={r['lambda']:>4} | "
        f"recon="
        f"{r['reconstruction_mean']:.6f}"
        f" +/- "
        f"{r['reconstruction_std']:.6f} | "
        f"MMD="
        f"{r['imq_mmd_mean']:.5f}"
        f" +/- "
        f"{r['imq_mmd_std']:.5f} | "
        f"var="
        f"{r['latent_variance_mean']:.3f}"
        f" +/- "
        f"{r['latent_variance_std']:.3f} | "
        f"active_min="
        f"{r['active_dims_min']}/30"
    )

print(
    f"\nSaved: {output_path}"
)
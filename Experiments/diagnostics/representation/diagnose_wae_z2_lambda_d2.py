import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import MatConvert, sample_hdgm_semi_t2


# ============================================================
# HDGM d=2 WAE z=2 lambda diagnostic
#
# PURPOSE
# -------
# The previous latent-dimension diagnostic showed:
#
#   WAE z=30, lambda=0.3:
#       strong reconstruction
#       poor prior matching
#
#   WAE z=2, lambda=0.3:
#       excellent prior matching
#       weaker reconstruction
#
# Therefore lambda=0.3 may be unnecessarily strong when the
# WAE latent dimension already matches the 2D HDGM input.
#
# This experiment sweeps lambda for z=2 BEFORE looking at
# downstream C2ST-S / C2ST-L power.
#
# IMPORTANT:
#   Lambda must be selected using Phase-1 criteria only:
#
#       reconstruction MSE
#       prior IMQ-MMD
#
# Covariance gap / energy distance are recorded diagnostically
# but must NOT be used to choose lambda, because they use the
# knowledge of which observations came from P and Q.
# ============================================================


# ------------------------------------------------------------
# Device / deterministic setup
# ------------------------------------------------------------

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

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
# HDGM / WAE configuration
# ------------------------------------------------------------

x_in = 2
H = 30
latent_dim = 2

n = 250
n_train = n
n_test = n

# Same paper-N convention as the existing HDGM diagnostics.
paper_N = 8 * n

batch_size = 1024
lr = 0.002

# Recovered historical HDGM d=2 representation-training schedule.
epochs = int(
    400 * 2000 / (n_train + n_test)
)

hdgm_seeds = [0, 1, 2]

# z=2-specific lambda sweep.
#
# lambda=0 gives the reconstruction-only z=2 bottleneck reference.
# lambda=0.3 is the value inherited from the earlier z=30 WAE.
lambda_values = [
    0.0,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
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

print(
    f"paper N={paper_N}, "
    f"z={latent_dim}, "
    f"epochs={epochs}",
    flush=True,
)

print(
    "lambda grid:",
    lambda_values,
    flush=True,
)


# ============================================================
# WAE
# ============================================================

class WAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_in, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, x_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ============================================================
# IMQ-MMD
# ============================================================

def pairwise_squared_distance(A, B):

    A2 = torch.sum(
        A * A,
        dim=1,
        keepdim=True,
    )

    B2 = torch.sum(
        B * B,
        dim=1,
        keepdim=True,
    ).T

    D = (
        A2
        + B2
        - 2.0 * A @ B.T
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

    # Same scaling rule as the existing WAE diagnostic.
    C_base = 2.0 * latent_dim

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
# Representation diagnostics
# ============================================================

def covariance_matrix(X):

    X = X - X.mean(
        dim=0,
        keepdim=True,
    )

    return (
        X.T @ X
        / (X.shape[0] - 1)
    )


def effective_rank(X):

    cov = covariance_matrix(X)

    eigvals = torch.linalg.eigvalsh(
        cov
    )

    eigvals = torch.clamp(
        eigvals,
        min=0.0,
    )

    total = eigvals.sum()

    if total <= 1e-12:
        return 0.0, 0.0

    p = eigvals / total

    p_positive = p[
        p > 1e-12
    ]

    entropy = -torch.sum(
        p_positive
        * torch.log(p_positive)
    )

    erank = torch.exp(entropy)

    eigvals_sorted = torch.sort(
        eigvals,
        descending=True,
    ).values

    top2_fraction = (
        eigvals_sorted[:2].sum()
        / total
    )

    return (
        float(erank.item()),
        float(top2_fraction.item()),
    )


def normalized_covariance_gap(P, Q):

    cov_p = covariance_matrix(P)
    cov_q = covariance_matrix(Q)

    pooled = torch.cat(
        [P, Q],
        dim=0,
    )

    cov_pool = covariance_matrix(
        pooled
    )

    numerator = torch.linalg.norm(
        cov_p - cov_q,
        ord="fro",
    )

    denominator = torch.linalg.norm(
        cov_pool,
        ord="fro",
    )

    return float(
        (
            numerator
            / (denominator + 1e-12)
        ).item()
    )


def normalized_energy_distance(P, Q):

    Dpq = torch.cdist(P, Q)
    Dpp = torch.cdist(P, P)
    Dqq = torch.cdist(Q, Q)

    np_ = P.shape[0]
    nq_ = Q.shape[0]

    cross = Dpq.mean()

    within_p = (
        Dpp.sum()
        - torch.diagonal(Dpp).sum()
    ) / (
        np_ * (np_ - 1)
    )

    within_q = (
        Dqq.sum()
        - torch.diagonal(Dqq).sum()
    ) / (
        nq_ * (nq_ - 1)
    )

    energy = (
        2.0 * cross
        - within_p
        - within_q
    )

    scale = 0.5 * (
        within_p
        + within_q
    )

    return float(
        (
            energy
            / (scale + 1e-12)
        ).item()
    )


def representation_metrics(P, Q):

    pooled = torch.cat(
        [P, Q],
        dim=0,
    )

    erank, top2_fraction = effective_rank(
        pooled
    )

    latent_mean_norm = torch.linalg.norm(
        pooled.mean(dim=0)
    ).item()

    latent_variance = (
        pooled.var(
            dim=0,
            unbiased=True,
        )
        .mean()
        .item()
    )

    return {
        "effective_rank":
            erank,

        "top2_variance_fraction":
            top2_fraction,

        "latent_mean_norm":
            float(latent_mean_norm),

        "mean_latent_variance":
            float(latent_variance),

        "normalized_covariance_gap":
            normalized_covariance_gap(
                P,
                Q,
            ),

        "normalized_energy_distance":
            normalized_energy_distance(
                P,
                Q,
            ),
    }


# ============================================================
# WAE training
# ============================================================

def train_wae(
    S,
    lambda_mmd,
    run_seed,
):

    # Reset before every lambda so all lambda values receive
    # matching initialization/randomness within an HDGM seed.
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

    dataset = (
        torch.utils.data.TensorDataset(
            S
        )
    )

    generator = torch.Generator()
    generator.manual_seed(
        run_seed
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    final_recon = np.nan
    final_mmd = np.nan

    for _ in range(epochs):

        recon_sum = 0.0
        mmd_sum = 0.0
        n_batches = 0

        for (x_batch,) in loader:

            reconstruction, z = model(
                x_batch
            )

            recon_loss = F.mse_loss(
                reconstruction,
                x_batch,
            )

            z_prior = torch.randn_like(
                z
            )

            mmd_loss = imq_mmd(
                z,
                z_prior,
            )

            loss = (
                recon_loss
                + lambda_mmd
                * mmd_loss
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite WAE loss: "
                    f"lambda={lambda_mmd}"
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

        final_recon = (
            recon_sum
            / n_batches
        )

        final_mmd = (
            mmd_sum
            / n_batches
        )

    return (
        model,
        final_recon,
        final_mmd,
    )


# ============================================================
# Main diagnostic
# ============================================================

records = []

total_jobs = (
    len(hdgm_seeds)
    * len(lambda_values)
)

progress = tqdm(
    total=total_jobs,
    desc="WAE z=2 lambda diagnostic",
    unit="model",
    dynamic_ncols=True,
)


for kk in hdgm_seeds:

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

    P_np = np.concatenate(
        [
            s1_tr,
            s1_te,
        ],
        axis=0,
    )

    Q_np = np.concatenate(
        [
            s2_tr,
            s2_te,
        ],
        axis=0,
    )

    S_np = np.concatenate(
        [
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ],
        axis=0,
    )

    P = MatConvert(
        P_np,
        device,
        dtype,
    )

    Q = MatConvert(
        Q_np,
        device,
        dtype,
    )

    S = MatConvert(
        S_np,
        device,
        dtype,
    )

    # Each HDGM realization gets its own deterministic
    # initialization. Within that realization, all lambdas
    # receive exactly the same initialization/random sequence.
    run_seed = (
        MASTER_SEED
        + kk
    )

    for lambda_mmd in lambda_values:

        (
            wae,
            recon_loss,
            training_mmd,
        ) = train_wae(
            S,
            lambda_mmd,
            run_seed,
        )

        wae.eval()

        with torch.no_grad():

            Pz = wae.encoder(P)
            Qz = wae.encoder(Q)

            pooled_z = torch.cat(
                [Pz, Qz],
                dim=0,
            )

            # Crucially, use the SAME evaluation-prior draw
            # across lambda values for a given HDGM seed.
            eval_seed = (
                MASTER_SEED
                + 10000
                + kk
            )

            torch.manual_seed(
                eval_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    eval_seed
                )

            prior = torch.randn_like(
                pooled_z
            )

            eval_prior_mmd = imq_mmd(
                pooled_z,
                prior,
            ).item()

        metrics = representation_metrics(
            Pz,
            Qz,
        )

        record = {
            "seed":
                kk,

            "paper_N":
                paper_N,

            "internal_n":
                n,

            "latent_dim":
                latent_dim,

            "lambda_mmd":
                lambda_mmd,

            "epochs":
                epochs,

            "reconstruction_mse":
                float(recon_loss),

            "training_imq_mmd":
                float(training_mmd),

            "eval_prior_imq_mmd":
                float(eval_prior_mmd),

            **metrics,
        }

        records.append(
            record
        )

        progress.set_postfix(
            seed=kk,
            lam=lambda_mmd,
            recon=f"{recon_loss:.2e}",
            mmd=f"{eval_prior_mmd:.3g}",
        )

        progress.update(1)


progress.close()


# ============================================================
# Aggregate over HDGM seeds
# ============================================================

summary = {}

summary_keys = [
    "reconstruction_mse",
    "eval_prior_imq_mmd",
    "effective_rank",
    "top2_variance_fraction",
    "latent_mean_norm",
    "mean_latent_variance",
    "normalized_covariance_gap",
    "normalized_energy_distance",
]


for lambda_mmd in lambda_values:

    rows = [
        r
        for r in records
        if r["lambda_mmd"]
        == lambda_mmd
    ]

    lambda_summary = {}

    for key in summary_keys:

        values = np.array(
            [
                r[key]
                for r in rows
            ],
            dtype=float,
        )

        lambda_summary[key] = {
            "mean":
                float(values.mean()),

            "std":
                float(values.std()),
        }

    summary[str(lambda_mmd)] = (
        lambda_summary
    )


# ============================================================
# Save JSON
# ============================================================

payload = {
    "experiment":
        "HDGM d=2 deterministic WAE z=2 lambda diagnostic",

    "purpose": (
        "Calibrate the WAE-MMD regularization strength for "
        "latent_dim=2 using Phase-1 criteria before any "
        "downstream C2ST-S or C2ST-L power evaluation."
    ),

    "selection_rule": (
        "Lambda must be selected from reconstruction MSE and "
        "prior IMQ-MMD only. P-vs-Q covariance gap and energy "
        "distance are descriptive diagnostics and must not "
        "be used for lambda selection."
    ),

    "paper_N":
        paper_N,

    "internal_n":
        n,

    "latent_dim":
        latent_dim,

    "seeds":
        hdgm_seeds,

    "lambda_values":
        lambda_values,

    "training": {
        "epochs":
            epochs,

        "batch_size":
            batch_size,

        "learning_rate":
            lr,
    },

    "wae": {
        "kernel":
            "multi-scale IMQ",

        "prior":
            "N(0,I)",

        "imq_scales":
            list(IMQ_SCALES),
    },

    "summary":
        summary,

    "records":
        records,
}


os.makedirs(
    "result",
    exist_ok=True,
)

output_path = (
    "result/"
    "diagnose_wae_z2_lambda_d2.json"
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
# Human-readable summary
# ============================================================

print("\n")
print(
    "============================================================"
)

print(
    "WAE z=2 LAMBDA DIAGNOSTIC"
)

print(
    "============================================================"
)

print(
    f"{'lambda':>8}"
    f"{'recon':>14}"
    f"{'prior MMD':>14}"
    f"{'mean|z|':>12}"
    f"{'z var':>12}"
    f"{'cov gap':>12}"
    f"{'energy':>12}"
)

for lambda_mmd in lambda_values:

    s = summary[
        str(lambda_mmd)
    ]

    print(
        f"{lambda_mmd:>8.3g}"
        f"{s['reconstruction_mse']['mean']:>14.6f}"
        f"{s['eval_prior_imq_mmd']['mean']:>14.5f}"
        f"{s['latent_mean_norm']['mean']:>12.4f}"
        f"{s['mean_latent_variance']['mean']:>12.4f}"
        f"{s['normalized_covariance_gap']['mean']:>12.4f}"
        f"{s['normalized_energy_distance']['mean']:>12.4f}"
    )


print("\n")
print(
    "SELECTION REMINDER:"
)

print(
    "Choose the promising lambda region using reconstruction "
    "and prior MMD only — NOT covariance gap, energy distance, "
    "or downstream C2ST power."
)

print(
    "\nSaved:",
    output_path,
)
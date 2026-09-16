import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import MatConvert, sample_hdgm_semi_t2


# ============================================================
# HDGM d=2 WAE latent-dimension diagnostic
#
# Goal:
#   Compare deterministic WAE-MMD/IMQ with:
#
#       z_dim = 30   (current reconstruction)
#       z_dim = 2    (intrinsic-dimension-matched candidate)
#
# No downstream C2ST power is used here.
#
# We ask:
#   1. Does z=30 really behave like a high-dimensional latent space?
#   2. Does z=30 suppress the P-vs-Q covariance/distributional signal?
#   3. Does z=2 preserve that signal better while still matching N(0,I)?
#
# IMPORTANT:
#   lambda=0.3 is used for BOTH models only as an initial controlled
#   comparison. It has NOT yet been calibrated for z=2.
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
# HDGM configuration
# ------------------------------------------------------------

x_in = 2
H = 30

n = 250
n_train = n
n_test = n

paper_N = 8 * n

batch_size = 1024
lr = 0.002

# Recovered historical HDGM d=2 epoch schedule
epochs = int(
    400 * 2000 / (n_train + n_test)
)

lambda_mmd = 0.3

hdgm_seeds = [0, 1, 2]
latent_dims = [30, 2]

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
    f"paper N={paper_N}, epochs={epochs}, "
    f"lambda={lambda_mmd}",
    flush=True,
)

# ============================================================
# Historical AE reference
#
# Architecture and training setup matched to the recovered
# HDGM d=2 RL-C2ST AE configuration.
# ============================================================

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(x_in, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, H),
            nn.Softplus(),

            nn.Linear(H, 30),
        )

        self.decoder = nn.Sequential(
            nn.Linear(30, H),
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


def train_ae(S):

    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MASTER_SEED)

    model = AutoEncoder().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    dataset = torch.utils.data.TensorDataset(S)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    final_recon = np.nan

    for _ in range(epochs):

        recon_sum = 0.0
        n_batches = 0

        for (x_batch,) in loader:

            reconstruction, _ = model(
                x_batch
            )

            recon_loss = F.mse_loss(
                reconstruction,
                x_batch,
            )

            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            recon_sum += recon_loss.item()
            n_batches += 1

        final_recon = (
            recon_sum / n_batches
        )

    return model, final_recon
# ============================================================
# WAE
# ============================================================

class WAE(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

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

    return torch.clamp(D, min=0.0)


def imq_mmd(qz, pz, latent_dim):

    nq = qz.shape[0]
    np_ = pz.shape[0]

    Dqq = pairwise_squared_distance(qz, qz)
    Dpp = pairwise_squared_distance(pz, pz)
    Dqp = pairwise_squared_distance(qz, pz)

    # Standard Gaussian prior N(0,I)
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

        result = result + qq + pp - 2.0 * qp

    return result


# ============================================================
# Representation diagnostics
# ============================================================

def covariance_matrix(X):

    X = X - X.mean(dim=0, keepdim=True)

    return (
        X.T @ X
        / (X.shape[0] - 1)
    )


def effective_rank(X):

    cov = covariance_matrix(X)

    eigvals = torch.linalg.eigvalsh(cov)

    eigvals = torch.clamp(
        eigvals,
        min=0.0,
    )

    total = eigvals.sum()

    if total <= 1e-12:
        return 0.0, 0.0

    p = eigvals / total

    p_positive = p[p > 1e-12]

    entropy = -torch.sum(
        p_positive * torch.log(p_positive)
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

    cov_pool = covariance_matrix(pooled)

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

    # Scale normalization so latent-space scale alone cannot
    # create an apparent improvement.
    scale = 0.5 * (
        within_p + within_q
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

    return {
        "effective_rank": erank,
        "top2_variance_fraction": top2_fraction,
        "normalized_covariance_gap":
            normalized_covariance_gap(P, Q),
        "normalized_energy_distance":
            normalized_energy_distance(P, Q),
    }


# ============================================================
# WAE training
# ============================================================

def train_wae(
    S,
    latent_dim,
):

    # Same initialization seed for every model so the comparison
    # is reproducible.
    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(MASTER_SEED)

    model = WAE(
        latent_dim=latent_dim
    ).to(device, dtype)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    dataset = torch.utils.data.TensorDataset(S)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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

            z_prior = torch.randn_like(z)

            mmd_loss = imq_mmd(
                z,
                z_prior,
                latent_dim,
            )

            loss = (
                recon_loss
                + lambda_mmd * mmd_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_sum += recon_loss.item()
            mmd_sum += mmd_loss.item()

            n_batches += 1

        final_recon = (
            recon_sum / n_batches
        )

        final_mmd = (
            mmd_sum / n_batches
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
ae_records = []

total_jobs = (
    len(hdgm_seeds)
    * (len(latent_dims) + 1)
)

progress = tqdm(
    total=total_jobs,
    desc="WAE latent-dim diagnostic",
    unit="model",
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
        [s1_tr, s1_te],
        axis=0,
    )

    Q_np = np.concatenate(
        [s2_tr, s2_te],
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

    # ----------------------------------------
    # Raw-space reference
    # ----------------------------------------

    raw_metrics = representation_metrics(
        P,
        Q,
    )

    print(
        f"\nseed={kk} RAW:"
        f" cov_gap={raw_metrics['normalized_covariance_gap']:.4f}"
        f" energy={raw_metrics['normalized_energy_distance']:.4f}",
        flush=True,
    )

    # ----------------------------------------
    # Historical AE reference
    # ----------------------------------------

    ae, ae_recon = train_ae(S)

    ae.eval()

    with torch.no_grad():
        P_ae = ae.encoder(P)
        Q_ae = ae.encoder(Q)

    ae_metrics = representation_metrics(
        P_ae,
        Q_ae,
    )

    ae_record = {
        "seed": kk,
        "paper_N": paper_N,
        "reconstruction_mse": float(ae_recon),
        **ae_metrics,
    }

    ae_records.append(ae_record)

    progress.set_postfix(
        seed=kk,
        model="AE",
        recon=f"{ae_recon:.2e}",
        cov=f"{ae_metrics['normalized_covariance_gap']:.3f}",
        energy=f"{ae_metrics['normalized_energy_distance']:.3f}",
    )

    progress.update(1)


    # ----------------------------------------
    # z = 30 and z = 2
    # ----------------------------------------

    for latent_dim in latent_dims:

        (
            wae,
            recon_loss,
            training_mmd,
        ) = train_wae(
            S,
            latent_dim,
        )

        wae.eval()

        with torch.no_grad():

            Pz = wae.encoder(P)
            Qz = wae.encoder(Q)

            pooled_z = torch.cat(
                [Pz, Qz],
                dim=0,
            )

            # Fresh reference prior for evaluation
            torch.manual_seed(
                MASTER_SEED + kk + latent_dim
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    MASTER_SEED
                    + kk
                    + latent_dim
                )

            prior = torch.randn_like(
                pooled_z
            )

            eval_prior_mmd = imq_mmd(
                pooled_z,
                prior,
                latent_dim,
            ).item()

        metrics = representation_metrics(
            Pz,
            Qz,
        )

        record = {
            "seed": kk,
            "paper_N": paper_N,
            "latent_dim": latent_dim,
            "lambda_mmd": lambda_mmd,
            "epochs": epochs,

            "reconstruction_mse":
                float(recon_loss),

            "training_imq_mmd":
                float(training_mmd),

            "eval_prior_imq_mmd":
                float(eval_prior_mmd),

            "raw_covariance_gap":
                raw_metrics[
                    "normalized_covariance_gap"
                ],

            "raw_energy_distance":
                raw_metrics[
                    "normalized_energy_distance"
                ],

            **metrics,
        }

        records.append(record)

        progress.set_postfix(
            seed=kk,
            z=latent_dim,
            recon=f"{recon_loss:.2e}",
            cov=f"{metrics['normalized_covariance_gap']:.3f}",
            energy=f"{metrics['normalized_energy_distance']:.3f}",
        )

        progress.update(1)


progress.close()


# ============================================================
# Aggregate over seeds
# ============================================================

# ------------------------------------------------------------
# WAE summaries
# ------------------------------------------------------------

wae_summary = {}

wae_keys = [
    "reconstruction_mse",
    "eval_prior_imq_mmd",
    "effective_rank",
    "top2_variance_fraction",
    "normalized_covariance_gap",
    "normalized_energy_distance",
]

for latent_dim in latent_dims:

    rows = [
        r
        for r in records
        if r["latent_dim"] == latent_dim
    ]

    wae_summary[str(latent_dim)] = {}

    for key in wae_keys:

        values = np.array(
            [r[key] for r in rows],
            dtype=float,
        )

        wae_summary[str(latent_dim)][key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }


# ------------------------------------------------------------
# Historical AE summary
# ------------------------------------------------------------

ae_summary = {}

ae_keys = [
    "reconstruction_mse",
    "effective_rank",
    "top2_variance_fraction",
    "normalized_covariance_gap",
    "normalized_energy_distance",
]

for key in ae_keys:

    values = np.array(
        [r[key] for r in ae_records],
        dtype=float,
    )

    ae_summary[key] = {
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


# ------------------------------------------------------------
# Raw-space reference
#
# Raw metrics are duplicated once for each WAE latent dimension.
# Take the z=30 row from each seed so every HDGM seed contributes
# exactly once.
# ------------------------------------------------------------

raw_rows = [
    r
    for r in records
    if r["latent_dim"] == 30
]

raw_cov = np.array(
    [
        r["raw_covariance_gap"]
        for r in raw_rows
    ],
    dtype=float,
)

raw_energy = np.array(
    [
        r["raw_energy_distance"]
        for r in raw_rows
    ],
    dtype=float,
)


# ============================================================
# Save complete diagnostic
# ============================================================

payload = {
    "experiment":
        "HDGM d=2 AE vs deterministic WAE latent-dimension diagnostic",

    "purpose": (
        "Compare the recovered historical AE representation "
        "against deterministic WAE-MMD/IMQ with z=30 and z=2 "
        "before further downstream C2ST power experiments."
    ),

    "paper_N": paper_N,

    "internal_n": n,

    "seeds": hdgm_seeds,

    "training": {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    },

    "wae": {
        "lambda_mmd": lambda_mmd,
        "kernel": "multi-scale IMQ",
        "prior": "N(0,I)",
        "latent_dims": latent_dims,
        "lambda_note": (
            "lambda=0.3 is the existing z=30 choice and is "
            "used for z=2 only as an untuned controlled "
            "screening value."
        ),
    },

    "raw_reference": {
        "normalized_covariance_gap_mean":
            float(raw_cov.mean()),

        "normalized_covariance_gap_std":
            float(raw_cov.std()),

        "normalized_energy_distance_mean":
            float(raw_energy.mean()),

        "normalized_energy_distance_std":
            float(raw_energy.std()),
    },

    "ae_summary": ae_summary,

    "wae_summary": wae_summary,

    "ae_records": ae_records,

    "wae_records": records,
}


os.makedirs(
    "result",
    exist_ok=True,
)

output_path = (
    "result/"
    "diagnose_wae_latent_dim_d2_with_ae.json"
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
print("======================================")
print("RAW REFERENCE")
print("======================================")

print(
    "covariance gap:",
    f"{raw_cov.mean():.4f}"
    f" +/- {raw_cov.std():.4f}",
)

print(
    "energy distance:",
    f"{raw_energy.mean():.4f}"
    f" +/- {raw_energy.std():.4f}",
)


# ------------------------------------------------------------
# AE
# ------------------------------------------------------------

print("\n")
print("======================================")
print("HISTORICAL AE")
print("======================================")

print(
    "reconstruction:",
    f"{ae_summary['reconstruction_mse']['mean']:.6f}"
    f" +/- "
    f"{ae_summary['reconstruction_mse']['std']:.6f}",
)

print(
    "effective rank:",
    f"{ae_summary['effective_rank']['mean']:.3f}"
    f" +/- "
    f"{ae_summary['effective_rank']['std']:.3f}",
)

print(
    "top-2 variance fraction:",
    f"{ae_summary['top2_variance_fraction']['mean']:.4f}"
    f" +/- "
    f"{ae_summary['top2_variance_fraction']['std']:.4f}",
)

print(
    "normalized covariance gap:",
    f"{ae_summary['normalized_covariance_gap']['mean']:.4f}"
    f" +/- "
    f"{ae_summary['normalized_covariance_gap']['std']:.4f}",
)

print(
    "normalized energy distance:",
    f"{ae_summary['normalized_energy_distance']['mean']:.4f}"
    f" +/- "
    f"{ae_summary['normalized_energy_distance']['std']:.4f}",
)


# ------------------------------------------------------------
# WAE z=30 and z=2
# ------------------------------------------------------------

for latent_dim in latent_dims:

    s = wae_summary[str(latent_dim)]

    print("\n")
    print("======================================")
    print(f"WAE z={latent_dim}")
    print("======================================")

    print(
        "reconstruction:",
        f"{s['reconstruction_mse']['mean']:.6f}"
        f" +/- "
        f"{s['reconstruction_mse']['std']:.6f}",
    )

    print(
        "prior IMQ-MMD:",
        f"{s['eval_prior_imq_mmd']['mean']:.5f}"
        f" +/- "
        f"{s['eval_prior_imq_mmd']['std']:.5f}",
    )

    print(
        "effective rank:",
        f"{s['effective_rank']['mean']:.3f}"
        f" +/- "
        f"{s['effective_rank']['std']:.3f}",
    )

    print(
        "top-2 variance fraction:",
        f"{s['top2_variance_fraction']['mean']:.4f}"
        f" +/- "
        f"{s['top2_variance_fraction']['std']:.4f}",
    )

    print(
        "normalized covariance gap:",
        f"{s['normalized_covariance_gap']['mean']:.4f}"
        f" +/- "
        f"{s['normalized_covariance_gap']['std']:.4f}",
    )

    print(
        "normalized energy distance:",
        f"{s['normalized_energy_distance']['mean']:.4f}"
        f" +/- "
        f"{s['normalized_energy_distance']['std']:.4f}",
    )


# ============================================================
# Direct comparison
# ============================================================

print("\n")
print("======================================")
print("DIRECT REPRESENTATION COMPARISON")
print("======================================")

print(
    f"{'representation':<18}"
    f"{'cov gap':>12}"
    f"{'energy':>12}"
    f"{'eff.rank':>12}"
    f"{'top2 frac':>12}"
)

print(
    f"{'RAW':<18}"
    f"{raw_cov.mean():>12.4f}"
    f"{raw_energy.mean():>12.4f}"
    f"{'-':>12}"
    f"{'-':>12}"
)

print(
    f"{'AE z=30':<18}"
    f"{ae_summary['normalized_covariance_gap']['mean']:>12.4f}"
    f"{ae_summary['normalized_energy_distance']['mean']:>12.4f}"
    f"{ae_summary['effective_rank']['mean']:>12.3f}"
    f"{ae_summary['top2_variance_fraction']['mean']:>12.4f}"
)

for latent_dim in latent_dims:

    s = wae_summary[str(latent_dim)]

    label = f"WAE z={latent_dim}"

    print(
        f"{label:<18}"
        f"{s['normalized_covariance_gap']['mean']:>12.4f}"
        f"{s['normalized_energy_distance']['mean']:>12.4f}"
        f"{s['effective_rank']['mean']:>12.3f}"
        f"{s['top2_variance_fraction']['mean']:>12.4f}"
    )


print(
    "\nSaved:",
    output_path,
)
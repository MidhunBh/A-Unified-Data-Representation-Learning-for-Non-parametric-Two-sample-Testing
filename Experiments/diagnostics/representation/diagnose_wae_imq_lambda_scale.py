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
# WAE-MMD/IMQ lambda diagnostic
#
# Purpose:
#   Calibrate the RELATIVE SCALE of the WAE regularization
#   without looking at downstream C2ST power.
#
# We inspect:
#   - reconstruction loss
#   - IMQ-MMD to N(0,I)
#   - weighted MMD contribution
#   - latent mean
#   - latent variance
#   - reconstruction/MMD balance
#
# This is a Phase-1-only diagnostic.
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
    torch.cuda.manual_seed_all(MASTER_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dtype = torch.float

x_in = 2
H = 30
x_out = 30

batch_size = 1024
lr = 0.002

n = 250
n_train = n
n_test = n

kk = 0

epochs = int(
    400 * 2000 / (n_train + n_test)
)

lambda_values = [
    0.0,
    0.1,
    0.3,
    0.6,
    1.0,
    3.0,
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
# Architecture matched to Tian's released AE
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
# IMQ MMD
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

    Dqq = pairwise_squared_distance(qz, qz)
    Dpp = pairwise_squared_distance(pz, pz)
    Dqp = pairwise_squared_distance(qz, pz)

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
# Fixed HDGM data
#
# Every lambda sees EXACTLY the same dataset.
# ============================================================

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

print(
    f"HDGM d=2, internal n={n}, "
    f"paper N={8*n}, pooled shape={tuple(S.shape)}"
)

print(
    f"Historical representation epochs = {epochs}"
)


# ============================================================
# Train one WAE for each lambda
# ============================================================

results = []

for lam in lambda_values:

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"lambda = {lam}"
    )

    print(
        "=" * 70,
        flush=True,
    )

    # Same initialization for every lambda.
    torch.manual_seed(MASTER_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            MASTER_SEED
        )

    model = WAE().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Deterministic loader generator.
    generator = torch.Generator()
    generator.manual_seed(MASTER_SEED)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(S),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history = []

    progress = tqdm(
        range(epochs),
        desc=f"lambda={lam}",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in progress:

        recon_sum = 0.0
        mmd_sum = 0.0
        total_sum = 0.0

        num_batches = 0

        for (x_batch,) in loader:

            recon, z = model(x_batch)

            recon_loss = F.mse_loss(
                recon,
                x_batch,
            )

            z_prior = torch.randn_like(z)

            mmd_loss = imq_mmd(
                z,
                z_prior,
            )

            total_loss = (
                recon_loss
                + lam * mmd_loss
            )

            if not torch.isfinite(
                total_loss
            ):
                raise RuntimeError(
                    f"Non-finite loss at "
                    f"lambda={lam}, "
                    f"epoch={epoch}"
                )

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            recon_sum += (
                recon_loss.item()
            )

            mmd_sum += (
                mmd_loss.item()
            )

            total_sum += (
                total_loss.item()
            )

            num_batches += 1

        recon_epoch = (
            recon_sum / num_batches
        )

        mmd_epoch = (
            mmd_sum / num_batches
        )

        total_epoch = (
            total_sum / num_batches
        )

        if (
            epoch == 0
            or (epoch + 1) % 100 == 0
            or epoch + 1 == epochs
        ):

            history.append(
                {
                    "epoch": epoch + 1,
                    "recon": recon_epoch,
                    "mmd": mmd_epoch,
                    "weighted_mmd": (
                        lam * mmd_epoch
                    ),
                    "total": total_epoch,
                }
            )

        progress.set_postfix(
            recon=f"{recon_epoch:.3g}",
            mmd=f"{mmd_epoch:.3g}",
            wmmd=f"{lam*mmd_epoch:.3g}",
        )


    # ========================================================
    # Final full-pool diagnostics
    # ========================================================

    model.eval()

    with torch.no_grad():

        recon_all, z_all = model(S)

        recon_final = (
            F.mse_loss(
                recon_all,
                S,
            ).item()
        )

        # Large reference prior sample
        # matching pool size.
        torch.manual_seed(
            MASTER_SEED + 999
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                MASTER_SEED + 999
            )

        prior_all = torch.randn_like(
            z_all
        )

        mmd_final = (
            imq_mmd(
                z_all,
                prior_all,
            ).item()
        )

        weighted_mmd_final = (
            lam * mmd_final
        )

        if recon_final > 0:
            ratio_final = (
                weighted_mmd_final
                / recon_final
            )
        else:
            ratio_final = float("inf")

        latent_mean = (
            z_all.mean(dim=0)
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

        mean_norm = (
            torch.linalg.vector_norm(
                latent_mean
            ).item()
        )

        var_mean = (
            latent_var.mean().item()
        )

        var_min = (
            latent_var.min().item()
        )

        var_max = (
            latent_var.max().item()
        )

        active_variance_dims = int(
            (
                latent_var > 1e-3
            )
            .sum()
            .item()
        )


    record = {
        "lambda": lam,

        "reconstruction_loss":
            recon_final,

        "imq_mmd":
            mmd_final,

        "weighted_mmd":
            weighted_mmd_final,

        "weighted_mmd_over_recon":
            ratio_final,

        "latent_mean_abs_average":
            mean_abs,

        "latent_mean_l2":
            mean_norm,

        "latent_variance_mean":
            var_mean,

        "latent_variance_min":
            var_min,

        "latent_variance_max":
            var_max,

        "active_variance_dims_gt_1e-3":
            active_variance_dims,

        "history":
            history,
    }

    results.append(record)

    print(
        "\nFINAL:"
    )

    print(
        f"  recon        = "
        f"{recon_final:.6f}"
    )

    print(
        f"  IMQ-MMD      = "
        f"{mmd_final:.6f}"
    )

    print(
        f"  lambda*MMD   = "
        f"{weighted_mmd_final:.6f}"
    )

    print(
        f"  weighted/recon = "
        f"{ratio_final:.4f}"
    )

    print(
        f"  |latent mean| avg = "
        f"{mean_abs:.6f}"
    )

    print(
        f"  latent mean L2 = "
        f"{mean_norm:.6f}"
    )

    print(
        f"  latent variance mean = "
        f"{var_mean:.6f}"
    )

    print(
        f"  latent variance range = "
        f"[{var_min:.6f}, "
        f"{var_max:.6f}]"
    )

    print(
        f"  active latent dims = "
        f"{active_variance_dims}/{x_out}"
    )


# ============================================================
# Save diagnostic
# ============================================================

os.makedirs(
    "result",
    exist_ok=True,
)

payload = {
    "experiment":
        "WAE-MMD/IMQ lambda scale diagnostic",

    "purpose":
        (
            "Calibrate WAE regularization scale "
            "without using downstream C2ST power."
        ),

    "dataset":
        "HDGM-D",

    "d":
        2,

    "internal_n":
        n,

    "paper_N":
        8 * n,

    "outer_trial":
        kk,

    "epochs":
        epochs,

    "batch_size":
        batch_size,

    "learning_rate":
        lr,

    "lambda_values":
        lambda_values,

    "IMQ_scales":
        list(IMQ_SCALES),

    "latent_dim":
        x_out,

    "C_base":
        2 * x_out,

    "results":
        results,

    "ts":
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
}

output_path = (
    "result/"
    "diagnose_wae_imq_lambda_scale.json"
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


print(
    "\n"
    + "=" * 70
)

print(
    "SUMMARY"
)

print(
    "=" * 70
)

for r in results:

    print(
        f"lambda={r['lambda']:>4} | "
        f"recon={r['reconstruction_loss']:.5f} | "
        f"MMD={r['imq_mmd']:.5f} | "
        f"lambda*MMD/recon="
        f"{r['weighted_mmd_over_recon']:.3f} | "
        f"var_mean="
        f"{r['latent_variance_mean']:.3f} | "
        f"active="
        f"{r['active_variance_dims_gt_1e-3']}/30"
    )

print(
    f"\nSaved: {output_path}"
)
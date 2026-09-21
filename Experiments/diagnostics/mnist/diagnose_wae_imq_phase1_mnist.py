import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

from utils import (
    Autoencoder_Img,
    Encoder_Img,
    Decoder_Img,
    sample_mnist_semi,
)

# ============================================================
# MNIST WAE Phase-1 diagnostic
#
# Purpose:
#   Compare the released MNIST AE representation learner against
#   an architecture-matched WAE-MMD before looking at test power.
#
# Author MNIST settings retained:
#   z = 100
#   epochs = 1000
#   lr = 0.002
#   batch = 200
#   seed = 1102
#
# Diagnostic WAE lambdas:
#   0, 0.01, 0.1
#
# No downstream C2ST is run here.
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float

SEED = 1102

CHANNELS = 1
IMG_SIZE = 32
Z_DIM = 100

EPOCHS = 1000
LR = 0.002
BATCH_SIZE = 200

INTERNAL_N = 100
PAPER_M = 2 * INTERNAL_N
KK = 0

LAMBDAS = [0.0, 0.01, 0.1]

IMQ_SCALES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)

os.makedirs("result", exist_ok=True)


def reset_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pairwise_squared_distance(A, B):
    A2 = (A * A).sum(dim=1, keepdim=True)
    B2 = (B * B).sum(dim=1, keepdim=True).t()
    D = A2 + B2 - 2.0 * A @ B.t()
    return torch.clamp(D, min=0.0)


def imq_mmd(qz, pz):
    nq = qz.shape[0]
    np_ = pz.shape[0]

    Dqq = pairwise_squared_distance(qz, qz)
    Dpp = pairwise_squared_distance(pz, pz)
    Dqp = pairwise_squared_distance(qz, pz)

    base = 2.0 * Z_DIM
    total = qz.new_tensor(0.0)

    for scale in IMQ_SCALES:
        C = base * scale

        Kqq = C / (C + Dqq)
        Kpp = C / (C + Dpp)
        Kqp = C / (C + Dqp)

        qq = (
            Kqq.sum() - torch.diagonal(Kqq).sum()
        ) / (nq * (nq - 1))

        pp = (
            Kpp.sum() - torch.diagonal(Kpp).sum()
        ) / (np_ * (np_ - 1))

        qp = Kqp.mean()

        total = total + qq + pp - 2.0 * qp

    return total


class WAEImg(nn.Module):
    """
    Same Encoder_Img / Decoder_Img architecture used by the
    released MNIST Autoencoder_Img, but exposes z so that an
    MMD penalty can be added.
    """

    def __init__(self):
        super().__init__()

        self.encoder = Encoder_Img(
            CHANNELS,
            IMG_SIZE,
            Z_DIM,
        )

        self.decoder = Decoder_Img(
            CHANNELS,
            IMG_SIZE,
            Z_DIM,
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def make_loader(S):
    # Fixed shuffle stream for fair comparison.
    g = torch.Generator()
    g.manual_seed(SEED)

    dataset = torch.utils.data.TensorDataset(S)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
    )


def train_author_ae(S):
    reset_seed()

    model = Autoencoder_Img(
        CHANNELS,
        IMG_SIZE,
        Z_DIM,
    ).to(DEVICE, DTYPE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loader = make_loader(S)

    for epoch in range(EPOCHS):
        model.train()

        for (x,) in loader:
            recon = model(x)

            loss = F.mse_loss(
                recon,
                x,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"[AE] epoch {epoch+1:4d}/{EPOCHS} "
                f"recon={loss.item():.6f}",
                flush=True,
            )

    return model


def train_wae(S, lambda_mmd):
    reset_seed()

    model = WAEImg().to(
        DEVICE,
        DTYPE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    loader = make_loader(S)

    for epoch in range(EPOCHS):
        model.train()

        for (x,) in loader:
            recon, z = model(x)

            recon_loss = F.mse_loss(
                recon,
                x,
            )

            if lambda_mmd > 0:
                prior = torch.randn_like(z)

                mmd_loss = imq_mmd(
                    z,
                    prior,
                )
            else:
                mmd_loss = z.new_tensor(0.0)

            loss = (
                recon_loss
                + lambda_mmd * mmd_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(
                f"[WAE lambda={lambda_mmd:g}] "
                f"epoch {epoch+1:4d}/{EPOCHS} "
                f"recon={recon_loss.item():.6f} "
                f"mmd={mmd_loss.item():.6f}",
                flush=True,
            )

    return model


@torch.no_grad()
def evaluate(model, S, is_wae):
    model.eval()

    if is_wae:
        recon, z = model(S)
    else:
        z = model.encoder(S)
        recon = model.decoder(z)

    recon_mse = F.mse_loss(
        recon,
        S,
    ).item()

    # Fixed evaluation prior so every condition is compared
    # against the same N(0,I) sample.
    reset_seed(SEED + 999)

    prior = torch.randn_like(z)

    mmd = imq_mmd(
        z,
        prior,
    ).item()

    z_mean = z.mean(dim=0)
    z_std = z.std(dim=0, unbiased=False)

    return {
        "recon_mse": float(recon_mse),
        "imq_mmd": float(mmd),
        "latent_abs_mean": float(
            z_mean.abs().mean().item()
        ),
        "latent_std_mean": float(
            z_std.mean().item()
        ),
        "latent_std_min": float(
            z_std.min().item()
        ),
        "latent_std_max": float(
            z_std.max().item()
        ),
    }


# ============================================================
# Data
# ============================================================

print("device:", DEVICE)
print(
    f"internal_n={INTERNAL_N}, "
    f"paper_M={PAPER_M}, kk={KK}"
)

reset_seed(819)

os.makedirs("./data/mnist", exist_ok=True)

real_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5],
                    [0.5],
                ),
            ]
        ),
    ),
    batch_size=60000,
    shuffle=True,
)

data_real_all, _ = next(iter(real_loader))

data_fake_all = pickle.load(
    open(
        "./data/Fake_MNIST_data_EP100_N10000.pckl",
        "rb",
    )
)[0]

data_fake_all = torch.from_numpy(
    data_fake_all
).float()

s1_tr, s1_te, s2_tr, s2_te = sample_mnist_semi(
    data_real_all,
    data_fake_all,
    INTERNAL_N,
    INTERNAL_N,
    kk=KK,
)

S_pool = torch.cat(
    [
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ],
    dim=0,
).to(DEVICE, DTYPE)

print("pooled Phase-1 sample:", tuple(S_pool.shape))


# ============================================================
# Run diagnostic
# ============================================================

results = []


print("\n=== A: released-author AE ===")

ae = train_author_ae(S_pool)

ae_metrics = evaluate(
    ae,
    S_pool,
    is_wae=False,
)

results.append(
    {
        "condition": "author_AE",
        "lambda_mmd": None,
        **ae_metrics,
    }
)

print(ae_metrics)

del ae

if torch.cuda.is_available():
    torch.cuda.empty_cache()


for lam in LAMBDAS:

    print(
        f"\n=== WAE lambda={lam:g} ==="
    )

    wae = train_wae(
        S_pool,
        lam,
    )

    metrics = evaluate(
        wae,
        S_pool,
        is_wae=True,
    )

    results.append(
        {
            "condition": "WAE",
            "lambda_mmd": lam,
            **metrics,
        }
    )

    print(metrics)

    del wae

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Save
# ============================================================

payload = {
    "dataset": "MNIST vs Fake MNIST",
    "purpose": "Phase-1-only WAE diagnostic",
    "internal_n": INTERNAL_N,
    "paper_M": PAPER_M,
    "kk": KK,
    "z_dim": Z_DIM,
    "epochs": EPOCHS,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "imq_scales": list(IMQ_SCALES),
    "results": results,
}

output_path = (
    "result/"
    "diagnose_wae_imq_phase1_mnist_n100.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        payload,
        f,
        indent=2,
    )

print("\n=== FINAL COMPARISON ===")

for r in results:
    print(
        f"{r['condition']:10s} "
        f"lambda={str(r['lambda_mmd']):>4s} "
        f"recon={r['recon_mse']:.6f} "
        f"MMD={r['imq_mmd']:.6f} "
        f"|mean|={r['latent_abs_mean']:.4f} "
        f"std_mean={r['latent_std_mean']:.4f} "
        f"std=[{r['latent_std_min']:.4f},"
        f"{r['latent_std_max']:.4f}]"
    )

print("\nSaved:", output_path)

import os
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm

from utils import (
    Encoder_Img,
    Decoder_Img,
    sample_mnist_semi,
    TST_C2ST_D,
    TST_LCE_D,
)

# ============================================================
# MNIST WAE-RL-C2ST pilot
#
# internal n = 100  -> paper M = 200
# 30 outer trials
#
# Phase 1:
#   architecture-matched WAE
#   z=100
#   1000 epochs
#   lr=.002
#   batch=200
#   lambda=.01
#   multi-scale IMQ MMD
#
# Phase 2:
#   match released MNIST AE/RL-C2ST wrapper
#   WAE + discriminator jointly optimized
#   lr=.0004
#   batch=200
#   epochs=2*n
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
dtype = torch.float

MASTER_SEED = 1102

alpha = 0.05

channels = 1
img_size = 32
z_size = 100

INTERNAL_N = 100
PAPER_M = 2 * INTERNAL_N

N_OUTER = 30
N_PER = 100

WAE_EPOCHS = 1000
WAE_LR = 0.002
WAE_BATCH = 200
LAMBDA_MMD = 0.01

C2ST_LR = 0.0004
C2ST_BATCH = 200
C2ST_EPOCHS = 2 * INTERNAL_N

IMQ_SCALES = (
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)

os.makedirs("result", exist_ok=True)


def reset_seed(seed=MASTER_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# WAE
# ============================================================

class WAEImg(nn.Module):

    def __init__(self):
        super().__init__()

        # Same image encoder/decoder architecture as author's AE
        self.encoder = Encoder_Img(
            channels,
            img_size,
            z_size,
        )

        self.decoder = Decoder_Img(
            channels,
            img_size,
            z_size,
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)

        return recon, z


def pairwise_squared_distance(A, B):

    A2 = (A * A).sum(
        dim=1,
        keepdim=True,
    )

    B2 = (B * B).sum(
        dim=1,
        keepdim=True,
    ).t()

    D = (
        A2
        + B2
        - 2.0 * A @ B.t()
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

    base = 2.0 * z_size

    total = qz.new_tensor(0.0)

    for scale in IMQ_SCALES:

        C = base * scale

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

        total = (
            total
            + qq
            + pp
            - 2.0 * qp
        )

    return total


def train_wae(S_pool):

    # Match author's Phase-1 initialization convention
    reset_seed(MASTER_SEED)

    model = WAEImg().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=WAE_LR,
    )

    dataset = torch.utils.data.TensorDataset(
        S_pool
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=WAE_BATCH,
        shuffle=True,
    )

    final_recon = None
    final_mmd = None

    for ep in range(WAE_EPOCHS):

        model.train()

        recon_sum = 0.0
        mmd_sum = 0.0
        nb = 0

        for (x,) in loader:

            recon, z = model(x)

            recon_loss = F.mse_loss(
                recon,
                x,
            )

            prior = torch.randn_like(z)

            mmd_loss = imq_mmd(
                z,
                prior,
            )

            loss = (
                recon_loss
                + LAMBDA_MMD * mmd_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_sum += recon_loss.item()
            mmd_sum += mmd_loss.item()
            nb += 1

        final_recon = recon_sum / nb
        final_mmd = mmd_sum / nb

    model.eval()

    return (
        model,
        final_recon,
        final_mmd,
    )


# ============================================================
# Author-style MNIST discriminator
# ============================================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        def discriminator_block(
            in_filters,
            out_filters,
            bn=True,
        ):
            block = [
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    3,
                    2,
                    1,
                ),
                nn.LeakyReLU(
                    0.2,
                    inplace=True,
                ),
                nn.Dropout2d(0),
            ]

            if bn:
                block.append(
                    nn.BatchNorm2d(
                        out_filters,
                        0.8,
                    )
                )

            return block

        self.model = nn.Sequential(
            *discriminator_block(
                channels,
                8,
                bn=False,
            ),
            *discriminator_block(
                8,
                16,
            ),
            *discriminator_block(
                16,
                32,
            ),
        )

        ds_size = img_size // 2 ** 3

        self.adv_layer = nn.Sequential(
            nn.Linear(
                32 * ds_size ** 2,
                100,
            ),
            nn.ReLU(),
            nn.Linear(
                100,
                20,
            ),
            nn.ReLU(),
            nn.Linear(
                20,
                2,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, img):

        out = self.model(img)

        out = out.view(
            out.shape[0],
            -1,
        )

        return self.adv_layer(out)


# ============================================================
# Data
# ============================================================

print("device:", device, flush=True)

reset_seed(819)

os.makedirs(
    "./data/mnist",
    exist_ok=True,
)

real_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    img_size
                ),
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

data_real_all, _ = next(
    iter(real_loader)
)

with open(
    "./data/Fake_MNIST_data_EP100_N10000.pckl",
    "rb",
) as f:

    data_fake_all = pickle.load(f)[0]

data_fake_all = torch.from_numpy(
    data_fake_all
).float()


# ============================================================
# Outer trials
# ============================================================

trial_records = []

for kk in tqdm.trange(
    N_OUTER,
    desc="MNIST WAE M=200",
):

    s1_tr, s1_te, s2_tr, s2_te = (
        sample_mnist_semi(
            data_real_all,
            data_fake_all,
            INTERNAL_N,
            INTERNAL_N,
            kk=kk,
        )
    )

    # --------------------------------------------------------
    # Phase 1: pooled unlabeled WAE
    # --------------------------------------------------------

    S_pool = torch.cat(
        [
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )

    wae, recon, mmd = train_wae(
        S_pool
    )

    # --------------------------------------------------------
    # Phase 2: author's MNIST joint wrapper
    # --------------------------------------------------------

    S_train = torch.cat(
        [
            s1_tr,
            s2_tr,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )

    y_train = torch.cat(
        [
            torch.zeros(
                INTERNAL_N
            ),
            torch.ones(
                INTERNAL_N
            ),
        ]
    ).to(
        device
    ).long()

    S_test = torch.cat(
        [
            s1_te,
            s2_te,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )

    # Critical:
    # reset before downstream model construction,
    # matching author's MNIST implementation.
    reset_seed(MASTER_SEED)

    discriminator = Discriminator().to(
        device,
        dtype,
    )

    # Whole WAE remains trainable here.
    # This matches the author's joint AE + discriminator Phase 2.
    for p in wae.parameters():
        p.requires_grad = True

    optimizer_D = torch.optim.Adam(
        list(discriminator.parameters())
        + list(wae.parameters()),
        lr=C2ST_LR,
    )

    criterion = nn.CrossEntropyLoss().to(
        device
    )

    dataset = torch.utils.data.TensorDataset(
        S_train,
        y_train,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=C2ST_BATCH,
        shuffle=True,
    )

    def c_model(x):
        reconstructed, _ = wae(x)
        return discriminator(reconstructed)

    for ep in range(C2ST_EPOCHS):

        for xb, yb in loader:

            output = c_model(xb)

            loss = criterion(
                output,
                yb,
            )

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

    # --------------------------------------------------------
    # Held-out test
    # --------------------------------------------------------

    wae.eval()
    discriminator.eval()

    h_s, _, stat_s = TST_C2ST_D(
        S_test,
        INTERNAL_N,
        N_PER,
        alpha,
        c_model,
        device,
        dtype,
    )

    h_l, _, stat_l = TST_LCE_D(
        S_test,
        INTERNAL_N,
        N_PER,
        alpha,
        c_model,
        device,
        dtype,
    )

    trial_records.append(
        {
            "trial": kk,
            "reject_S": int(h_s),
            "reject_L": int(h_l),
            "phase1_recon": float(recon),
            "phase1_mmd": float(mmd),
            "stat_S": float(stat_s),
            "stat_L": float(stat_l),
        }
    )

    running_s = np.mean(
        [
            r["reject_S"]
            for r in trial_records
        ]
    )

    running_l = np.mean(
        [
            r["reject_L"]
            for r in trial_records
        ]
    )

    tqdm.tqdm.write(
        f"trial={kk:02d} "
        f"S={h_s} L={h_l} "
        f"running={running_s:.3f}/{running_l:.3f} "
        f"recon={recon:.6f} "
        f"mmd={mmd:.6f}"
    )


# ============================================================
# Results
# ============================================================

power_s = float(
    np.mean(
        [
            r["reject_S"]
            for r in trial_records
        ]
    )
)

power_l = float(
    np.mean(
        [
            r["reject_L"]
            for r in trial_records
        ]
    )
)

payload = {
    "method": "MNIST WAE-RL-C2ST author-wrapper pilot",
    "internal_n": INTERNAL_N,
    "paper_M": PAPER_M,
    "n_outer_trials": N_OUTER,

    "phase1": {
        "z_dim": z_size,
        "epochs": WAE_EPOCHS,
        "lr": WAE_LR,
        "batch_size": WAE_BATCH,
        "lambda_mmd": LAMBDA_MMD,
        "kernel": "multi-scale IMQ",
    },

    "phase2": {
        "joint_WAE_discriminator": True,
        "lr": C2ST_LR,
        "batch_size": C2ST_BATCH,
        "epochs": C2ST_EPOCHS,
    },

    "RL-C2ST-S": power_s,
    "RL-C2ST-L": power_l,

    "trials": trial_records,

    "timestamp": time.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
}

stem = (
    "result/"
    "pilot_wae_imq_mnist_"
    "M200_lam001_authorwrapper_30trials"
)

with open(
    stem + ".json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        payload,
        f,
        indent=2,
    )

with open(
    stem + ".pkl",
    "wb",
) as f:
    pickle.dump(
        payload,
        f,
    )


print("\n=== DONE ===")

print(
    {
        "internal_n": INTERNAL_N,
        "paper_M": PAPER_M,
        "n_outer_trials": N_OUTER,
        "RL-C2ST-S": power_s,
        "RL-C2ST-L": power_l,
    }
)

print("\nSaved:")
print(" ", stem + ".json")
print(" ", stem + ".pkl")

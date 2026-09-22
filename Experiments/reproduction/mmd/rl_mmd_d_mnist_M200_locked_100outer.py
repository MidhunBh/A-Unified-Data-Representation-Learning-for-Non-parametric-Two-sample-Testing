
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange

from utils import (
    Autoencoder_Img,
    ModelLatentF,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

dev = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

n = 100
M = 200

outer = 100
n_cal = 100
n_perm = 100
alpha = .05

ae_epochs = 1000
ae_lr = .002
ae_batch = 200

mmd_epochs = 1000
mmd_lr = 5e-6

out = (
    "result/"
    "rl_mmd_d_mnist_M200_"
    "locked_100outer.json"
)

os.makedirs("result", exist_ok=True)


def fit_ae(x):
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    ae = Autoencoder_Img(
        1, 32, 100
    ).to(dev, dtype)

    opt = torch.optim.Adam(
        ae.parameters(),
        lr=ae_lr,
    )

    mse = nn.MSELoss()

    dl = DataLoader(
        TensorDataset(x),
        batch_size=ae_batch,
        shuffle=True,
    )

    for _ in range(ae_epochs):
        for (b,) in dl:
            loss = mse(ae(b), b)

            opt.zero_grad()
            loss.backward()
            opt.step()

    enc = ae.encoder
    enc.eval()

    for p in enc.parameters():
        p.requires_grad = False

    return enc


def init_mmd(eps_init):
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    f = ModelLatentF(
        100, 30, 30
    ).to(dev, dtype)

    eps = torch.log(
        torch.tensor(
            [eps_init],
            device=dev,
            dtype=dtype,
        )
    )
    eps.requires_grad_(True)

    sig = torch.tensor(
        [np.sqrt(200.)],
        device=dev,
        dtype=dtype,
        requires_grad=True,
    )

    sig0 = torch.tensor(
        [np.sqrt(.005)],
        device=dev,
        dtype=dtype,
        requires_grad=True,
    )

    opt = torch.optim.Adam(
        list(f.parameters())
        + [eps, sig, sig0],
        lr=mmd_lr,
    )

    return f, eps, sig, sig0, opt


def fit_mmd(z, eps_init):
    f, eps, sig, sig0, opt = init_mmd(
        eps_init
    )

    j0 = j1 = None

    for epoch in range(mmd_epochs):
        e = torch.sigmoid(eps)

        tmp = MMDu(
            f(z),
            n,
            z,
            sig ** 2,
            sig0 ** 2,
            e,
        )

        J = (
            tmp[0] + 1e-8
        ) / torch.sqrt(
            tmp[1] + 1e-8
        )

        if epoch == 0:
            j0 = float(J.detach())

        opt.zero_grad()
        (-J).backward()
        opt.step()

        j1 = float(J.detach())

    return (
        f,
        (sig ** 2).detach(),
        (sig0 ** 2).detach(),
        torch.sigmoid(eps).detach(),
        j0,
        j1,
    )


def test_mmd(
    f,
    z,
    sig,
    sig0,
    eps,
):
    f.eval()

    with torch.no_grad():
        feat = f(z)

    H = np.zeros(n_cal)

    for k in range(n_cal):
        H[k], _, _ = TST_MMD_u(
            feat,
            n,
            n_perm,
            z,
            sig,
            sig0,
            eps,
            alpha,
            k,
        )

    return float(H.mean())


def save(records):
    rates = [
        r["conditional_rejection"]
        for r in records
    ]

    payload = {
        "method": "RL-MMD-D",
        "dataset": "MNIST vs Fake MNIST",

        "internal_n": n,
        "paper_M": M,

        "configuration": {
            "AE": "100-D image AE",
            "AE_epochs": ae_epochs,
            "AE_lr": ae_lr,
            "AE_batch": ae_batch,
            "encoder": "frozen",

            "MMD_head": "100-30-30-30-30",
            "MMD_epochs": mmd_epochs,
            "MMD_lr": mmd_lr,

            "epsilon": "log/sigmoid",
            "kernel_scalars_trainable": True,
            "data_pool": "full",
        },

        "evaluation": {
            "outer_target": outer,
            "outer_completed": len(records),
            "calibrations": n_cal,
            "permutations": n_perm,
            "calibration_seed": "k",
        },

        "rates": rates,

        "mean": (
            float(np.mean(rates))
            if rates else None
        ),

        "records": records,

        "references": {
            "our_locked_MMD_D": .2454,
            "paper_MMD_D": .290,
            "paper_RL_MMD_D": .420,
        },
    }

    tmp = out + ".tmp"

    with open(tmp, "w") as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    os.replace(tmp, out)


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

np.random.seed(819)
torch.manual_seed(819)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(819)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

loader = DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize([.5], [.5]),
        ]),
    ),
    batch_size=60000,
    shuffle=True,
)

real, _ = next(iter(loader))

with open(
    "./data/Fake_MNIST_data_EP100_N10000.pckl",
    "rb",
) as f:
    fake = torch.from_numpy(
        pickle.load(f)[0]
    ).float()


# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------

records = []

if os.path.exists(out):
    with open(out, "r") as f:
        old = json.load(f)

    records = old.get(
        "records",
        [],
    )

start = len(records)

print(
    f"starting outer trial "
    f"{start}/{outer}"
)


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

for kk in trange(
    start,
    outer,
    initial=start,
    total=outer,
    desc="RL-MMD-D M=200",
):
    a, b, c, d = sample_mnist_semi(
        real,
        fake,
        n,
        n,
        kk=kk,
    )

    pool = torch.cat(
        [a, b, c, d]
    ).to(dev, dtype)

    tr = torch.cat(
        [a, c]
    ).to(dev, dtype)

    te = torch.cat(
        [b, d]
    ).to(dev, dtype)

    enc = fit_ae(pool)

    with torch.no_grad():
        ztr = enc(tr).detach()
        zte = enc(te).detach()

    eps_init = max(
        np.random.rand() * 1e-10,
        1e-30,
    )

    (
        f,
        sig,
        sig0,
        eps,
        j0,
        j1,
    ) = fit_mmd(
        ztr,
        eps_init,
    )

    rate = test_mmd(
        f,
        zte,
        sig,
        sig0,
        eps,
    )

    records.append({
        "trial": kk,
        "conditional_rejection": rate,
        "J0": j0,
        "J1": j1,
        "epsilon": float(eps),
        "sigma": float(sig),
        "sigma0": float(sig0),
    })

    save(records)

    mean = np.mean([
        r["conditional_rejection"]
        for r in records
    ])

    print(
        f"\nkk={kk:02d} "
        f"power={rate:.2f} "
        f"running={mean:.3f} "
        f"J={j0:.3f}->{j1:.3f}"
    )


rates = [
    r["conditional_rejection"]
    for r in records
]

mean = float(np.mean(rates))

print("\n==============================")
print("FINAL RL-MMD-D M=200")
print("==============================")
print(f"outer   = {len(rates)}")
print(f"mean    = {mean:.4f}")
print("vanilla = .2454")
print("paper   = .4200")
print("saved   =", out)

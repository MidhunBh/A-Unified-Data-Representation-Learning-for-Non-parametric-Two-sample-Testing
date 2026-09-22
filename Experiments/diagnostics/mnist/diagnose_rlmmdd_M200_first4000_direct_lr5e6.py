
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

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

n = 100
M = 200
alpha = .05

outer = 10
n_cal = 100
n_perm = 100

ae_epochs = 1000
ae_lr = .002
ae_batch = 200

mmd_epochs = 1000
mmd_lr = 5e-6

out = (
    "result/"
    "diagnose_rlmmdd_M200_"
    "first4000_direct_lr5e6.json"
)

os.makedirs("result", exist_ok=True)


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

# Historical Table-3 MMD-D universe.
real = real[:4000]
fake = fake[:4000]


# ------------------------------------------------------------
# Phase 1
# ------------------------------------------------------------

def fit_ae(pool):

    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    ae = Autoencoder_Img(
        1, 32, 100
    ).to(device, dtype)

    opt = torch.optim.Adam(
        ae.parameters(),
        lr=ae_lr,
    )

    loss_fn = nn.MSELoss()

    dl = DataLoader(
        TensorDataset(pool),
        batch_size=ae_batch,
        shuffle=True,
    )

    for _ in range(ae_epochs):

        for (x,) in dl:

            y = ae(x)
            loss = loss_fn(y, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return ae.encoder


# ------------------------------------------------------------
# Phase 2
#
# Same original RL structure:
#   IR -> trainable MMD MLP
#
# But now use:
#   first4000
#   direct epsilon
#   trainable kernel params
#   lr=5e-6
# ------------------------------------------------------------

def fit_mmd(z, kk):

    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    np.random.seed(
        1102 * (kk + 10) + n
    )

    head = ModelLatentF(
        100, 30, 30
    ).to(device, dtype)

    eps = nn.Parameter(
        torch.tensor(
            [np.random.rand() * 1e-8],
            device=device,
            dtype=dtype,
        )
    )

    sigma = nn.Parameter(
        torch.tensor(
            [np.sqrt(2 * 100)],
            device=device,
            dtype=dtype,
        )
    )

    sigma0 = nn.Parameter(
        torch.tensor(
            [np.sqrt(.005)],
            device=device,
            dtype=dtype,
        )
    )

    opt = torch.optim.Adam(
        list(head.parameters())
        + [eps, sigma, sigma0],
        lr=mmd_lr,
    )

    first_j = None
    last_j = None

    for ep in range(mmd_epochs):

        fz = head(z)

        tmp = MMDu(
            fz,
            n,
            z,
            sigma ** 2,
            sigma0 ** 2,
            eps,
        )

        j = (
            tmp[0]
            / torch.sqrt(
                tmp[1] + 1e-8
            )
        )

        if ep == 0:
            first_j = float(
                j.detach()
            )

        opt.zero_grad()
        (-j).backward()
        opt.step()

        last_j = float(
            j.detach()
        )

    return (
        head,
        (sigma ** 2).detach(),
        (sigma0 ** 2).detach(),
        eps.detach(),
        first_j,
        last_j,
    )


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

records = []

for kk in trange(
    outer,
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
    ).to(device, dtype)

    enc = fit_ae(pool)

    for p in enc.parameters():
        p.requires_grad = False

    enc.eval()

    train = torch.cat(
        [a, c]
    ).to(device, dtype)

    test = torch.cat(
        [b, d]
    ).to(device, dtype)

    with torch.no_grad():
        z_tr = enc(train).detach()
        z_te = enc(test).detach()

    (
        head,
        sigma,
        sigma0,
        eps,
        j0,
        j1,
    ) = fit_mmd(z_tr, kk)

    head.eval()

    with torch.no_grad():
        f_te = head(z_te)

    h = np.zeros(n_cal)

    for k in range(n_cal):

        h[k], _, _ = TST_MMD_u(
            f_te,
            n,
            n_perm,
            z_te,
            sigma,
            sigma0,
            eps,
            alpha,
            k,
        )

    rate = float(h.mean())

    records.append({
        "trial": kk,
        "power": rate,
        "J0": j0,
        "J1": j1,
        "eps": float(eps.item()),
        "sigma": float(sigma.item()),
        "sigma0": float(sigma0.item()),
    })

    mean = np.mean([
        r["power"]
        for r in records
    ])

    print(
        f"\nkk={kk:02d} "
        f"power={rate:.2f} "
        f"mean={mean:.3f} "
        f"J={j0:.3f}->{j1:.3f}"
    )


rates = [
    r["power"]
    for r in records
]

mean = float(np.mean(rates))

result = {
    "method": "RL-MMD-D",
    "paper_M": M,
    "internal_n": n,
    "outer": outer,
    "AE": {
        "z": 100,
        "epochs": ae_epochs,
        "lr": ae_lr,
        "batch": ae_batch,
    },
    "MMD": {
        "head": "100-30-30-30-30",
        "epochs": mmd_epochs,
        "lr": mmd_lr,
        "first4000": True,
        "epsilon": "direct",
        "kernel_params_trainable": True,
    },
    "mean": mean,
    "rates": rates,
    "records": records,
    "references": {
        "ours_vanilla": .2454,
        "paper_vanilla": .290,
        "paper_RL_MMD_D": .420,
    },
}

with open(out, "w") as f:
    json.dump(
        result,
        f,
        indent=2,
    )


print(
    "\n============================="
)
print(
    "FINAL RL-MMD-D M=200"
)
print(
    "============================="
)
print("rates =", rates)
print(f"mean  = {mean:.3f}")
print()
print("ours vanilla = .2454")
print("paper vanilla = .290")
print("paper RL-MMD-D = .420")
print()
print("saved:", out)

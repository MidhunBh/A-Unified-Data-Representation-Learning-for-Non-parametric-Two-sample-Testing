
import json, os, pickle
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
    "cuda:0" if torch.cuda.is_available()
    else "cpu"
)
dtype = torch.float32

n = 100
outer = 10
cal = perm = 100
alpha = .05

ae_epochs = 1000
ae_lr = .002
ae_batch = 200

mmd_lr = 5e-6

arms = {
    "legacy_best": (1000, False),
    "gradfix": (1000, True),
    "gradfix_2000": (2000, True),
}

out = (
    "result/"
    "audit_rlmmdd_M200_core_settings.json"
)
os.makedirs("result", exist_ok=True)


def train_ae(pool):
    torch.manual_seed(1102)

    ae = Autoencoder_Img(
        1, 32, 100
    ).to(dev, dtype)

    opt = torch.optim.Adam(
        ae.parameters(),
        lr=ae_lr,
    )
    mse = nn.MSELoss()

    dl = DataLoader(
        TensorDataset(pool),
        batch_size=ae_batch,
        shuffle=True,
    )

    ae.train()

    for _ in range(ae_epochs):
        for (x,) in dl:
            loss = mse(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()

    enc = ae.encoder
    enc.eval()

    for p in enc.parameters():
        p.requires_grad = False

    return enc


def train_mmd(z, epochs, scalars_train, eps_init):
    torch.manual_seed(1102)

    head = ModelLatentF(
        100, 30, 30
    ).to(dev, dtype)

    eps = torch.log(
        torch.tensor(
            [eps_init],
            device=dev,
            dtype=dtype,
        )
    )

    sig = torch.tensor(
        [np.sqrt(2 * z.shape[1])],
        device=dev,
        dtype=dtype,
    )

    sig0 = torch.tensor(
        [np.sqrt(.005)],
        device=dev,
        dtype=dtype,
    )

    if scalars_train:
        eps.requires_grad_(True)
        sig.requires_grad_(True)
        sig0.requires_grad_(True)

    params = list(head.parameters())

    if scalars_train:
        params += [eps, sig, sig0]

    opt = torch.optim.Adam(
        params,
        lr=mmd_lr,
    )

    j0 = j1 = None

    for ep in range(epochs):
        e = torch.sigmoid(eps)
        s = sig ** 2
        s0 = sig0 ** 2

        tmp = MMDu(
            head(z),
            n,
            z,
            s,
            s0,
            e,
        )

        J = (
            tmp[0] + 1e-8
        ) / torch.sqrt(
            tmp[1] + 1e-8
        )

        if ep == 0:
            j0 = float(J.detach())

        opt.zero_grad()
        (-J).backward()
        opt.step()

        j1 = float(J.detach())

    return (
        head,
        (sig ** 2).detach(),
        (sig0 ** 2).detach(),
        torch.sigmoid(eps).detach(),
        j0,
        j1,
    )


def test(head, z, sig, sig0, eps, kk):
    head.eval()

    with torch.no_grad():
        feat = head(z)

    H = np.zeros(cal)

    for k in range(cal):
        H[k], _, _ = TST_MMD_u(
            feat,
            n,
            perm,
            z,
            sig,
            sig0,
            eps,
            alpha,
            k * kk + 2024,
        )

    return float(H.mean())


# Data
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


results = {
    name: []
    for name in arms
}


for kk in trange(
    outer,
    desc="RL-MMD-D core audit",
):
    a, b, c, d = sample_mnist_semi(
        real, fake, n, n, kk=kk
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

    enc = train_ae(pool)

    with torch.no_grad():
        ztr = enc(tr).detach()
        zte = enc(te).detach()

    # Identical epsilon initialization for all arms.
    eps_init = max(
        np.random.rand() * 1e-10,
        1e-30,
    )

    for name, (
        epochs,
        scalar_grad,
    ) in arms.items():

        head, sig, sig0, eps, j0, j1 = train_mmd(
            ztr,
            epochs,
            scalar_grad,
            eps_init,
        )

        power = test(
            head,
            zte,
            sig,
            sig0,
            eps,
            kk,
        )

        results[name].append({
            "trial": kk,
            "power": power,
            "J0": j0,
            "J1": j1,
            "epsilon": float(eps),
            "sigma": float(sig),
            "sigma0": float(sig0),
        })

    msg = []

    for name in arms:
        vals = [
            x["power"]
            for x in results[name]
        ]

        msg.append(
            f"{name}="
            f"{vals[-1]:.2f}/"
            f"{np.mean(vals):.3f}"
        )

    print(
        f"\nkk={kk:02d} | "
        + " | ".join(msg)
    )


summary = {}

for name in arms:
    vals = [
        x["power"]
        for x in results[name]
    ]

    summary[name] = {
        "mean": float(np.mean(vals)),
        "rates": vals,
    }


payload = {
    "paper_M": 200,
    "internal_n": 100,
    "mmd_lr": mmd_lr,
    "arms": {
        "legacy_best": {
            "epochs": 1000,
            "kernel_scalars_trainable": False,
        },
        "gradfix": {
            "epochs": 1000,
            "kernel_scalars_trainable": True,
        },
        "gradfix_2000": {
            "epochs": 2000,
            "kernel_scalars_trainable": True,
        },
    },
    "fixed": {
        "data_pool": "full",
        "AE": "100-D, 1000 ep, lr .002, batch 200",
        "encoder": "frozen",
        "MMD_head": "100-30-30-30-30",
        "epsilon": "log/sigmoid",
        "mmd_lr": 5e-6,
        "test_seed": "k*kk+2024",
    },
    "summary": summary,
    "results": results,
    "references": {
        "vanilla_locked": .2454,
        "paper_vanilla": .290,
        "paper_RL_MMD_D": .420,
        "previous_RL_lr5e6_10trial": .259,
    },
}

with open(out, "w") as f:
    json.dump(payload, f, indent=2)


print("\n==============================")
print("RL-MMD-D CORE SETTINGS AUDIT")
print("==============================")

for name in arms:
    print(
        f"{name:16s}: "
        f"{summary[name]['mean']:.3f}"
    )
    print(
        "  ",
        summary[name]["rates"],
    )

print()
print("locked vanilla = .2454")
print("previous RL    = .259")
print("paper RL       = .420")
print("saved          =", out)


import json, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange

from utils import (
    Autoencoder_Img, MMDu, Pdist2,
    sample_mnist_semi
)

dev = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

n = 100
outer = 10
alpha = .05
n_perm = 100
n_cal = 100

lambdas = [0, .03, .1, .3, 1, 3]

out = (
    "result/"
    "rl_mmdd_kernel_fusion_M200_10trials.json"
)

os.makedirs("result", exist_ok=True)


class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()

        def block(a, b, bn=True):
            x = [
                nn.Conv2d(a, b, 3, 2, 1),
                nn.LeakyReLU(.2, inplace=True),
                nn.Dropout2d(0),
            ]
            if bn:
                x += [nn.BatchNorm2d(b, .8)]
            return x

        self.net = nn.Sequential(
            *block(1, 16, False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        self.fc = nn.Linear(
            128 * 2 * 2, 100
        )

    def forward(self, x):
        return self.fc(
            self.net(x).flatten(1)
        )


def mmd2(K):
    a = K[:n, :n]
    b = K[n:, n:]
    c = K[:n, n:]

    xx = (
        a.sum() - a.diag().sum()
    ) / (n * (n - 1))

    yy = (
        b.sum() - b.diag().sum()
    ) / (n * (n - 1))

    xy = (
        c.sum() - c.diag().sum()
    ) / (n * (n - 1))

    return xx + yy - 2 * xy


def J(K):
    a = K[:n, :n]
    b = K[n:, n:]
    c = K[:n, n:]

    h = a + b - c - c.T
    v1 = (h.sum(1) / n).square().sum() / n
    v2 = h.sum() / (n * n)
    var = 4 * (v1 - v2.square())

    return mmd2(K) / torch.sqrt(
        var.clamp_min(0) + 1e-8
    )


def test_K(K, seed):
    obs = float(mmd2(K))
    null = np.zeros(n_perm)
    N = 2 * n

    for r in range(n_perm):
        np.random.seed(
            1102 + 3 * r + seed
        )

        idx = np.random.choice(
            N, N, replace=False
        )

        P = K[idx][:, idx]
        null[r] = float(mmd2(P))

    q = np.sort(null)[
        int(np.ceil(
            n_perm * (1 - alpha)
        ))
    ]

    return int(obs > q)


def fit_ae(pool):
    torch.manual_seed(1102)

    ae = Autoencoder_Img(
        1, 32, 100
    ).to(dev)

    opt = torch.optim.Adam(
        ae.parameters(), lr=.002
    )

    dl = DataLoader(
        TensorDataset(pool),
        200,
        shuffle=True,
    )

    mse = nn.MSELoss()

    for _ in range(1000):
        for (x,) in dl:
            loss = mse(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()

    ae.encoder.eval()

    for p in ae.encoder.parameters():
        p.requires_grad = False

    return ae.encoder


def fit_mmd(x, kk):
    torch.manual_seed(kk * 19 + n)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            kk * 19 + n
        )

    np.random.seed(
        1102 * (kk + 10) + n
    )

    f = Featurizer().to(dev)

    eps = nn.Parameter(
        torch.tensor(
            [np.random.rand() * 1e-8],
            device=dev,
        )
    )

    sig = nn.Parameter(
        torch.tensor(
            [np.sqrt(2 * 32 * 32)],
            device=dev,
        )
    )

    sig0 = nn.Parameter(
        torch.tensor(
            [np.sqrt(.005)],
            device=dev,
        )
    )

    opt = torch.optim.Adam(
        list(f.parameters())
        + [eps, sig, sig0],
        lr=.001,
    )

    raw = x.flatten(1)

    for _ in range(2000):
        feat = f(x)

        tmp = MMDu(
            feat, n, raw,
            sig ** 2,
            sig0 ** 2,
            eps,
        )

        j = tmp[0] / torch.sqrt(
            tmp[1] + 1e-8
        )

        opt.zero_grad()
        (-j).backward()
        opt.step()

    return (
        f,
        (sig ** 2).detach(),
        (sig0 ** 2).detach(),
        eps.detach(),
    )


def base_kernel(f, x, sig, sig0, eps):
    raw = x.flatten(1)

    with torch.no_grad():
        feat = f(x)

    return MMDu(
        feat, n, raw,
        sig, sig0, eps
    )[2].detach()


def ae_kernel(z, bw):
    return torch.exp(
        -Pdist2(z, z) / bw
    )


# Data
np.random.seed(819)
torch.manual_seed(819)

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
    60000,
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

real = real[:4000]
fake = fake[:4000]

vanilla = []
fusion = []
chosen = []


for kk in trange(
    outer,
    desc="Kernel-fusion RL-MMD-D",
):
    a, b, c, d = sample_mnist_semi(
        real, fake, n, n, kk=kk
    )

    pool = torch.cat(
        [a, b, c, d]
    ).to(dev)

    tr = torch.cat(
        [a, c]
    ).to(dev)

    te = torch.cat(
        [b, d]
    ).to(dev)

    enc = fit_ae(pool)
    f, sig, sig0, eps = fit_mmd(
        tr, kk
    )

    f.eval()

    K0tr = base_kernel(
        f, tr, sig, sig0, eps
    )

    K0te = base_kernel(
        f, te, sig, sig0, eps
    )

    with torch.no_grad():
        ztr = enc(tr)
        zte = enc(te)

    D = Pdist2(ztr, ztr)

    mask = ~torch.eye(
        2 * n,
        dtype=torch.bool,
        device=dev,
    )

    bw = torch.median(
        D[mask][D[mask] > 0]
    )

    Katr = ae_kernel(ztr, bw)
    Kate = ae_kernel(zte, bw)

    scores = [
        float(J(K0tr + l * Katr))
        for l in lambdas
    ]

    lam = lambdas[
        int(np.argmax(scores))
    ]

    H0 = np.zeros(n_cal)
    HR = np.zeros(n_cal)

    Krl = K0te + lam * Kate

    for k in range(n_cal):
        H0[k] = test_K(K0te, k)
        HR[k] = test_K(Krl, k)

    v = float(H0.mean())
    r = float(HR.mean())

    vanilla.append(v)
    fusion.append(r)
    chosen.append(lam)

    print(
        f"\nkk={kk:02d} "
        f"lambda={lam:g} "
        f"vanilla={v:.2f} "
        f"RL={r:.2f} | "
        f"means={np.mean(vanilla):.3f}/"
        f"{np.mean(fusion):.3f}"
    )


result = {
    "paper_M": 200,
    "internal_n": 100,
    "method": (
        "kernel-fusion RL-MMD-D extension"
    ),
    "lambda_grid": lambdas,
    "lambda": chosen,
    "vanilla": vanilla,
    "RL": fusion,
    "mean_vanilla": float(
        np.mean(vanilla)
    ),
    "mean_RL": float(
        np.mean(fusion)
    ),
    "paper": {
        "MMD_D": .290,
        "RL_MMD_D": .420,
    },
}

with open(out, "w") as f:
    json.dump(result, f, indent=2)


print("\n============================")
print("KERNEL-FUSION RL-MMD-D")
print("============================")
print(
    f"vanilla = "
    f"{np.mean(vanilla):.3f}"
)
print(
    f"RL      = "
    f"{np.mean(fusion):.3f}"
)
print("lambda  =", chosen)
print("paper   = .290 -> .420")
print("saved   =", out)

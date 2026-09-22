
import json, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange

from utils import Autoencoder_Img, MMDu, TST_MMD_u, sample_mnist_semi

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

n = 100
outer = 10
alpha = .05
n_cal = n_perm = 100

os.makedirs("result", exist_ok=True)
out = "result/rl_mmdd_residual_M200_10trials.json"


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

        self.conv = nn.Sequential(
            *block(1, 16, False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )
        self.fc = nn.Linear(128 * 2 * 2, 100)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.flatten(1))


def fit_ae(pool):
    torch.manual_seed(1102)

    ae = Autoencoder_Img(1, 32, 100).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=.002)
    dl = DataLoader(TensorDataset(pool), 200, shuffle=True)
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


def fit_test(a, b, c, d, kk, enc, residual):
    torch.manual_seed(kk * 19 + n)
    np.random.seed(1102 * (kk + 10) + n)

    f = Featurizer().to(dev)

    eps = nn.Parameter(
        torch.tensor(
            [np.random.rand() * 1e-8],
            device=dev,
        )
    )
    sig = nn.Parameter(
        torch.tensor([np.sqrt(2 * 32 * 32)], device=dev)
    )
    sig0 = nn.Parameter(
        torch.tensor([np.sqrt(.005)], device=dev)
    )

    # gamma?0 initially => almost exact vanilla MMD-D.
    g = nn.Parameter(torch.tensor([-12.0], device=dev))

    params = list(f.parameters()) + [eps, sig, sig0]
    if residual:
        params += [g]

    opt = torch.optim.Adam(params, lr=.001)

    x = torch.cat([a, c]).to(dev)
    raw = x.flatten(1)

    with torch.no_grad():
        z = enc(x)

        # Label-free scale normalization.
        mu = z.mean(0, keepdim=True)
        sd = z.std(0, keepdim=True).clamp_min(1e-4)
        z = (z - mu) / sd

    for _ in range(2000):
        feat = f(x)

        if residual:
            gamma = torch.sigmoid(g)
            feat = torch.cat(
                [feat, torch.sqrt(gamma) * z],
                dim=1,
            )

        tmp = MMDu(
            feat, n, raw,
            sig ** 2,
            sig0 ** 2,
            eps,
        )

        J = tmp[0] / torch.sqrt(tmp[1] + 1e-8)

        opt.zero_grad()
        (-J).backward()
        opt.step()

    xt = torch.cat([b, d]).to(dev)
    raw_t = xt.flatten(1)

    f.eval()

    with torch.no_grad():
        feat_t = f(xt)

        if residual:
            zt = (enc(xt) - mu) / sd
            gamma = torch.sigmoid(g)
            feat_t = torch.cat(
                [feat_t, torch.sqrt(gamma) * zt],
                1,
            )
        else:
            gamma = torch.tensor(0.0)

        sigma = (sig ** 2).detach()
        sigma0 = (sig0 ** 2).detach()
        ep = eps.detach()

    H = np.zeros(n_cal)

    for k in range(n_cal):
        H[k], _, _ = TST_MMD_u(
            feat_t, n, n_perm, raw_t,
            sigma, sigma0, ep, alpha, k,
        )

    return float(H.mean()), float(gamma)


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

with open("./data/Fake_MNIST_data_EP100_N10000.pckl", "rb") as f:
    fake = torch.from_numpy(pickle.load(f)[0]).float()

real = real[:4000]
fake = fake[:4000]

vanilla, residual, gammas = [], [], []

for kk in trange(outer, desc="Residual RL-MMD-D"):

    a, b, c, d = sample_mnist_semi(
        real, fake, n, n, kk=kk
    )

    pool = torch.cat([a, b, c, d]).to(dev)
    enc = fit_ae(pool)

    v, _ = fit_test(
        a, b, c, d, kk, enc, False
    )

    r, g = fit_test(
        a, b, c, d, kk, enc, True
    )

    vanilla.append(v)
    residual.append(r)
    gammas.append(g)

    print(
        f"\nkk={kk:02d} "
        f"vanilla={v:.2f} "
        f"RL={r:.2f} "
        f"gamma={g:.4f} | "
        f"means={np.mean(vanilla):.3f}/"
        f"{np.mean(residual):.3f}"
    )


result = {
    "paper_M": 200,
    "internal_n": 100,
    "vanilla": vanilla,
    "residual_RL": residual,
    "gamma": gammas,
    "mean_vanilla": float(np.mean(vanilla)),
    "mean_residual_RL": float(np.mean(residual)),
    "paper": {
        "MMD_D": .290,
        "RL_MMD_D": .420,
    },
}

with open(out, "w") as f:
    json.dump(result, f, indent=2)

print("\n==============================")
print("RESIDUAL RL-MMD-D")
print("==============================")
print(f"vanilla : {np.mean(vanilla):.3f}")
print(f"RL      : {np.mean(residual):.3f}")
print(f"gamma   : {np.mean(gammas):.5f}")
print("paper   : .290 -> .420")
print("saved   :", out)

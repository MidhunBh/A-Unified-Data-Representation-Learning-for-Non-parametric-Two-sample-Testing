
import json, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange

from utils import MatConvert, MMDu, TST_MMD_u, sample_mnist_semi

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

n = 100
outer = 30
cal = perm = 100
alpha = .05

epochs = 2000
lr = .001

out = "result/audit_mmdd_bnmode_M200_30outer.json"
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
                x.append(nn.BatchNorm2d(b, .8))
            return x

        self.conv = nn.Sequential(
            *block(1, 16, False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        self.fc = nn.Linear(128 * 2 * 2, 100)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def train_model(a, c, kk):
    seed = kk * 19 + n

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    np.random.seed(1102 * (kk + 10) + n)

    f = Featurizer().to(dev, dtype)

    eps = MatConvert(
        np.random.rand(1) * 1e-8,
        dev, dtype,
    )
    sig = MatConvert(
        np.ones(1) * np.sqrt(2048),
        dev, dtype,
    )
    sig0 = MatConvert(
        np.ones(1) * np.sqrt(.005),
        dev, dtype,
    )

    eps.requires_grad_(True)
    sig.requires_grad_(True)
    sig0.requires_grad_(True)

    opt = torch.optim.Adam(
        list(f.parameters()) + [eps, sig, sig0],
        lr=lr,
    )

    ds = TensorDataset(a, c)
    dl = DataLoader(ds, batch_size=100, shuffle=True)

    J = None

    f.train()

    for _ in range(epochs):
        for x, y in dl:
            x = x.to(dev, dtype)
            y = y.to(dev, dtype)

            z = torch.cat([x, y])
            feat = f(z)

            tmp = MMDu(
                feat,
                x.shape[0],
                z.flatten(1),
                sig ** 2,
                sig0 ** 2,
                eps,
            )

            J = tmp[0] / torch.sqrt(tmp[1] + 1e-8)

            opt.zero_grad()
            (-J).backward()
            opt.step()

    return (
        f,
        (sig ** 2).detach(),
        (sig0 ** 2).detach(),
        eps.detach(),
        float(J.detach()),
    )


def test_mode(f, z, sig, sig0, eps, train_mode):
    if train_mode:
        f.train()
    else:
        f.eval()

    with torch.no_grad():
        feat = f(z)

    H = np.zeros(cal)

    for k in range(cal):
        H[k], _, _ = TST_MMD_u(
            feat,
            n,
            perm,
            z.flatten(1),
            sig,
            sig0,
            eps,
            alpha,
            k,
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

with open("./data/Fake_MNIST_data_EP100_N10000.pckl", "rb") as f:
    fake = pickle.load(f)[0]

if not torch.is_tensor(fake):
    fake = torch.from_numpy(fake)

fake = fake.float()

# Historical notebook universe.
real = real[:4000]
fake = fake[:4000]


eval_rates = []
train_rates = []
records = []

for kk in trange(outer, desc="MMD-D BN-mode audit"):
    a, b, c, d = sample_mnist_semi(
        real, fake, n, n, kk=kk
    )

    a = a.to(dev, dtype)
    b = b.to(dev, dtype)
    c = c.to(dev, dtype)
    d = d.to(dev, dtype)

    f, sig, sig0, eps, J = train_model(a, c, kk)

    zte = torch.cat([b, d])

    p_eval = test_mode(
        f, zte, sig, sig0, eps, False
    )

    p_train = test_mode(
        f, zte, sig, sig0, eps, True
    )

    eval_rates.append(p_eval)
    train_rates.append(p_train)

    records.append({
        "trial": kk,
        "eval": p_eval,
        "train": p_train,
        "delta": p_train - p_eval,
        "J": J,
        "epsilon": float(eps),
        "sigma": float(sig),
        "sigma0": float(sig0),
    })

    print(
        f"\nkk={kk:02d} "
        f"eval={p_eval:.2f} "
        f"train={p_train:.2f} "
        f"delta={p_train-p_eval:+.2f} | "
        f"means={np.mean(eval_rates):.3f}/"
        f"{np.mean(train_rates):.3f}"
    )


delta = np.array(train_rates) - np.array(eval_rates)

result = {
    "paper_M": 200,
    "internal_n": 100,
    "outer": outer,
    "configuration": {
        "pool": "first4000",
        "epsilon": "direct",
        "epochs": epochs,
        "lr": lr,
        "architecture": "16-32-64-128-100",
        "calibrations": cal,
        "permutations": perm,
    },
    "eval_mode": {
        "mean": float(np.mean(eval_rates)),
        "rates": eval_rates,
    },
    "train_mode": {
        "mean": float(np.mean(train_rates)),
        "rates": train_rates,
    },
    "paired": {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "train_better": int(np.sum(delta > 0)),
        "equal": int(np.sum(delta == 0)),
        "eval_better": int(np.sum(delta < 0)),
    },
    "records": records,
    "references": {
        "locked_eval_100outer": .2454,
        "paper_MMD_D": .290,
        "paper_RL_MMD_D": .420,
    },
}

with open(out, "w") as f:
    json.dump(result, f, indent=2)


print("\n==============================")
print("PAIRED BN-MODE AUDIT")
print("==============================")
print(f"eval mode   = {np.mean(eval_rates):.4f}")
print(f"train mode  = {np.mean(train_rates):.4f}")
print(f"mean delta  = {np.mean(delta):+.4f}")
print(
    "wins        = "
    f"{np.sum(delta > 0)} train / "
    f"{np.sum(delta == 0)} tie / "
    f"{np.sum(delta < 0)} eval"
)
print()
print("paper MMD-D    = .290")
print("paper RL-MMD-D = .420")
print("saved          =", out)

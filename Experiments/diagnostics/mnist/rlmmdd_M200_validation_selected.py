
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


# ============================================================
# CONFIG
# ============================================================

dev = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

n = 100
paper_M = 200

outer = 10
cal = 100
perm = 100
alpha = .05

ae_epochs = 1000
ae_lr = .002
ae_batch = 200

mmd_lr = 5e-6

max_epochs = 2000
check_every = 50
min_epoch = 100

val_n = 20
fit_n = n - val_n

out = (
    "result/"
    "rlmmdd_M200_validation_selected_10trials.json"
)

os.makedirs("result", exist_ok=True)


# ============================================================
# PHASE 1 ? AUTHOR MNIST AE
# ============================================================

def fit_ae(pool):
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
        TensorDataset(pool),
        batch_size=ae_batch,
        shuffle=True,
    )

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


# ============================================================
# MMD MODEL
# ============================================================

def init_mmd(eps_init):
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

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
        list(head.parameters())
        + [eps, sig, sig0],
        lr=mmd_lr,
    )

    return head, eps, sig, sig0, opt


def J_value(
    head,
    z,
    n_side,
    eps,
    sig,
    sig0,
):
    tmp = MMDu(
        head(z),
        n_side,
        z,
        sig ** 2,
        sig0 ** 2,
        torch.sigmoid(eps),
    )

    return (
        tmp[0] + 1e-8
    ) / torch.sqrt(
        tmp[1] + 1e-8
    )


# ============================================================
# TRAINING-ONLY MODEL SELECTION
# ============================================================

def choose_epoch(z, kk, eps_init):
    rng = np.random.default_rng(
        7000 + kk
    )

    p = rng.permutation(n)
    q = rng.permutation(n)

    p_fit = p[:fit_n]
    p_val = p[fit_n:]

    q_fit = q[:fit_n] + n
    q_val = q[fit_n:] + n

    fit_idx = np.r_[
        p_fit,
        q_fit,
    ]

    val_idx = np.r_[
        p_val,
        q_val,
    ]

    fit_idx = torch.tensor(
        fit_idx,
        device=dev,
    )

    val_idx = torch.tensor(
        val_idx,
        device=dev,
    )

    z_fit = z[fit_idx]
    z_val = z[val_idx]

    head, eps, sig, sig0, opt = (
        init_mmd(eps_init)
    )

    best_epoch = min_epoch
    best_val = -np.inf
    trace = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        head.train()

        J = J_value(
            head,
            z_fit,
            fit_n,
            eps,
            sig,
            sig0,
        )

        opt.zero_grad()
        (-J).backward()
        opt.step()

        if epoch % check_every:
            continue

        if epoch < min_epoch:
            continue

        head.eval()

        with torch.no_grad():
            Jv = J_value(
                head,
                z_val,
                val_n,
                eps,
                sig,
                sig0,
            )

        score = float(Jv)

        trace.append({
            "epoch": epoch,
            "val_J": score,
        })

        if np.isfinite(score):
            if score > best_val:
                best_val = score
                best_epoch = epoch

    return (
        best_epoch,
        best_val,
        trace,
    )


# ============================================================
# REFIT ON ALL TRAINING DATA
# ============================================================

def fit_full(
    z,
    epochs,
    eps_init,
):
    head, eps, sig, sig0, opt = (
        init_mmd(eps_init)
    )

    first_J = None
    last_J = None

    for epoch in range(
        1,
        epochs + 1,
    ):
        J = J_value(
            head,
            z,
            n,
            eps,
            sig,
            sig0,
        )

        if epoch == 1:
            first_J = float(
                J.detach()
            )

        opt.zero_grad()
        (-J).backward()
        opt.step()

        last_J = float(
            J.detach()
        )

    return (
        head,
        (sig ** 2).detach(),
        (sig0 ** 2).detach(),
        torch.sigmoid(eps).detach(),
        first_J,
        last_J,
    )


# ============================================================
# PAPER-STYLE TEST
#
# Important:
# rd_seed = k
#
# NOT k * kk + 2024
# ============================================================

def test(
    head,
    z,
    sig,
    sig0,
    eps,
):
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
            k,
        )

    return float(H.mean())


# ============================================================
# DATA
# ============================================================

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
            T.Normalize(
                [.5],
                [.5],
            ),
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


# ============================================================
# EXPERIMENT
#
# fixed1000:
#   corrected gradfix configuration
#
# selected:
#   epoch chosen using training-only validation
#   then refit from scratch on all 100+100 training samples
# ============================================================

fixed = []
selected = []
records = []


for kk in trange(
    outer,
    desc="RL-MMD-D validation selection",
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

    train = torch.cat(
        [a, c]
    ).to(dev, dtype)

    test_z = torch.cat(
        [b, d]
    ).to(dev, dtype)

    enc = fit_ae(pool)

    with torch.no_grad():
        ztr = enc(train).detach()
        zte = enc(test_z).detach()

    # Preserve global RNG behavior of the working RL branch.
    eps_init = max(
        np.random.rand() * 1e-10,
        1e-30,
    )

    best_epoch, best_val, trace = (
        choose_epoch(
            ztr,
            kk,
            eps_init,
        )
    )

    # ---------------------------
    # Fixed 1000-epoch control
    # ---------------------------

    (
        h0,
        s0,
        s00,
        e0,
        j00,
        j01,
    ) = fit_full(
        ztr,
        1000,
        eps_init,
    )

    p_fixed = test(
        h0,
        zte,
        s0,
        s00,
        e0,
    )

    # ---------------------------
    # Validation-selected model
    # ---------------------------

    (
        hs,
        ss,
        ss0,
        es,
        js0,
        js1,
    ) = fit_full(
        ztr,
        best_epoch,
        eps_init,
    )

    p_sel = test(
        hs,
        zte,
        ss,
        ss0,
        es,
    )

    fixed.append(p_fixed)
    selected.append(p_sel)

    records.append({
        "trial": kk,
        "selected_epoch": best_epoch,
        "best_val_J": best_val,
        "fixed_power": p_fixed,
        "selected_power": p_sel,
        "fixed_J": [j00, j01],
        "selected_J": [js0, js1],
        "val_trace": trace,
    })

    print(
        f"\nkk={kk:02d} "
        f"epoch={best_epoch:4d} "
        f"valJ={best_val:.3f} | "
        f"fixed={p_fixed:.2f} "
        f"selected={p_sel:.2f} | "
        f"means="
        f"{np.mean(fixed):.3f}/"
        f"{np.mean(selected):.3f}"
    )


# ============================================================
# SAVE
# ============================================================

payload = {
    "method": (
        "RL-MMD-D validation-selected "
        "training duration"
    ),

    "paper_M": paper_M,
    "internal_n": n,
    "outer": outer,

    "phase1": {
        "AE_latent": 100,
        "epochs": ae_epochs,
        "lr": ae_lr,
        "batch": ae_batch,
        "pooled_unlabelled": True,
        "encoder_frozen": True,
    },

    "phase2": {
        "head": "100-30-30-30-30",
        "lr": mmd_lr,
        "max_epochs": max_epochs,
        "validation_per_group": val_n,
        "check_every": check_every,
        "minimum_epoch": min_epoch,
        "kernel_scalars_trainable": True,
        "epsilon": "log/sigmoid",
    },

    "evaluation": {
        "calibrations": cal,
        "permutations": perm,
        "rd_seed": "k",
    },

    "fixed1000": {
        "mean": float(np.mean(fixed)),
        "rates": fixed,
    },

    "validation_selected": {
        "mean": float(np.mean(selected)),
        "rates": selected,
        "epochs": [
            r["selected_epoch"]
            for r in records
        ],
    },

    "records": records,

    "references": {
        "locked_vanilla": .2454,
        "paper_vanilla": .290,
        "paper_RL_MMD_D": .420,
    },
}

with open(out, "w") as f:
    json.dump(
        payload,
        f,
        indent=2,
    )


print("\n================================")
print("RL-MMD-D VALIDATION SELECTION")
print("================================")

print(
    f"fixed 1000 : "
    f"{np.mean(fixed):.3f}"
)
print(
    f"selected   : "
    f"{np.mean(selected):.3f}"
)
print(
    "epochs     :",
    [
        r["selected_epoch"]
        for r in records
    ],
)

print()
print("locked vanilla = .2454")
print("paper vanilla  = .290")
print("paper RL       = .420")
print("saved          =", out)

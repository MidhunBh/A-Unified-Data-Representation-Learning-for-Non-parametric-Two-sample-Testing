import os
import json
import pickle
import platform

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from utils import (
    MatConvert,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# HISTORICAL MMD-D MNIST DIAGNOSTIC
#
# paper M = 200
# internal n = 100
#
# Compare two historical choices found in released code:
#
# data pool:
#   FULL       = all available MNIST / fake MNIST
#   FIRST4000  = first 4000 of each
#
# epsilon parameterization:
#   SIGMOID = log(U(0,1e-10)) -> sigmoid
#   DIRECT  = U(0,1e-8) used directly
#
# Four variants, same kk = 0,...,9:
#
# A full_sigmoid
# B first4000_sigmoid
# C full_direct
# D first4000_direct
#
# D corresponds to the Tian notebook cell whose recorded
# 100-outer-trial result finishes at ~0.2885.
#
# We also compare against the notebook's recorded first
# ten conditional rejection rates.
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float

ALPHA = 0.05
CHANNELS = 1
IMG_SIZE = 32

INTERNAL_N = 100
PAPER_M = 200

N_OUTER = 10
N_PER = 100
N_CAL = 100

EPOCHS = 2000
BATCH = 100
LR = 0.001

OUTPUT = (
    "result/"
    "diagnose_mmdd_mnist_M200_"
    "historical_2x2_10trials.json"
)

os.makedirs("result", exist_ok=True)

print("device:", device)
print("torch:", torch.__version__)
print("python:", platform.python_version())


# ============================================================
# MMD-D featurizer from released code
# ============================================================

class Featurizer(nn.Module):

    def __init__(self):

        super().__init__()

        def block(
            in_filters,
            out_filters,
            bn=True,
        ):

            layers = [
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
                layers.append(
                    nn.BatchNorm2d(
                        out_filters,
                        0.8,
                    )
                )

            return layers

        self.model = nn.Sequential(
            *block(CHANNELS, 16, bn=False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        ds_size = IMG_SIZE // 2 ** 4

        self.adv_layer = nn.Sequential(
            nn.Linear(
                128 * ds_size ** 2,
                100,
            )
        )

    def forward(self, x):

        x = self.model(x)

        x = x.view(
            x.shape[0],
            -1,
        )

        return self.adv_layer(x)


# ============================================================
# Load data exactly once
# ============================================================

np.random.seed(819)
torch.manual_seed(819)

if torch.cuda.is_available():
    torch.cuda.manual_seed(819)

torch.backends.cudnn.deterministic = True


loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(
                    IMG_SIZE
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
    iter(loader)
)


with open(
    "./data/Fake_MNIST_data_EP100_N10000.pckl",
    "rb",
) as f:

    data_fake_all = pickle.load(f)[0]


data_fake_all = torch.from_numpy(
    data_fake_all
).float()


print(
    "real:",
    tuple(data_real_all.shape),
)

print(
    "fake:",
    tuple(data_fake_all.shape),
)


# ============================================================
# Recorded Tian notebook fingerprint
#
# first ten outer datasets of successful M=200 cell
# ============================================================

AUTHOR_FIRST10 = np.array(
    [
        0.00,
        0.85,
        0.00,
        1.00,
        0.00,
        0.00,
        0.00,
        0.05,
        0.00,
        0.61,
    ],
    dtype=float,
)

print(
    "\nAuthor notebook first-10 mean:",
    AUTHOR_FIRST10.mean(),
)


# ============================================================
# Variants
# ============================================================

VARIANTS = [
    {
        "name": "full_sigmoid",
        "first4000": False,
        "epsilon_mode": "sigmoid",
    },
    {
        "name": "first4000_sigmoid",
        "first4000": True,
        "epsilon_mode": "sigmoid",
    },
    {
        "name": "full_direct",
        "first4000": False,
        "epsilon_mode": "direct",
    },
    {
        "name": "first4000_direct",
        "first4000": True,
        "epsilon_mode": "direct",
    },
]


def run_trial(
    kk,
    first4000,
    epsilon_mode,
):

    # --------------------------------------------------------
    # Data universe
    # --------------------------------------------------------

    if first4000:

        real_pool = (
            data_real_all[:4000]
        )

        fake_pool = (
            data_fake_all[:4000]
        )

    else:

        real_pool = data_real_all
        fake_pool = data_fake_all


    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_mnist_semi(
        real_pool,
        fake_pool,
        INTERNAL_N,
        INTERNAL_N,
        kk=kk,
    )


    # --------------------------------------------------------
    # Released initialization convention
    # --------------------------------------------------------

    seed = (
        kk * 19
        + INTERNAL_N
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


    np.random.seed(
        1102 * (kk + 10)
        + INTERNAL_N
    )


    featurizer = (
        Featurizer()
        .to(device, dtype)
    )


    # --------------------------------------------------------
    # Historical epsilon variants
    # --------------------------------------------------------

    if epsilon_mode == "sigmoid":

        # Earlier / original deep-kernel parameterization:
        #
        # epsilonOPT = log(U(0,1e-10))
        # ep = sigmoid(epsilonOPT)

        epsilonOPT = torch.log(
            MatConvert(
                np.random.rand(1)
                * 10 ** (-10),
                device,
                dtype,
            )
        )

    elif epsilon_mode == "direct":

        # Tian notebook cell that produced ~0.2885:
        #
        # epsilonOPT = U(0,1e-8)
        # ep = epsilonOPT

        epsilonOPT = MatConvert(
            np.random.rand(1)
            * 10 ** (-8),
            device,
            dtype,
        )

    else:
        raise ValueError(
            epsilon_mode
        )


    epsilonOPT.requires_grad = True


    sigmaOPT = MatConvert(
        np.ones(1)
        * np.sqrt(
            2 * 32 * 32
        ),
        device,
        dtype,
    )

    sigmaOPT.requires_grad = True


    sigma0OPT = MatConvert(
        np.ones(1)
        * np.sqrt(0.005),
        device,
        dtype,
    )

    sigma0OPT.requires_grad = True


    optimizer = torch.optim.Adam(
        list(
            featurizer.parameters()
        )
        + [
            epsilonOPT,
            sigmaOPT,
            sigma0OPT,
        ],
        lr=LR,
    )


    dataset = (
        torch.utils.data.TensorDataset(
            s1_tr,
            s2_tr,
        )
    )


    train_loader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH,
            shuffle=True,
        )
    )


    # --------------------------------------------------------
    # Direct deep-MMD training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        for (
            real_imgs,
            fake_imgs,
        ) in train_loader:

            real_imgs = real_imgs.to(
                device,
                dtype,
            )

            fake_imgs = fake_imgs.to(
                device,
                dtype,
            )


            X = torch.cat(
                [
                    real_imgs,
                    fake_imgs,
                ],
                dim=0,
            )


            optimizer.zero_grad()


            features = featurizer(X)


            if epsilon_mode == "sigmoid":

                ep = (
                    torch.exp(
                        epsilonOPT
                    )
                    /
                    (
                        1
                        + torch.exp(
                            epsilonOPT
                        )
                    )
                )

            else:

                ep = epsilonOPT


            sigma = (
                sigmaOPT ** 2
            )

            sigma0 = (
                sigma0OPT ** 2
            )


            TEMP = MMDu(
                features,
                real_imgs.shape[0],
                X.view(
                    X.shape[0],
                    -1,
                ),
                sigma,
                sigma0,
                ep,
            )


            objective = (
                -TEMP[0]
                /
                torch.sqrt(
                    TEMP[1]
                    + 10 ** (-8)
                )
            )


            objective.backward()

            optimizer.step()


    # --------------------------------------------------------
    # Fixed held-out test sample
    # --------------------------------------------------------

    featurizer.eval()


    with torch.no_grad():

        if epsilon_mode == "sigmoid":

            ep_final = (
                torch.exp(
                    epsilonOPT
                )
                /
                (
                    1
                    + torch.exp(
                        epsilonOPT
                    )
                )
            ).detach()

        else:

            ep_final = (
                epsilonOPT.detach()
            )


        sigma_final = (
            sigmaOPT ** 2
        ).detach()

        sigma0_final = (
            sigma0OPT ** 2
        ).detach()


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


        test_features = (
            featurizer(
                S_test
            )
            .detach()
        )


        test_raw = (
            S_test.view(
                S_test.shape[0],
                -1,
            )
            .detach()
        )


    # --------------------------------------------------------
    # Tian notebook evaluation:
    # 100 permutation calibrations on same held-out dataset
    # --------------------------------------------------------

    H = np.zeros(
        N_CAL
    )


    for k in range(N_CAL):

        H[k], _, _ = (
            TST_MMD_u(
                test_features,
                INTERNAL_N,
                N_PER,
                test_raw,
                sigma_final,
                sigma0_final,
                ep_final,
                ALPHA,
                k,
            )
        )


    rate = float(
        H.mean()
    )


    result = {
        "trial": kk,
        "conditional_rejection": rate,
        "epsilon": float(
            ep_final.item()
        ),
        "sigma": float(
            sigma_final.item()
        ),
        "sigma0": float(
            sigma0_final.item()
        ),
    }


    del featurizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return result


# ============================================================
# Run 2x2 diagnostic
# ============================================================

all_results = {}


for variant in VARIANTS:

    name = variant["name"]

    print(
        "\n"
        "======================================"
    )

    print(
        "VARIANT:",
        name,
    )

    print(
        "first4000 =",
        variant["first4000"],
        "| epsilon =",
        variant["epsilon_mode"],
    )

    print(
        "======================================"
    )


    records = []


    for kk in range(N_OUTER):

        r = run_trial(
            kk=kk,
            first4000=(
                variant["first4000"]
            ),
            epsilon_mode=(
                variant["epsilon_mode"]
            ),
        )

        records.append(r)


        rates = np.array(
            [
                x[
                    "conditional_rejection"
                ]
                for x in records
            ]
        )


        print(
            f"{name:20s} "
            f"trial={kk:02d} "
            f"rate="
            f"{r['conditional_rejection']:.2f} "
            f"running="
            f"{rates.mean():.3f} "
            f"eps="
            f"{r['epsilon']:.2e} "
            f"sigma0="
            f"{r['sigma0']:.4f}",
            flush=True,
        )


    rates = np.array(
        [
            x[
                "conditional_rejection"
            ]
            for x in records
        ],
        dtype=float,
    )


    mae = float(
        np.mean(
            np.abs(
                rates
                - AUTHOR_FIRST10
            )
        )
    )


    all_results[name] = {
        "first4000": (
            variant["first4000"]
        ),

        "epsilon_mode": (
            variant["epsilon_mode"]
        ),

        "rates": (
            rates.tolist()
        ),

        "mean": float(
            rates.mean()
        ),

        "mae_vs_author_first10": (
            mae
        ),

        "records": records,
    }


# ============================================================
# Final comparison
# ============================================================

print(
    "\n"
    "======================================"
)

print(
    "FINAL HISTORICAL 2x2 COMPARISON"
)

print(
    "======================================"
)

print(
    "Author first-10:",
    AUTHOR_FIRST10.tolist(),
)

print(
    f"Author mean: "
    f"{AUTHOR_FIRST10.mean():.3f}\n"
)


for name, r in all_results.items():

    print(
        f"{name:20s} "
        f"mean={r['mean']:.3f} "
        f"MAE_vs_author="
        f"{r['mae_vs_author_first10']:.3f}"
    )

    print(
        "  rates =",
        r["rates"],
    )


payload = {
    "method": (
        "MMD-D historical implementation diagnostic"
    ),

    "dataset": (
        "MNIST vs Fake MNIST"
    ),

    "internal_n": INTERNAL_N,
    "paper_M": PAPER_M,

    "n_outer": N_OUTER,

    "permutation_calibrations_per_outer": (
        N_CAL
    ),

    "training_epochs": EPOCHS,
    "training_batch": BATCH,
    "training_lr": LR,

    "author_first10_recorded": (
        AUTHOR_FIRST10.tolist()
    ),

    "author_first10_mean": float(
        AUTHOR_FIRST10.mean()
    ),

    "author_final_100outer_reference": (
        0.2885
    ),

    "torch_version": (
        torch.__version__
    ),

    "python_version": (
        platform.python_version()
    ),

    "results": all_results,
}


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


print(
    "\nSaved:",
    OUTPUT,
)

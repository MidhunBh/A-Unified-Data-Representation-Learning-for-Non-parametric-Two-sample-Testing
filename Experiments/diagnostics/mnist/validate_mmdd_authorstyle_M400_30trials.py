import os
import json
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm

from utils import (
    MatConvert,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# AUTHOR-STYLE MMD-D VALIDATION — MNIST
#
# paper M = 400
# internal n = 200
#
# 30 INDEPENDENT outer trials
# ONE permutation test per outer trial
#
# This is now a power estimate:
#
#   trial 0 -> reject / not reject
#   trial 1 -> reject / not reject
#   ...
#   trial 29 -> reject / not reject
#
# power = total rejections / 30
#
# Locked after successful 3-trial diagnostic:
#
#   no cross-entropy pretraining
#   CNN featurizer trained directly by standardized MMD
#   epochs = 2000
#   paired batch = 100
#   lr = .001
#   N_PER = 100 permutations
#
# ============================================================


parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from existing checkpoint."
)

args = parser.parse_args()


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dtype = torch.float

alpha = 0.05

channels = 1
img_size = 32

INTERNAL_N = 200
PAPER_M = 400

N_OUTER = 30

TRAIN_EPOCHS = 2000
TRAIN_BATCH = 100
TRAIN_LR = 0.001

N_PER = 100

OUTPUT_STEM = (
    "result/"
    "validate_mmdd_authorstyle_"
    "mnist_M400_30trials"
)

JSON_PATH = OUTPUT_STEM + ".json"
PKL_PATH = OUTPUT_STEM + ".pkl"

os.makedirs("result", exist_ok=True)

print("device:", device, flush=True)


# ============================================================
# Author MMD-D image featurizer
# ============================================================

class Featurizer(nn.Module):

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
                16,
                bn=False,
            ),
            *discriminator_block(
                16,
                32,
            ),
            *discriminator_block(
                32,
                64,
            ),
            *discriminator_block(
                64,
                128,
            ),
        )

        ds_size = img_size // 2 ** 4

        self.adv_layer = nn.Sequential(
            nn.Linear(
                128 * ds_size ** 2,
                100,
            )
        )

    def forward(self, img):

        out = self.model(img)

        out = out.view(
            out.shape[0],
            -1,
        )

        return self.adv_layer(out)


# ============================================================
# Save / checkpoint
# ============================================================

def save_results(records):

    power = (
        float(
            np.mean(
                [
                    r["reject"]
                    for r in records
                ]
            )
        )
        if records
        else 0.0
    )

    payload = {
        "method": "MMD-D author-style validation",
        "dataset": "MNIST vs Fake MNIST",

        "internal_n": INTERNAL_N,
        "paper_M": PAPER_M,

        "n_outer_target": N_OUTER,
        "n_outer_completed": len(records),

        "power": power,

        "training": {
            "objective": "direct standardized deep-MMD",
            "classifier_pretraining": False,
            "epochs": TRAIN_EPOCHS,
            "paired_batch_size": TRAIN_BATCH,
            "lr": TRAIN_LR,
            "feature_dim": 100,
        },

        "testing": {
            "one_test_per_outer_trial": True,
            "N_PER": N_PER,
            "alpha": alpha,
        },

        "records": records,
    }

    tmp_json = JSON_PATH + ".tmp"

    with open(
        tmp_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    os.replace(
        tmp_json,
        JSON_PATH,
    )

    tmp_pkl = PKL_PATH + ".tmp"

    with open(
        tmp_pkl,
        "wb",
    ) as f:
        pickle.dump(
            payload,
            f,
        )

    os.replace(
        tmp_pkl,
        PKL_PATH,
    )


# ============================================================
# Load MNIST
# ============================================================

np.random.seed(819)
torch.manual_seed(819)

if torch.cuda.is_available():
    torch.cuda.manual_seed(819)
    torch.cuda.manual_seed_all(819)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
# Resume
# ============================================================

records = []

if (
    args.resume
    and os.path.exists(JSON_PATH)
):

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        previous = json.load(f)

    records = previous.get(
        "records",
        [],
    )

    print(
        f"Resuming from "
        f"{len(records)}/{N_OUTER} trials.",
        flush=True,
    )


start_trial = len(records)


# ============================================================
# Independent outer trials
# ============================================================

for kk in tqdm.trange(
    start_trial,
    N_OUTER,
    desc="MMD-D M=400 independent trials",
):

    # --------------------------------------------------------
    # Independent train / held-out dataset
    # --------------------------------------------------------

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_mnist_semi(
        data_real_all,
        data_fake_all,
        INTERNAL_N,
        INTERNAL_N,
        kk=kk,
    )


    # --------------------------------------------------------
    # Author initialization convention
    # --------------------------------------------------------

    init_seed = (
        kk * 19
        + INTERNAL_N
    )

    torch.manual_seed(
        init_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            init_seed
        )


    np.random.seed(
        1102 * (kk + 10)
        + INTERNAL_N
    )


    # --------------------------------------------------------
    # Initialize MMD-D featurizer
    # --------------------------------------------------------

    featurizer = Featurizer().to(
        device,
        dtype,
    )


    epsilonOPT = torch.log(
        MatConvert(
            np.random.rand(1)
            * 10 ** (-10),
            device,
            dtype,
        )
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


    optimizer_F = torch.optim.Adam(
        list(
            featurizer.parameters()
        )
        + [
            epsilonOPT,
            sigmaOPT,
            sigma0OPT,
        ],
        lr=TRAIN_LR,
    )


    # --------------------------------------------------------
    # Paired P/Q training loader
    # --------------------------------------------------------

    train_dataset = (
        torch.utils.data.TensorDataset(
            s1_tr,
            s2_tr,
        )
    )

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH,
            shuffle=True,
        )
    )


    # --------------------------------------------------------
    # Train directly on standardized MMD
    # --------------------------------------------------------

    final_J = np.nan

    for epoch in range(
        TRAIN_EPOCHS
    ):

        featurizer.train()

        J_values = []

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


            optimizer_F.zero_grad()


            features = featurizer(
                X
            )


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

            sigma = (
                sigmaOPT ** 2
            )

            sigma0_u = (
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
                sigma0_u,
                ep,
            )


            negative_J = (
                -TEMP[0]
                /
                torch.sqrt(
                    TEMP[1]
                    + 10 ** (-8)
                )
            )


            negative_J.backward()

            optimizer_F.step()


            J_values.append(
                -negative_J.detach().item()
            )


        final_J = float(
            np.mean(
                J_values
            )
        )


    # --------------------------------------------------------
    # Final kernel parameters
    # --------------------------------------------------------

    featurizer.eval()

    with torch.no_grad():

        ep = torch.sigmoid(
            epsilonOPT
        ).detach()

        sigma = (
            sigmaOPT ** 2
        ).detach()

        sigma0_u = (
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


        test_features = featurizer(
            S_test
        ).detach()


        test_raw = S_test.view(
            S_test.shape[0],
            -1,
        ).detach()


    # --------------------------------------------------------
    # ONE permutation test for this independent outer trial
    # --------------------------------------------------------

    reject, threshold, statistic = (
        TST_MMD_u(
            test_features,
            INTERNAL_N,
            N_PER,
            test_raw,
            sigma,
            sigma0_u,
            ep,
            alpha,
            kk,
        )
    )


    record = {
        "trial": kk,

        "reject": int(
            reject
        ),

        "statistic": float(
            statistic
        ),

        "threshold": float(
            threshold
        ),

        "final_J": float(
            final_J
        ),

        "epsilon": float(
            ep.item()
        ),

        "sigma": float(
            sigma.item()
        ),

        "sigma0": float(
            sigma0_u.item()
        ),
    }


    records.append(
        record
    )


    running_power = float(
        np.mean(
            [
                r["reject"]
                for r in records
            ]
        )
    )


    tqdm.tqdm.write(
        f"trial={kk:02d} "
        f"reject={int(reject)} "
        f"running_power="
        f"{running_power:.3f} "
        f"J={final_J:.4f} "
        f"eps={ep.item():.2e} "
        f"sigma={sigma.item():.2f} "
        f"sigma0={sigma0_u.item():.4f}"
    )


    # Save after every independent outer trial.
    save_results(
        records
    )


    del featurizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Final
# ============================================================

power = float(
    np.mean(
        [
            r["reject"]
            for r in records
        ]
    )
)


print(
    "\n=== FINAL 30-TRIAL VALIDATION ==="
)

print(
    f"internal_n = {INTERNAL_N}"
)

print(
    f"paper_M    = {PAPER_M}"
)

print(
    f"rejections = "
    f"{sum(r['reject'] for r in records)}"
    f"/{len(records)}"
)

print(
    f"MMD-D power = {power:.3f}"
)

print(
    "\nReference:"
)

print(
    "paper M=400 MMD-D ~= 0.996"
)

print(
    "old reproduction       ~= 0.236"
)

print(
    "\nSaved:"
)

print(
    " ",
    JSON_PATH
)

print(
    " ",
    PKL_PATH
)

import os
import json
import pickle
import time

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
# MMD-D MNIST AUTHOR-STYLE DIAGNOSTIC
#
# Purpose:
#   Test whether the large Table-3 MMD-D discrepancy is caused
#   by the current reproduction's classifier-pretraining stage.
#
# Paper point:
#   internal n = 200
#   paper M    = 400
#
# This diagnostic follows the released author's MMD-D pathway:
#
#   raw images
#       -> convolutional Featurizer
#       -> deep-kernel MMD objective
#
# Train jointly:
#   - featurizer
#   - epsilon
#   - sigma
#   - sigma0
#
# NO cross-entropy classifier pretraining.
# NO frozen classifier features.
#
# Author-style settings:
#   epochs = 2000
#   paired batch = 100
#   lr = 0.001
#   feature dim = 100
#
# 3 outer trials first: enough to diagnose a discrepancy
# as large as current ~.24 versus paper ~1.0 at M=400.
# ============================================================


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

dtype = torch.float

print("device:", device, flush=True)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

alpha = 0.05

channels = 1
img_size = 32

INTERNAL_N = 200
PAPER_M = 2 * INTERNAL_N

N_OUTER = 3

TRAIN_EPOCHS = 2000
TRAIN_BATCH = 100
TRAIN_LR = 0.001

N_PER = 100

# For diagnosis, repeat the permutation test with different
# permutation seeds on the same held-out set, matching the
# style of the existing reproduction.
N_TEST = 100

os.makedirs("result", exist_ok=True)


# ============================================================
# Author MMD-D image featurizer
#
# Released MNIST notebook architecture:
#
#   1 -> 16 -> 32 -> 64 -> 128 convolution blocks
#   -> Linear(..., 100)
#
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

                # Preserve released-author construction exactly.
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

        ds_size = (
            img_size // 2 ** 4
        )

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
# Data
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
# Outer trials
# ============================================================

records = []


for kk in tqdm.trange(
    N_OUTER,
    desc="Author-style MMD-D M=400",
):

    # --------------------------------------------------------
    # Same MNIST sample split
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
    # Initialize deep-kernel featurizer
    # --------------------------------------------------------

    featurizer = Featurizer().to(
        device,
        dtype,
    )


    # --------------------------------------------------------
    # Initialize deep-kernel parameters exactly in the
    # released-author style.
    #
    # Actual kernel parameters used below are:
    #
    #   epsilon = sigmoid(epsilonOPT)
    #   sigma   = sigmaOPT^2
    #   sigma0  = sigma0OPT^2
    #
    # --------------------------------------------------------

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
    # IMPORTANT:
    # paired batches keep P and Q sample counts identical.
    #
    # Each item = (one real image, one fake image)
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
    # Direct deep-MMD training
    # --------------------------------------------------------

    checkpoints = []

    for epoch in range(
        TRAIN_EPOCHS
    ):

        featurizer.train()

        epoch_J = []

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


            # Deep representation
            modelu_output = featurizer(
                X
            )


            # Author parameterization
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


            # Deep-kernel MMD
            TEMP = MMDu(
                modelu_output,
                real_imgs.shape[0],
                X.view(
                    X.shape[0],
                    -1,
                ),
                sigma,
                sigma0_u,
                ep,
            )


            # Author minimizes -MMD/std,
            # equivalent to maximizing standardized MMD.
            mmd_value_temp = (
                -1.0 * TEMP[0]
            )

            mmd_std_temp = torch.sqrt(
                TEMP[1]
                + 10 ** (-8)
            )

            STAT_u = torch.div(
                mmd_value_temp,
                mmd_std_temp,
            )


            STAT_u.backward()

            optimizer_F.step()


            # Positive J for easier interpretation.
            J = (
                -STAT_u.detach().item()
            )

            epoch_J.append(J)


        # Save a few diagnostic checkpoints only.
        if (
            epoch == 0
            or epoch + 1
            in [
                100,
                500,
                1000,
                2000,
            ]
        ):

            with torch.no_grad():

                ep_now = torch.sigmoid(
                    epsilonOPT
                ).item()

                sigma_now = (
                    sigmaOPT ** 2
                ).item()

                sigma0_now = (
                    sigma0OPT ** 2
                ).item()


            checkpoint = {
                "epoch": epoch + 1,
                "J": float(
                    np.mean(
                        epoch_J
                    )
                ),
                "epsilon": float(
                    ep_now
                ),
                "sigma": float(
                    sigma_now
                ),
                "sigma0": float(
                    sigma0_now
                ),
            }

            checkpoints.append(
                checkpoint
            )

            tqdm.tqdm.write(
                f"kk={kk} "
                f"epoch={epoch+1:4d} "
                f"J={checkpoint['J']:.6f} "
                f"eps={ep_now:.3e} "
                f"sigma={sigma_now:.4f} "
                f"sigma0={sigma0_now:.6f}"
            )


    # --------------------------------------------------------
    # Final learned kernel parameters
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
    # Held-out permutation test
    #
    # We repeat permutation seeds here only as a diagnostic,
    # so this is called a conditional rejection rate rather
    # than independent 100-trial power.
    # --------------------------------------------------------

    H = np.zeros(
        N_TEST
    )


    for k in range(
        N_TEST
    ):

        H[k], _, _ = TST_MMD_u(
            test_features,
            INTERNAL_N,
            N_PER,
            test_raw,
            sigma,
            sigma0_u,
            ep,
            alpha,
            k,
        )


    conditional_rate = float(
        H.mean()
    )


    record = {
        "trial": kk,
        "internal_n": INTERNAL_N,
        "paper_M": PAPER_M,

        "conditional_rejection_rate": (
            conditional_rate
        ),

        "final_epsilon": float(
            ep.item()
        ),

        "final_sigma": float(
            sigma.item()
        ),

        "final_sigma0": float(
            sigma0_u.item()
        ),

        "training_checkpoints": (
            checkpoints
        ),
    }


    records.append(
        record
    )


    running = float(
        np.mean(
            [
                r[
                    "conditional_rejection_rate"
                ]
                for r in records
            ]
        )
    )


    tqdm.tqdm.write(
        "\n"
        f"TRIAL {kk} DONE: "
        f"conditional rejection="
        f"{conditional_rate:.3f}, "
        f"running mean={running:.3f}\n"
    )


    del featurizer

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# Save diagnostic
# ============================================================

mean_rate = float(
    np.mean(
        [
            r[
                "conditional_rejection_rate"
            ]
            for r in records
        ]
    )
)


payload = {
    "method": (
        "MMD-D author-style diagnostic"
    ),

    "dataset": (
        "MNIST vs Fake MNIST"
    ),

    "internal_n": INTERNAL_N,
    "paper_M": PAPER_M,

    "n_outer_trials": (
        N_OUTER
    ),

    "training": {
        "objective": (
            "direct standardized deep-MMD"
        ),

        "classifier_pretraining": False,

        "epochs": TRAIN_EPOCHS,
        "paired_batch_size": (
            TRAIN_BATCH
        ),

        "lr": TRAIN_LR,

        "feature_dim": 100,

        "kernel_parameterization": {
            "epsilon": (
                "sigmoid(epsilonOPT)"
            ),
            "sigma": (
                "sigmaOPT^2"
            ),
            "sigma0": (
                "sigma0OPT^2"
            ),
        },
    },

    "evaluation": {
        "N_TEST": N_TEST,
        "N_PER": N_PER,

        "note": (
            "N_TEST repeats permutation seeds "
            "on each fixed held-out dataset; "
            "reported value is diagnostic "
            "conditional rejection rate, not "
            "100 independent outer-trial power."
        ),
    },

    "mean_conditional_rejection_rate": (
        mean_rate
    ),

    "records": records,

    "timestamp": time.strftime(
        "%Y-%m-%d %H:%M:%S"
    ),
}


output_path = (
    "result/"
    "diagnose_mmdd_authorstyle_"
    "mnist_M400_3trials.json"
)


with open(
    output_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        payload,
        f,
        indent=2,
    )


print(
    "\n=== FINAL DIAGNOSTIC ==="
)

for r in records:

    print(
        f"trial={r['trial']} "
        f"conditional_rejection="
        f"{r['conditional_rejection_rate']:.3f} "
        f"epsilon={r['final_epsilon']:.3e} "
        f"sigma={r['final_sigma']:.4f} "
        f"sigma0={r['final_sigma0']:.6f}"
    )


print(
    "\nMean conditional rejection rate:",
    f"{mean_rate:.3f}"
)

print(
    "\nPaper M=400 MMD-D reference "
    "is approximately 0.996."
)

print(
    "\nOld reproduction at this point "
    "was approximately 0.236."
)

print(
    "\nSaved:",
    output_path
)

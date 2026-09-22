import os
import json
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets

from utils import (
    Autoencoder_Img,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# RL-MMD-D MNIST — M=200 PILOT
#
# Goal:
#   Test the literal RL-TST kernel construction:
#
#       pooled unlabeled data
#            ↓
#       standard AE
#            ↓
#       frozen encoder phi*
#            ↓
#       MMD-D kernel optimized on labelled training split
#            ↓
#       held-out permutation test
#
# paper M = 200
# internal n = 100
#
# 10 outer trials
# 100 permutation calibrations per held-out dataset
#
# Phase 1: released MNIST AE settings
#   z=100
#   epochs=1000
#   lr=.002
#   batch=200
#
# Phase 2: carry forward solved MNIST MMD-D conventions
#   first4000 data universe
#   direct epsilon U(0,1e-8)
#   lr=.001
#   epochs=2000
#   paired batch=100
#
# IMPORTANT:
#   Encoder is fine-tuned in Phase 2 using the MMD-D objective.
#   No classifier pretraining.
#   No extra MMD feature network.
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

POOL_SIZE = 4000

N_OUTER = 10

N_PER = 100
N_CAL = 100

AE_Z = 100
AE_EPOCHS = 1000
AE_LR = 0.002
AE_BATCH = 200

MMD_EPOCHS = 2000
MMD_LR = 0.001
MMD_BATCH = 100

OUTPUT = (
    "result/"
    "diagnose_rl_mmdd_mnist_M200_"
    "jointAE_first4000_directeps_10trials.json"
)

os.makedirs(
    "result",
    exist_ok=True,
)


print(
    "device:",
    device,
    flush=True,
)


# ============================================================
# Load MNIST / fake MNIST
# ============================================================

np.random.seed(819)
torch.manual_seed(819)

if torch.cuda.is_available():
    torch.cuda.manual_seed(819)
    torch.cuda.manual_seed_all(819)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


real_loader = torch.utils.data.DataLoader(
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


# Historical MMD-D universe selected before RL-MMD-D work.
data_real_pool = data_real_all[:POOL_SIZE]
data_fake_pool = data_fake_all[:POOL_SIZE]


print(
    "real pool:",
    tuple(data_real_pool.shape),
)

print(
    "fake pool:",
    tuple(data_fake_pool.shape),
)


# ============================================================
# One outer trial
# ============================================================

def run_trial(kk):

    # --------------------------------------------------------
    # Independent train / held-out split
    # --------------------------------------------------------

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_mnist_semi(
        data_real_pool,
        data_fake_pool,
        INTERNAL_N,
        INTERNAL_N,
        kk=kk,
    )


    # ========================================================
    # PHASE 1
    # Learn IRs from ALL sampled points without labels
    # ========================================================

    S_encoder = torch.cat(
        [
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )


    # Released MNIST RL-C2ST AE seed.
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)


    ae = Autoencoder_Img(
        CHANNELS,
        IMG_SIZE,
        AE_Z,
    ).to(
        device,
        dtype,
    )


    ae_optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=AE_LR,
    )

    criterion = torch.nn.MSELoss()


    ae_dataset = (
        torch.utils.data.TensorDataset(
            S_encoder
        )
    )

    ae_loader = (
        torch.utils.data.DataLoader(
            ae_dataset,
            batch_size=AE_BATCH,
            shuffle=True,
        )
    )


    final_recon = np.nan


    for epoch in range(
        AE_EPOCHS
    ):

        ae.train()

        losses = []

        for (x,) in ae_loader:

            output = ae(x)

            loss = criterion(
                output,
                x,
            )

            ae_optimizer.zero_grad()

            loss.backward()

            ae_optimizer.step()

            losses.append(
                loss.detach().item()
            )


        final_recon = float(
            np.mean(losses)
        )


    # --------------------------------------------------------
    # Freeze phi*
    # --------------------------------------------------------

    encoder = ae.encoder

    # Phase 2: fine-tune the AE-pretrained feature map using
    # the standardized MMD-D power objective.
    for p in encoder.parameters():
        p.requires_grad = True

    encoder.train()


    # ========================================================
    # PHASE 2
    # Optimize deep-kernel parameters on TRAINING split only
    # ========================================================

    np.random.seed(
        1102 * (kk + 10)
        + INTERNAL_N
    )


    # Later Tian MNIST MMD-D historical convention:
    # direct epsilon, no log/sigmoid.
    epsilonOPT = torch.tensor(
        np.random.rand(1)
        * 1e-8,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )


    # Raw-image bandwidth component.
    sigmaOPT = torch.tensor(
        [np.sqrt(2 * 32 * 32)],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )


    # Learned-feature bandwidth component.
    sigma0OPT = torch.tensor(
        [np.sqrt(0.005)],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )


    kernel_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + [
            epsilonOPT,
            sigmaOPT,
            sigma0OPT,
        ],
        lr=MMD_LR,
    )


    train_dataset = (
        torch.utils.data.TensorDataset(
            s1_tr,
            s2_tr,
        )
    )

    train_loader = (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=MMD_BATCH,
            shuffle=True,
        )
    )


    final_J = np.nan


    for epoch in range(
        MMD_EPOCHS
    ):

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


            # Fine-tune the AE-pretrained representation using
            # the standardized MMD-D objective.
            encoder.train()

            Z = encoder(
                X
            )


            ep = epsilonOPT

            sigma = (
                sigmaOPT ** 2
            )

            sigma0 = (
                sigma0OPT ** 2
            )


            TEMP = MMDu(
                Z,
                real_imgs.shape[0],

                X.view(
                    X.shape[0],
                    -1,
                ),

                sigma,
                sigma0,
                ep,
            )


            negative_J = (
                -TEMP[0]
                /
                torch.sqrt(
                    TEMP[1]
                    + 1e-8
                )
            )


            kernel_optimizer.zero_grad()

            negative_J.backward()

            kernel_optimizer.step()


            J_values.append(
                -negative_J
                .detach()
                .item()
            )


        final_J = float(
            np.mean(
                J_values
            )
        )


    # ========================================================
    # PHASE 3
    # Held-out permutation test
    # ========================================================

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


    with torch.no_grad():

        Z_test = encoder(
            S_test
        ).detach()


        raw_test = S_test.view(
            S_test.shape[0],
            -1,
        ).detach()


        ep_final = (
            epsilonOPT.detach()
        )

        sigma_final = (
            sigmaOPT ** 2
        ).detach()

        sigma0_final = (
            sigma0OPT ** 2
        ).detach()


    H = np.zeros(
        N_CAL
    )


    for k in range(
        N_CAL
    ):

        H[k], _, _ = TST_MMD_u(
            Z_test,
            INTERNAL_N,
            N_PER,
            raw_test,
            sigma_final,
            sigma0_final,
            ep_final,
            ALPHA,
            k,
        )


    rate = float(
        H.mean()
    )


    result = {
        "trial": kk,

        "conditional_rejection": (
            rate
        ),

        "final_reconstruction_mse": (
            final_recon
        ),

        "final_J": (
            final_J
        ),

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


    del ae
    del encoder


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    return result


# ============================================================
# Run pilot
# ============================================================

records = []


for kk in range(
    N_OUTER
):

    result = run_trial(
        kk
    )

    records.append(
        result
    )


    running = float(
        np.mean(
            [
                r[
                    "conditional_rejection"
                ]
                for r in records
            ]
        )
    )


    print(
        f"trial={kk:02d} "
        f"conditional="
        f"{result['conditional_rejection']:.2f} "
        f"running="
        f"{running:.3f} "
        f"recon="
        f"{result['final_reconstruction_mse']:.5f} "
        f"J="
        f"{result['final_J']:.3f} "
        f"eps="
        f"{result['epsilon']:.2e} "
        f"sigma="
        f"{result['sigma']:.2f} "
        f"sigma0="
        f"{result['sigma0']:.5f}",
        flush=True,
    )


mean_rate = float(
    np.mean(
        [
            r[
                "conditional_rejection"
            ]
            for r in records
        ]
    )
)


payload = {
    "method": (
        "RL-MMD-D reconstruction"
    ),

    "dataset": (
        "MNIST vs Fake MNIST"
    ),

    "paper_M": PAPER_M,
    "internal_n": INTERNAL_N,

    "n_outer": N_OUTER,

    "phase1": {
        "model": (
            "standard Autoencoder_Img"
        ),
        "latent_dim": AE_Z,
        "epochs": AE_EPOCHS,
        "lr": AE_LR,
        "batch": AE_BATCH,
        "pooled_train_and_test": True,
    },

    "phase2": {
        "encoder_frozen": False,

        "extra_feature_network": False,

        "kernel": (
            "MMD-D using fine-tuned AE "
            "features + raw image distance"
        ),

        "epsilon": (
            "direct Uniform(0,1e-8)"
        ),

        "epochs": MMD_EPOCHS,
        "lr": MMD_LR,
        "batch": MMD_BATCH,
    },

    "evaluation": {
        "calibrations_per_outer": (
            N_CAL
        ),
        "permutations_per_calibration": (
            N_PER
        ),
        "alpha": ALPHA,
    },

    "mean_conditional_rejection": (
        mean_rate
    ),

    "references": {
        "vanilla_MMD_D_reproduction_M200": (
            0.2454
        ),

        "paper_MMD_D_M200": (
            0.290
        ),

        "paper_RL_MMD_D_M200": (
            0.420
        ),
    },

    "records": records,
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
    "\n"
    "======================================"
)

print(
    "FINAL RL-MMD-D M=200 PILOT"
)

print(
    "======================================"
)

for r in records:

    print(
        f"trial={r['trial']:02d} "
        f"conditional="
        f"{r['conditional_rejection']:.2f}"
    )


print(
    f"\nMean conditional rejection: "
    f"{mean_rate:.3f}"
)

print(
    "\nReferences:"
)

print(
    "our locked vanilla MMD-D M=200 = 0.2454"
)

print(
    "paper vanilla MMD-D M=200      = 0.290"
)

print(
    "paper RL-MMD-D M=200           = 0.420"
)

print(
    "\nSaved:",
    OUTPUT,
)


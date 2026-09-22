
import os
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from utils import (
    Autoencoder_Img,
    ModelLatentF,
    MatConvert,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# MNIST RL-MMD-D ? corrected intended-template diagnostic
#
# Goal:
# Recover the original RL-MMD-D composition:
#
#   x
#    -> pooled unlabeled AE.encoder
#    -> frozen 100-D IR
#    -> TRAINABLE MMD head (100 -> 30 -> 30 -> 30 -> 30)
#    -> deep-kernel MMD
#
# Structural correction relative to old MMD_D_fit:
#   epsilonOPT.requires_grad = True
#   sigmaOPT.requires_grad   = True
#   sigma0OPT.requires_grad  = True
#
# Everything else intentionally stays near the old MNIST
# RL-MMD-D setup:
#
#   AE: 1000 epochs, lr=.002, batch=200
#   MMD head: 1000 epochs
#   MMD lr: 5e-5
#   log/sigmoid epsilon parameterization
#   full MNIST/fake pool
#   frozen AE encoder
#
# paper M=200 <=> internal n=100
# 10 outer datasets
# 100 repeated permutation calibrations each
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
N_CAL = 100
N_PER = 100

AE_Z = 100
AE_EPOCHS = 1000
AE_BATCH = 200
AE_LR = 0.002

HEAD_H = 30
HEAD_OUT = 30

MMD_EPOCHS = 1000
MMD_LR = 0.00005

OUTPUT = (
    "result/"
    "diagnose_rl_mmdd_mnist_M200_"
    "corrected_latent_head_kernelgrads_10trials.json"
)

os.makedirs("result", exist_ok=True)

print("device:", device)
print("MMD LR:", MMD_LR)


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


loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
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

data_real_all, _ = next(iter(loader))


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
# Corrected MMD_D_fit
#
# Deliberately mirrors generic utils.MMD_D_fit except that
# ALL kernel parameters really require gradients.
# ============================================================

def corrected_mmd_fit(S):

    # Match old generic MMD_D_fit model seed.
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    model = ModelLatentF(
        AE_Z,
        HEAD_H,
        HEAD_OUT,
    ).to(
        device,
        dtype,
    )

    N, d = S.shape

    # Historical generic MMD-D initialization.
    epsilonOPT = torch.log(
        MatConvert(
            np.random.rand(1)
            * 1e-10,
            device,
            dtype,
        )
    ).detach()

    epsilonOPT.requires_grad_(True)


    sigmaOPT = MatConvert(
        np.ones(1)
        * np.sqrt(2 * d),
        device,
        dtype,
    )

    sigmaOPT.requires_grad_(True)


    sigma0OPT = MatConvert(
        np.ones(1)
        * np.sqrt(0.005),
        device,
        dtype,
    )

    sigma0OPT.requires_grad_(True)


    optimizer = torch.optim.Adam(
        list(model.parameters())
        + [
            epsilonOPT,
            sigmaOPT,
            sigma0OPT,
        ],
        lr=MMD_LR,
    )


    first_J = None
    last_J = None

    first_kernel = None


    for epoch in range(
        MMD_EPOCHS
    ):

        sigma = sigmaOPT ** 2
        sigma0 = sigma0OPT ** 2

        ep = (
            torch.exp(epsilonOPT)
            /
            (
                1
                + torch.exp(epsilonOPT)
            )
        )


        features = model(S)


        temp = MMDu(
            features,
            N // 2,
            S,
            sigma,
            sigma0,
            ep,
        )


        negative_J = (
            -1.0
            * (
                temp[0]
                + 1e-8
            )
            /
            torch.sqrt(
                temp[1]
                + 1e-8
            )
        )


        if epoch == 0:

            first_J = float(
                -negative_J
                .detach()
                .item()
            )

            first_kernel = {
                "epsilon": float(
                    ep.detach().item()
                ),

                "sigma": float(
                    sigma.detach().item()
                ),

                "sigma0": float(
                    sigma0.detach().item()
                ),
            }


        optimizer.zero_grad()

        negative_J.backward()

        optimizer.step()


        last_J = float(
            -negative_J
            .detach()
            .item()
        )


    sigma = (
        sigmaOPT ** 2
    ).detach()

    sigma0 = (
        sigma0OPT ** 2
    ).detach()

    ep = (
        torch.exp(epsilonOPT)
        /
        (
            1
            + torch.exp(epsilonOPT)
        )
    ).detach()


    final_kernel = {
        "epsilon": float(
            ep.item()
        ),

        "sigma": float(
            sigma.item()
        ),

        "sigma0": float(
            sigma0.item()
        ),
    }


    return (
        model,
        sigma,
        sigma0,
        ep,
        first_J,
        last_J,
        first_kernel,
        final_kernel,
    )


# ============================================================
# Outer experiment
# ============================================================

records = []


for kk in range(
    N_OUTER
):

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


    # ========================================================
    # Phase 1: pooled unlabeled AE
    # ========================================================

    pooled = torch.cat(
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


    criterion = nn.MSELoss()


    ae_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            pooled
        ),
        batch_size=AE_BATCH,
        shuffle=True,
    )


    final_recon = np.nan


    for epoch in range(
        AE_EPOCHS
    ):

        losses = []

        ae.train()


        for (x,) in ae_loader:

            recon = ae(x)

            loss = criterion(
                recon,
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


    encoder = ae.encoder


    for param in encoder.parameters():
        param.requires_grad = False


    encoder.eval()


    # ========================================================
    # Obtain intrinsic representations
    # ========================================================

    S_train = torch.cat(
        [
            s1_tr,
            s2_tr,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )


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

        IR_train = encoder(
            S_train
        ).detach()

        IR_test = encoder(
            S_test
        ).detach()


    # ========================================================
    # Phase 2: train MMD head on the IRs
    # ========================================================

    # Keep old outer-dependent numpy initialization explicit.
    np.random.seed(
        1102 * (kk + 10)
        + INTERNAL_N
    )


    (
        model_mmd,
        sigma,
        sigma0,
        ep,
        first_J,
        last_J,
        first_kernel,
        final_kernel,
    ) = corrected_mmd_fit(
        IR_train
    )


    model_mmd.eval()


    with torch.no_grad():

        test_features = model_mmd(
            IR_test
        ).detach()


    # ========================================================
    # Phase 3: notebook-style repeated permutation calibration
    #
    # Important correction versus the old local script:
    # use k directly, rather than k*kk+2024.
    # ========================================================

    H = np.zeros(
        N_CAL
    )


    for k in range(
        N_CAL
    ):

        H[k], _, _ = TST_MMD_u(
            test_features,
            INTERNAL_N,
            N_PER,
            IR_test,
            sigma,
            sigma0,
            ep,
            ALPHA,
            k,
        )


    conditional = float(
        H.mean()
    )


    record = {
        "trial": kk,

        "conditional_rejection": (
            conditional
        ),

        "reconstruction_mse": (
            final_recon
        ),

        "first_J": (
            first_J
        ),

        "last_J": (
            last_J
        ),

        "initial_kernel": (
            first_kernel
        ),

        "final_kernel": (
            final_kernel
        ),
    }


    records.append(
        record
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
        f"{conditional:.2f} "
        f"running="
        f"{running:.3f} "
        f"recon="
        f"{final_recon:.5f} "
        f"J="
        f"{first_J:.2f}->{last_J:.2f} "
        f"eps="
        f"{first_kernel['epsilon']:.2e}"
        f"->{final_kernel['epsilon']:.2e} "
        f"sigma="
        f"{first_kernel['sigma']:.3f}"
        f"->{final_kernel['sigma']:.3f} "
        f"sigma0="
        f"{first_kernel['sigma0']:.5f}"
        f"->{final_kernel['sigma0']:.5f}",
        flush=True,
    )


    del ae
    del encoder
    del model_mmd


    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Final
# ============================================================

rates = [
    r[
        "conditional_rejection"
    ]
    for r in records
]


mean_rate = float(
    np.mean(rates)
)


payload = {
    "method": (
        "RL-MMD-D corrected latent-head diagnostic"
    ),

    "paper_M": PAPER_M,
    "internal_n": INTERNAL_N,

    "n_outer": N_OUTER,

    "phase1": {
        "AE_latent": AE_Z,
        "epochs": AE_EPOCHS,
        "batch": AE_BATCH,
        "lr": AE_LR,
        "pooled_unlabeled": True,
        "encoder_frozen": True,
    },

    "phase2": {
        "architecture": (
            "AE IR 100 -> ModelLatentF(100,30,30)"
        ),

        "epochs": MMD_EPOCHS,

        "lr": MMD_LR,

        "kernel_parameters_trainable": True,

        "epsilon_parameterization": (
            "historical log/sigmoid"
        ),

        "Fea_org": (
            "100-D AE intrinsic representation"
        ),
    },

    "evaluation": {
        "calibrations_per_outer": N_CAL,
        "permutations": N_PER,
        "calibration_seed": "k",
    },

    "mean_conditional_rejection": (
        mean_rate
    ),

    "references": {
        "locked_vanilla_MMD_D": 0.2454,
        "paper_vanilla_MMD_D": 0.290,
        "paper_RL_MMD_D": 0.420,
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
    "=========================================="
)

print(
    "FINAL CORRECTED LATENT-HEAD DIAGNOSTIC"
)

print(
    "=========================================="
)


print(
    "rates =",
    rates,
)


print(
    f"\nmean conditional rejection = "
    f"{mean_rate:.3f}"
)


print(
    "\nReferences:"
)

print(
    "locked vanilla MMD-D M=200 = 0.2454"
)

print(
    "paper vanilla MMD-D M=200  = 0.290"
)

print(
    "paper RL-MMD-D M=200       = 0.420"
)


print(
    "\nSaved:",
    OUTPUT,
)

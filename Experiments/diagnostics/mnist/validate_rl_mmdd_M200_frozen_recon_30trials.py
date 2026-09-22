import copy
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from utils import (
    Autoencoder_Img,
    MatConvert,
    MMDu,
    TST_MMD_u,
    Pdist2,
    sample_mnist_semi,
)

# ============================================================
# RL-MMD-D MNIST: reconstruction-wrapper diagnostic
#
# paper M = 200 <=> internal n = 100
#
# Shared Phase 1:
#   pooled unlabeled AE
#   z=100, epochs=1000, lr=.002, batch=200
#
# Phase 2:
#   AE(x) reconstructed image
#          ->
#   solved vanilla MMD-D CNN
#          ->
#   standardized deep-MMD objective
#
# Compare:
#   frozen_recon : AE frozen in Phase 2
#   joint_recon  : AE jointly fine-tuned in Phase 2
#
# Everything else identical.
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float

N = 100
PAPER_M = 200

POOL_SIZE = 4000

N_OUTER = 30
N_CAL = 100
N_PER = 100

ALPHA = 0.05

AE_EPOCHS = 1000
AE_BATCH = 200
AE_LR = 0.002

MMD_EPOCHS = 2000
MMD_BATCH = 100
MMD_LR = 0.001

IMG_SIZE = 32
CHANNELS = 1

OUTPUT = (
    "result/"
    "validate_rl_mmdd_mnist_M200_"
    "frozen_reconstruction_30trials.json"
)

os.makedirs("result", exist_ok=True)


# ============================================================
# Exact successful vanilla MMD-D image network
# ============================================================

class Featurizer(nn.Module):

    def __init__(self):
        super().__init__()

        def block(in_f, out_f, bn=True):

            layers = [
                nn.Conv2d(
                    in_f,
                    out_f,
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
                # Preserve author's literal code:
                # 0.8 is positional eps.
                layers.append(
                    nn.BatchNorm2d(
                        out_f,
                        0.8,
                    )
                )

            return layers

        self.model = nn.Sequential(
            *block(
                CHANNELS,
                16,
                bn=False,
            ),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        ds_size = IMG_SIZE // 2**4

        self.adv_layer = nn.Linear(
            128 * ds_size**2,
            100,
        )

    def forward(self, x):

        x = self.model(x)

        x = x.view(
            x.shape[0],
            -1,
        )

        return self.adv_layer(x)


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


# Carry forward locked MMD-D historical universe.
data_real_pool = data_real_all[:POOL_SIZE]
data_fake_pool = data_fake_all[:POOL_SIZE]


print("device:", device)
print("real pool:", tuple(data_real_pool.shape))
print("fake pool:", tuple(data_fake_pool.shape))


# ============================================================
# Phase 1
# ============================================================

def train_phase1(
    s1_tr,
    s1_te,
    s2_tr,
    s2_te,
):

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

    # Released MNIST RL seed.
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    ae = Autoencoder_Img(
        CHANNELS,
        IMG_SIZE,
        100,
    ).to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=AE_LR,
    )

    criterion = nn.MSELoss()

    loader_ae = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            pooled
        ),
        batch_size=AE_BATCH,
        shuffle=True,
    )

    final_loss = np.nan

    ae.train()

    for epoch in range(AE_EPOCHS):

        losses = []

        for (x,) in loader_ae:

            recon = ae(x)

            loss = criterion(
                recon,
                x,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(
                loss.detach().item()
            )

        final_loss = float(
            np.mean(losses)
        )

    return ae, final_loss


# ============================================================
# Phase 2 + testing
# ============================================================

def run_branch(
    pretrained_ae,
    s1_tr,
    s1_te,
    s2_tr,
    s2_te,
    kk,
    joint,
):

    # Each branch starts from EXACT same Phase-1 AE.
    ae = copy.deepcopy(
        pretrained_ae
    ).to(
        device,
        dtype,
    )

    # Reset same MMD-D initialization for both branches.
    model_seed = (
        kk * 19
        + N
    )

    torch.manual_seed(
        model_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            model_seed
        )

    np.random.seed(
        1102 * (kk + 10)
        + N
    )

    featurizer = Featurizer().to(
        device,
        dtype,
    )

    epsilonOPT = MatConvert(
        np.random.rand(1)
        * 1e-8,
        device,
        dtype,
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


    if joint:

        params = (
            list(ae.parameters())
            + list(
                featurizer.parameters()
            )
            + [
                epsilonOPT,
                sigmaOPT,
                sigma0OPT,
            ]
        )

    else:

        for p in ae.parameters():
            p.requires_grad = False

        params = (
            list(
                featurizer.parameters()
            )
            + [
                epsilonOPT,
                sigmaOPT,
                sigma0OPT,
            ]
        )


    optimizer = torch.optim.Adam(
        params,
        lr=MMD_LR,
    )


    phase2_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            s1_tr,
            s2_tr,
        ),
        batch_size=MMD_BATCH,
        shuffle=True,
    )


    final_J = np.nan


    for epoch in range(
        MMD_EPOCHS
    ):

        featurizer.train()

        if joint:
            ae.train()
        else:
            ae.eval()

        Js = []


        for real_imgs, fake_imgs in phase2_loader:

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


            if joint:

                # Full differentiable RL wrapper.
                R = ae(X)

            else:

                with torch.no_grad():
                    R = ae(X)

                R = R.detach()


            # Critical correction:
            #
            # Do NOT replace the MMD-D featurizer by the
            # 100-D AE encoder.
            #
            # Preserve the successful vanilla image CNN.
            features = featurizer(
                R
            )


            ep = epsilonOPT
            sigma = sigmaOPT**2
            sigma0 = sigma0OPT**2


            # MMD-D now operates IN THE LEARNED
            # representation/reconstruction space.
            TEMP = MMDu(
                features,
                real_imgs.shape[0],
                R.view(
                    R.shape[0],
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


            negative_J.backward()

            optimizer.step()


            Js.append(
                -negative_J
                .detach()
                .item()
            )


        final_J = float(
            np.mean(Js)
        )


    # ========================================================
    # Held-out evaluation
    # ========================================================

    ae.eval()
    featurizer.eval()


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

        R_test = ae(
            S_test
        )

        F_test = featurizer(
            R_test
        )

        R_flat = R_test.view(
            R_test.shape[0],
            -1,
        )

        ep_final = epsilonOPT.detach()
        sigma_final = (
            sigmaOPT**2
        ).detach()

        sigma0_final = (
            sigma0OPT**2
        ).detach()


        # Useful scale diagnostics.
        d_rep = Pdist2(
            R_flat[:N],
            R_flat[N:],
        )

        d_feat = Pdist2(
            F_test[:N],
            F_test[N:],
        )

        median_rep_dist = float(
            torch.median(
                d_rep
            ).item()
        )

        median_feat_dist = float(
            torch.median(
                d_feat
            ).item()
        )


    H = np.zeros(
        N_CAL
    )


    for k in range(
        N_CAL
    ):

        H[k], _, _ = TST_MMD_u(
            F_test,
            N,
            N_PER,
            R_flat,
            sigma_final,
            sigma0_final,
            ep_final,
            ALPHA,
            k,
        )


    result = {
        "conditional_rejection": float(
            H.mean()
        ),

        "final_J": final_J,

        "epsilon": float(
            ep_final.item()
        ),

        "sigma": float(
            sigma_final.item()
        ),

        "sigma0": float(
            sigma0_final.item()
        ),

        "median_rep_distance": (
            median_rep_dist
        ),

        "median_feature_distance": (
            median_feat_dist
        ),
    }


    del ae
    del featurizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Paired outer trials
# ============================================================
#
# Validation run: frozen reconstruction branch only.
#
# Same structure selected by the previous diagnostic:
#
#   x -> pretrained frozen AE reconstruction
#     -> solved vanilla MMD-D CNN
#     -> MMD-D permutation test
#
# No hyperparameter changes.
# ============================================================

results = {
    "frozen_recon": [],
}


for kk in range(
    N_OUTER
):

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_mnist_semi(
        data_real_pool,
        data_fake_pool,
        N,
        N,
        kk=kk,
    )


    pretrained_ae, recon_loss = train_phase1(
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    )


    frozen = run_branch(
        pretrained_ae,
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
        kk,
        joint=False,
    )


    frozen["trial"] = kk

    frozen["phase1_recon_mse"] = (
        recon_loss
    )


    results[
        "frozen_recon"
    ].append(
        frozen
    )


    rates = [
        x[
            "conditional_rejection"
        ]
        for x in results[
            "frozen_recon"
        ]
    ]


    mean_frozen = float(
        np.mean(rates)
    )


    print(
        f"trial={kk:02d} "
        f"AE_mse={recon_loss:.5f} | "
        f"conditional="
        f"{frozen['conditional_rejection']:.2f} "
        f"running="
        f"{mean_frozen:.3f} "
        f"J="
        f"{frozen['final_J']:.2f} "
        f"eps="
        f"{frozen['epsilon']:.2e} "
        f"sigma0="
        f"{frozen['sigma0']:.5f} "
        f"repdist="
        f"{frozen['median_rep_distance']:.3f} "
        f"featdist="
        f"{frozen['median_feature_distance']:.3f}",
        flush=True,
    )


payload = {
    "method": (
        "RL-MMD-D frozen reconstruction validation"
    ),

    "paper_M": PAPER_M,
    "internal_n": N,

    "n_outer": N_OUTER,

    "shared_phase1": {
        "model": "Autoencoder_Img",
        "latent_dim": 100,
        "epochs": AE_EPOCHS,
        "batch": AE_BATCH,
        "lr": AE_LR,
        "pooled_unlabeled": True,
    },

    "phase2": {
        "wrapper": (
            "x -> full AE reconstruction -> "
            "vanilla MMD-D CNN"
        ),

        "mmd_cnn": (
            "16-32-64-128-100"
        ),

        "epochs": MMD_EPOCHS,
        "paired_batch": MMD_BATCH,
        "lr": MMD_LR,

        "epsilon": (
            "direct Uniform(0,1e-8)"
        ),

        "kernel_original_component": (
            "AE reconstruction flattened"
        ),
    },

    "references": {
        "our_vanilla_M200": 0.2454,
        "paper_vanilla_M200": 0.290,
        "paper_RL_MMD_D_M200": 0.420,
    },

    "results": results,
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
    "FINAL FROZEN RECONSTRUCTION VALIDATION"
)

print(
    "=========================================="
)


for name in [
    "frozen_recon",
]:

    rates = [
        x[
            "conditional_rejection"
        ]
        for x in results[name]
    ]

    print(
        f"{name:14s}: "
        f"mean={np.mean(rates):.3f}"
    )

    print(
        " rates =",
        rates,
    )


print(
    "\nReferences:"
)

print(
    "our vanilla MMD-D M=200 = 0.2454"
)

print(
    "paper vanilla MMD-D M=200 = 0.290"
)

print(
    "paper RL-MMD-D M=200 = 0.420"
)

print(
    "\nSaved:",
    OUTPUT,
)


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
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# MNIST RL-MMD-D ? MATCHED ENCODER DIAGNOSTIC
#
# Idea:
#
# Vanilla:
#   random MMD-D Featurizer
#       -> direct deep-MMD training
#
# RL:
#   same exact Featurizer architecture
#       -> pretrained as AE encoder on pooled unlabeled data
#       -> SAME direct deep-MMD training
#
# Nothing is stacked on top of the AE representation.
#
# paper M=200 <=> internal n=100
# 10 paired outer trials.
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

N = 100
PAPER_M = 200

N_OUTER = 10
N_CAL = 100
N_PER = 100

ALPHA = 0.05

AE_EPOCHS = 1000
AE_BATCH = 200
AE_LR = 0.002

MMD_EPOCHS = 2000
MMD_LR = 0.001

IMG_SIZE = 32
CHANNELS = 1
Z = 100

POOL_SIZE = 4000

OUTPUT = (
    "result/"
    "diagnose_rl_mmdd_matched_encoder_"
    "M200_10trials.json"
)

os.makedirs(
    "result",
    exist_ok=True,
)


# ============================================================
# EXACT MMD-D FEATURIZER ARCHITECTURE
# ============================================================

class Featurizer(nn.Module):

    def __init__(self):
        super().__init__()

        def block(
            in_f,
            out_f,
            bn=True,
        ):

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

                # Preserve author's literal call:
                # second positional argument = eps=0.8.
                layers.append(
                    nn.BatchNorm2d(
                        out_f,
                        0.8,
                    )
                )

            return layers

        self.model = nn.Sequential(
            *block(1, 16, False),
            *block(16, 32),
            *block(32, 64),
            *block(64, 128),
        )

        self.adv_layer = nn.Linear(
            128 * 2 * 2,
            Z,
        )

    def forward(self, x):

        x = self.model(x)

        x = x.reshape(
            x.shape[0],
            -1,
        )

        return self.adv_layer(x)


# ============================================================
# MATCHING DECODER
#
# Mirrors the deeper encoder/decoder path left commented in
# Tian's utils.py.
# ============================================================

class MatchedDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(
            Z,
            128 * 2 * 2,
        )

        def block(
            in_f,
            out_f,
        ):

            return [
                nn.ConvTranspose2d(
                    in_f,
                    out_f,
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.LeakyReLU(
                    0.2,
                    inplace=True,
                ),
                nn.BatchNorm2d(
                    out_f,
                    0.8,
                ),
            ]

        self.decoder = nn.Sequential(
            *block(128, 64),
            *block(64, 32),
            *block(32, 16),

            nn.ConvTranspose2d(
                16,
                1,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),

            nn.Tanh(),
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.reshape(
            z.shape[0],
            128,
            2,
            2,
        )

        return self.decoder(x)


class MatchedAE(nn.Module):

    def __init__(self):
        super().__init__()

        # This is literally the MMD-D feature network.
        self.encoder = Featurizer()

        self.decoder = MatchedDecoder()

    def forward(self, x):

        return self.decoder(
            self.encoder(x)
        )


# ============================================================
# DATA
# ============================================================

np.random.seed(819)

torch.manual_seed(819)

if torch.cuda.is_available():
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

    data_fake_all = pickle.load(
        f
    )[0]


data_fake_all = torch.from_numpy(
    data_fake_all
).float()


# Locked historical MMD-D universe.
data_real_pool = data_real_all[
    :POOL_SIZE
]

data_fake_pool = data_fake_all[
    :POOL_SIZE
]


print(
    "device:",
    device,
)

print(
    "real pool:",
    tuple(
        data_real_pool.shape
    ),
)

print(
    "fake pool:",
    tuple(
        data_fake_pool.shape
    ),
)


# ============================================================
# PHASE 1
# ============================================================

def pretrain_ae(
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

    # Same representation-learning seed used by released
    # MNIST RL implementation.
    torch.manual_seed(1102)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1102)

    ae = MatchedAE().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=AE_LR,
    )

    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        pooled
    )

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=AE_BATCH,
        shuffle=True,
    )

    final_loss = np.nan

    for epoch in range(
        AE_EPOCHS
    ):

        ae.train()

        losses = []

        for (x,) in dl:

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
# EXACT LOCKED PHASE-2 MMD-D TRAINING
# ============================================================

def run_mmd(
    featurizer,
    s1_tr,
    s1_te,
    s2_tr,
    s2_te,
    kk,
):

    featurizer = featurizer.to(
        device,
        dtype,
    )

    # Same direct-epsilon initialization as locked
    # first4000_direct MNIST MMD-D.
    np.random.seed(
        1102 * (kk + 10)
        + N
    )

    epsilonOPT = torch.tensor(
        np.random.rand(1)
        * 1e-8,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    sigmaOPT = torch.tensor(
        [
            np.sqrt(
                2 * 32 * 32
            )
        ],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    sigma0OPT = torch.tensor(
        [
            np.sqrt(
                0.005
            )
        ],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    optimizer = torch.optim.Adam(
        list(
            featurizer.parameters()
        )
        + [
            epsilonOPT,
            sigmaOPT,
            sigma0OPT,
        ],
        lr=MMD_LR,
    )


    X_train = torch.cat(
        [
            s1_tr,
            s2_tr,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )

    raw_train = X_train.reshape(
        2 * N,
        -1,
    )


    first_J = np.nan
    last_J = np.nan


    for epoch in range(
        MMD_EPOCHS
    ):

        featurizer.train()

        features = featurizer(
            X_train
        )

        sigma = (
            sigmaOPT ** 2
        )

        sigma0 = (
            sigma0OPT ** 2
        )

        ep = epsilonOPT

        temp = MMDu(
            features,
            N,
            raw_train,
            sigma,
            sigma0,
            ep,
        )

        negative_J = (
            -temp[0]
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

        optimizer.zero_grad()

        negative_J.backward()

        optimizer.step()

        last_J = float(
            -negative_J
            .detach()
            .item()
        )


    # --------------------------------------------------------
    # Held-out evaluation
    # --------------------------------------------------------

    featurizer.eval()

    X_test = torch.cat(
        [
            s1_te,
            s2_te,
        ],
        dim=0,
    ).to(
        device,
        dtype,
    )

    raw_test = X_test.reshape(
        2 * N,
        -1,
    )

    with torch.no_grad():

        features_test = featurizer(
            X_test
        )

        sigma = (
            sigmaOPT ** 2
        ).detach()

        sigma0 = (
            sigma0OPT ** 2
        ).detach()

        ep = epsilonOPT.detach()


    H = np.zeros(
        N_CAL
    )


    for k in range(
        N_CAL
    ):

        H[k], _, _ = TST_MMD_u(
            features_test,
            N,
            N_PER,
            raw_test,
            sigma,
            sigma0,
            ep,
            ALPHA,
            k,
        )


    return {
        "conditional_rejection":
            float(
                H.mean()
            ),

        "first_J":
            first_J,

        "last_J":
            last_J,

        "epsilon":
            float(
                ep.item()
            ),

        "sigma":
            float(
                sigma.item()
            ),

        "sigma0":
            float(
                sigma0.item()
            ),
    }


# ============================================================
# PAIRED DIAGNOSTIC
# ============================================================

results = {
    "random_init": [],
    "ae_pretrained_init": [],
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


    # ========================================================
    # Phase 1
    # ========================================================

    ae, recon_loss = pretrain_ae(
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    )


    pretrained_encoder = copy.deepcopy(
        ae.encoder
    )


    # ========================================================
    # ARM A ? exact random-init vanilla MMD-D
    # ========================================================

    torch.manual_seed(
        kk * 19
        + N
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(
            kk * 19
            + N
        )

    random_model = Featurizer()

    random_result = run_mmd(
        random_model,
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
        kk,
    )


    # ========================================================
    # ARM B ? SAME MMD-D, AE-pretrained initialization
    # ========================================================

    pretrained_result = run_mmd(
        pretrained_encoder,
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
        kk,
    )


    random_result[
        "trial"
    ] = kk

    pretrained_result[
        "trial"
    ] = kk

    pretrained_result[
        "phase1_reconstruction_mse"
    ] = recon_loss


    results[
        "random_init"
    ].append(
        random_result
    )

    results[
        "ae_pretrained_init"
    ].append(
        pretrained_result
    )


    mean_random = np.mean(
        [
            x[
                "conditional_rejection"
            ]
            for x
            in results[
                "random_init"
            ]
        ]
    )

    mean_ae = np.mean(
        [
            x[
                "conditional_rejection"
            ]
            for x
            in results[
                "ae_pretrained_init"
            ]
        ]
    )


    print(
        f"trial={kk:02d} "
        f"recon={recon_loss:.5f} | "
        f"random="
        f"{random_result['conditional_rejection']:.2f} "
        f"(run={mean_random:.3f}) | "
        f"AE-init="
        f"{pretrained_result['conditional_rejection']:.2f} "
        f"(run={mean_ae:.3f})",
        flush=True,
    )


    del ae
    del random_model
    del pretrained_encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SUMMARY
# ============================================================

summary = {}

for name in [
    "random_init",
    "ae_pretrained_init",
]:

    rates = [
        x[
            "conditional_rejection"
        ]
        for x
        in results[name]
    ]

    summary[name] = {
        "mean":
            float(
                np.mean(rates)
            ),

        "rates":
            rates,
    }


payload = {
    "method":
        "RL-MMD-D matched-encoder diagnostic",

    "paper_M":
        PAPER_M,

    "internal_n":
        N,

    "n_outer":
        N_OUTER,

    "phase1": {
        "architecture":
            "same 16-32-64-128-100 encoder as MMD-D",
        "epochs":
            AE_EPOCHS,
        "batch":
            AE_BATCH,
        "lr":
            AE_LR,
        "pooled_unlabeled":
            True,
    },

    "phase2": {
        "architecture":
            "locked MNIST MMD-D",
        "epochs":
            MMD_EPOCHS,
        "lr":
            MMD_LR,
        "epsilon":
            "direct Uniform(0,1e-8)",
        "raw_q":
            "original flattened pixels",
        "first4000":
            True,
    },

    "summary":
        summary,

    "results":
        results,

    "references": {
        "locked_vanilla_full100":
            0.2454,
        "paper_MMD_D":
            0.290,
        "paper_RL_MMD_D":
            0.420,
    },
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
    "FINAL MATCHED-ENCODER RL-MMD-D DIAGNOSTIC"
)

print(
    "=========================================="
)


for name in [
    "random_init",
    "ae_pretrained_init",
]:

    print(
        f"{name:20s}: "
        f"mean="
        f"{summary[name]['mean']:.3f}"
    )

    print(
        " rates =",
        summary[name]["rates"],
    )


print(
    "\nUseful historical check:"
)

print(
    "previous locked first4000_direct first-10"
)

print(
    "[0, 1, 0, 1, 0, .07, 0, .61, 0, 1]"
)

print(
    "mean = .368"
)

print(
    "\nFull-100 references:"
)

print(
    "our vanilla MMD-D = .2454"
)

print(
    "paper vanilla      = .290"
)

print(
    "paper RL-MMD-D     = .420"
)

print(
    "\nSaved:",
    OUTPUT,
)

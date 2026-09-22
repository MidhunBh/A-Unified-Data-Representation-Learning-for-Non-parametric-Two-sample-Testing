
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
    Pdist2,
    MMDu,
    TST_MMD_u,
    sample_mnist_semi,
)

# ============================================================
# RL-MMD-D STRUCTURAL DIAGNOSTIC
#
# paper M=200 <=> internal n=100
#
# Phase 1:
#   pooled unlabeled AE -> phi*(x)
#
# Phase 2:
#   NO extra MLP.
#
#   k(x,y) =
#       [(1-eps) kappa(phi*(x),phi*(y)) + eps] q(x,y)
#
#   q(x,y) remains on ORIGINAL PIXELS.
#
# Two paired arms:
#
#   1) hybrid_legacy_scale
#      raw sigma = 2*32*32
#      phi sigma = .005
#
#   2) hybrid_median_scale
#      raw sigma = median raw squared distance
#      phi sigma = median AE squared distance
#
# Encoder frozen in both arms.
# This isolates the MMD/RL geometry + scale interaction.
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float32

INTERNAL_N = 100
PAPER_M = 200

N_OUTER = 10
N_CAL = 100
N_PER = 100

ALPHA = 0.05

AE_Z = 100
AE_EPOCHS = 1000
AE_BATCH = 200
AE_LR = 0.002

KERNEL_EPOCHS = 2000
KERNEL_LR = 0.001

EPS_INIT = 1e-10

IMG_SIZE = 32
CHANNELS = 1

OUT = (
    "result/"
    "diagnose_rl_mmdd_direct_hybrid_M200_10trials.json"
)

os.makedirs("result", exist_ok=True)

print("device:", device)


# ============================================================
# Utilities
# ============================================================

def median_positive_offdiag(D):

    n = D.shape[0]

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=D.device,
    )

    x = D[mask]

    x = x[x > 0]

    if len(x) == 0:
        return torch.tensor(
            1.0,
            device=D.device,
            dtype=D.dtype,
        )

    return torch.median(x)


def geometry_stats(
    D_raw,
    D_phi,
    sigma_raw,
    sigma_phi,
    eps,
):

    with torch.no_grad():

        Kq = torch.exp(
            -D_raw / sigma_raw
        )

        Kphi = torch.exp(
            -D_phi / sigma_phi
        )

        K = (
            (1.0 - eps)
            * Kphi
            * Kq
            + eps * Kq
        )

        n = K.shape[0]

        off = ~torch.eye(
            n,
            dtype=torch.bool,
            device=K.device,
        )

        kp = Kphi[off]
        kq = Kq[off]
        kd = K[off]

        return {
            "Kphi_mean": float(
                kp.mean().item()
            ),
            "Kphi_median": float(
                kp.median().item()
            ),
            "Kphi_frac_lt_1e-6": float(
                (kp < 1e-6)
                .float()
                .mean()
                .item()
            ),

            "Kq_mean": float(
                kq.mean().item()
            ),

            "Kdeep_mean": float(
                kd.mean().item()
            ),
        }


def fit_kernel_scalars(
    phi_train,
    raw_train,
    init_sigma_raw,
    init_sigma_phi,
):

    # Same parameterization style as legacy MMD-D:
    # sigma = sigmaOPT ** 2

    sigma_raw_root = nn.Parameter(
        torch.tensor(
            [np.sqrt(init_sigma_raw)],
            device=device,
            dtype=dtype,
        )
    )

    sigma_phi_root = nn.Parameter(
        torch.tensor(
            [np.sqrt(init_sigma_phi)],
            device=device,
            dtype=dtype,
        )
    )

    epsilon_log = nn.Parameter(
        torch.tensor(
            [np.log(EPS_INIT)],
            device=device,
            dtype=dtype,
        )
    )

    optimizer = torch.optim.Adam(
        [
            sigma_raw_root,
            sigma_phi_root,
            epsilon_log,
        ],
        lr=KERNEL_LR,
    )

    D_raw = Pdist2(
        raw_train,
        raw_train,
    ).detach()

    D_phi = Pdist2(
        phi_train,
        phi_train,
    ).detach()

    initial_stats = geometry_stats(
        D_raw,
        D_phi,
        sigma_raw_root.detach() ** 2,
        sigma_phi_root.detach() ** 2,
        torch.sigmoid(
            epsilon_log.detach()
        ),
    )

    negative_variance_epochs = 0

    first_J = None
    last_J = None

    for epoch in range(
        KERNEL_EPOCHS
    ):

        sigma_raw = (
            sigma_raw_root ** 2
            + 1e-12
        )

        sigma_phi = (
            sigma_phi_root ** 2
            + 1e-12
        )

        eps = torch.sigmoid(
            epsilon_log
        )

        TEMP = MMDu(
            phi_train,
            INTERNAL_N,
            raw_train,
            sigma_raw,
            sigma_phi,
            eps,
        )

        mmd2 = TEMP[0]
        var = TEMP[1]

        if var.detach().item() < 0:
            negative_variance_epochs += 1

        # Keep optimization finite while recording if
        # the finite-sample variance estimate becomes bad.
        std = torch.sqrt(
            torch.clamp(
                var,
                min=0.0,
            )
            + 1e-8
        )

        J = mmd2 / std
        loss = -J

        if first_J is None:
            first_J = float(
                J.detach().item()
            )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        last_J = float(
            J.detach().item()
        )

    sigma_raw = (
        sigma_raw_root.detach() ** 2
        + 1e-12
    )

    sigma_phi = (
        sigma_phi_root.detach() ** 2
        + 1e-12
    )

    eps = torch.sigmoid(
        epsilon_log.detach()
    )

    final_stats = geometry_stats(
        D_raw,
        D_phi,
        sigma_raw,
        sigma_phi,
        eps,
    )

    return {
        "sigma_raw": sigma_raw,
        "sigma_phi": sigma_phi,
        "eps": eps,

        "first_J": first_J,
        "last_J": last_J,

        "negative_variance_epochs":
            negative_variance_epochs,

        "initial_geometry":
            initial_stats,

        "final_geometry":
            final_stats,
    }


# ============================================================
# Load MNIST / fake MNIST
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


# ============================================================
# Experiment
# ============================================================

arm_names = [
    "hybrid_legacy_scale",
    "hybrid_median_scale",
]

all_results = {
    a: []
    for a in arm_names
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
        data_real_all,
        data_fake_all,
        INTERNAL_N,
        INTERNAL_N,
        kk=kk,
    )


    # --------------------------------------------------------
    # Phase 1: author-style pooled unlabeled AE
    # --------------------------------------------------------

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


    for epoch in range(
        AE_EPOCHS
    ):

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


    # --------------------------------------------------------
    # phi*(x) = AE encoder DIRECTLY
    # --------------------------------------------------------

    encoder = ae.encoder

    for p in encoder.parameters():
        p.requires_grad = False

    encoder.eval()


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


    raw_train = S_train.reshape(
        2 * INTERNAL_N,
        -1,
    )

    raw_test = S_test.reshape(
        2 * INTERNAL_N,
        -1,
    )


    with torch.no_grad():

        phi_train = encoder(
            S_train
        ).detach()

        phi_test = encoder(
            S_test
        ).detach()


    D_raw = Pdist2(
        raw_train,
        raw_train,
    )

    D_phi = Pdist2(
        phi_train,
        phi_train,
    )


    med_raw = float(
        median_positive_offdiag(
            D_raw
        ).item()
    )

    med_phi = float(
        median_positive_offdiag(
            D_phi
        ).item()
    )


    print(
        f"\ntrial={kk:02d} "
        f"median_D_raw={med_raw:.6f} "
        f"median_D_phi={med_phi:.6f}"
    )


    configs = {
        "hybrid_legacy_scale": {
            "sigma_raw": (
                2.0
                * IMG_SIZE
                * IMG_SIZE
            ),
            "sigma_phi": 0.005,
        },

        "hybrid_median_scale": {
            "sigma_raw": med_raw,
            "sigma_phi": med_phi,
        },
    }


    for arm in arm_names:

        cfg = configs[arm]

        fitted = fit_kernel_scalars(
            phi_train,
            raw_train,
            cfg["sigma_raw"],
            cfg["sigma_phi"],
        )


        H = np.zeros(
            N_CAL
        )


        for k in range(
            N_CAL
        ):

            H[k], _, _ = TST_MMD_u(
                phi_test,
                INTERNAL_N,
                N_PER,
                raw_test,
                fitted["sigma_raw"],
                fitted["sigma_phi"],
                fitted["eps"],
                ALPHA,
                k,
            )


        conditional = float(
            H.mean()
        )


        rec = {
            "trial": kk,

            "conditional_rejection":
                conditional,

            "median_D_raw":
                med_raw,

            "median_D_phi":
                med_phi,

            "init_sigma_raw":
                cfg["sigma_raw"],

            "init_sigma_phi":
                cfg["sigma_phi"],

            "final_sigma_raw":
                float(
                    fitted[
                        "sigma_raw"
                    ].item()
                ),

            "final_sigma_phi":
                float(
                    fitted[
                        "sigma_phi"
                    ].item()
                ),

            "final_epsilon":
                float(
                    fitted[
                        "eps"
                    ].item()
                ),

            "first_J":
                fitted["first_J"],

            "last_J":
                fitted["last_J"],

            "negative_variance_epochs":
                fitted[
                    "negative_variance_epochs"
                ],

            "initial_geometry":
                fitted[
                    "initial_geometry"
                ],

            "final_geometry":
                fitted[
                    "final_geometry"
                ],
        }


        all_results[
            arm
        ].append(
            rec
        )


        running = np.mean(
            [
                r[
                    "conditional_rejection"
                ]
                for r
                in all_results[arm]
            ]
        )


        ig = rec[
            "initial_geometry"
        ]


        print(
            f"  {arm:22s} "
            f"cond={conditional:.2f} "
            f"running={running:.3f} "
            f"Kphi0_mean="
            f"{ig['Kphi_mean']:.3e} "
            f"Kphi0_<1e-6="
            f"{ig['Kphi_frac_lt_1e-6']:.3f} "
            f"J="
            f"{rec['first_J']:.3f}"
            f"->{rec['last_J']:.3f}"
        )


    del ae
    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Save / summarize
# ============================================================

summary = {}

for arm in arm_names:

    rates = [
        r[
            "conditional_rejection"
        ]
        for r in all_results[
            arm
        ]
    ]

    summary[arm] = {
        "mean": float(
            np.mean(rates)
        ),

        "rates": rates,
    }


payload = {
    "method":
        "RL-MMD-D structural hybrid diagnostic",

    "paper_M": PAPER_M,
    "internal_n": INTERNAL_N,

    "n_outer": N_OUTER,
    "n_calibration": N_CAL,
    "n_permutations": N_PER,

    "AE": {
        "z": AE_Z,
        "epochs": AE_EPOCHS,
        "batch": AE_BATCH,
        "lr": AE_LR,
        "pooled_unlabelled": True,
        "encoder_frozen": True,
    },

    "kernel": {
        "extra_MLP": False,
        "phi": "AE encoder directly",
        "q_space": "original flattened pixels",
        "epochs": KERNEL_EPOCHS,
        "lr": KERNEL_LR,
        "epsilon_initial": EPS_INIT,
    },

    "summary": summary,
    "records": all_results,

    "references": {
        "our_locked_vanilla_MMD_D":
            0.2454,
        "paper_MMD_D":
            0.290,
        "paper_RL_MMD_D":
            0.420,
    },
}


with open(
    OUT,
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
    "==============================================="
)

print(
    "FINAL RL-MMD-D HYBRID GEOMETRY DIAGNOSTIC"
)

print(
    "==============================================="
)


for arm in arm_names:

    print(
        f"{arm:22s}: "
        f"mean="
        f"{summary[arm]['mean']:.3f}"
    )

    print(
        " rates =",
        summary[arm]["rates"],
    )


print(
    "\nReferences:"
)

print(
    "our vanilla MMD-D M=200 = 0.2454"
)

print(
    "paper MMD-D M=200       = 0.290"
)

print(
    "paper RL-MMD-D M=200    = 0.420"
)

print(
    "\nSaved:",
    OUT,
)

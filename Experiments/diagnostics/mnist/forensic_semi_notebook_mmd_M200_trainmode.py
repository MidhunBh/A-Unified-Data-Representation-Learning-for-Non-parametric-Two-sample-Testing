import argparse
import json
import os
import pickle
import platform

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
# LOCKED MNIST MMD-D FULL-GRID REPRODUCTION
#
# Historical implementation selected BEFORE full-grid run:
#
#   data universe:
#       first 4000 real MNIST
#       first 4000 fake MNIST
#
#   epsilon:
#       epsilonOPT ~ U(0, 1e-8)
#       ep = epsilonOPT directly
#
#   MMD-D:
#       CNN 1->16->32->64->128->100
#       2000 epochs
#       paired batch = 100
#       lr = 0.001
#
#   evaluation:
#       100 independent outer datasets
#       100 permutation calibrations per outer dataset
#       100 permutations per calibration
#
# Grid accounting:
#
#   internal n = [100,200,300,400,500]
#   paper M    = [200,400,600,800,1000]
#
# This matches the later Tian MNIST notebook code path
# associated with the ~0.2885 M=200 result.
#
# NO further hyperparameter tuning after this point.
# ============================================================


parser = argparse.ArgumentParser()

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume an interrupted full-grid run.",
)

parser.add_argument(
    "--reuse-m200-diagnostic",
    action="store_true",
    help=(
        "Reuse the first 10 M=200 first4000_direct trials "
        "from the historical 2x2 diagnostic."
    ),
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

dtype = torch.float

ALPHA = 0.05

CHANNELS = 1
IMG_SIZE = 32

INTERNAL_N_LIST = [100]

PAPER_M_LIST = [
    2 * n
    for n in INTERNAL_N_LIST
]

N_OUTER = 10

# Author notebook-style repeated calibration.
N_CAL = 100

# Number of permutations inside each test.
N_PER = 100

TRAIN_EPOCHS = 2000
TRAIN_BATCH = 100
TRAIN_LR = 0.001

POOL_SIZE = 4000

OUTPUT_STEM = (
    "result/"
    "forensic_semi_notebook_"
    "M200_trainmode_10outer"
)

JSON_PATH = OUTPUT_STEM + ".json"
PKL_PATH = OUTPUT_STEM + ".pkl"

DIAGNOSTIC_PATH = (
    "result/"
    "diagnose_mmdd_mnist_M200_"
    "historical_2x2_10trials.json"
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

print(
    "torch:",
    torch.__version__,
    flush=True,
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )


# ============================================================
# Released MMD-D image featurizer
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

            *block(
                CHANNELS,
                16,
                bn=False,
            ),

            *block(
                16,
                32,
            ),

            *block(
                32,
                64,
            ),

            *block(
                64,
                128,
            ),
        )


        ds_size = (
            IMG_SIZE // 2 ** 4
        )


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
# Atomic checkpointing
# ============================================================

def build_payload(points):

    results = []

    for n in INTERNAL_N_LIST:

        key = str(n)

        if key not in points:
            continue

        records = points[key][
            "records"
        ]

        if len(records) == 0:
            continue

        values = [
            r["conditional_rejection"]
            for r in records
        ]

        results.append(
            {
                "internal_n": n,
                "paper_M": 2 * n,

                "n_outer_completed": (
                    len(records)
                ),

                "paper_protocol_estimate": float(
                    np.mean(values)
                ),

                "std_across_outer": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                    if len(values) > 1
                    else 0.0
                ),
            }
        )


    payload = {
        "method": (
            "Forensic semi-MNIST notebook MMD path"
        ),

        "dataset": (
            "MNIST vs Fake MNIST"
        ),

        "implementation": {
            "data_pool": (
                "first 4000 samples "
                "from real and fake pools"
            ),

            "pool_size": POOL_SIZE,

            "epsilon_parameterization": (
                "direct"
            ),

            "epsilon_initialization": (
                "Uniform(0, 1e-8)"
            ),

            "architecture": (
                "CNN 16-32-64-128 "
                "with 100-D output"
            ),

            "objective": (
                "standardized deep MMD"
            ),

            "epochs": TRAIN_EPOCHS,

            "paired_batch_size": (
                TRAIN_BATCH
            ),

            "lr": TRAIN_LR,
        },

        "evaluation": {
            "outer_datasets": N_OUTER,

            "permutation_calibrations_per_outer": (
                N_CAL
            ),

            "permutations_per_calibration": (
                N_PER
            ),

            "alpha": ALPHA,

            "note": (
                "Each outer dataset contributes "
                "the mean of 100 permutation-test "
                "rejection indicators, matching "
                "the released Tian notebook protocol."
            ),
        },

        "internal_n_grid": (
            INTERNAL_N_LIST
        ),

        "paper_M_grid": (
            PAPER_M_LIST
        ),

        "reference_table3": {
            "M200": 0.290,
            "M400": 0.996,
            "M600": 1.0,
            "M800": 1.0,
            "M1000": 1.0,
        },

        "torch_version": (
            torch.__version__
        ),

        "python_version": (
            platform.python_version()
        ),

        "results": results,

        "points": points,
    }

    return payload


def save_checkpoint(points):

    payload = build_payload(
        points
    )


    temp_json = (
        JSON_PATH + ".tmp"
    )

    with open(
        temp_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )

    os.replace(
        temp_json,
        JSON_PATH,
    )


    temp_pkl = (
        PKL_PATH + ".tmp"
    )

    with open(
        temp_pkl,
        "wb",
    ) as f:

        pickle.dump(
            payload,
            f,
        )

    os.replace(
        temp_pkl,
        PKL_PATH,
    )


# ============================================================
# Load MNIST
#
# Important:
# the real MNIST loader is shuffled under seed 819.
# first4000 therefore refers to the first 4000 samples of
# this deterministic shuffled 60k tensor, matching the
# notebook structure.
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


real_loader = (
    torch.utils.data.DataLoader(

        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=False,

            transform=(
                transforms.Compose(
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
                )
            ),
        ),

        batch_size=60000,
        shuffle=True,
    )
)


data_real_all, _ = next(
    iter(real_loader)
)


with open(
    "./data/Fake_MNIST_data_EP100_N10000.pckl",
    "rb",
) as f:

    data_fake_all = (
        pickle.load(f)[0]
    )


data_fake_all = (
    torch.from_numpy(
        data_fake_all
    ).float()
)


# LOCKED historical pool.
data_real_pool = (
    data_real_all[
        :POOL_SIZE
    ]
)

data_fake_pool = (
    data_fake_all[
        :POOL_SIZE
    ]
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
# Initialize / resume results
# ============================================================

points = {}


if (
    args.resume
    and os.path.exists(
        JSON_PATH
    )
):

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        previous = json.load(f)


    points = previous.get(
        "points",
        {},
    )


    print(
        "\nLoaded checkpoint:",
        JSON_PATH,
        flush=True,
    )


# ============================================================
# Optional reuse of exact M=200 diagnostic trials
# ============================================================

elif (
    args.reuse_m200_diagnostic
    and os.path.exists(
        DIAGNOSTIC_PATH
    )
):

    with open(
        DIAGNOSTIC_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        diag = json.load(f)


    candidate = (
        diag
        .get("results", {})
        .get(
            "first4000_direct",
            {}
        )
    )


    old_records = candidate.get(
        "records",
        []
    )


    if len(old_records) != 10:

        raise RuntimeError(
            "Expected exactly 10 "
            "first4000_direct diagnostic "
            "records, found "
            f"{len(old_records)}."
        )


    normalized = []

    for r in old_records:

        normalized.append(
            {
                "trial": int(
                    r["trial"]
                ),

                "conditional_rejection": float(
                    r[
                        "conditional_rejection"
                    ]
                ),

                "epsilon": float(
                    r["epsilon"]
                ),

                "sigma": float(
                    r["sigma"]
                ),

                "sigma0": float(
                    r["sigma0"]
                ),

                "source": (
                    "historical_2x2_diagnostic"
                ),
            }
        )


    points["100"] = {
        "internal_n": 100,
        "paper_M": 200,
        "records": normalized,
    }


    print(
        "\nReused first 10 M=200 "
        "first4000_direct trials."
    )

    print(
        "Diagnostic mean:",
        np.mean(
            [
                r[
                    "conditional_rejection"
                ]
                for r in normalized
            ]
        ),
    )


    save_checkpoint(
        points
    )


# ============================================================
# One independent outer dataset
# ============================================================

def run_outer_trial(
    n,
    kk,
):

    # --------------------------------------------------------
    # Independent train / test sample
    # --------------------------------------------------------

    (
        s1_tr,
        s1_te,
        s2_tr,
        s2_te,
    ) = sample_mnist_semi(
        data_real_pool,
        data_fake_pool,
        n,
        n,
        kk=kk,
    )


    # --------------------------------------------------------
    # Released initialization
    # --------------------------------------------------------

    model_seed = (
        kk * 19
        + n
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
        + n
    )


    featurizer = (
        Featurizer()
        .to(
            device,
            dtype,
        )
    )


    # --------------------------------------------------------
    # DIRECT epsilon path from later Tian notebook
    #
    # NO log()
    # NO sigmoid()
    # --------------------------------------------------------

    epsilonOPT = MatConvert(
        np.random.rand(1)
        * 10 ** (-8),
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


    optimizer = torch.optim.Adam(

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


    dataset = (
        torch.utils.data.TensorDataset(
            s1_tr,
            s2_tr,
        )
    )


    train_loader = (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=TRAIN_BATCH,
            shuffle=True,
        )
    )


    # --------------------------------------------------------
    # Train MMD-D
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

            real_imgs = (
                real_imgs.to(
                    device,
                    dtype,
                )
            )

            fake_imgs = (
                fake_imgs.to(
                    device,
                    dtype,
                )
            )


            X = torch.cat(
                [
                    real_imgs,
                    fake_imgs,
                ],
                dim=0,
            )


            optimizer.zero_grad()


            features = (
                featurizer(X)
            )


            # Historical direct epsilon.
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


            negative_J = (
                -TEMP[0]
                /
                torch.sqrt(
                    TEMP[1]
                    + 10 ** (-8)
                )
            )


            negative_J.backward()

            optimizer.step()


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


    # --------------------------------------------------------
    # Fixed held-out dataset
    # --------------------------------------------------------

    featurizer.train()


    with torch.no_grad():

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
    #
    # Same held-out dataset,
    # 100 different finite-permutation calibrations.
    # --------------------------------------------------------

    H = np.zeros(
        N_CAL
    )


    for k in range(
        N_CAL
    ):

        H[k], _, _ = (
            TST_MMD_u(
                test_features,
                n,
                N_PER,
                test_raw,
                sigma_final,
                sigma0_final,
                ep_final,
                ALPHA,
                k,
            )
        )


    conditional_rate = float(
        H.mean()
    )


    result = {
        "trial": kk,

        "conditional_rejection": (
            conditional_rate
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

        "source": (
            "full_grid_run"
        ),
    }


    del featurizer


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# Full paper grid
# ============================================================

for n in INTERNAL_N_LIST:

    paper_M = (
        2 * n
    )

    key = str(n)


    if key not in points:

        points[key] = {
            "internal_n": n,
            "paper_M": paper_M,
            "records": [],
        }


    records = points[key][
        "records"
    ]


    start_trial = len(
        records
    )


    if start_trial >= N_OUTER:

        print(
            f"\nM={paper_M} already complete.",
            flush=True,
        )

        continue


    print(
        "\n"
        "=========================================="
    )

    print(
        f"paper M={paper_M} "
        f"| internal n={n}"
    )

    print(
        f"starting outer trial "
        f"{start_trial}/100"
    )

    print(
        "=========================================="
    )


    for kk in tqdm.trange(
        start_trial,
        N_OUTER,
        desc=(
            f"MMD-D paper M={paper_M}"
        ),
    ):

        record = run_outer_trial(
            n,
            kk,
        )


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


        tqdm.tqdm.write(
            f"M={paper_M:4d} "
            f"trial={kk:02d} "
            f"conditional="
            f"{record['conditional_rejection']:.2f} "
            f"running="
            f"{running:.4f} "
            f"J="
            f"{record['final_J']:.3f} "
            f"eps="
            f"{record['epsilon']:.2e} "
            f"sigma0="
            f"{record['sigma0']:.5f}"
        )


        # Save after every outer dataset.
        save_checkpoint(
            points
        )


    final_value = float(
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
        f"\nCOMPLETED M={paper_M}: "
        f"{final_value:.4f}"
    )


    save_checkpoint(
        points
    )


# ============================================================
# Final table
# ============================================================

payload = build_payload(
    points
)


print(
    "\n"
    "=========================================="
)

print(
    "FINAL MMD-D FULL GRID"
)

print(
    "=========================================="
)


for r in payload["results"]:

    print(
        f"M={r['paper_M']:4d} "
        f"(internal n="
        f"{r['internal_n']:3d}) "
        f"MMD-D="
        f"{r['paper_protocol_estimate']:.4f} "
        f"outer="
        f"{r['n_outer_completed']}"
    )


print(
    "\nPaper reference:"
)

print(
    "M = [200, 400, 600, 800, 1000]"
)

print(
    "MMD-D ~= "
    "[.290, .996, 1, 1, 1]"
)


save_checkpoint(
    points
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

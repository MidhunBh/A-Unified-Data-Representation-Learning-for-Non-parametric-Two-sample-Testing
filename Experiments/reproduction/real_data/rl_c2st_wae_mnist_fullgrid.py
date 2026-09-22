import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import tqdm

from utils import (
    Encoder_Img,
    Decoder_Img,
    sample_mnist_semi,
    TST_C2ST_D,
    TST_LCE_D,
)

# ============================================================
# LOCKED FULL-GRID MNIST WAE-RL-C2ST RECONSTRUCTION
#
# Paper grid:
#   M = [200, 400, 600, 800, 1000]
#
# Internal code grid:
#   n = [100, 200, 300, 400, 500]
#
# because:
#   paper M = n_train + n_test = 2*n
#
# WAE choices were locked BEFORE the full-grid run:
#
# Phase 1:
#   author MNIST Encoder_Img / Decoder_Img architecture
#   z = 100
#   epochs = 1000
#   lr = .002
#   batch = 200
#   lambda_MMD = .01
#   Gaussian N(0,I) prior
#   multi-scale IMQ MMD
#
# Phase 2:
#   author's MNIST joint AE/RL-C2ST wrapper
#   WAE reconstruction -> convolutional discriminator
#   WAE + discriminator jointly optimized
#   lr = .0004
#   batch = 200
#   epochs = 2*n
#
# 100 independent outer trials per grid point.
#
# IMPORTANT:
# Tian et al. did not release their WAE implementation.
# This is a principled reconstruction, not recovered source.
# ============================================================


parser = argparse.ArgumentParser()

parser.add_argument(
    "--reuse-existing",
    action="store_true",
    help=(
        "Reuse the already-completed 100-trial M=200 and M=400 "
        "diagnostic results after checking their locked metadata."
    ),
)

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume an interrupted full-grid checkpoint.",
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

dtype = torch.float

MASTER_SEED = 1102

alpha = 0.05

channels = 1
img_size = 32
z_size = 100

INTERNAL_N_LIST = [
    100,
    200,
    300,
    400,
    500,
]

N_OUTER = 100
N_PER = 100

WAE_EPOCHS = 1000
WAE_LR = 0.002
WAE_BATCH = 200
LAMBDA_MMD = 0.01

C2ST_LR = 0.0004
C2ST_BATCH = 200

IMQ_SCALES = (
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)

OUTPUT_STEM = (
    "result/"
    "rl_c2st_wae_mnist_fullgrid_"
    "z100_lam001_authorwrapper_100trials"
)

JSON_PATH = OUTPUT_STEM + ".json"
PKL_PATH = OUTPUT_STEM + ".pkl"

os.makedirs("result", exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

def reset_seed(seed=MASTER_SEED):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# WAE architecture
#
# Same Encoder_Img / Decoder_Img architecture as the released
# MNIST Autoencoder_Img. No extra output BatchNorm.
# ============================================================

class WAEImg(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder_Img(
            channels,
            img_size,
            z_size,
        )

        self.decoder = Decoder_Img(
            channels,
            img_size,
            z_size,
        )

    def forward(self, x):

        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction, z


# ============================================================
# Multi-scale IMQ MMD
# ============================================================

def pairwise_squared_distance(A, B):

    A2 = (
        A * A
    ).sum(
        dim=1,
        keepdim=True,
    )

    B2 = (
        B * B
    ).sum(
        dim=1,
        keepdim=True,
    ).t()

    D = (
        A2
        + B2
        - 2.0 * A @ B.t()
    )

    return torch.clamp(
        D,
        min=0.0,
    )


def imq_mmd(qz, pz):

    nq = qz.shape[0]
    np_ = pz.shape[0]

    Dqq = pairwise_squared_distance(
        qz,
        qz,
    )

    Dpp = pairwise_squared_distance(
        pz,
        pz,
    )

    Dqp = pairwise_squared_distance(
        qz,
        pz,
    )

    # Gaussian N(0,I) prior.
    base = 2.0 * z_size

    total = qz.new_tensor(0.0)

    for scale in IMQ_SCALES:

        C = base * scale

        Kqq = C / (
            C + Dqq
        )

        Kpp = C / (
            C + Dpp
        )

        Kqp = C / (
            C + Dqp
        )

        # Unbiased within-sample terms.
        qq = (
            Kqq.sum()
            - torch.diagonal(Kqq).sum()
        ) / (
            nq * (nq - 1)
        )

        pp = (
            Kpp.sum()
            - torch.diagonal(Kpp).sum()
        ) / (
            np_ * (np_ - 1)
        )

        # Full cross-sample term.
        qp = Kqp.mean()

        total = (
            total
            + qq
            + pp
            - 2.0 * qp
        )

    return total


# ============================================================
# Phase 1: WAE
# ============================================================

def train_wae(S_pool):

    # Match released author's Phase-1 initialization convention.
    reset_seed(MASTER_SEED)

    model = WAEImg().to(
        device,
        dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=WAE_LR,
    )

    dataset = torch.utils.data.TensorDataset(
        S_pool
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=WAE_BATCH,
        shuffle=True,
    )

    final_recon = np.nan
    final_mmd = np.nan

    for ep in range(WAE_EPOCHS):

        model.train()

        recon_sum = 0.0
        mmd_sum = 0.0
        n_batches = 0

        for (x,) in loader:

            reconstruction, z = model(x)

            recon_loss = F.mse_loss(
                reconstruction,
                x,
            )

            prior = torch.randn_like(z)

            mmd_loss = imq_mmd(
                z,
                prior,
            )

            loss = (
                recon_loss
                + LAMBDA_MMD * mmd_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            recon_sum += recon_loss.item()
            mmd_sum += mmd_loss.item()
            n_batches += 1

        final_recon = (
            recon_sum / n_batches
        )

        final_mmd = (
            mmd_sum / n_batches
        )

    model.eval()

    return (
        model,
        float(final_recon),
        float(final_mmd),
    )


# ============================================================
# Author MNIST discriminator
# ============================================================

class Discriminator(nn.Module):

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
                8,
                bn=False,
            ),
            *discriminator_block(
                8,
                16,
            ),
            *discriminator_block(
                16,
                32,
            ),
        )

        ds_size = (
            img_size // 2 ** 3
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(
                32 * ds_size ** 2,
                100,
            ),
            nn.ReLU(),

            nn.Linear(
                100,
                20,
            ),
            nn.ReLU(),

            nn.Linear(
                20,
                2,
            ),
            nn.Softmax(
                dim=1
            ),
        )

    def forward(self, img):

        out = self.model(img)

        out = out.view(
            out.shape[0],
            -1,
        )

        return self.adv_layer(out)


# ============================================================
# Checkpoint helpers
# ============================================================

def atomic_json_dump(obj, path):

    tmp = path + ".tmp"

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
        )

    os.replace(
        tmp,
        path,
    )


def atomic_pickle_dump(obj, path):

    tmp = path + ".tmp"

    with open(
        tmp,
        "wb",
    ) as f:

        pickle.dump(
            obj,
            f,
        )

    os.replace(
        tmp,
        path,
    )


def build_payload(points):

    results = []

    for n in INTERNAL_N_LIST:

        key = str(n)

        if key not in points:
            continue

        trials = points[key]["trials"]

        if len(trials) == 0:
            continue

        power_s = float(
            np.mean(
                [
                    r["reject_S"]
                    for r in trials
                ]
            )
        )

        power_l = float(
            np.mean(
                [
                    r["reject_L"]
                    for r in trials
                ]
            )
        )

        results.append(
            {
                "internal_n": n,
                "paper_M": 2 * n,
                "n_completed": len(
                    trials
                ),
                "RL-C2ST-S": power_s,
                "RL-C2ST-L": power_l,
                "source": points[key].get(
                    "source",
                    "full_grid_run",
                ),
            }
        )

    return {
        "method": (
            "WAE-RL-C2ST "
            "(MNIST author-wrapper reconstruction)"
        ),

        "dataset": (
            "MNIST vs Fake MNIST"
        ),

        "paper_M_grid": [
            2 * n
            for n in INTERNAL_N_LIST
        ],

        "internal_n_grid": (
            INTERNAL_N_LIST
        ),

        "n_outer_target": (
            N_OUTER
        ),

        "phase1": {
            "architecture": (
                "Encoder_Img + Decoder_Img"
            ),
            "z_dim": z_size,
            "epochs": WAE_EPOCHS,
            "lr": WAE_LR,
            "batch_size": WAE_BATCH,
            "lambda_mmd": LAMBDA_MMD,
            "prior": "N(0,I)",
            "kernel": (
                "multi-scale IMQ"
            ),
            "imq_scales": list(
                IMQ_SCALES
            ),
            "pooled_unlabelled_data": (
                "P_train + P_test + "
                "Q_train + Q_test"
            ),
        },

        "phase2": {
            "wrapper": (
                "released MNIST "
                "joint AE/RL-C2ST structure"
            ),
            "joint_WAE_discriminator": True,
            "lr": C2ST_LR,
            "batch_size": C2ST_BATCH,
            "epochs": "2 * internal_n",
        },

        "note": (
            "Tian et al. did not release their WAE "
            "implementation. lambda=.01 was locked using "
            "Phase-1 representation diagnostics before "
            "the full-grid power experiment."
        ),

        "results": results,
        "points": points,

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


def save_checkpoint(points):

    payload = build_payload(
        points
    )

    atomic_json_dump(
        payload,
        JSON_PATH,
    )

    atomic_pickle_dump(
        payload,
        PKL_PATH,
    )


# ============================================================
# Optionally import the two completed 100-trial diagnostics.
#
# These are the same locked configuration:
#
# M=200 -> S=.27 L=.34
# M=400 -> S=.92 L=.95
#
# Metadata is checked before reuse.
# ============================================================

EXISTING_FILES = {
    100: (
        "result/"
        "pilot_wae_imq_mnist_"
        "M200_lam001_authorwrapper_100trials.json"
    ),

    200: (
        "result/"
        "pilot_wae_imq_mnist_"
        "M400_lam001_authorwrapper_100trials.json"
    ),
}


def validate_existing_result(
    obj,
    expected_n,
):

    if obj.get(
        "internal_n"
    ) != expected_n:

        return False

    if obj.get(
        "paper_M"
    ) != 2 * expected_n:

        return False

    if obj.get(
        "n_outer_trials"
    ) != 100:

        return False

    p1 = obj.get(
        "phase1",
        {},
    )

    p2 = obj.get(
        "phase2",
        {},
    )

    checks = [
        p1.get("z_dim") == 100,
        p1.get("epochs") == 1000,
        p1.get("lr") == 0.002,
        p1.get("batch_size") == 200,
        p1.get("lambda_mmd") == 0.01,

        p2.get(
            "joint_WAE_discriminator"
        ) is True,

        p2.get("lr") == 0.0004,
        p2.get("batch_size") == 200,
        p2.get("epochs") == (
            2 * expected_n
        ),
    ]

    return all(checks)


def import_existing_points(points):

    for n, path in EXISTING_FILES.items():

        if not os.path.exists(path):

            print(
                "Existing result not found:",
                path,
                flush=True,
            )

            continue

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            obj = json.load(f)

        if not validate_existing_result(
            obj,
            n,
        ):

            raise RuntimeError(
                "Existing result failed "
                f"locked-config validation: {path}"
            )

        trials = obj.get(
            "trials",
            [],
        )

        if len(trials) != 100:

            raise RuntimeError(
                "Expected exactly 100 trial "
                f"records in {path}, got "
                f"{len(trials)}."
            )

        points[str(n)] = {
            "internal_n": n,
            "paper_M": 2 * n,
            "trials": trials,
            "source": path,
        }

        print(
            f"Reused M={2*n}: "
            f"S={obj['RL-C2ST-S']:.3f}, "
            f"L={obj['RL-C2ST-L']:.3f}",
            flush=True,
        )


# ============================================================
# Load / initialize checkpoint
# ============================================================

points = {}

if args.resume and os.path.exists(
    JSON_PATH
):

    with open(
        JSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        old = json.load(f)

    points = old.get(
        "points",
        {},
    )

    print(
        "Loaded existing full-grid checkpoint.",
        flush=True,
    )

elif args.reuse_existing:

    import_existing_points(
        points
    )

    save_checkpoint(
        points
    )


# ============================================================
# Load MNIST data
# ============================================================

print(
    "device:",
    device,
    flush=True,
)

reset_seed(819)

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

    data_fake_all = pickle.load(
        f
    )[0]

data_fake_all = torch.from_numpy(
    data_fake_all
).float()


# ============================================================
# Full grid
# ============================================================

for n in INTERNAL_N_LIST:

    paper_M = 2 * n

    key = str(n)

    if key not in points:

        points[key] = {
            "internal_n": n,
            "paper_M": paper_M,
            "trials": [],
            "source": "full_grid_run",
        }

    trial_records = points[key][
        "trials"
    ]

    start_trial = len(
        trial_records
    )

    if start_trial >= N_OUTER:

        power_s = np.mean(
            [
                r["reject_S"]
                for r in trial_records
            ]
        )

        power_l = np.mean(
            [
                r["reject_L"]
                for r in trial_records
            ]
        )

        print(
            f"\nM={paper_M} already complete: "
            f"S={power_s:.3f}, "
            f"L={power_l:.3f}",
            flush=True,
        )

        continue

    print(
        f"\n======================================\n"
        f"Running internal n={n}, paper M={paper_M}\n"
        f"Starting at trial {start_trial}/100\n"
        f"======================================",
        flush=True,
    )

    for kk in tqdm.trange(
        start_trial,
        N_OUTER,
        desc=f"MNIST WAE M={paper_M}",
    ):

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        (
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ) = sample_mnist_semi(
            data_real_all,
            data_fake_all,
            n,
            n,
            kk=kk,
        )

        # ----------------------------------------------------
        # Phase 1
        # ----------------------------------------------------

        S_pool = torch.cat(
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

        wae, recon, mmd = train_wae(
            S_pool
        )

        # ----------------------------------------------------
        # Phase 2
        # ----------------------------------------------------

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

        y_train = torch.cat(
            [
                torch.zeros(n),
                torch.ones(n),
            ]
        ).to(
            device
        ).long()

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

        # Author downstream init convention.
        reset_seed(
            MASTER_SEED
        )

        discriminator = (
            Discriminator().to(
                device,
                dtype,
            )
        )

        # Joint Phase-2 optimization.
        for p in wae.parameters():

            p.requires_grad = True

        optimizer_D = torch.optim.Adam(
            list(
                discriminator.parameters()
            )
            + list(
                wae.parameters()
            ),
            lr=C2ST_LR,
        )

        criterion = (
            nn.CrossEntropyLoss().to(
                device
            )
        )

        dataset = (
            torch.utils.data.TensorDataset(
                S_train,
                y_train,
            )
        )

        loader = (
            torch.utils.data.DataLoader(
                dataset,
                batch_size=C2ST_BATCH,
                shuffle=True,
            )
        )

        def c_model(x):

            reconstructed, _ = wae(x)

            return discriminator(
                reconstructed
            )

        c2st_epochs = 2 * n

        for ep in range(
            c2st_epochs
        ):

            for xb, yb in loader:

                output = c_model(
                    xb
                )

                loss = criterion(
                    output,
                    yb,
                )

                optimizer_D.zero_grad()
                loss.backward()
                optimizer_D.step()

        # ----------------------------------------------------
        # Held-out test
        # ----------------------------------------------------

        wae.eval()
        discriminator.eval()

        h_s, _, stat_s = TST_C2ST_D(
            S_test,
            n,
            N_PER,
            alpha,
            c_model,
            device,
            dtype,
        )

        h_l, _, stat_l = TST_LCE_D(
            S_test,
            n,
            N_PER,
            alpha,
            c_model,
            device,
            dtype,
        )

        trial_records.append(
            {
                "trial": kk,
                "reject_S": int(h_s),
                "reject_L": int(h_l),
                "phase1_recon": (
                    float(recon)
                ),
                "phase1_mmd": (
                    float(mmd)
                ),
                "stat_S": float(
                    stat_s
                ),
                "stat_L": float(
                    stat_l
                ),
            }
        )

        running_s = float(
            np.mean(
                [
                    r["reject_S"]
                    for r in trial_records
                ]
            )
        )

        running_l = float(
            np.mean(
                [
                    r["reject_L"]
                    for r in trial_records
                ]
            )
        )

        tqdm.tqdm.write(
            f"M={paper_M} "
            f"trial={kk:02d} "
            f"S={h_s} L={h_l} "
            f"running="
            f"{running_s:.3f}/"
            f"{running_l:.3f} "
            f"recon={recon:.6f} "
            f"mmd={mmd:.6f}"
        )

        # Save after every outer trial.
        save_checkpoint(
            points
        )

        del wae
        del discriminator

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    power_s = float(
        np.mean(
            [
                r["reject_S"]
                for r in trial_records
            ]
        )
    )

    power_l = float(
        np.mean(
            [
                r["reject_L"]
                for r in trial_records
            ]
        )
    )

    print(
        f"\nCOMPLETED M={paper_M}: "
        f"S={power_s:.3f}, "
        f"L={power_l:.3f}",
        flush=True,
    )

    save_checkpoint(
        points
    )


# ============================================================
# Final grid
# ============================================================

payload = build_payload(
    points
)

print(
    "\n=== FINAL FULL GRID ==="
)

for r in payload["results"]:

    print(
        f"M={r['paper_M']:4d} "
        f"(internal n={r['internal_n']:3d}) "
        f"S={r['RL-C2ST-S']:.3f} "
        f"L={r['RL-C2ST-L']:.3f} "
        f"trials={r['n_completed']}"
    )

save_checkpoint(
    points
)

print("\nSaved:")
print(" ", JSON_PATH)
print(" ", PKL_PATH)

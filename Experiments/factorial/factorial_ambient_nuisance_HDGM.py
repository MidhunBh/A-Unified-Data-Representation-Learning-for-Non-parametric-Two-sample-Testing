import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from utils import (
    MatConvert,
    train_autoencoder,
    sample_hdgm_semi_t1,
    sample_hdgm_semi_t2,
    C2ST_NN_fit,
    TST_C2ST,
    TST_LCE,
)


# ---------------------------------------------------------------------
# Controlled ambient-nuisance sensitivity
#
# Signal distribution is ALWAYS Tian hard HDGM in d=2.
# For total dimension d > 2, append independent N(0, I_{d-2})
# coordinates identically to P and Q.
#
# Thus only nuisance dimension changes; the underlying P-vs-Q signal
# remains fixed.
# ---------------------------------------------------------------------

DEFAULT_DIMS = [2, 10, 20, 50, 100]

H = 30
AE_LATENT_DIM = 30
FINAL_DR_DIM = 30

N_TRAIN = 500
N_TEST = 500

DEFAULT_AE_EPOCHS = 2000
DEFAULT_C2ST_EPOCHS = 1000
DEFAULT_TRIALS = 100
DEFAULT_PERMUTATIONS = 100

BATCH_SIZE = 1024
ALPHA = 0.05
LR_AE = 0.002
LR_C2ST = 0.002


class ExtendedModelFixed(torch.nn.Module):
    def __init__(self, encoder, ae_latent_dim, H, final_out):
        super().__init__()
        self.encoder = encoder
        self.additional_layers = torch.nn.Sequential(
            torch.nn.Linear(ae_latent_dim, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, final_out, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.additional_layers(x)


def append_nested_nuisance(base_arrays, max_dim, kk, null_mode):
    """
    Generate one maximum-width nuisance realization for each split/source.
    Lower-dimensional conditions use prefixes of the SAME nuisance draws.

    This creates paired/nested dimension conditions within each outer trial,
    while P and Q nuisance draws remain independent.
    """
    nuisance_width = max_dim - 2

    if nuisance_width <= 0:
        return {
            2: tuple(np.asarray(a, dtype=np.float32) for a in base_arrays)
        }

    # Separate seed families for alternative and null experiments.
    mode_offset = 50_000_000 if null_mode else 10_000_000

    max_noise = []
    for j, arr in enumerate(base_arrays):
        seed = mode_offset + 100_003 * kk + 10_007 * j + 17
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(
            size=(len(arr), nuisance_width)
        ).astype(np.float32)
        max_noise.append(z)

    out = {}

    for d in DEFAULT_DIMS:
        if d > max_dim:
            continue

        width = d - 2
        arrays_d = []

        for arr, z in zip(base_arrays, max_noise):
            arr = np.asarray(arr, dtype=np.float32)

            if width == 0:
                x = arr.copy()
            else:
                x = np.concatenate(
                    [arr, z[:, :width]],
                    axis=1
                )

            arrays_d.append(x)

        out[d] = tuple(arrays_d)

    return out


def make_trial_data(kk, dims, null_mode):
    """
    Generate the fixed 2-D signal once, then append nested nuisance
    coordinates for all requested dimensions.
    """
    if null_mode:
        base = sample_hdgm_semi_t1(
            N_TRAIN,
            N_TEST,
            d=2,
            kk=kk,
        )
    else:
        base = sample_hdgm_semi_t2(
            N_TRAIN,
            N_TEST,
            d=2,
            kk=kk,
            level="hard",
        )

    max_dim = max(dims)

    # Generate all default nested conditions, then select requested dims.
    nested = append_nested_nuisance(
        base,
        max_dim=max_dim,
        kk=kk,
        null_mode=null_mode,
    )

    return {d: nested[d] for d in dims}


def run_cell(
    arrays,
    d,
    kk,
    ae_epochs,
    c2st_epochs,
    permutations,
    device,
    dtype,
):
    s1_tr, s1_te, s2_tr, s2_te = arrays

    # Preserve Tian-style transductive unsupervised representation fitting:
    # AE sees pooled train + held-out covariates, but no labels.
    S_encoder = np.concatenate(
        (s1_tr, s1_te, s2_tr, s2_te),
        axis=0,
    )
    S_encoder = MatConvert(S_encoder, device, dtype)

    encoder = train_autoencoder(
        S_encoder,
        ae_epochs,
        d,
        H,
        AE_LATENT_DIM,
        512,
        device,
        dtype,
        lr=LR_AE,
    )

    # Validated HDGM Phase-2 setting: freeze AE encoder.
    for p in encoder.parameters():
        p.requires_grad = False

    S_train = np.concatenate(
        (s1_tr, s2_tr),
        axis=0,
    )
    S_train = MatConvert(S_train, device, dtype)

    y = torch.cat(
        [
            torch.zeros(len(s1_tr)),
            torch.ones(len(s2_tr)),
        ]
    ).to(device, dtype).long()

    base_model = ExtendedModelFixed(
        encoder,
        AE_LATENT_DIM,
        H,
        FINAL_DR_DIM,
    )

    model, w, b = C2ST_NN_fit(
        S_train,
        y,
        d,
        H,
        FINAL_DR_DIM,
        c2st_epochs,
        BATCH_SIZE,
        device,
        dtype,
        base_model,
        lr_c2st=LR_C2ST,
    )

    S_test = np.concatenate(
        (s1_te, s2_te),
        axis=0,
    )
    S_test = MatConvert(S_test, device, dtype)

    # One test decision per independent outer trial.
    # This deliberately avoids the old redundant N_TEST loop.
    h_s, _, stat_s = TST_C2ST(
        S_test,
        len(s1_te),
        permutations,
        ALPHA,
        model,
        w,
        b,
        rd_seed=0,
    )

    h_l, _, stat_l = TST_LCE(
        S_test,
        len(s1_te),
        permutations,
        ALPHA,
        model,
        w,
        b,
        rd_seed=0,
    )

    return {
        "reject_S": int(h_s),
        "reject_L": int(h_l),
        "stat_S": float(stat_s),
        "stat_L": float(stat_l),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=DEFAULT_DIMS,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=DEFAULT_AE_EPOCHS,
    )
    parser.add_argument(
        "--c2st-epochs",
        type=int,
        default=DEFAULT_C2ST_EPOCHS,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result/factorial_ambient_nuisance_HDGM_N4000.json",
    )

    args = parser.parse_args()

    dims = list(args.dims)

    invalid = [d for d in dims if d < 2]
    if invalid:
        raise ValueError(f"All dimensions must be >=2: {invalid}")

    # append_nested_nuisance currently builds prefixes using this grid.
    unknown = [d for d in dims if d not in DEFAULT_DIMS]
    if unknown:
        raise ValueError(
            f"Use planned dimensions {DEFAULT_DIMS}; got unsupported {unknown}"
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dtype = torch.float

    print("================================================")
    print("CONTROLLED HDGM AMBIENT-NUISANCE SENSITIVITY")
    print("================================================")
    print("device:", device)
    print("CUDA_VISIBLE_DEVICES:",
          __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("dims:", dims)
    print("trials:", args.trials)
    print("AE epochs:", args.ae_epochs)
    print("C2ST epochs:", args.c2st_epochs)
    print("permutations:", args.permutations)
    print()

    trial_records = []

    t0 = time.time()

    for kk in range(args.trials):
        print()
        print(f"========== OUTER TRIAL {kk + 1}/{args.trials} ==========")

        alt_data = make_trial_data(
            kk,
            dims,
            null_mode=False,
        )
        null_data = make_trial_data(
            kk,
            dims,
            null_mode=True,
        )

        for d in dims:
            print()
            print(f"--- trial={kk} d={d} POWER ---")

            power = run_cell(
                alt_data[d],
                d,
                kk,
                args.ae_epochs,
                args.c2st_epochs,
                args.permutations,
                device,
                dtype,
            )

            print(
                f"d={d} power:"
                f" S={power['reject_S']}"
                f" L={power['reject_L']}"
            )

            print(f"--- trial={kk} d={d} TYPE-I ---")

            null = run_cell(
                null_data[d],
                d,
                kk,
                args.ae_epochs,
                args.c2st_epochs,
                args.permutations,
                device,
                dtype,
            )

            print(
                f"d={d} null:"
                f" S={null['reject_S']}"
                f" L={null['reject_L']}"
            )

            trial_records.append(
                {
                    "trial": kk,
                    "d": d,
                    "power": power,
                    "null": null,
                }
            )

        # Save after every outer trial.
        by_dim = {}

        for d in dims:
            rows = [
                r for r in trial_records
                if r["d"] == d
            ]

            by_dim[str(d)] = {
                "completed_trials": len(rows),
                "power_S": float(
                    np.mean([r["power"]["reject_S"] for r in rows])
                ),
                "power_L": float(
                    np.mean([r["power"]["reject_L"] for r in rows])
                ),
                "type1_S": float(
                    np.mean([r["null"]["reject_S"] for r in rows])
                ),
                "type1_L": float(
                    np.mean([r["null"]["reject_L"] for r in rows])
                ),
            }

        meta = {
            "schema_version": 1,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip(),
            "experiment":
                "controlled_ambient_nuisance_dimension_HDGM",
            "classification":
                "Tian-style sensitivity extension",
            "signal":
                "fixed hard HDGM d=2",
            "nuisance":
                "independent standard Gaussian coordinates identical in law for P and Q",
            "paired_design":
                "within each outer trial, dimensions use prefixes of the same maximum-width nuisance draws",
            "dims": dims,
            "n_train_internal": N_TRAIN,
            "n_test_internal": N_TEST,
            "paper_N_convention": 8 * N_TRAIN,
            "H": H,
            "ae_latent_dim": AE_LATENT_DIM,
            "final_dr_dim": FINAL_DR_DIM,
            "ae_epochs": args.ae_epochs,
            "c2st_epochs": args.c2st_epochs,
            "outer_trials_planned": args.trials,
            "permutations": args.permutations,
            "alpha": ALPHA,
            "encoder_phase2": "frozen",
            "representation_protocol":
                "AE fitted on pooled unlabeled train+test covariates, preserving Tian reproduction protocol",
            "implementation_note":
                "one permutation-test decision per independent outer trial; redundant deterministic N_TEST loop removed",
            "summary_by_dim": by_dim,
            "trial_records": trial_records,
            "runtime_seconds_so_far": time.time() - t0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w") as f:
            json.dump(meta, f, indent=2)

        print()
        print("CURRENT SUMMARY")
        for d in dims:
            print(d, by_dim[str(d)])

        print("saved:", out)

    print()
    print("================================================")
    print("EXPERIMENT FINISHED")
    print("================================================")
    print("runtime seconds:", time.time() - t0)


if __name__ == "__main__":
    main()

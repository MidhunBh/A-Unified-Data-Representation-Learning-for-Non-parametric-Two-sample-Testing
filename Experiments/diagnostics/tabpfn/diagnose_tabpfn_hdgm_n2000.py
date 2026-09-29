"""
Controlled TabPFN representation diagnostic on HDGM d=2, N=2000.

Purpose
-------
Determine whether the weak low-N TabPFN two-sample power is caused by:

1. loss of the HDGM covariance signal in the pretrained representation, or
2. preservation of the signal but poor downstream sample efficiency caused
   by the 512-dimensional TabPFN representation.

Representations compared on the SAME HDGM splits:
    raw        : original 2-D HDGM observations
    tabpfn512  : historical TabPFN v1 decoder-input representation
    pca30      : TabPFN 512-D -> PCA 30-D
    pca10      : TabPFN 512-D -> PCA 10-D
    rp30       : TabPFN 512-D -> random projection 30-D

PCA is fitted on the full unlabeled pooled trial, matching the semi-supervised
representation-learning convention used by the existing Tian reproduction.

Both:
    HDGM-D alternative -> rejection rate / power
    HDGM-S null        -> Type-I rejection rate

are evaluated.

This is a diagnostic, not a reproduction result.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from tabpfn import TabPFNClassifier

from utils import (
    C2ST_NN_fit,
    MatConvert,
    TST_C2ST,
    TST_LCE,
    sample_hdgm_semi_t1,
    sample_hdgm_semi_t2,
)


# ---------------------------------------------------------------------
# Fixed reproduction settings
# ---------------------------------------------------------------------

SEED = 1102

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

alpha = 0.05
batch_size = 1024

x_in = 2
H = 30
x_out = 30

N_EPOCH = 1000
N_PER = 100

# Existing repository convention:
# total reported N = 8*n.
# Hence n=250 -> N=2000.
n = 250
n_train = n
n_test = n
TOTAL_N = 8 * n

TABPFN_HIDDEN = 512
CONTEXT_SIZE = 500

EXPECTED_MODEL_HASH = (
    "adc33028590f56b7b7562d701a542ef6c00d5104fae9de8460ab00f6888aff94"
)
EXPECTED_CHECKPOINT_HASH = (
    "3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392"
)


# ---------------------------------------------------------------------
# TabPFN model + provenance
# ---------------------------------------------------------------------

print("device:", device)
print("N:", TOTAL_N)
print("TabPFN:", md.version("tabpfn"))

tabpfn_clf = TabPFNClassifier(device=device)
transformer = tabpfn_clf.model[2]


def model_state_hash(model):
    h = hashlib.sha256()

    for name, tensor in sorted(model.state_dict().items()):
        t = tensor.detach().cpu().contiguous()

        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())

    return h.hexdigest()


MODEL_HASH = model_state_hash(transformer)

if MODEL_HASH.lower() != EXPECTED_MODEL_HASH.lower():
    raise RuntimeError(
        f"Unexpected TabPFN model hash:\n"
        f"got      {MODEL_HASH}\n"
        f"expected {EXPECTED_MODEL_HASH}"
    )

tabpfn_root = pathlib.Path(
    sys.modules["tabpfn"].__file__
).resolve().parent

checkpoint_candidates = list(
    tabpfn_root.rglob("prior_diff_real_checkpoint_n_0_epoch_42.cpkt")
)

if len(checkpoint_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one TabPFN checkpoint, found "
        f"{len(checkpoint_candidates)}"
    )

checkpoint_path = checkpoint_candidates[0]

hc = hashlib.sha256()
with checkpoint_path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        hc.update(chunk)

CHECKPOINT_HASH = hc.hexdigest()

if CHECKPOINT_HASH.lower() != EXPECTED_CHECKPOINT_HASH.lower():
    raise RuntimeError(
        f"Unexpected checkpoint hash:\n"
        f"got      {CHECKPOINT_HASH}\n"
        f"expected {EXPECTED_CHECKPOINT_HASH}"
    )

print("model hash:      MATCH")
print("checkpoint hash: MATCH")


# ---------------------------------------------------------------------
# Historical real-context representation extraction
# ---------------------------------------------------------------------

def tabpfn_extract_realcontext(S_pool_np, kk):
    """
    Preserve the existing real-context TabPFN implementation.

    Context:
        500-point random subsample of this trial's full unlabeled pool.

    Labels:
        alternating parity placeholders, unrelated to P/Q identity.

    Query:
        full pooled trial.

    Representation:
        decoder input, ensemble-averaged -> 512 dimensions.
    """

    n_pool = len(S_pool_np)

    np.random.seed(SEED * kk + 7)

    ctx_idx = np.random.choice(
        n_pool,
        min(CONTEXT_SIZE, n_pool),
        replace=False,
    )

    X_context = S_pool_np[ctx_idx].astype(np.float32)
    y_placeholder = np.arange(len(ctx_idx)) % 2

    tabpfn_clf.fit(X_context, y_placeholder)

    captured = {}

    def hook(module, input):
        captured["ir"] = input[0].detach()

    handle = transformer.decoder.register_forward_pre_hook(hook)

    with torch.no_grad():
        _ = tabpfn_clf.predict_proba(
            S_pool_np.astype(np.float32)
        )

    handle.remove()

    ctx_len = len(ctx_idx)

    if "ir" not in captured:
        raise RuntimeError("TabPFN decoder hook captured nothing.")

    ir_full = captured["ir"]

    ir_query = ir_full[ctx_len:].mean(dim=1)

    if ir_query.shape[0] != n_pool:
        raise RuntimeError(
            f"Unexpected TabPFN query shape: "
            f"{tuple(ir_query.shape)}, expected first dimension {n_pool}"
        )

    return ir_query.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------
# Representation construction
# ---------------------------------------------------------------------

def make_representations(S_pool_np, kk):
    raw = S_pool_np.astype(np.float32)

    tab = tabpfn_extract_realcontext(S_pool_np, kk)

    # Unsupervised transforms fitted on the same full pooled observations.
    pca30 = PCA(
        n_components=30,
        svd_solver="full",
    ).fit_transform(tab).astype(np.float32)

    pca10 = PCA(
        n_components=10,
        svd_solver="full",
    ).fit_transform(tab).astype(np.float32)

    # Control: dimension reduction without variance-directed PCA.
    rp30 = GaussianRandomProjection(
        n_components=30,
        random_state=SEED + kk,
    ).fit_transform(tab).astype(np.float32)

    return {
        "raw": raw,
        "tabpfn512": tab,
        "pca30": pca30,
        "pca10": pca10,
        "rp30": rp30,
    }


def split_pool(Z, lengths):
    n1tr, n1te, n2tr, n2te = lengths

    i0 = 0
    i1 = i0 + n1tr
    i2 = i1 + n1te
    i3 = i2 + n2tr
    i4 = i3 + n2te

    return (
        Z[i0:i1],
        Z[i1:i2],
        Z[i2:i3],
        Z[i3:i4],
    )


# ---------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------

def geometry(P, Q):
    """
    Scale-normalized descriptive geometry.

    mean_gap:
        mean separation divided by pooled RMS variation.

    covariance_gap:
        Frobenius covariance difference divided by average covariance
        Frobenius norm.

    HDGM-D is predominantly a covariance-shift problem, so covariance_gap
    is especially informative here.
    """

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)

    cov_p = np.atleast_2d(np.cov(P, rowvar=False))
    cov_q = np.atleast_2d(np.cov(Q, rowvar=False))

    pooled_cov = 0.5 * (cov_p + cov_q)

    mean_denom = np.sqrt(
        max(float(np.trace(pooled_cov)), 0.0)
    ) + 1e-12

    mean_gap = float(
        np.linalg.norm(mu_p - mu_q) / mean_denom
    )

    cov_denom = (
        0.5
        * (
            np.linalg.norm(cov_p, ord="fro")
            + np.linalg.norm(cov_q, ord="fro")
        )
        + 1e-12
    )

    covariance_gap = float(
        np.linalg.norm(cov_p - cov_q, ord="fro")
        / cov_denom
    )

    return {
        "mean_gap_normalized": mean_gap,
        "covariance_gap_normalized": covariance_gap,
    }


# ---------------------------------------------------------------------
# C2ST evaluation
# ---------------------------------------------------------------------

def accuracy(model, w, b, X, y):
    with torch.no_grad():
        logits = model(X).mm(w) + b
        pred = logits.argmax(dim=1)

    return float(
        (pred == y).float().mean().item()
    )


def evaluate_representation(
    name,
    Z,
    lengths,
):
    z1tr, z1te, z2tr, z2te = split_pool(Z, lengths)

    dim = Z.shape[1]

    S_train_np = np.concatenate(
        [z1tr, z2tr],
        axis=0,
    )

    S_test_np = np.concatenate(
        [z1te, z2te],
        axis=0,
    )

    S_train = MatConvert(
        S_train_np.astype(np.float32),
        device,
        dtype,
    )

    S_test = MatConvert(
        S_test_np.astype(np.float32),
        device,
        dtype,
    )

    y_train = torch.cat(
        [
            torch.zeros(len(z1tr)),
            torch.ones(len(z2tr)),
        ]
    ).to(device, dtype).long()

    y_test = torch.cat(
        [
            torch.zeros(len(z1te)),
            torch.ones(len(z2te)),
        ]
    ).to(device, dtype).long()

    model, w, b = C2ST_NN_fit(
        S_train,
        y_train,
        dim,
        H,
        x_out,
        N_EPOCH,
        batch_size,
        device,
        dtype,
        model=None,
        lr_c2st=0.002,
    )

    train_acc = accuracy(
        model, w, b, S_train, y_train
    )

    test_acc = accuracy(
        model, w, b, S_test, y_test
    )

    # The repository TST functions use deterministic permutations at the
    # default rd_seed=0. Repeating the same call 100 times therefore gives
    # the same decision. One call is exactly sufficient for this diagnostic.
    h_s, threshold_s, stat_s = TST_C2ST(
        S_test,
        len(z1te),
        N_PER,
        alpha,
        model,
        w,
        b,
    )

    h_l, threshold_l, stat_l = TST_LCE(
        S_test,
        len(z1te),
        N_PER,
        alpha,
        model,
        w,
        b,
    )

    geom = geometry(z1te, z2te)

    downstream_params = (
        sum(p.numel() for p in model.parameters())
        + w.numel()
        + b.numel()
    )

    result = {
        "representation": name,
        "dim": int(dim),
        "downstream_parameters": int(downstream_params),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "reject_C2ST_S": int(h_s),
        "reject_C2ST_L": int(h_l),
        "stat_C2ST_S": float(stat_s.item()),
        "stat_C2ST_L": float(stat_l.item()),
        "threshold_C2ST_S": float(threshold_s),
        "threshold_C2ST_L": float(threshold_l),
        **geom,
    }

    del model, w, b, S_train, S_test
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------
# One HDGM trial
# ---------------------------------------------------------------------

def run_one_trial(kk, null=False):
    if null:
        s1tr, s1te, s2tr, s2te = sample_hdgm_semi_t1(
            n_train,
            n_test,
            d=x_in,
            kk=kk,
        )
    else:
        s1tr, s1te, s2tr, s2te = sample_hdgm_semi_t2(
            n_train,
            n_test,
            d=x_in,
            kk=kk,
            level="hard",
        )

    S_pool = np.concatenate(
        [s1tr, s1te, s2tr, s2te],
        axis=0,
    )

    lengths = (
        len(s1tr),
        len(s1te),
        len(s2tr),
        len(s2te),
    )

    reps = make_representations(
        S_pool,
        kk + (100000 if null else 0),
    )

    outputs = {}

    for name, Z in reps.items():
        print(
            f"\n[{('NULL' if null else 'ALT')}] "
            f"trial={kk} representation={name} "
            f"shape={Z.shape}"
        )

        outputs[name] = evaluate_representation(
            name,
            Z,
            lengths,
        )

        r = outputs[name]

        print(
            f"  train_acc={r['train_accuracy']:.3f} "
            f"test_acc={r['test_accuracy']:.3f} "
            f"S={r['reject_C2ST_S']} "
            f"L={r['reject_C2ST_L']} "
            f"mean_gap={r['mean_gap_normalized']:.4f} "
            f"cov_gap={r['covariance_gap_normalized']:.4f}"
        )

    return outputs


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def summarize(trials):
    method_names = list(trials[0].keys())

    summary = {}

    metrics = [
        "train_accuracy",
        "test_accuracy",
        "reject_C2ST_S",
        "reject_C2ST_L",
        "mean_gap_normalized",
        "covariance_gap_normalized",
    ]

    for method in method_names:
        s = {
            "dim": trials[0][method]["dim"],
            "downstream_parameters":
                trials[0][method]["downstream_parameters"],
        }

        for metric in metrics:
            values = [
                float(t[method][metric])
                for t in trials
            ]

            s[metric + "_mean"] = float(
                np.mean(values)
            )

            s[metric + "_std"] = float(
                np.std(values, ddof=1)
                if len(values) > 1
                else 0.0
            )

        summary[method] = s

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trials",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    started = time.time()

    alt_trials = []
    null_trials = []

    for kk in range(args.trials):
        print("\n" + "=" * 72)
        print(
            f"ALTERNATIVE HDGM-D: trial "
            f"{kk + 1}/{args.trials}"
        )
        print("=" * 72)

        alt_trials.append(
            run_one_trial(kk, null=False)
        )

        print("\n" + "=" * 72)
        print(
            f"NULL HDGM-S: trial "
            f"{kk + 1}/{args.trials}"
        )
        print("=" * 72)

        null_trials.append(
            run_one_trial(kk, null=True)
        )

    alt_summary = summarize(alt_trials)
    null_summary = summarize(null_trials)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    result = {
        "diagnostic": "TabPFN HDGM N=2000 dimensionality/generalization diagnostic",
        "git_commit_at_run": commit,
        "N": TOTAL_N,
        "internal_n": n,
        "d": x_in,
        "level_alternative": "hard",
        "null": "HDGM-S via sample_hdgm_semi_t1",
        "trials": args.trials,
        "N_EPOCH": N_EPOCH,
        "N_PER": N_PER,
        "context_size": CONTEXT_SIZE,
        "representations": [
            "raw",
            "tabpfn512",
            "pca30",
            "pca10",
            "rp30",
        ],
        "pca_protocol":
            "PCA fitted on full unlabeled pooled trial, matching semi-supervised Phase-1 convention.",
        "tabpfn_protocol":
            "TabPFN 0.1.11 real-context extraction: 500 pooled observations, parity placeholder labels, decoder-input hook, ensemble mean.",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": md.version("scikit-learn"),
            "tabpfn": md.version("tabpfn"),
            "device": str(device),
            "gpu":
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None,
        },
        "provenance": {
            "tabpfn_checkpoint":
                str(checkpoint_path),
            "checkpoint_sha256":
                CHECKPOINT_HASH,
            "loaded_model_state_sha256":
                MODEL_HASH,
        },
        "alternative_trials": alt_trials,
        "null_trials": null_trials,
        "alternative_summary": alt_summary,
        "null_summary": null_summary,
        "runtime_seconds":
            float(time.time() - started),
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    print("\nALTERNATIVE / HDGM-D")
    for method, s in alt_summary.items():
        print(
            f"{method:12s} "
            f"dim={s['dim']:3d} "
            f"train={s['train_accuracy_mean']:.3f} "
            f"test={s['test_accuracy_mean']:.3f} "
            f"S={s['reject_C2ST_S_mean']:.3f} "
            f"L={s['reject_C2ST_L_mean']:.3f} "
            f"covgap={s['covariance_gap_normalized_mean']:.4f}"
        )

    print("\nNULL / HDGM-S")
    for method, s in null_summary.items():
        print(
            f"{method:12s} "
            f"dim={s['dim']:3d} "
            f"train={s['train_accuracy_mean']:.3f} "
            f"test={s['test_accuracy_mean']:.3f} "
            f"S={s['reject_C2ST_S_mean']:.3f} "
            f"L={s['reject_C2ST_L_mean']:.3f}"
        )

    print("\nSaved:", output)
    print(
        "Runtime:",
        result["runtime_seconds"],
        "seconds",
    )


if __name__ == "__main__":
    main()

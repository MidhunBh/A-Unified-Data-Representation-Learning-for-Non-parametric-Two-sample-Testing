import os
import json
import time
import pickle
import itertools
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from utils import C2ST_NN_fit


# ============================================================
# Frozen DINOv2 small-m exploratory pilot
#
# PURPOSE:
#   Find a non-saturated sample-size region BEFORE LoRA.
#
# This exploratory sweep MUST NOT later be compared directly
# against fresh LoRA trials. Once m is selected, frozen and LoRA
# will both be rerun on fresh paired trial seeds.
# ============================================================

DINO_REPO = (
    "facebookresearch/dinov2:"
    "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)

M_LIST = [5, 10, 15, 25, 50]
N_TRIALS = 20

NULL_M = 25
NULL_TRIALS = 10

HEAD_EPOCHS = 1000
HEAD_LR = 0.002
HEAD_H = 30
HEAD_OUT = 30
HEAD_BATCH = 128

MC_PERMUTATIONS = 999
ALPHA = 0.05

DINO_DIM = 384
EMBED_BATCH = 256

OUTPUT = Path("result/dinov2_frozen_small_m_exploratory_v2.json")
CACHE = Path(
    "data/dinov2_vits14_7764ea0_"
    "correctnorm_mnist_fake_embeddings.pt"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
DTYPE = torch.float32


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def as_nchw(x):
    x = torch.as_tensor(x)

    if x.ndim == 3:
        x = x.unsqueeze(1)

    if x.ndim != 4:
        raise ValueError(f"Expected 4-D image tensor; got {x.shape}")

    # Handle channel-last if necessary.
    if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)

    if x.shape[1] not in (1, 3):
        raise ValueError(f"Cannot identify image channels: {x.shape}")

    return x


def convert_fake_to_unit_interval(x):
    """
    Convert retained fake-MNIST tensor to [0,1] explicitly.

    We log the rule rather than silently assuming its original range.
    """
    x = as_nchw(x).float()

    lo = float(x.min())
    hi = float(x.max())

    print(f"raw fake range: [{lo:.6f}, {hi:.6f}]")

    if lo >= -1.05 and hi <= 1.05 and lo < -0.05:
        rule = "mapped [-1,1] -> [0,1]"
        x = (x + 1.0) / 2.0

    elif lo >= -1e-6 and hi <= 1.05:
        rule = "already [0,1]"

    elif lo >= -1e-6 and hi <= 255.5:
        rule = "mapped [0,255] -> [0,1]"
        x = x / 255.0

    else:
        raise ValueError(
            "Unexpected fake-MNIST range; refusing automatic preprocessing: "
            f"[{lo}, {hi}]"
        )

    x = x.clamp(0.0, 1.0)

    print("fake conversion rule:", rule)
    print(
        "converted fake range:",
        float(x.min()),
        float(x.max()),
    )

    return x, rule


def prepare_dino_batch(x):
    """
    Correct DINOv2 input:
      raw grayscale [0,1]
      -> RGB
      -> 224x224
      -> ImageNet normalization
    """
    x = as_nchw(x).to(DEVICE, DTYPE)

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    x = F.interpolate(
        x,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=DEVICE,
        dtype=DTYPE,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=DEVICE,
        dtype=DTYPE,
    ).view(1, 3, 1, 1)

    return (x - mean) / std


@torch.no_grad()
def embed_pool(model, images, label):
    out = []
    total = len(images)
    nb = (total + EMBED_BATCH - 1) // EMBED_BATCH

    for j, start in enumerate(
        range(0, total, EMBED_BATCH),
        start=1,
    ):
        batch = images[start:start + EMBED_BATCH]
        batch = prepare_dino_batch(batch)

        z = model(batch)
        out.append(z.cpu())

        if j % 25 == 0 or j == nb:
            print(f"{label}: embedding batch {j}/{nb}", flush=True)

    z = torch.cat(out, dim=0)

    if z.shape[1] != DINO_DIM:
        raise RuntimeError(f"Unexpected DINO embedding shape {z.shape}")

    return z


def load_data_and_embeddings():
    print("Loading raw MNIST...")
    mnist = datasets.MNIST(
        "data/mnist",
        train=True,
        download=False,
    )

    real = mnist.data.unsqueeze(1).float() / 255.0

    print("real shape:", tuple(real.shape))
    print(
        "real range:",
        float(real.min()),
        float(real.max()),
    )

    fake_path = Path(
        "data/Fake_MNIST_data_EP100_N10000.pckl"
    )

    if not fake_path.exists():
        raise FileNotFoundError(fake_path)

    with fake_path.open("rb") as f:
        obj = pickle.load(f)

    fake_raw = obj[0]
    fake, fake_rule = convert_fake_to_unit_interval(fake_raw)

    print("fake shape:", tuple(fake.shape))

    if CACHE.exists():
        print("Loading frozen embedding cache:", CACHE)
        cached = torch.load(CACHE, map_location="cpu")

        real_emb = cached["real"]
        fake_emb = cached["fake"]

        print("cached real:", tuple(real_emb.shape))
        print("cached fake:", tuple(fake_emb.shape))

        return real_emb, fake_emb, fake_rule

    print()
    print("Loading pinned DINOv2...")
    model = torch.hub.load(
        DINO_REPO,
        "dinov2_vits14",
        trust_repo=True,
    ).to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False

    print("Embedding full real/fake pools once...")

    t0 = time.time()

    real_emb = embed_pool(model, real, "real")
    fake_emb = embed_pool(model, fake, "fake")

    print(
        f"embedding time: {(time.time() - t0) / 60:.2f} min"
    )

    CACHE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "real": real_emb,
            "fake": fake_emb,
            "dinov2_source": DINO_REPO,
            "preprocessing": (
                "raw [0,1] -> RGB -> resize 224 -> "
                "ImageNet mean/std"
            ),
            "fake_rule": fake_rule,
        },
        CACHE,
    )

    print("saved local embedding cache:", CACHE)

    return real_emb, fake_emb, fake_rule


def sample_alternative(real, fake, m, trial):
    """
    Disjoint train/test IDs within each source.
    Exploratory seeds are intentionally separate from future
    paired frozen-vs-LoRA evaluation seeds.
    """
    rng = np.random.default_rng(
        230_000 + 10_000 * m + trial
    )

    rp = rng.choice(len(real), size=2 * m, replace=False)
    fq = rng.choice(len(fake), size=2 * m, replace=False)

    return (
        real[rp[:m]],
        real[rp[m:]],
        fake[fq[:m]],
        fake[fq[m:]],
        rp.tolist(),
        fq.tolist(),
    )


def sample_same_source(source, m, trial, source_code):
    """
    Four completely distinct sets:
      P_train, P_test, Q_train, Q_test

    All observations come from the same source distribution.
    """
    rng = np.random.default_rng(
        510_000
        + source_code * 100_000
        + 10_000 * m
        + trial
    )

    ids = rng.choice(
        len(source),
        size=4 * m,
        replace=False,
    )

    p_tr = ids[:m]
    p_te = ids[m:2*m]
    q_tr = ids[2*m:3*m]
    q_te = ids[3*m:4*m]

    assert len(set(ids.tolist())) == 4 * m

    return (
        source[p_tr],
        source[p_te],
        source[q_tr],
        source[q_te],
        ids.tolist(),
    )


def statistic(values, group_a):
    """
    Absolute difference in group means.

    For hard C2ST values = predicted class.
    For soft C2ST values = P(class 0).

    This matches the statistic form used by the Tian utilities.
    """
    group_a = np.asarray(group_a, dtype=np.int64)
    n = len(values)

    mask = np.ones(n, dtype=bool)
    mask[group_a] = False

    return abs(
        float(np.mean(values[group_a]))
        - float(np.mean(values[mask]))
    )


def permutation_pvalue(values, m, seed):
    """
    m=5:
        enumerate all C(10,5)=252 group allocations exactly.

    m>5:
        B=999 Monte Carlo permutations with
        p = (1 + #{T_perm >= T_obs}) / (B + 1).

    Ties are explicitly counted.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    observed_group = np.arange(m)
    obs = statistic(values, observed_group)

    tol = 1e-12

    if m == 5 and n == 10:
        stats = []

        for comb in itertools.combinations(range(n), m):
            stats.append(
                statistic(values, np.asarray(comb))
            )

        stats = np.asarray(stats)

        p = float(
            np.mean(stats >= obs - tol)
        )

        return p, obs, "exact-252", len(stats)

    rng = np.random.default_rng(seed)
    ge = 0

    for _ in range(MC_PERMUTATIONS):
        group = rng.choice(n, size=m, replace=False)
        t = statistic(values, group)

        if t >= obs - tol:
            ge += 1

    p = (1.0 + ge) / (MC_PERMUTATIONS + 1.0)

    return p, obs, "MC-plus-one", MC_PERMUTATIONS


def fit_and_test(p_tr, p_te, q_tr, q_te, seed):
    Xtr = torch.cat([p_tr, q_tr], dim=0).to(
        DEVICE, DTYPE
    )
    Xte = torch.cat([p_te, q_te], dim=0).to(
        DEVICE, DTYPE
    )

    ytr = torch.cat([
        torch.zeros(len(p_tr)),
        torch.ones(len(q_tr)),
    ]).to(DEVICE).long()

    yte = torch.cat([
        torch.zeros(len(p_te)),
        torch.ones(len(q_te)),
    ]).to(DEVICE).long()

    model, w, b = C2ST_NN_fit(
        Xtr,
        ytr,
        DINO_DIM,
        HEAD_H,
        HEAD_OUT,
        N_epoch=HEAD_EPOCHS,
        batch_size=HEAD_BATCH,
        device=DEVICE,
        dtype=DTYPE,
        model=None,
        lr_c2st=HEAD_LR,
    )

    model.eval()

    with torch.no_grad():
        logits_tr = model(Xtr).mm(w) + b
        logits_te = model(Xte).mm(w) + b

        prob_tr = torch.softmax(logits_tr, dim=1)
        prob_te = torch.softmax(logits_te, dim=1)

        pred_tr = prob_tr.argmax(dim=1)
        pred_te = prob_te.argmax(dim=1)

        train_acc = float(
            (pred_tr == ytr).float().mean().item()
        )
        test_acc = float(
            (pred_te == yte).float().mean().item()
        )

    hard = pred_te.detach().cpu().numpy().astype(float)
    soft = prob_te[:, 0].detach().cpu().numpy()

    m = len(p_te)

    p_s, stat_s, perm_s, b_s = permutation_pvalue(
        hard,
        m,
        seed=seed + 17,
    )

    p_l, stat_l, perm_l, b_l = permutation_pvalue(
        soft,
        m,
        seed=seed + 31,
    )

    return {
        "train_accuracy": train_acc,
        "heldout_accuracy": test_acc,
        "p_S": p_s,
        "p_L": p_l,
        "stat_S": stat_s,
        "stat_L": stat_l,
        "reject_S": int(p_s <= ALPHA),
        "reject_L": int(p_l <= ALPHA),
        "perm_S": perm_s,
        "perm_L": perm_l,
        "B_S": b_s,
        "B_L": b_l,
    }


def summarize(rows):
    if not rows:
        return {}

    return {
        "n_trials": len(rows),
        "power_or_type1_S": float(
            np.mean([r["reject_S"] for r in rows])
        ),
        "power_or_type1_L": float(
            np.mean([r["reject_L"] for r in rows])
        ),
        "mean_train_accuracy": float(
            np.mean([r["train_accuracy"] for r in rows])
        ),
        "mean_heldout_accuracy": float(
            np.mean([r["heldout_accuracy"] for r in rows])
        ),
    }


def save(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w") as f:
        json.dump(payload, f, indent=2)

    print("saved:", OUTPUT, flush=True)


def main():
    t0 = time.time()

    print("=" * 68)
    print("FROZEN DINOv2 SMALL-m EXPLORATORY PILOT v2")
    print("=" * 68)
    print("device:", DEVICE)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("git:", git_commit())
    print("m grid:", M_LIST)
    print("exploratory trials:", N_TRIALS)
    print("null m:", NULL_M)
    print("null trials/source:", NULL_TRIALS)
    print("head epochs:", HEAD_EPOCHS)
    print()

    real, fake, fake_rule = load_data_and_embeddings()

    payload = {
        "experiment":
            "dinov2_frozen_small_m_exploratory_v2",
        "classification":
            "exploratory sample-size selection; NOT final LoRA comparison",
        "git_commit": git_commit(),
        "dinov2_source": DINO_REPO,
        "dinov2_dim": DINO_DIM,
        "preprocessing":
            "raw [0,1] grayscale -> RGB -> 224x224 -> ImageNet normalization",
        "fake_input_conversion": fake_rule,
        "m_definition":
            "m observations per source in train and m per source in held-out test",
        "m_list": M_LIST,
        "outer_trials_per_m": N_TRIALS,
        "head_epochs": HEAD_EPOCHS,
        "head_lr": HEAD_LR,
        "head_architecture":
            "Tian ModelLatentF 384->30 hidden family + learned final 2-class weights",
        "alpha": ALPHA,
        "mc_permutations": MC_PERMUTATIONS,
        "m5_test":
            "exact enumeration of all C(10,5)=252 allocations",
        "mc_test":
            "plus-one Monte Carlo p-value with ties counted",
        "null_protocol":
            "same-source, all four train/test groups use distinct image IDs",
        "null": {},
        "alternative": {},
        "runtime_seconds": None,
    }

    # --------------------------------------------------------
    # Null check FIRST
    # --------------------------------------------------------
    print()
    print("=" * 68)
    print("NULL CHECK FIRST")
    print("=" * 68)

    for source_name, source, code in [
        ("real_vs_real", real, 1),
        ("fake_vs_fake", fake, 2),
    ]:
        rows = []

        for trial in range(NULL_TRIALS):
            print(
                f"NULL {source_name} "
                f"trial {trial+1}/{NULL_TRIALS}",
                flush=True,
            )

            p_tr, p_te, q_tr, q_te, ids = (
                sample_same_source(
                    source,
                    NULL_M,
                    trial,
                    code,
                )
            )

            result = fit_and_test(
                p_tr,
                p_te,
                q_tr,
                q_te,
                seed=700_000 + code * 10_000 + trial,
            )

            result["trial"] = trial
            result["all_ids_distinct"] = (
                len(set(ids)) == len(ids)
            )

            rows.append(result)

            print(result, flush=True)

        payload["null"][source_name] = {
            "m": NULL_M,
            "trials": rows,
            "summary": summarize(rows),
        }

        print(
            source_name,
            payload["null"][source_name]["summary"],
            flush=True,
        )

        save(payload)

    # --------------------------------------------------------
    # Alternative exploratory sweep
    # --------------------------------------------------------
    print()
    print("=" * 68)
    print("REAL vs FAKE EXPLORATORY SMALL-m SWEEP")
    print("=" * 68)

    for m in M_LIST:
        rows = []

        for trial in range(N_TRIALS):
            print(
                f"ALT m={m} trial {trial+1}/{N_TRIALS}",
                flush=True,
            )

            (
                p_tr,
                p_te,
                q_tr,
                q_te,
                real_ids,
                fake_ids,
            ) = sample_alternative(
                real,
                fake,
                m,
                trial,
            )

            # Train/test are disjoint within each source.
            assert set(real_ids[:m]).isdisjoint(
                set(real_ids[m:])
            )
            assert set(fake_ids[:m]).isdisjoint(
                set(fake_ids[m:])
            )

            result = fit_and_test(
                p_tr,
                p_te,
                q_tr,
                q_te,
                seed=900_000 + 10_000 * m + trial,
            )

            result["trial"] = trial
            result["m"] = m
            result["real_train_test_disjoint"] = True
            result["fake_train_test_disjoint"] = True

            rows.append(result)

            print(result, flush=True)

        payload["alternative"][str(m)] = {
            "trials": rows,
            "summary": summarize(rows),
        }

        payload["runtime_seconds"] = (
            time.time() - t0
        )

        print()
        print(
            f"SUMMARY m={m}:",
            payload["alternative"][str(m)]["summary"],
            flush=True,
        )

        save(payload)

    payload["runtime_seconds"] = time.time() - t0
    save(payload)

    print()
    print("=" * 68)
    print("FINAL SUMMARY")
    print("=" * 68)

    print("NULL:")
    for k, v in payload["null"].items():
        print(k, v["summary"])

    print()
    print("ALTERNATIVE:")
    for m in M_LIST:
        print(
            "m =", m,
            payload["alternative"][str(m)]["summary"],
        )

    print()
    print(
        f"runtime = {payload['runtime_seconds']/60:.2f} minutes"
    )
    print("DONE")


if __name__ == "__main__":
    main()

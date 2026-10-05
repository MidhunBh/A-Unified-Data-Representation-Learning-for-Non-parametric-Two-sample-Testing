import sys
import json
import time
import math
import pickle
import itertools
import subprocess
from pathlib import Path

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets


DINO_REPO = (
    "facebookresearch/dinov2:"
    "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

DTYPE = torch.float32

M = 5

# Precommitted engineering-smoke configuration.
LORA_RANK = 4
LORA_ALPHA = 4.0
LORA_DROPOUT = 0.0

ADAPTER_LR = 1e-4
HEAD_LR = 2e-3
EPOCHS = 300

H = 30
DINO_DIM = 384
ALPHA_TEST = 0.05

RESULT_PATH = Path("result/dinov2_lora_mnist_smoke.json")
STATE_PATH = Path("/tmp/dinov2_lora_smoke_trainable.pt")


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
        raise ValueError(f"Expected NCHW-like input, got {x.shape}")

    if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
        x = x.permute(0, 3, 1, 2)

    if x.shape[1] not in (1, 3):
        raise ValueError(f"Cannot identify channels: {x.shape}")

    return x


def fake_to_unit(x):
    x = as_nchw(x).float()

    lo = float(x.min())
    hi = float(x.max())

    if lo >= -1.05 and hi <= 1.05 and lo < -0.05:
        rule = "[-1,1] -> [0,1]"
        x = (x + 1.0) / 2.0
    elif lo >= -1e-6 and hi <= 1.05:
        rule = "already [0,1]"
    elif lo >= -1e-6 and hi <= 255.5:
        rule = "[0,255] -> [0,1]"
        x = x / 255.0
    else:
        raise ValueError(
            f"Unexpected fake-MNIST range [{lo}, {hi}]"
        )

    return x.clamp(0, 1), rule


def preprocess(x):
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


class LoRALinear(nn.Module):
    """
    W x + scaling * B(A(dropout(x)))

    Original Linear is retained and frozen.

    A is initialized nonzero.
    B is initialized exactly zero.

    Therefore initial LoRA update is zero, while gradients can flow
    immediately into B. Both factors are NOT initialized to zero.
    """
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError(type(base))

        self.base = base

        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Parameter(
            torch.empty(rank, base.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank)
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

    def forward(self, x):
        base = self.base(x)

        update = F.linear(
            F.linear(self.dropout(x), self.lora_A),
            self.lora_B,
        )

        return base + self.scaling * update


def inject_lora(model):
    targets = []

    # Architecture was audited before defining this rule.
    for i, block in enumerate(model.blocks):
        old = block.attn.qkv

        if not isinstance(old, nn.Linear):
            raise TypeError(
                f"blocks.{i}.attn.qkv is {type(old)}"
            )

        block.attn.qkv = LoRALinear(
            old,
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            dropout=LORA_DROPOUT,
        )

        targets.append(f"blocks.{i}.attn.qkv")

    if len(targets) != 12:
        raise RuntimeError(
            f"Expected 12 qkv targets, found {len(targets)}"
        )

    return targets


class C2STHead(nn.Module):
    """
    Same nonlinear readout family as Tian ModelLatentF:
    384 -> 30 -> 30 -> 30 -> 30, Softplus between layers,
    followed by a separate 2-class linear readout.
    """
    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(DINO_DIM, H),
            nn.Softplus(),
            nn.Linear(H, H),
            nn.Softplus(),
            nn.Linear(H, H),
            nn.Softplus(),
            nn.Linear(H, H),
        )

        self.classifier = nn.Linear(H, 2)

        # Tian C2ST uses random-normal final w,b.
        with torch.no_grad():
            self.classifier.weight.copy_(
                torch.randn_like(self.classifier.weight)
            )
            self.classifier.bias.copy_(
                torch.randn_like(self.classifier.bias)
            )

    def forward(self, z):
        return self.classifier(self.feature(z))


class DinoLoRAC2ST(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        # Reproduce fixed head initialization rule.
        torch.manual_seed(1102)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1102)

        self.head = C2STHead()

    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)


def load_backbone():
    model = torch.hub.load(
        DINO_REPO,
        "dinov2_vits14",
        trust_repo=True,
    ).to(DEVICE)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


def load_raw_data():
    mnist = datasets.MNIST(
        "data/mnist",
        train=True,
        download=False,
    )

    real = mnist.data.unsqueeze(1).float() / 255.0

    with open(
        "data/Fake_MNIST_data_EP100_N10000.pckl",
        "rb",
    ) as f:
        fake_raw = pickle.load(f)[0]

    fake, fake_rule = fake_to_unit(fake_raw)

    return real, fake, fake_rule


def alt_sample(real, fake, seed):
    rng = np.random.default_rng(seed)

    rid = rng.choice(
        len(real),
        size=2*M,
        replace=False,
    )

    fid = rng.choice(
        len(fake),
        size=2*M,
        replace=False,
    )

    return (
        real[rid[:M]],
        real[rid[M:]],
        fake[fid[:M]],
        fake[fid[M:]],
        rid.tolist(),
        fid.tolist(),
    )


def null_sample(source, seed):
    rng = np.random.default_rng(seed)

    ids = rng.choice(
        len(source),
        size=4*M,
        replace=False,
    )

    assert len(set(ids.tolist())) == 4*M

    return (
        source[ids[:M]],
        source[ids[M:2*M]],
        source[ids[2*M:3*M]],
        source[ids[3*M:]],
        ids.tolist(),
    )


def stat(values, group):
    group = np.asarray(group)
    mask = np.ones(len(values), dtype=bool)
    mask[group] = False

    return abs(
        float(np.mean(values[group]))
        - float(np.mean(values[mask]))
    )


def exact_pvalue(values):
    values = np.asarray(values, dtype=float)
    obs_group = np.arange(M)
    obs = stat(values, obs_group)

    vals = []

    for c in itertools.combinations(range(2*M), M):
        vals.append(
            stat(values, np.asarray(c))
        )

    vals = np.asarray(vals)

    # Exact conditional permutation p-value.
    p = float(
        np.mean(vals >= obs - 1e-12)
    )

    return p, obs, len(vals)


def trainable_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if (
            "lora_A" in k
            or "lora_B" in k
            or k.startswith("head.")
        )
    }


def snapshot_base(model):
    """
    Snapshot all original DINO parameters, excluding LoRA matrices.
    """
    snap = {}

    for name, p in model.backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue

        snap[name] = p.detach().cpu().clone()

    return snap


def base_unchanged(model, snap):
    bad = []

    for name, p in model.backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue

        if not torch.equal(
            p.detach().cpu(),
            snap[name],
        ):
            bad.append(name)

    return bad


def adapter_snapshot(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if "lora_A" in n or "lora_B" in n
    }


def adapter_change(model, before):
    out = {}

    for n, p in model.named_parameters():
        if n in before:
            out[n] = float(
                torch.norm(
                    p.detach().cpu() - before[n]
                ).item()
            )

    return out


def build_lora_model(check_initial_batch=None):
    backbone = load_backbone()

    reference = None

    if check_initial_batch is not None:
        with torch.no_grad():
            reference = backbone(
                check_initial_batch
            ).detach().clone()

    targets = inject_lora(backbone)

    if check_initial_batch is not None:
        with torch.no_grad():
            adapted = backbone(
                check_initial_batch
            )

        max_initial_diff = float(
            (reference - adapted).abs().max().item()
        )
    else:
        max_initial_diff = None

    model = DinoLoRAC2ST(backbone).to(DEVICE)

    return model, targets, max_initial_diff


def fit_one(p_tr, p_te, q_tr, q_te, label):
    # Real MNIST is 28x28 while the retained fake-MNIST pool is 32x32.
    # Preprocess each source separately to the common DINO 224x224
    # representation before concatenating them.
    p_tr = preprocess(p_tr)
    q_tr = preprocess(q_tr)
    p_te = preprocess(p_te)
    q_te = preprocess(q_te)

    xtr = torch.cat([p_tr, q_tr], dim=0)
    xte = torch.cat([p_te, q_te], dim=0)

    ytr = torch.cat([
        torch.zeros(M),
        torch.ones(M),
    ]).long().to(DEVICE)

    yte = torch.cat([
        torch.zeros(M),
        torch.ones(M),
    ]).long().to(DEVICE)

    print()
    print(f"===== BUILD {label} MODEL =====")

    model, targets, initial_diff = build_lora_model(
        check_initial_batch=xtr[:2]
    )

    print("LoRA targets:", len(targets))
    print("initial frozen-vs-LoRA max abs diff:", initial_diff)

    if initial_diff > 1e-6:
        raise RuntimeError(
            f"Initial LoRA is not zero-update: {initial_diff}"
        )

    trainable = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad
    ]

    lora_params = [
        p for n, p in trainable
        if "lora_A" in n or "lora_B" in n
    ]

    head_params = [
        p for n, p in trainable
        if n.startswith("head.")
    ]

    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)

    print("trainable LoRA parameters:", n_lora)
    print("trainable head parameters:", n_head)
    print("total trainable:", n_lora + n_head)

    expected_lora = 12 * LORA_RANK * (384 + 1152)

    if n_lora != expected_lora:
        raise RuntimeError(
            f"Unexpected LoRA parameter count "
            f"{n_lora} != {expected_lora}"
        )

    base_before = snapshot_base(model)
    adapter_before = adapter_snapshot(model)

    optimizer = torch.optim.Adam([
        {
            "params": lora_params,
            "lr": ADAPTER_LR,
        },
        {
            "params": head_params,
            "lr": HEAD_LR,
        },
    ])

    criterion = nn.CrossEntropyLoss()

    model.train()

    t0 = time.time()
    first_grad_norm = None
    losses = []

    for epoch in range(EPOCHS):
        logits = model(xtr)
        loss = criterion(logits, ytr)

        optimizer.zero_grad()
        loss.backward()

        if epoch == 0:
            grad_sq = 0.0

            for p in lora_params:
                if p.grad is not None:
                    grad_sq += float(
                        p.grad.detach().pow(2).sum().item()
                    )

            first_grad_norm = math.sqrt(grad_sq)

            print(
                "first-step LoRA gradient norm:",
                first_grad_norm,
            )

            if first_grad_norm <= 0:
                raise RuntimeError(
                    "No gradient reached LoRA parameters"
                )

        optimizer.step()

        losses.append(float(loss.item()))

        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
            or epoch + 1 == EPOCHS
        ):
            print(
                f"{label} epoch {epoch+1:3d}/{EPOCHS}: "
                f"loss={loss.item():.6f}",
                flush=True,
            )

    train_seconds = time.time() - t0

    changes = adapter_change(model, adapter_before)

    changed = {
        k: v
        for k, v in changes.items()
        if v > 0
    }

    print(
        "changed adapter tensors:",
        len(changed),
        "/",
        len(changes),
    )

    if len(changed) == 0:
        raise RuntimeError(
            "LoRA parameters did not update"
        )

    bad_base = base_unchanged(model, base_before)

    print(
        "changed original backbone tensors:",
        len(bad_base),
    )

    if bad_base:
        raise RuntimeError(
            "Frozen DINO parameters changed: "
            + ", ".join(bad_base[:5])
        )

    model.eval()

    with torch.no_grad():
        logits_tr = model(xtr)
        logits_te = model(xte)

        prob_te = torch.softmax(
            logits_te,
            dim=1,
        )

        pred_tr = logits_tr.argmax(dim=1)
        pred_te = logits_te.argmax(dim=1)

        train_acc = float(
            (pred_tr == ytr).float().mean().item()
        )

        test_acc = float(
            (pred_te == yte).float().mean().item()
        )

    hard = pred_te.cpu().numpy().astype(float)
    soft = prob_te[:, 0].cpu().numpy()

    p_s, stat_s, n_perm_s = exact_pvalue(hard)
    p_l, stat_l, n_perm_l = exact_pvalue(soft)

    # --------------------------------------------------------
    # SAVE / RELOAD CHECK
    # --------------------------------------------------------
    state = trainable_state(model)
    torch.save(state, STATE_PATH)

    fresh, _, fresh_initial_diff = build_lora_model(
        check_initial_batch=xtr[:2]
    )

    fresh_sd = fresh.state_dict()

    missing = [
        k for k in state
        if k not in fresh_sd
    ]

    if missing:
        raise RuntimeError(
            f"State keys absent in fresh model: {missing[:5]}"
        )

    for k, v in state.items():
        fresh_sd[k] = v.to(
            fresh_sd[k].device,
            fresh_sd[k].dtype,
        )

    fresh.load_state_dict(fresh_sd)
    fresh.eval()

    with torch.no_grad():
        reload_logits = fresh(xte)

    reload_diff = float(
        (reload_logits - logits_te).abs().max().item()
    )

    print("save/reload max logits diff:", reload_diff)

    if reload_diff > 1e-5:
        raise RuntimeError(
            f"Save/reload mismatch: {reload_diff}"
        )

    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label": label,
        "epochs": EPOCHS,
        "adapter_lr": ADAPTER_LR,
        "head_lr": HEAD_LR,
        "train_seconds": train_seconds,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "first_lora_grad_norm": first_grad_norm,
        "initial_zero_update_max_abs_diff": initial_diff,
        "fresh_zero_update_max_abs_diff": fresh_initial_diff,
        "n_lora_params": n_lora,
        "n_head_params": n_head,
        "adapter_tensors_total": len(changes),
        "adapter_tensors_changed": len(changed),
        "max_adapter_tensor_change_norm": max(changes.values()),
        "original_backbone_tensors_changed": len(bad_base),
        "train_accuracy": train_acc,
        "heldout_accuracy": test_acc,
        "p_S": p_s,
        "p_L": p_l,
        "stat_S": stat_s,
        "stat_L": stat_l,
        "reject_S": int(p_s <= ALPHA_TEST),
        "reject_L": int(p_l <= ALPHA_TEST),
        "permutation_allocations_S": n_perm_s,
        "permutation_allocations_L": n_perm_l,
        "save_reload_max_logits_diff": reload_diff,
    }


def main():
    total_t0 = time.time()

    print("=" * 70)
    print("DINOv2 LoRA MNIST ENGINEERING / TIMING SMOKE")
    print("=" * 70)
    print("device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("git:", git_commit())
    print("m:", M)
    print("rank:", LORA_RANK)
    print("alpha:", LORA_ALPHA)
    print("dropout:", LORA_DROPOUT)
    print("epochs:", EPOCHS)
    print("adapter lr:", ADAPTER_LR)
    print("head lr:", HEAD_LR)

    real, fake, fake_rule = load_raw_data()

    print()
    print("fake conversion:", fake_rule)

    # Fresh seed namespace, separate from exploratory frozen sweep.
    alt = alt_sample(
        real,
        fake,
        seed=1_500_001,
    )

    p_tr, p_te, q_tr, q_te, rid, fid = alt

    assert set(rid[:M]).isdisjoint(set(rid[M:]))
    assert set(fid[:M]).isdisjoint(set(fid[M:]))

    alternative = fit_one(
        p_tr,
        p_te,
        q_tr,
        q_te,
        label="alternative-real-vs-fake",
    )

    # One same-source correctness smoke.
    null = null_sample(
        real,
        seed=1_600_001,
    )

    n_p_tr, n_p_te, n_q_tr, n_q_te, nids = null

    assert len(set(nids)) == 4*M

    null_result = fit_one(
        n_p_tr,
        n_p_te,
        n_q_tr,
        n_q_te,
        label="null-real-vs-real",
    )

    payload = {
        "experiment":
            "dinov2_lora_mnist_engineering_timing_smoke",
        "classification":
            "engineering correctness and runtime smoke; NOT efficacy evidence",
        "git_commit": git_commit(),
        "dinov2_source": DINO_REPO,
        "dataset": "MNIST vs generated MNIST",
        "m": M,
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "scaling": LORA_ALPHA / LORA_RANK,
            "dropout": LORA_DROPOUT,
            "target_rule":
                "all 12 DINOv2 ViT-S/14 fused attention qkv projections",
            "adapter_lr": ADAPTER_LR,
            "epochs": EPOCHS,
        },
        "head": {
            "architecture":
                "Tian-style nonlinear C2ST head 384->30->30->30->30 plus 2-class readout",
            "lr": HEAD_LR,
        },
        "preprocessing":
            "raw [0,1] grayscale -> RGB -> resize 224 -> ImageNet normalization",
        "fake_input_conversion": fake_rule,
        "permutation_test":
            "exact enumeration of all C(10,5)=252 held-out allocations; ties counted",
        "alternative": alternative,
        "null": null_result,
        "total_runtime_seconds": time.time() - total_t0,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)

    print()
    print("=" * 70)
    print("SMOKE COMPLETE")
    print("=" * 70)

    print(
        "alternative train/test acc:",
        alternative["train_accuracy"],
        alternative["heldout_accuracy"],
    )
    print(
        "alternative p S/L:",
        alternative["p_S"],
        alternative["p_L"],
    )
    print(
        "null train/test acc:",
        null_result["train_accuracy"],
        null_result["heldout_accuracy"],
    )
    print(
        "null p S/L:",
        null_result["p_S"],
        null_result["p_L"],
    )

    print(
        "alternative LoRA training minutes:",
        alternative["train_seconds"] / 60,
    )
    print(
        "null LoRA training minutes:",
        null_result["train_seconds"] / 60,
    )
    print(
        "total smoke minutes:",
        payload["total_runtime_seconds"] / 60,
    )

    print("result:", RESULT_PATH)
    print("DONE")


if __name__ == "__main__":
    main()

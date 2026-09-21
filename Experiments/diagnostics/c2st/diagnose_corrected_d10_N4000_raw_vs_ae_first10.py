import json
import os
import numpy as np
import torch

from utils import (
    MatConvert,
    sample_hdgm_semi_t2,
    C2ST_NN_fit,
    TST_C2ST,
    TST_LCE,
    train_autoencoder,
    ExtendedModel,
)

# ============================================================
# CORRECTED d=10 / N=4000 COMPARATOR DIAGNOSTIC
#
# d=10
# internal n=100
# paper N = 4*n*d = 4000
#
# Compare:
#   1. vanilla C2ST
#   2. recovered historical AE -> C2ST
#
# Same dataset seeds for both methods.
# ============================================================

MASTER_SEED = 1102

np.random.seed(MASTER_SEED)
torch.manual_seed(MASTER_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(MASTER_SEED)
    torch.cuda.manual_seed_all(MASTER_SEED)
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dtype = torch.float
alpha = 0.05

D = 10
H = 30
C2ST_FEATURE_DIM = 30

INTERNAL_N = 100
PAPER_N = 4 * INTERNAL_N * D

N_OUTER = 10
N_PER = 100
C2ST_EPOCHS = 1000

# Recovered historical settings
RAW_BATCH = 2048
RAW_C2ST_LR = 0.005

AE_EPOCHS = 2000
AE_BATCH = 512
AE_LR = 0.002

AE_C2ST_BATCH = 1024
AE_C2ST_LR = 0.002

os.makedirs("result", exist_ok=True)

records = []

print()
print("=" * 60)
print("CORRECTED HDGM d=10 COMPARATOR PILOT")
print("=" * 60)
print(f"d           = {D}")
print(f"internal n  = {INTERNAL_N}")
print(f"paper N     = {PAPER_N}")
print(f"outer trials= {N_OUTER}")
print()

assert PAPER_N == 4000


def test_once(S_test, n1, model, w, b):
    h_s, _, _ = TST_C2ST(
        S_test,
        n1,
        N_PER,
        alpha,
        model,
        w,
        b,
    )

    h_l, _, _ = TST_LCE(
        S_test,
        n1,
        N_PER,
        alpha,
        model,
        w,
        b,
    )

    return int(h_s), int(h_l)


for kk in range(N_OUTER):

    print(f"----- dataset seed {kk} -----", flush=True)

    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(
        INTERNAL_N,
        INTERNAL_N,
        d=D,
        kk=kk,
        level="hard",
    )

    if kk == 0:
        print(
            "split sizes:",
            len(s1_tr),
            len(s1_te),
            len(s2_tr),
            len(s2_te),
        )

    # --------------------------------------------------------
    # Common supervised train/test tensors
    # --------------------------------------------------------

    S_train_np = np.concatenate(
        (s1_tr, s2_tr),
        axis=0,
    )

    S_test_np = np.concatenate(
        (s1_te, s2_te),
        axis=0,
    )

    S_train = MatConvert(
        S_train_np,
        device,
        dtype,
    )

    S_test = MatConvert(
        S_test_np,
        device,
        dtype,
    )

    y = torch.cat(
        [
            torch.zeros(len(s1_tr)),
            torch.ones(len(s2_tr)),
        ]
    ).to(device, dtype).long()

    n1_test = len(s1_te)

    # ========================================================
    # 1. VANILLA C2ST
    # ========================================================

    raw_model, raw_w, raw_b = C2ST_NN_fit(
        S_train,
        y,
        D,
        H,
        C2ST_FEATURE_DIM,
        C2ST_EPOCHS,
        RAW_BATCH,
        device,
        dtype,
        model=None,
        lr_c2st=RAW_C2ST_LR,
    )

    raw_s, raw_l = test_once(
        S_test,
        n1_test,
        raw_model,
        raw_w,
        raw_b,
    )

    # ========================================================
    # 2. HISTORICAL AE -> C2ST
    # ========================================================

    S_encoder_np = np.concatenate(
        (
            s1_tr,
            s1_te,
            s2_tr,
            s2_te,
        ),
        axis=0,
    )

    S_encoder = MatConvert(
        S_encoder_np,
        device,
        dtype,
    )

    encoder = train_autoencoder(
        S_encoder,
        AE_EPOCHS,
        D,
        H,
        C2ST_FEATURE_DIM,
        AE_BATCH,
        device,
        dtype,
        lr=AE_LR,
    )

    # Match recovered historical d=10 AE implementation.
    for p in encoder.parameters():
        p.requires_grad = False

    ae_phase2 = ExtendedModel(
        encoder,
        H,
        C2ST_FEATURE_DIM,
    )

    ae_model, ae_w, ae_b = C2ST_NN_fit(
        S_train,
        y,
        D,
        H,
        C2ST_FEATURE_DIM,
        C2ST_EPOCHS,
        AE_C2ST_BATCH,
        device,
        dtype,
        model=ae_phase2,
        lr_c2st=AE_C2ST_LR,
    )

    ae_s, ae_l = test_once(
        S_test,
        n1_test,
        ae_model,
        ae_w,
        ae_b,
    )

    records.append(
        {
            "seed": kk,
            "raw_s": raw_s,
            "raw_l": raw_l,
            "ae_s": ae_s,
            "ae_l": ae_l,
        }
    )

    print(
        f"RAW  S/L = {raw_s}/{raw_l}    "
        f"AE  S/L = {ae_s}/{ae_l}",
        flush=True,
    )


raw_s_mean = float(np.mean([r["raw_s"] for r in records]))
raw_l_mean = float(np.mean([r["raw_l"] for r in records]))

ae_s_mean = float(np.mean([r["ae_s"] for r in records]))
ae_l_mean = float(np.mean([r["ae_l"] for r in records]))


print()
print("=" * 60)
print("MEAN REJECTION RATE OVER 10 OUTER DATASETS")
print("=" * 60)
print(f"Vanilla C2ST-S = {raw_s_mean:.3f}")
print(f"Vanilla C2ST-L = {raw_l_mean:.3f}")
print(f"AE RL-C2ST-S   = {ae_s_mean:.3f}")
print(f"AE RL-C2ST-L   = {ae_l_mean:.3f}")
print()


payload = {
    "diagnostic": "corrected d10 N4000 vanilla vs historical AE",
    "d": D,
    "internal_n": INTERNAL_N,
    "paper_N": PAPER_N,
    "mapping": "paper_N = 4 * internal_n * d",
    "n_outer": N_OUTER,
    "n_permutations": N_PER,
    "raw": {
        "C2ST-S": raw_s_mean,
        "C2ST-L": raw_l_mean,
    },
    "historical_AE": {
        "epochs": AE_EPOCHS,
        "C2ST-S": ae_s_mean,
        "C2ST-L": ae_l_mean,
    },
    "records": records,
}

out_path = "result/diagnose_corrected_d10_N4000_raw_vs_ae_first10.json"

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)

print("Saved:")
print(f"  {out_path}")

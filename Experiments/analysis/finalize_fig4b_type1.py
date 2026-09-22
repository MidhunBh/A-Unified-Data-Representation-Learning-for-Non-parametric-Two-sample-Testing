import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
FIGURE_DIR = ROOT / "figure"

FIGURE_DIR.mkdir(exist_ok=True)

N = [1000, 2000, 4000, 6000, 8000, 10000]
ALPHA = 0.05


def load_json(name):
    path = RESULT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ------------------------------------------------------------
# Load the final/corrected Figure 4(b) result files
# ------------------------------------------------------------

c2st = load_json("c2st_HDGM_d2_type1.json")
rl_c2st = load_json("c2st_semi_HDGM_d2_type1.json")
mmd_d = load_json("mmd_d_HDGM_d2_type1.json")

# Corrected forensic reconstruction:
# author-style RNG, only sample-size bug fixed.
mmd_fuse = load_json("mmdfuse_HDGM_type1_authorstyle_nfix.json")

# Use the same independently selected lr_mmd=5e-6 as the final
# RL-MMD-D HDGM power experiment.
rl_mmd_d = load_json("rl_mmd_d_HDGM_lrfix_d2_type1.json")


# ------------------------------------------------------------
# Normalize into seven Figure 4(b) curves
# ------------------------------------------------------------

series = {
    "C2ST": [
        float(r["C2ST-S"]) for r in c2st["result"]
    ],
    "C2ST-L": [
        float(r["C2ST-L"]) for r in c2st["result"]
    ],
    "MMD-D": [
        float(x) for x in mmd_d["result"]
    ],
    "MMD-FUSE": [
        float(r["type1"]) for r in mmd_fuse["results"]
    ],
    "RL-C2ST": [
        float(r["RL-C2ST-S"]) for r in rl_c2st["result"]
    ],
    "RL-C2ST-L": [
        float(r["RL-C2ST-L"]) for r in rl_c2st["result"]
    ],
    "RL-MMD-D": [
        float(x) for x in rl_mmd_d["result"]
    ],
}


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

assert c2st["N"] == N
assert rl_c2st["N"] == N
assert mmd_d["N"] == N
assert mmd_fuse["N_completed"] == N
assert rl_mmd_d["N"] == N

for method, values in series.items():
    assert len(values) == len(N), (
        f"{method}: expected {len(N)} values, got {len(values)}"
    )


# ------------------------------------------------------------
# Print final Figure 4(b) table
# ------------------------------------------------------------

print()
print("=" * 96)
print("TIAN ET AL. FIGURE 4(b) REPRODUCTION")
print("HDGM-S NULL: P = Q | d = 2 | nominal alpha = 0.05 | 100 trials")
print("=" * 96)

header = (
    f"{'N':>7} "
    + " ".join(f"{m:>11}" for m in series)
)

print(header)
print("-" * len(header))

for i, n in enumerate(N):
    row = f"{n:>7} " + " ".join(
        f"{series[m][i]:>11.4f}" for m in series
    )
    print(row)

print()
print("Across-grid calibration summary")
print("-" * 60)

for method, values in series.items():
    arr = np.asarray(values, dtype=float)
    mean_level = arr.mean()
    max_dev = np.abs(arr - ALPHA).max()

    print(
        f"{method:<12} "
        f"mean={mean_level:.4f}   "
        f"max |level-alpha|={max_dev:.4f}"
    )


# ------------------------------------------------------------
# Save consolidated JSON
# ------------------------------------------------------------

summary = {
    "experiment": "Tian et al. Figure 4(b) reproduction",
    "dataset": "HDGM-S",
    "null": "P = Q",
    "dimension": 2,
    "nominal_alpha": ALPHA,
    "paper_N": N,
    "internal_m": [125, 250, 500, 750, 1000, 1250],
    "sample_accounting": (
        "For the HDGM sampler, n_train=n_test=m and the two-component "
        "construction gives paper N=8*m."
    ),
    "n_trials": 100,
    "methods": series,
    "source_files": {
        "C2ST/C2ST-L": "result/c2st_HDGM_d2_type1.json",
        "MMD-D": "result/mmd_d_HDGM_d2_type1.json",
        "MMD-FUSE": "result/mmdfuse_HDGM_type1_authorstyle_nfix.json",
        "RL-C2ST/RL-C2ST-L": "result/c2st_semi_HDGM_d2_type1.json",
        "RL-MMD-D": "result/rl_mmd_d_HDGM_lrfix_d2_type1.json",
    },
    "notes": {
        "MMD-FUSE": (
            "Final forensic author-style reconstruction with the "
            "sample-size bug corrected."
        ),
        "RL-MMD-D": (
            "Uses lr_mmd=5e-6, the same independently selected learning "
            "rate used for the final HDGM power reproduction."
        ),
    },
}

json_path = RESULT_DIR / "figure4b_HDGM_d2_type1_final.json"

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)


# ------------------------------------------------------------
# Save CSV for thesis/report plotting
# ------------------------------------------------------------

csv_path = RESULT_DIR / "figure4b_HDGM_d2_type1_final.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["N"] + list(series.keys()))

    for i, n in enumerate(N):
        writer.writerow(
            [n] + [series[m][i] for m in series]
        )


# ------------------------------------------------------------
# Plot Figure 4(b)
# ------------------------------------------------------------

plt.figure(figsize=(9, 5.5))

for method, values in series.items():
    plt.plot(
        N,
        values,
        marker="o",
        linewidth=1.5,
        label=method,
    )

plt.axhline(
    ALPHA,
    linestyle="--",
    linewidth=1.5,
    label=r"nominal $\alpha=0.05$",
)

plt.xlabel("Sample size N")
plt.ylabel("Type-I error")
plt.title("HDGM-S (d=2): Figure 4(b) Type-I Error Reproduction")
plt.ylim(0.0, 0.10)
plt.xticks(N)
plt.grid(alpha=0.25)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

fig_path = FIGURE_DIR / "figure4b_HDGM_d2_type1_final.png"
plt.savefig(fig_path, dpi=250, bbox_inches="tight")
plt.close()


print()
print("=" * 96)
print("SAVED")
print(f"JSON : {json_path.relative_to(ROOT)}")
print(f"CSV  : {csv_path.relative_to(ROOT)}")
print(f"PLOT : {fig_path.relative_to(ROOT)}")
print("=" * 96)

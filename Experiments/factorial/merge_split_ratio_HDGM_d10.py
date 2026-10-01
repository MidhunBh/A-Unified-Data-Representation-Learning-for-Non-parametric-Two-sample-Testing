import json
import pickle
import subprocess
import time
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = EXPERIMENTS_DIR.parent

MAIN_JSON = (
    EXPERIMENTS_DIR
    / "result"
    / "factorial_split_ratio_HDGM_d10_N4000.json"
)

MAIN_PKL = (
    EXPERIMENTS_DIR
    / "result"
    / "factorial_split_ratio_HDGM_d10_N4000.pkl"
)

CELL_DIR = EXPERIMENTS_DIR / "result" / "split_ratio_cells"

SPLITS = [
    (300, 700),
    (400, 600),
    (500, 500),
    (600, 400),
    (700, 300),
]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_DIR),
        ).decode().strip()
    except Exception:
        return "unknown"


# Preserve the already completed historical 300/700 cell.
with open(MAIN_JSON) as f:
    prior = json.load(f)

prior_by_split = {
    r["split"]: r
    for r in prior.get("results", [])
}

results = []

for n_train, n_test in SPLITS:
    label = f"{n_train}_{n_test}"

    if label == "300_700":
        if label not in prior_by_split:
            raise RuntimeError(
                "Historical 300_700 cell missing from main JSON."
            )
        results.append(prior_by_split[label])
        continue

    cell_path = CELL_DIR / f"split_{label}.json"

    if not cell_path.exists():
        raise RuntimeError(
            f"Missing completed cell: {cell_path}"
        )

    with open(cell_path) as f:
        cell_meta = json.load(f)

    results.append(cell_meta["cell"])


meta = {
    "schema_version": 2,
    "commit": git_commit(),
    "factor": "data_splitting (train/test ratio)",
    "levels_planned": [f"{a}_{b}" for a, b in SPLITS],
    "levels_completed": [r["split"] for r in results],
    "dataset": "HDGM",
    "level": "hard",
    "d": 10,
    "total_budget": 1000,
    "N_TRAIL": 100,
    "AE_EPOCHS": 2000,
    "N_EPOCH": 1000,
    "N_PER": 100,
    "encoder_phase2": "frozen",
    "implementation_note": (
        "300_700 is retained from the original runner. "
        "The remaining cells use the corrected runner that removes "
        "a redundant outer N_TEST=100 loop. TST_C2ST/TST_LCE already "
        "perform N_PER=100 deterministic permutations internally, so "
        "the legacy outer loop duplicated the same decision and did "
        "not change each outer-trial result."
    ),
    "note": (
        "Fixed-budget sensitivity study. n_train+n_test=1000 "
        "(paper total N=4000 under this HDGM sampling convention). "
        "Tests the statistical tradeoff between fitting the learned "
        "test statistic and retaining observations for independent testing."
    ),
    "results": results,
    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(MAIN_JSON, "w") as f:
    json.dump(meta, f, indent=2)

with open(MAIN_PKL, "wb") as f:
    pickle.dump(results, f)

print("=== MERGED SPLIT-RATIO STUDY ===")
for r in results:
    print(
        f"{r['split']}: "
        f"power S/L={r['power_S']:.3f}/{r['power_L']:.3f} | "
        f"type-I S/L={r['type1_S']:.3f}/{r['type1_L']:.3f}"
    )

print()
print(f"saved {MAIN_JSON}")
print(f"saved {MAIN_PKL}")

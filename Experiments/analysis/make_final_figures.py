import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
FIGURE = ROOT / "figure"
FIGURE.mkdir(exist_ok=True)

FIG3_OUT = FIGURE / "figure3_final.png"
FIG4_OUT = FIGURE / "figure4_final.png"


def read_json(filename):
    path = RESULT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file is missing:\n  {path}"
        )

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# PAPER-LIKE COLORS
# ============================================================

# Figure 3
BAR_C2ST = "#c6dfe8"       # light blue
BAR_AE   = "#f3d2b3"       # peach
BAR_WAE  = "#ece7b3"       # pale yellow

# Figure 4
COLORS = {
    "C2ST": "#ff1f1f",       # red
    "C2ST-L": "#178517",     # green
    "MMD-D": "#ff8c00",      # orange
    "MMD-FUSE": "#00caca",   # cyan
    "RL-C2ST": "#ff00dd",    # magenta
    "RL-C2ST-L": "#1529e8",  # blue
    "RL-MMD-D": "#111111",   # black
}

MARKERS = {
    "C2ST": "o",
    "C2ST-L": "x",
    "MMD-D": "*",
    "MMD-FUSE": "^",
    "RL-C2ST": "P",
    "RL-C2ST-L": "s",
    "RL-MMD-D": "d",
}


# ============================================================
# FIGURE 3
# Reuse the finalized bar-chart construction.
# ============================================================

MNIST_M = [200, 400, 600, 800, 1000]
HDGM_N = [1000, 2000, 4000, 6000, 8000, 10000]

# MNIST
F3A_C2ST = [0.14, 0.74, 0.99, 1.00, 1.00]
F3A_AE   = [0.27, 0.92, 1.00, 1.00, 1.00]
F3A_WAE  = [0.27, 0.92, 1.00, 1.00, 0.99]

# HDGM d=2
F3B_C2ST = [0.09, 0.29, 0.81, 0.93, 0.99, 1.00]
F3B_AE   = [0.32, 0.96, 1.00, 1.00, 1.00, 1.00]
F3B_WAE  = [0.24, 0.57, 0.93, 0.98, 1.00, 0.99]

# HDGM d=10
F3C_C2ST = [0.03, 0.04, 0.18, 0.33, 0.51, 0.72]
F3C_AE   = [0.08, 0.11, 0.60, 0.80, 0.95, 1.00]
F3C_WAE  = [0.08, 0.11, 0.60, 0.80, 0.95, 1.00]


def draw_bar_panel(ax, xvals, vanilla, ae, wae, title):
    x = np.arange(len(xvals))
    width = 0.25

    ax.bar(
        x - width,
        vanilla,
        width,
        color=BAR_C2ST,
        label="C2ST",
    )

    ax.bar(
        x,
        ae,
        width,
        color=BAR_AE,
        label="RL-C2ST (AE)",
    )

    ax.bar(
        x + width,
        wae,
        width,
        color=BAR_WAE,
        label="RL-C2ST (WAE)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(xvals)

    ax.set_ylim(0, 1.05)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Test Power")
    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.20,
    )


fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.0, 4.3),
)

draw_bar_panel(
    axes[0],
    MNIST_M,
    F3A_C2ST,
    F3A_AE,
    F3A_WAE,
    "(a) Power vs. N; MNIST",
)

draw_bar_panel(
    axes[1],
    HDGM_N,
    F3B_C2ST,
    F3B_AE,
    F3B_WAE,
    "(b) Power vs. N; HDGM; d = 2",
)

draw_bar_panel(
    axes[2],
    HDGM_N,
    F3C_C2ST,
    F3C_AE,
    F3C_WAE,
    "(c) Power vs. N; HDGM; d = 10",
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.01),
    frameon=True,
)

fig.subplots_adjust(
    top=0.82,
    bottom=0.15,
    wspace=0.30,
)

fig.savefig(
    FIG3_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4(a)
# Exact final HDGM d=2 source files.
# ============================================================

c2st = read_json(
    "c2st_HDGM_d2_power.json"
)

rl_c2st = read_json(
    "c2st_semi_HDGM_d2_power.json"
)

mmd_d = read_json(
    "mmd-d_HDGM_baseline_0.00005_d2.json"
)

mmd_fuse = read_json(
    "mmdfuse_HDGM_d2_power_authorstyle.json"
)

rl_mmd_d = read_json(
    "rl_mmd_d_HDGM_lrfix_d2_power.json"
)


# C2ST
N = [
    int(r["N"])
    for r in c2st["result"]
]

power = {}

power["C2ST"] = [
    float(r["C2ST-S"])
    for r in c2st["result"]
]

power["C2ST-L"] = [
    float(r["C2ST-L"])
    for r in c2st["result"]
]


# RL-C2ST
assert rl_c2st["N"] == N

power["RL-C2ST"] = [
    float(r["RL-C2ST-S"])
    for r in rl_c2st["result"]
]

power["RL-C2ST-L"] = [
    float(r["RL-C2ST-L"])
    for r in rl_c2st["result"]
]


# Vanilla MMD-D
#
# File stores internal HDGM n:
# [125,250,500,750,1000,1250]
#
# paper N = 8*n
mmd_N = [
    8 * int(n)
    for n in mmd_d["n_list"]
]

assert mmd_N == N

power["MMD-D"] = [
    float(v)
    for v in mmd_d["result"]
]


# MMD-FUSE
fuse_N = [
    int(r["paper_N"])
    for r in mmd_fuse["results"]
]

assert fuse_N == N

power["MMD-FUSE"] = [
    float(r["power"])
    for r in mmd_fuse["results"]
]


# RL-MMD-D
assert rl_mmd_d["N"] == N

power["RL-MMD-D"] = [
    float(v)
    for v in rl_mmd_d["result"]
]


# ============================================================
# FIGURE 4(b)
# Already finalized HDGM-S d=2 Type-I result.
# ============================================================

type1 = read_json(
    "figure4b_HDGM_d2_type1_final.json"
)

assert type1["paper_N"] == N

level = {
    method: [float(v) for v in vals]
    for method, vals in type1["methods"].items()
}

alpha = float(
    type1["nominal_alpha"]
)


# ============================================================
# VALIDATE ALL SEVEN METHODS
# ============================================================

METHODS = [
    "C2ST",
    "C2ST-L",
    "MMD-D",
    "MMD-FUSE",
    "RL-C2ST",
    "RL-C2ST-L",
    "RL-MMD-D",
]

assert list(power.keys()) != []


for method in METHODS:
    if method not in power:
        raise RuntimeError(
            f"Missing Figure 4(a) power curve: {method}"
        )

    if method not in level:
        raise RuntimeError(
            f"Missing Figure 4(b) Type-I curve: {method}"
        )

    assert len(power[method]) == 6
    assert len(level[method]) == 6


# ============================================================
# PRINT EXACT DATA BEFORE PLOTTING
# ============================================================

print()
print("=" * 94)
print("FIGURE 4(a): HDGM-D d=2 POWER")
print("=" * 94)

print(
    f"{'N':>7} "
    + " ".join(
        f"{m:>11}"
        for m in METHODS
    )
)

print("-" * 94)

for i, n in enumerate(N):
    print(
        f"{n:>7} "
        + " ".join(
            f"{power[m][i]:>11.4f}"
            for m in METHODS
        )
    )


print()
print("=" * 94)
print("FIGURE 4(b): HDGM-S d=2 TYPE-I ERROR")
print("=" * 94)

print(
    f"{'N':>7} "
    + " ".join(
        f"{m:>11}"
        for m in METHODS
    )
)

print("-" * 94)

for i, n in enumerate(N):
    print(
        f"{n:>7} "
        + " ".join(
            f"{level[m][i]:>11.4f}"
            for m in METHODS
        )
    )


# ============================================================
# FIGURE 4 — paper-style two-panel composite
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12.8, 5.0),
)


# ---------- 4(a): Power ----------

ax = axes[0]

for method in METHODS:
    ax.plot(
        N,
        power[method],
        color=COLORS[method],
        marker=MARKERS[method],
        linewidth=1.5,
        markersize=5,
        label=method,
    )

ax.set_xlabel("Number of samples")
ax.set_ylabel("Average test power")
ax.set_title("(a) Power vs. N; d = 2")

ax.set_ylim(0, 1.05)
ax.set_xticks(N)

ax.grid(alpha=0.35)


# ---------- 4(b): Level ----------

ax = axes[1]

for method in METHODS:
    ax.plot(
        N,
        level[method],
        color=COLORS[method],
        marker=MARKERS[method],
        linewidth=1.5,
        markersize=5,
        label=method,
    )

# Useful reference line, subtle enough not to obscure paper-style curves.
ax.axhline(
    alpha,
    linestyle="--",
    linewidth=1.0,
    color="gray",
    alpha=0.7,
)

ax.set_xlabel("Number of samples")
ax.set_ylabel("Average Type-I error rate")
ax.set_title("(b) Level vs. N; d = 2")

ax.set_ylim(0, 0.105)
ax.set_xticks(N)

ax.grid(alpha=0.35)


# Shared legend, like the paper
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.02),
    frameon=True,
)

fig.subplots_adjust(
    top=0.82,
    bottom=0.14,
    wspace=0.28,
)

fig.savefig(
    FIG4_OUT,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 94)
print("FINAL FIGURES CREATED")
print("=" * 94)

print(
    "Figure 3:",
    FIG3_OUT.relative_to(ROOT),
)

print(
    "Figure 4:",
    FIG4_OUT.relative_to(ROOT),
)

print()
print("Figure 4(a) sources:")
print("  C2ST / C2ST-L : c2st_HDGM_d2_power.json")
print("  MMD-D          : mmd-d_HDGM_baseline_0.00005_d2.json")
print("  MMD-FUSE       : mmdfuse_HDGM_d2_power_authorstyle.json")
print("  RL-C2ST / L    : c2st_semi_HDGM_d2_power.json")
print("  RL-MMD-D       : rl_mmd_d_HDGM_lrfix_d2_power.json")

print()
print("Figure 4(b) source:")
print("  figure4b_HDGM_d2_type1_final.json")

print("=" * 94)

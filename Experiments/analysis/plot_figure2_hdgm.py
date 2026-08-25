import numpy as np
import matplotlib.pyplot as plt
from utils import sample_hdgm_semi_t2

d = 10
n = 100  # samples per cluster per split, matching the driver convention elsewhere

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

levels = ["easy", "medium", "hard"]
titles = ["(a) HDGM-Easy", "(b) HDGM-Medium", "(c) HDGM-Hard"]

for ax, level, title in zip(axes, levels, titles):
    s1_tr, s1_te, s2_tr, s2_te = sample_hdgm_semi_t2(n, n, d=d, kk=0, level=level)
    P = np.concatenate([s1_tr, s1_te], axis=0)
    Q = np.concatenate([s2_tr, s2_te], axis=0)

    ax.scatter(P[:, 0], P[:, 1], s=10, alpha=0.6, label="Distribution P")
    ax.scatter(Q[:, 0], Q[:, 1], s=10, alpha=0.6, label="Distribution Q")
    ax.set_xlabel("Dimension1")
    ax.set_ylabel("Dimension2")
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
plt.savefig("figure/figure2_hdgm_levels.png", dpi=150)
print("saved to figure/figure2_hdgm_levels.png")

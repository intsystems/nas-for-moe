"""Plot found cluster split for slides (single plot)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "toy_experiment"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA_DIR = Path(__file__).resolve().parent.parent / "code" / "toy_experiment" / "data"
OUT_DIR = Path(__file__).resolve().parent / "figures"

X = np.load(DATA_DIR / "data_X.npy")
cluster_centers = np.load(DATA_DIR / "cluster_centers.npy")
train_indices = np.load(DATA_DIR / "train_indices.npy")
train_cluster_ids = np.load(DATA_DIR / "train_cluster_ids.npy")
val_indices = np.load(DATA_DIR / "val_indices.npy")
val_cluster_ids = np.load(DATA_DIR / "val_cluster_ids.npy")

M = len(cluster_centers)

# Found assignment (EM v4 result)
found_assignment = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])

# Combine all points
all_X = np.vstack([X[train_indices], X[val_indices]])
all_cluster = np.hstack([train_cluster_ids, val_cluster_ids])

fig, ax = plt.subplots(figsize=(7, 5.5))

expert_colors = {0: "#1f77b4", 1: "#d62728"}
point_expert = found_assignment[all_cluster]

for exp in [0, 1]:
    mask = point_expert == exp
    ax.scatter(
        all_X[mask, 0], all_X[mask, 1],
        c=expert_colors[exp], alpha=0.35, s=10, rasterized=True,
    )

# Cluster centers
for m in range(M):
    color = expert_colors[found_assignment[m]]
    ax.scatter(
        cluster_centers[m, 0], cluster_centers[m, 1],
        c=color, marker="X", s=150, edgecolors="black", linewidths=1.2, zorder=10,
    )

ax.set_title("Найденное разбиение (SGEM)", fontsize=18)
ax.set_xlabel("$x_1$", fontsize=16)
ax.set_ylabel("$x_2$", fontsize=16)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.3)

legend_elements = [
    Patch(facecolor="#1f77b4", edgecolor="black",
          label=f"Эксперт 0 — rbf ({int(np.sum(found_assignment == 0))} кл.)"),
    Patch(facecolor="#d62728", edgecolor="black",
          label=f"Эксперт 1 — linear ({int(np.sum(found_assignment == 1))} кл.)"),
]
ax.legend(handles=legend_elements, fontsize=14, loc="upper left")

plt.tight_layout()
out_path = OUT_DIR / "found_split.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close(fig)

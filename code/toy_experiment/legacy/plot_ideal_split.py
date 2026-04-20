"""Plot ideal cluster split: linear vs ring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("./data")

X = np.load(DATA_DIR / "data_X.npy")
y = np.load(DATA_DIR / "data_y.npy")
cluster_centers = np.load(DATA_DIR / "cluster_centers.npy")
train_indices = np.load(DATA_DIR / "train_indices.npy")
train_cluster_ids = np.load(DATA_DIR / "train_cluster_ids.npy")
val_indices = np.load(DATA_DIR / "val_indices.npy")
val_cluster_ids = np.load(DATA_DIR / "val_cluster_ids.npy")

M = len(cluster_centers)
threshold = -2.0

# Ideal assignment: 0 = linear (x < threshold), 1 = ring (x >= threshold)
cluster_expert = np.array([0 if cluster_centers[m, 0] < threshold else 1 for m in range(M)])

# Combine all points with their cluster ids
all_X = np.vstack([X[train_indices], X[val_indices]])
all_y = np.hstack([y[train_indices], y[val_indices]])
all_cluster = np.hstack([train_cluster_ids, val_cluster_ids])
all_expert = cluster_expert[all_cluster]

# Use distinct colormaps for two experts
colors_expert0 = ['#1f77b4', '#aec7e8']  # blue shades (linear)
colors_expert1 = ['#d62728', '#ff9896']  # red shades (ring)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Data colored by expert assignment
ax = axes[0]
for exp, label, colors in [(0, 'Linear (expert 0)', colors_expert0),
                            (1, 'Ring (expert 1)', colors_expert1)]:
    mask = all_expert == exp
    for cls in [0, 1]:
        cmask = mask & (all_y == cls)
        ax.scatter(all_X[cmask, 0], all_X[cmask, 1],
                   c=colors[cls], alpha=0.4, s=15,
                   label=f'Expert {exp} ({label.split()[0]}), class {cls}')

# Draw cluster centers
for m in range(M):
    color = '#1f77b4' if cluster_expert[m] == 0 else '#d62728'
    ax.scatter(cluster_centers[m, 0], cluster_centers[m, 1],
               c=color, marker='X', s=200, edgecolors='black', linewidths=1.5, zorder=10)
    ax.annotate(str(m), (cluster_centers[m, 0], cluster_centers[m, 1]),
                fontsize=8, ha='center', va='bottom', fontweight='bold',
                xytext=(0, 8), textcoords='offset points')

ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold x={threshold}')
ax.set_title('Ideal split: Linear vs Ring clusters', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Cluster centers only, colored by expert
ax = axes[1]
for m in range(M):
    color = '#1f77b4' if cluster_expert[m] == 0 else '#d62728'
    ax.scatter(cluster_centers[m, 0], cluster_centers[m, 1],
               c=color, marker='o', s=300, edgecolors='black', linewidths=1.5)
    ax.annotate(str(m), (cluster_centers[m, 0], cluster_centers[m, 1]),
                fontsize=10, ha='center', va='center', fontweight='bold')

ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold x={threshold}')
ax.set_title('Cluster centers with ideal assignment', fontsize=14)
ax.set_xlabel('x')
ax.set_ylabel('y')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', edgecolor='black', label=f'Expert 0 (linear, {sum(cluster_expert==0)} clusters)'),
    Patch(facecolor='#d62728', edgecolor='black', label=f'Expert 1 (ring, {sum(cluster_expert==1)} clusters)'),
]
ax.legend(handles=legend_elements, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(DATA_DIR / "ideal_split.png", dpi=150)
print(f"Saved to {DATA_DIR / 'ideal_split.png'}")
plt.close(fig)

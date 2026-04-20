"""Представители кластеров MNIST с разметкой экспертов из последнего K=3 v2 прогона."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mnist_data"
RESULTS = ROOT / "runs" / "results_mnist_sgem_k3_v2.json"
OUT = ROOT / "runs" / "mnist_cluster_reps_k3_v2.png"
N_REP = 8

X = np.load(DATA / "data_X.npy")
y = np.load(DATA / "data_y.npy")
train_idx = np.load(DATA / "train_indices.npy")
train_cid = np.load(DATA / "train_cluster_ids.npy")
centers = np.load(DATA / "cluster_centers.npy")

hard_assignments = json.load(open(RESULTS))["mnist_sgem"]["hard_assignments"]

M = centers.shape[0]
X_train = X[train_idx, 0]
y_train = y[train_idx]

expert_of = {c: hard_assignments[c] for c in range(len(hard_assignments))}
expert_colors = ["#d62728", "#2ca02c", "#1f77b4"]

fig, axes = plt.subplots(M, N_REP + 1, figsize=(N_REP + 1, M),
                          gridspec_kw={"wspace": 0.05, "hspace": 0.05})

for m in range(M):
    mask = train_cid == m
    idx_m = np.where(mask)[0]
    imgs = X_train[idx_m]
    labels_m = y_train[idx_m]
    n_m = len(imgs)

    flat = imgs.reshape(n_m, -1).astype(np.float32) / 255.0
    mean = flat.mean(axis=0)
    dists = np.linalg.norm(flat - mean, axis=1)
    order = np.argsort(dists)[: N_REP]
    sel = imgs[order]
    proto = mean.reshape(28, 28)

    uniq, cnt = np.unique(labels_m, return_counts=True)
    top = uniq[np.argsort(-cnt)][:3]
    top_str = "/".join(str(int(t)) for t in top)

    expert = expert_of[m]
    color = expert_colors[expert]

    ax0 = axes[m, 0]
    ax0.imshow(proto, cmap="gray", vmin=0, vmax=1)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel(f"c{m}\nE{expert}\nn={n_m}\n{top_str}",
                    rotation=0, labelpad=28, fontsize=7, va="center", color=color)
    for spine in ax0.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2)

    for j in range(N_REP):
        ax = axes[m, j + 1]
        if j < len(sel):
            ax.imshow(sel[j], cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])

axes[0, 0].set_title("mean", fontsize=8)
for j in range(N_REP):
    axes[0, j + 1].set_title(f"#{j}", fontsize=8)

plt.suptitle(
    "MNIST cluster representatives (K=3 v2 run, 20 KMeans clusters on PCA-50)\n"
    "rows: cluster id / expert / size / top-3 classes   |   col 0: mean   cols 1-8: nearest to mean",
    fontsize=9, y=0.995,
)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")

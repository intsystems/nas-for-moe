"""Нарисовать представителей кластеров MNIST."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parent / "mnist_data"
OUT = Path(__file__).resolve().parent / "mnist_cluster_reps_v2.png"
N_REP = 8

X = np.load(DATA / "data_X.npy")  # [70000, 1, 28, 28] uint8
y = np.load(DATA / "data_y.npy")
train_idx = np.load(DATA / "train_indices.npy")
train_cid = np.load(DATA / "train_cluster_ids.npy")
centers = np.load(DATA / "cluster_centers.npy")  # [M, 50]

M = centers.shape[0]
X_train = X[train_idx, 0]                # [N_tr, 28, 28]
y_train = y[train_idx]

# Цвет заголовка по назначению эксперта (из results_mnist_sgem_k3.json)
hard_assignments = [0, 1, 0, 0, 2, 0, 1, 0, 1, 0, 0, 2, 0, 1, 1, 2, 1, 1, 2, 2]
expert_of = {c: hard_assignments[c] for c in range(len(hard_assignments))}
expert_colors = ["#d62728", "#2ca02c", "#1f77b4"]

# Для поиска представителей: PCA-50 через ту же PCA (эквивалент сделаем по flat-пикселям),
# но у нас уже train_cid, а центры в PCA-50. Просто возьмём примеры ближайшие к медиоиду по
# расстоянию в пиксельном пространстве (это нормально визуально).
fig, axes = plt.subplots(M, N_REP + 1, figsize=(N_REP + 1, M),
                          gridspec_kw={"wspace": 0.05, "hspace": 0.05})

for m in range(M):
    mask = train_cid == m
    idx_m = np.where(mask)[0]
    imgs = X_train[idx_m]                # [n_m, 28, 28]
    labels_m = y_train[idx_m]
    n_m = len(imgs)

    # медоид в пиксельном пространстве
    flat = imgs.reshape(n_m, -1).astype(np.float32) / 255.0
    mean = flat.mean(axis=0)
    dists = np.linalg.norm(flat - mean, axis=1)
    order = np.argsort(dists)[: N_REP]
    sel = imgs[order]

    # среднее по кластеру (как "prototype")
    proto = mean.reshape(28, 28)

    # top-3 классов
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
    "MNIST cluster representatives (20 KMeans clusters on PCA-50, K=3 experts)\n"
    "rows: cluster id / expert / size / top-3 digit classes   |   col 0: mean image   cols 1-8: nearest to mean",
    fontsize=9, y=0.995,
)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")

"""Представители кластеров CIFAR-100 с разметкой экспертов из results_*.json.

Аналог plot_mnist_clusters_v2.py для CIFAR-100. Отличия:
    - RGB 3x32x32 (CHW) → imshow требует транспонирования в HWC;
    - читаем class_list из meta.json (если есть), чтобы показать названия классов;
    - пути задаются через CLI.

Пример:
    python plot_cifar100_clusters.py \
        --data-dir /pbabkin/.../code/data/cifar100_data \
        --results /pbabkin/.../code/runs/results_cifar100_sgem.json \
        --output  /pbabkin/.../code/runs/cifar100_cluster_reps.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


EXPERT_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Директория с data_X.npy / train_indices / cluster_ids / centers.")
    ap.add_argument("--results", type=Path, required=True,
                    help="JSON с результатами запуска (results_cifar100_*.json).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Путь для PNG.")
    ap.add_argument("--n-rep", type=int, default=8,
                    help="Сколько представителей на кластер (кроме mean).")
    ap.add_argument("--method-key", type=str, default=None,
                    help="Ключ метода в JSON (по умолчанию — первый).")
    args = ap.parse_args()

    # --- Данные ---
    X = np.load(args.data_dir / "data_X.npy")          # [N, 3, 32, 32] uint8
    y = np.load(args.data_dir / "data_y.npy")
    train_idx = np.load(args.data_dir / "train_indices.npy")
    train_cid = np.load(args.data_dir / "train_cluster_ids.npy")
    centers = np.load(args.data_dir / "cluster_centers.npy")
    M = centers.shape[0]

    # CIFAR: [N, 3, 32, 32] → выберем train, оставим CHW, транспонируем при imshow
    X_train = X[train_idx]                              # [N_tr, 3, 32, 32]
    y_train = y[train_idx]

    # --- Классы (имена, если доступны) ---
    class_names = None
    meta_path = args.data_dir / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        # class_list — исходные CIFAR-индексы; имена CIFAR-100:
        try:
            from torchvision import datasets
            cifar = datasets.CIFAR100(
                str(args.data_dir / "cifar100_raw"), train=True, download=False,
            )
            orig_classes = cifar.classes  # 100 имён
            class_names = [orig_classes[c] for c in meta["class_list"]]
        except Exception:
            class_names = None

    # --- Hard assignments ---
    results = json.load(open(args.results))
    key = args.method_key or next(iter(results))
    hard_assignments = results[key]["hard_assignments"]
    K = int(max(hard_assignments)) + 1
    print(f"[plot] method key: {key}, K={K}, M={M}")

    # --- Рисунок ---
    n_rep = args.n_rep
    fig, axes = plt.subplots(
        M, n_rep + 1, figsize=(n_rep + 1, M),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for m in range(M):
        mask = train_cid == m
        idx_m = np.where(mask)[0]
        imgs = X_train[idx_m]                            # [n_m, 3, 32, 32]
        labels_m = y_train[idx_m]
        n_m = len(imgs)

        # медиоид в пиксельном пространстве (flat RGB [0,1])
        flat = imgs.reshape(n_m, -1).astype(np.float32) / 255.0
        mean = flat.mean(axis=0)
        dists = np.linalg.norm(flat - mean, axis=1)
        order = np.argsort(dists)[: n_rep]
        sel = imgs[order]

        proto = mean.reshape(3, 32, 32).transpose(1, 2, 0)

        uniq, cnt = np.unique(labels_m, return_counts=True)
        top = uniq[np.argsort(-cnt)][:3]
        if class_names is not None:
            top_str = "/".join(class_names[int(t)][:6] for t in top)
        else:
            top_str = "/".join(str(int(t)) for t in top)

        expert = hard_assignments[m]
        color = EXPERT_COLORS[expert % len(EXPERT_COLORS)]

        ax0 = axes[m, 0]
        ax0.imshow(np.clip(proto, 0, 1))
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_ylabel(
            f"c{m}\nE{expert}\nn={n_m}\n{top_str}",
            rotation=0, labelpad=34, fontsize=6, va="center", color=color,
        )
        for spine in ax0.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)

        for j in range(n_rep):
            ax = axes[m, j + 1]
            if j < len(sel):
                ax.imshow(sel[j].transpose(1, 2, 0))
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_title("mean", fontsize=8)
    for j in range(n_rep):
        axes[0, j + 1].set_title(f"#{j}", fontsize=8)

    plt.suptitle(
        f"CIFAR-100 cluster representatives ({key}, M={M} clusters, K={K} experts)\n"
        "rows: cluster id / expert / size / top-3 classes   |   col 0: mean   cols 1-N: nearest to mean",
        fontsize=9, y=0.995,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

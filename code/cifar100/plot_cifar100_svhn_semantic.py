"""Визуализация semantic-кластеров для микс-датасета CIFAR-100 + SVHN.

Грузит ResNet-50 эмбеддинги (data_X.npy, [N, 2048]), проецирует в 2D через t-SNE
(на случайной подвыборке для скорости) и рисует две панели:
    (1) точки, окрашенные по cluster_id (M цветов);
    (2) те же точки, окрашенные по источнику (CIFAR / SVHN).

Использование:
    python plot_cifar100_svhn_semantic.py \\
        --data-dir ./cifar100_svhn_data_semantic_testsplit \\
        --output  ./cifar100_svhn_clusters.png \\
        --n-sample 8000 --seed 322
"""
import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from prepare_cifar100_semantic import _extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=8000,
                    help="Сколько точек оставить для t-SNE (равномерная подвыборка).")
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--embed-batch-size", type=int, default=256)
    args = ap.parse_args()

    X = np.load(args.data_dir / "data_X.npy")   # [N, 3, 32, 32] uint8 — raw images
    cid = np.load(args.data_dir / "train_cluster_ids.npy")
    train_idx = np.load(args.data_dir / "train_indices.npy")
    src_all = np.load(args.data_dir / "source_ids.npy")
    src = src_all[train_idx]
    Xtr_img = X[train_idx]

    rng = np.random.default_rng(args.seed)
    n = len(Xtr_img)
    if args.n_sample < n:
        sel = rng.choice(n, size=args.n_sample, replace=False)
        Xtr_img, cid, src = Xtr_img[sel], cid[sel], src[sel]
    print(f"[viz] extracting ResNet-50 embeddings on {len(Xtr_img)} samples (device={args.device})")
    Xtr_hwc = Xtr_img.transpose(0, 2, 3, 1)  # CHW → HWC, as expected by _extract_features
    feats = _extract_features(Xtr_hwc, device=args.device, batch_size=args.embed_batch_size)
    print(f"[viz] running t-SNE on {feats.shape[0]} points, dim={feats.shape[1]}, perplexity={args.perplexity}")
    Xtr = feats

    Z = TSNE(n_components=2, perplexity=args.perplexity,
             init="pca", random_state=args.seed, n_jobs=-1).fit_transform(Xtr)

    M = int(cid.max()) + 1

    meta_path = args.data_dir / "meta.json"
    ideal = None
    if meta_path.exists():
        meta = json.load(open(meta_path))
        ideal = meta.get("ideal_split_by_source")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # --- (1) по кластерам ---
    ax = axes[0]
    cmap = plt.get_cmap("tab20", M) if M <= 20 else plt.get_cmap("nipy_spectral", M)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=cid, cmap=cmap, s=4, alpha=0.7, linewidths=0)
    for m in range(M):
        mask = cid == m
        if mask.sum() == 0:
            continue
        cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
        ax.text(cx, cy, str(m), fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.85))
    ax.set_title(f"Semantic clusters (KMeans on ResNet-50, M={M})")
    ax.set_xticks([]); ax.set_yticks([])

    # --- (2) по источникам ---
    ax = axes[1]
    colors = np.where(src == 0, "#1f77b4", "#d62728")  # CIFAR=blue, SVHN=red
    ax.scatter(Z[src == 0, 0], Z[src == 0, 1], c="#1f77b4", s=4, alpha=0.6,
               linewidths=0, label=f"CIFAR-100 (n={int((src==0).sum())})")
    ax.scatter(Z[src == 1, 0], Z[src == 1, 1], c="#d62728", s=4, alpha=0.6,
               linewidths=0, label=f"SVHN (n={int((src==1).sum())})")
    if ideal is not None:
        for m in range(M):
            mask = cid == m
            if mask.sum() == 0:
                continue
            cx, cy = Z[mask, 0].mean(), Z[mask, 1].mean()
            tag = "C" if m in ideal.get("cifar_clusters", []) else "S"
            ax.text(cx, cy, f"{m}\n{tag}", fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.5, alpha=0.85))
    ax.legend(loc="upper right", fontsize=9, markerscale=3)
    ax.set_title("Source: CIFAR-100 vs SVHN  (per-cluster majority label C/S)")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(
        f"t-SNE of ResNet-50 embeddings — {args.data_dir.name}  "
        f"(n_sample={len(Xtr)}, perplexity={args.perplexity})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

"""
Скачивание, подмножество и кластеризация CIFAR-100.

Выходная директория содержит те же файлы, что и для MNIST:
    data_X.npy              — изображения uint8 [N, 3, 32, 32] (CHW)
    data_y.npy              — метки int64 [N] (перенумерованы с 0)
    train_indices.npy       — индексы train-точек в data_X
    val_indices.npy         — индексы val-точек в data_X
    train_cluster_ids.npy   — ID кластера для каждой train-точки
    val_cluster_ids.npy     — ID кластера для каждой val-точки
    cluster_centers.npy     — центроиды KMeans [M, pca_dim]
    meta.json               — мета-информация (num_classes, class_list и т.д.)

Пример:
    python prepare_cifar100.py --output-dir ./cifar100_data \
        --n-classes 20 --fraction 0.5 --n-clusters 20 --seed 322
"""

import argparse
import json
from pathlib import Path

import numpy as np


def prepare_cifar100(
    output_dir: str,
    n_clusters: int = 20,
    pca_dim: int = 50,
    seed: int = 322,
    n_classes: int = 100,
    selected_classes: list = None,
    fraction: float = 1.0,
    test_size: float = 0.2,
):
    from torchvision import datasets
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Скачивание CIFAR-100 ---
    cifar_root = output_dir / "cifar100_raw"
    print("[1/5] Downloading CIFAR-100...")
    train_ds = datasets.CIFAR100(str(cifar_root), train=True, download=True)
    test_ds = datasets.CIFAR100(str(cifar_root), train=False, download=True)

    X_full = np.concatenate([train_ds.data, test_ds.data], axis=0)  # [60000, 32, 32, 3]
    y_full = np.concatenate(
        [train_ds.targets, test_ds.targets], axis=0
    ).astype(np.int64)

    print(f"    Total: {len(X_full)} images, 100 classes")

    # --- 2. Отбор классов ---
    if selected_classes is not None:
        class_list = sorted(selected_classes)
    else:
        all_classes = sorted(np.unique(y_full).tolist())
        class_list = all_classes[:n_classes]

    mask = np.isin(y_full, class_list)
    X_sub = X_full[mask]
    y_sub = y_full[mask]

    # Перенумерация: 0, 1, ..., len(class_list)-1
    class_map = {c: i for i, c in enumerate(class_list)}
    y_sub = np.array([class_map[c] for c in y_sub], dtype=np.int64)
    num_classes = len(class_list)

    print(f"[2/5] Selected {num_classes} classes → {len(X_sub)} images")

    # --- 3. Подвыборка (fraction) ---
    rng = np.random.RandomState(seed)
    if 0 < fraction < 1.0:
        n_keep = max(1, int(len(X_sub) * fraction))
        keep_idx = rng.choice(len(X_sub), n_keep, replace=False)
        keep_idx.sort()
        X_sub = X_sub[keep_idx]
        y_sub = y_sub[keep_idx]

    N = len(X_sub)
    print(f"[3/5] After fraction={fraction}: {N} images")

    # HWC → CHW (uint8)
    X_img = X_sub.transpose(0, 3, 1, 2)  # [N, 3, 32, 32]

    # --- 4. PCA ---
    X_flat = X_sub.reshape(N, -1).astype(np.float32) / 255.0
    actual_pca_dim = min(pca_dim, N, X_flat.shape[1])
    print(f"[4/5] PCA {actual_pca_dim}-d on {N} images...")
    pca = PCA(n_components=actual_pca_dim, random_state=seed).fit(X_flat)
    X_pca = pca.transform(X_flat)

    # --- 5. Train/val split + KMeans ---
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y_sub,
    )

    actual_n_clusters = min(n_clusters, len(train_idx))
    print(f"[5/5] KMeans {actual_n_clusters} clusters on {len(train_idx)} train images...")
    km = KMeans(n_clusters=actual_n_clusters, random_state=seed, n_init=10).fit(
        X_pca[train_idx]
    )
    train_cluster_ids = km.labels_.astype(np.int64)
    val_cluster_ids = km.predict(X_pca[val_idx]).astype(np.int64)
    cluster_centers = km.cluster_centers_.astype(np.float32)

    # --- Сохранение ---
    np.save(output_dir / "data_X.npy", X_img)
    np.save(output_dir / "data_y.npy", y_sub)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "train_cluster_ids.npy", train_cluster_ids)
    np.save(output_dir / "val_cluster_ids.npy", val_cluster_ids)
    np.save(output_dir / "cluster_centers.npy", cluster_centers)

    meta = {
        "num_classes": num_classes,
        "class_list": class_list,
        "total_samples": N,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_clusters": actual_n_clusters,
        "pca_dim": actual_pca_dim,
        "fraction": fraction,
        "seed": seed,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  data_X.npy:            {X_img.shape} uint8")
    print(f"  data_y.npy:            {y_sub.shape} int64 ({num_classes} classes)")
    print(f"  train/val:             {len(train_idx)} / {len(val_idx)}")
    print(f"  clusters:              {actual_n_clusters}")
    print(f"  cluster_centers.npy:   {cluster_centers.shape}")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Download, subset and cluster CIFAR-100"
    )
    parser.add_argument("--output-dir", type=str, default="./cifar100_data",
                        help="Директория для сохранения данных")
    parser.add_argument("--n-classes", type=int, default=20,
                        help="Число классов (первые N из 100)")
    parser.add_argument("--selected-classes", type=int, nargs="+", default=None,
                        help="Явный список классов (перебивает --n-classes)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Доля датасета (0.0–1.0). Например, 0.5 = 50%% данных")
    parser.add_argument("--n-clusters", type=int, default=20,
                        help="Число кластеров KMeans")
    parser.add_argument("--pca-dim", type=int, default=50,
                        help="Размерность PCA")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Доля валидации (train/val split)")
    parser.add_argument("--seed", type=int, default=322)
    args = parser.parse_args()

    prepare_cifar100(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        pca_dim=args.pca_dim,
        seed=args.seed,
        n_classes=args.n_classes,
        selected_classes=args.selected_classes,
        fraction=args.fraction,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()

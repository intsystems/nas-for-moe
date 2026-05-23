"""
Кастомный микс CIFAR-100 + SVHN с семантической кластеризацией (ResNet-50 ImageNet).

Цель эксперимента: проверить, выделит ли SGEM разбиение кластеров **по источнику
данных** (один эксперт → CIFAR-100, другой → SVHN). SVHN (32x32 RGB, цифры на
фасадах домов) — максимально далёкий домен от природных категорий CIFAR-100, так
что ResNet-50-эмбеддинги почти не смешивают источники в один кластер → "идеальное
разбиение" чётко определено (при K=2: кластеры-CIFAR одному эксперту, кластеры-SVHN
другому).

Отличия от prepare_cifar100_semantic.py:
    * данные = смесь CIFAR-100 и SVHN в заданной пропорции (--cifar-frac);
    * метки SVHN сдвинуты на +100 (классы 100..109) → num_classes = 110;
    * дополнительно сохраняется source_ids.npy (0 = CIFAR-100, 1 = SVHN);
    * в конце печатается состав каждого кластера по источнику + majority-mapping
      (это ожидаемый "ideal split" для K=2).

Эмбеддинги, кластеризация и трёхчастный split — как в prepare_cifar100_semantic:
bicubic 32->224 + ImageNet-norm → ResNet-50.avgpool → L2-norm → KMeans на train,
val/test размечаются ближайшим центроидом (test нигде не участвует в кластеризации).

Выходная директория содержит:
    data_X.npy              — [N, 3, 32, 32] uint8 (CHW)
    data_y.npy              — [N] int64 (0..99 = CIFAR-100, 100..109 = SVHN)
    source_ids.npy          — [N] int64 (0 = CIFAR-100, 1 = SVHN)
    train_indices.npy       — индексы train-точек в data_X
    val_indices.npy         — индексы val-точек
    test_indices.npy        — индексы test-точек (отложенная выборка)
    train_cluster_ids.npy   — [N_train] int64
    val_cluster_ids.npy     — [N_val] int64
    test_cluster_ids.npy    — [N_test] int64
    cluster_centers.npy     — [M, 2048] float32 (в L2-нормированном пространстве)
    meta.json               — мета-информация (включая ideal_split_by_source)

Пример:
    python prepare_cifar100_svhn.py --output-dir ./cifar100_svhn_data_semantic_testsplit \\
        --total-samples 42000 --cifar-frac 0.5 --n-clusters 30 \\
        --val-size 0.15 --test-size 0.15 --seed 322 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Переиспользуем эмбеддер из семантического препроцессора (тот же препроцесс).
from prepare_cifar100_semantic import _extract_features, IMAGENET_MEAN, IMAGENET_STD


SVHN_LABEL_OFFSET = 100  # метки SVHN сдвигаем за пределы 100 классов CIFAR-100


def _load_cifar100_pool(raw_root: Path):
    """CIFAR-100 train+test → (X_hwc uint8 [N,32,32,3], y int64 [N])."""
    from torchvision import datasets

    train_ds = datasets.CIFAR100(str(raw_root), train=True, download=True)
    test_ds = datasets.CIFAR100(str(raw_root), train=False, download=True)
    X = np.concatenate([train_ds.data, test_ds.data], axis=0)  # HWC uint8
    y = np.concatenate([train_ds.targets, test_ds.targets], axis=0).astype(np.int64)
    return X, y


def _load_svhn_pool(raw_root: Path, split: str):
    """SVHN → (X_hwc uint8 [N,32,32,3], y int64 [N], метки 0..9)."""
    from torchvision import datasets

    ds = datasets.SVHN(str(raw_root), split=split, download=True)
    X = ds.data.transpose(0, 2, 3, 1)  # SVHN .data — CHW → переводим в HWC
    y = np.asarray(ds.labels, dtype=np.int64)  # torchvision уже ремапит '0'→0
    return np.ascontiguousarray(X), y


def _subsample(n_pool: int, n_take: int, rng: np.random.RandomState) -> np.ndarray:
    n_take = min(n_take, n_pool)
    idx = rng.choice(n_pool, n_take, replace=False)
    idx.sort()
    return idx


def prepare_cifar100_svhn(
    output_dir: str,
    total_samples: int = 42000,
    cifar_frac: float = 0.5,
    n_clusters: int = 30,
    seed: int = 322,
    val_size: float = 0.15,
    test_size: float = 0.15,
    svhn_split: str = "train",
    device: str = "cuda",
    embed_batch_size: int = 256,
):
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    if not (0.0 < cifar_frac < 1.0):
        raise ValueError(f"--cifar-frac должен быть в (0, 1), получено {cifar_frac}")
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0 and val_size + test_size < 1.0):
        raise ValueError(
            f"Некорректные доли split: val_size={val_size}, test_size={test_size} "
            f"(нужно val_size>0, test_size>0, val_size+test_size<1)"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # --- 1. Загрузка пулов ---
    print("[1/5] Downloading CIFAR-100 + SVHN...")
    X_cifar_pool, y_cifar_pool = _load_cifar100_pool(output_dir / "cifar100_raw")
    X_svhn_pool, y_svhn_pool = _load_svhn_pool(output_dir / "svhn_raw", svhn_split)
    print(f"    CIFAR-100 pool: {len(X_cifar_pool)} (100 classes)")
    print(f"    SVHN[{svhn_split}] pool: {len(X_svhn_pool)} (10 classes)")

    # --- 2. Подвыборка под заданную пропорцию ---
    n_cifar = int(round(total_samples * cifar_frac))
    n_svhn = total_samples - n_cifar
    cifar_idx = _subsample(len(X_cifar_pool), n_cifar, rng)
    svhn_idx = _subsample(len(X_svhn_pool), n_svhn, rng)
    if len(cifar_idx) < n_cifar:
        print(f"    [warn] CIFAR pool меньше запрошенного: {len(cifar_idx)} < {n_cifar}")
    if len(svhn_idx) < n_svhn:
        print(f"    [warn] SVHN pool меньше запрошенного: {len(svhn_idx)} < {n_svhn}")

    X_cifar = X_cifar_pool[cifar_idx]
    y_cifar = y_cifar_pool[cifar_idx]
    X_svhn = X_svhn_pool[svhn_idx]
    y_svhn = y_svhn_pool[svhn_idx] + SVHN_LABEL_OFFSET

    X_hwc = np.concatenate([X_cifar, X_svhn], axis=0)  # [N, 32, 32, 3] uint8
    y = np.concatenate([y_cifar, y_svhn], axis=0).astype(np.int64)
    source_ids = np.concatenate(
        [np.zeros(len(X_cifar), dtype=np.int64), np.ones(len(X_svhn), dtype=np.int64)]
    )
    N = len(X_hwc)
    num_classes = 100 + 10  # 0..99 CIFAR, 100..109 SVHN
    print(f"[2/5] Mix: {len(X_cifar)} CIFAR + {len(X_svhn)} SVHN = {N} "
          f"(cifar_frac≈{len(X_cifar) / N:.3f}), num_classes={num_classes}")

    X_img = X_hwc.transpose(0, 3, 1, 2).copy()  # CHW uint8 — формат data_X.npy

    # --- 3. ResNet-50 эмбеддинги ---
    print(f"[3/5] ResNet-50 (ImageNet) embeddings on {N} images, device={device}...")
    feats = _extract_features(X_hwc, device=device, batch_size=embed_batch_size)
    print(f"    feats: {feats.shape}, mean L2-norm: {np.linalg.norm(feats, axis=1).mean():.4f}")

    # --- 4. Трёхчастный split + KMeans на train-эмбеддингах ---
    indices = np.arange(N)
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y,
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, random_state=seed, stratify=y[trainval_idx],
    )

    actual_n_clusters = min(n_clusters, len(train_idx))
    print(f"[4/5] KMeans {actual_n_clusters} clusters on {len(train_idx)} train embeddings "
          f"(val={len(val_idx)}, test={len(test_idx)} by nearest centroid)...")
    km = KMeans(n_clusters=actual_n_clusters, random_state=seed, n_init=10).fit(feats[train_idx])
    train_cluster_ids = km.labels_.astype(np.int64)
    val_cluster_ids = km.predict(feats[val_idx]).astype(np.int64)
    test_cluster_ids = km.predict(feats[test_idx]).astype(np.int64)
    cluster_centers = km.cluster_centers_.astype(np.float32)

    # --- 5. Состав кластеров по источнику + ideal split (K=2) ---
    src_train = source_ids[train_idx]
    cifar_share = np.zeros(actual_n_clusters)
    cluster_total = np.zeros(actual_n_clusters, dtype=np.int64)
    for c in range(actual_n_clusters):
        mask = train_cluster_ids == c
        cluster_total[c] = int(mask.sum())
        if cluster_total[c] > 0:
            cifar_share[c] = float((src_train[mask] == 0).mean())
    majority_source = (cifar_share < 0.5).astype(int)  # 0 = CIFAR-majority, 1 = SVHN-majority
    purity = np.where(majority_source == 0, cifar_share, 1.0 - cifar_share)
    ideal_split_by_source = {
        "cifar_clusters": np.where(majority_source == 0)[0].tolist(),
        "svhn_clusters": np.where(majority_source == 1)[0].tolist(),
        "mean_cluster_purity": float(np.nanmean(purity)),
        "min_cluster_purity": float(np.nanmin(purity)),
    }

    # --- сохранение ---
    np.save(output_dir / "data_X.npy", X_img)
    np.save(output_dir / "data_y.npy", y)
    np.save(output_dir / "source_ids.npy", source_ids)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "test_indices.npy", test_idx)
    np.save(output_dir / "train_cluster_ids.npy", train_cluster_ids)
    np.save(output_dir / "val_cluster_ids.npy", val_cluster_ids)
    np.save(output_dir / "test_cluster_ids.npy", test_cluster_ids)
    np.save(output_dir / "cluster_centers.npy", cluster_centers)

    meta = {
        "dataset": "cifar100+svhn",
        "num_classes": num_classes,
        "cifar_classes": 100,
        "svhn_classes": 10,
        "svhn_label_offset": SVHN_LABEL_OFFSET,
        "svhn_split": svhn_split,
        "total_samples": N,
        "n_cifar": int(len(X_cifar)),
        "n_svhn": int(len(X_svhn)),
        "cifar_frac_requested": cifar_frac,
        "cifar_frac_actual": float(len(X_cifar) / N),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "val_size": val_size,
        "test_size": test_size,
        "n_clusters": actual_n_clusters,
        "embed_model": "resnet50_imagenet1k_v2",
        "embed_dim": int(feats.shape[1]),
        "embed_normalize": "l2",
        "input_resize": 224,
        "input_resize_mode": "bicubic",
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "seed": seed,
        "ideal_split_by_source": ideal_split_by_source,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- отчёт ---
    print(f"\nDone! Saved to {output_dir}/")
    print(f"  data_X.npy:           {X_img.shape} uint8 (CHW)")
    print(f"  data_y.npy:           {y.shape} int64 ({num_classes} classes; 0..99 CIFAR, 100..109 SVHN)")
    print(f"  source_ids.npy:       {source_ids.shape} int64 (0=CIFAR, 1=SVHN)")
    print(f"  train/val/test:       {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"  clusters:             {actual_n_clusters}")
    print(f"  cluster_centers.npy:  {cluster_centers.shape}")
    print(f"\n  Per-cluster source composition (train):")
    for c in range(actual_n_clusters):
        tag = "CIFAR" if majority_source[c] == 0 else "SVHN "
        print(f"    cluster {c:2d}: n={cluster_total[c]:5d}  cifar={cifar_share[c]:.3f}  "
              f"svhn={1.0 - cifar_share[c]:.3f}  -> {tag} (purity {purity[c]:.3f})")
    print(f"\n  Ideal split by source (K=2):")
    print(f"    CIFAR-majority clusters ({len(ideal_split_by_source['cifar_clusters'])}): "
          f"{ideal_split_by_source['cifar_clusters']}")
    print(f"    SVHN-majority  clusters ({len(ideal_split_by_source['svhn_clusters'])}): "
          f"{ideal_split_by_source['svhn_clusters']}")
    print(f"    mean cluster purity: {ideal_split_by_source['mean_cluster_purity']:.4f}  "
          f"(min {ideal_split_by_source['min_cluster_purity']:.4f})")
    if ideal_split_by_source["mean_cluster_purity"] < 0.9:
        print("    [warn] кластеры заметно смешивают источники — увеличь n_clusters "
              "или проверь данные; 'разбиение по датасетам' будет размытым.")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Build a CIFAR-100 + SVHN mix with semantic (ResNet-50) clustering"
    )
    parser.add_argument("--output-dir", type=str,
                        default="./cifar100_svhn_data_semantic_testsplit")
    parser.add_argument("--total-samples", type=int, default=42000,
                        help="Размер итогового микса (до split)")
    parser.add_argument("--cifar-frac", type=float, default=0.5,
                        help="Доля CIFAR-100 в миксе (SVHN получает 1 - cifar_frac); дефолт 0.5/0.5")
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Доля val (от полного микса)")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Доля отложенного test (от полного микса)")
    parser.add_argument("--svhn-split", type=str, default="train",
                        choices=["train", "test", "extra"],
                        help="Из какого SVHN-сплита тянуть пул изображений")
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    args = parser.parse_args()

    prepare_cifar100_svhn(
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        cifar_frac=args.cifar_frac,
        n_clusters=args.n_clusters,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        svhn_split=args.svhn_split,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()

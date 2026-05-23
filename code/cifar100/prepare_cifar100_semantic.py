"""
Семантическая кластеризация CIFAR-100 на признаках ResNet-50 (ImageNet pretrained).

Отличия от prepare_cifar100.py:
    * вместо PCA на сырых пикселях — извлечение 2048-мерных признаков из
      pre-pool layer ResNet-50 (torchvision, ImageNet weights);
    * перед извлечением: bicubic upsample 32→224 + ImageNet normalization;
    * L2-нормализация эмбеддингов перед KMeans (≈ spherical KMeans);
    * cluster_centers сохраняются в пространстве эмбеддингов (2048-d).

Выходная директория содержит:
    data_X.npy              — [N, 3, 32, 32] uint8
    data_y.npy              — [N] int64
    train_indices.npy       — индексы train-точек в data_X
    val_indices.npy         — индексы val-точек
    test_indices.npy        — индексы test-точек (отложенная выборка)
    train_cluster_ids.npy   — [N_train] int64
    val_cluster_ids.npy     — [N_val] int64
    test_cluster_ids.npy    — [N_test] int64
    cluster_centers.npy     — [M, 2048] float32 (в L2-нормированном пространстве)
    meta.json               — мета-информация

Сплит трёхчастный: сначала отделяется test (test_size), затем остаток
делится на train/val (val_size — доля от ПОЛНОГО датасета). KMeans
обучается ТОЛЬКО на train-эмбеддингах; val и test размечаются ближайшим
центроидом — test нигде не участвует в кластеризации/поиске.

Для маршрутизации новых изображений к кластеру (см. project_to_clusters):
повторяется тот же препроцесс: resize→normalize→ResNet-50.avgpool→L2-norm→
ближайший центроид по евклидовой дистанции (= косинусной, т.к. норма=1).

Пример:
    python prepare_cifar100_semantic.py --output-dir ./cifar100_data_semantic_testsplit \\
        --fraction 0.7 --n-clusters 30 --val-size 0.15 --test-size 0.15 \\
        --seed 322 --device cuda:0
"""

import argparse
import json
from pathlib import Path

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_resnet50(device: str):
    import torch
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    # Заменим финальный fc на Identity — out теперь 2048-d (после avgpool+flatten)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _extract_features(
    X_uint8_hwc: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """X_uint8_hwc: [N, 32, 32, 3] uint8 → [N, 2048] float32 (ResNet-50 fc-input, L2-normalized)."""
    import torch
    import torch.nn.functional as F

    model = _build_resnet50(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    N = len(X_uint8_hwc)
    out = np.empty((N, 2048), dtype=np.float32)
    for i in range(0, N, batch_size):
        xb = X_uint8_hwc[i:i + batch_size]                              # [B, 32, 32, 3]
        xb = torch.from_numpy(xb).to(device, non_blocking=True)
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)                  # [B, 3, 32, 32]
        xb = F.interpolate(xb, size=224, mode="bicubic", align_corners=False)
        xb = (xb - mean) / std
        with torch.no_grad():
            feat = model(xb)                                             # [B, 2048]
        feat = F.normalize(feat, p=2, dim=1)
        out[i:i + xb.size(0)] = feat.cpu().numpy().astype(np.float32)
        if (i // batch_size) % 20 == 0:
            print(f"    [embed] {min(i + batch_size, N)}/{N}")
    return out


def prepare_cifar100_semantic(
    output_dir: str,
    n_clusters: int = 30,
    seed: int = 322,
    fraction: float = 1.0,
    val_size: float = 0.15,
    test_size: float = 0.15,
    device: str = "cuda",
    embed_batch_size: int = 256,
):
    from torchvision import datasets
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Скачивание CIFAR-100 ---
    cifar_root = output_dir / "cifar100_raw"
    print("[1/4] Downloading CIFAR-100...")
    train_ds = datasets.CIFAR100(str(cifar_root), train=True, download=True)
    test_ds = datasets.CIFAR100(str(cifar_root), train=False, download=True)

    X_sub = np.concatenate([train_ds.data, test_ds.data], axis=0)  # [60000, 32, 32, 3]
    y_sub = np.concatenate(
        [train_ds.targets, test_ds.targets], axis=0
    ).astype(np.int64)
    num_classes = 100
    print(f"    Total: {len(X_sub)} images, {num_classes} classes")

    # --- 2. Подвыборка (fraction) ---
    rng = np.random.RandomState(seed)
    if 0 < fraction < 1.0:
        n_keep = max(1, int(len(X_sub) * fraction))
        keep_idx = rng.choice(len(X_sub), n_keep, replace=False)
        keep_idx.sort()
        X_sub = X_sub[keep_idx]
        y_sub = y_sub[keep_idx]

    N = len(X_sub)
    print(f"[2/4] After fraction={fraction}: {N} images")

    X_img = X_sub.transpose(0, 3, 1, 2)  # CHW uint8 — как в pixel-варианте

    # --- 3. ResNet-50 эмбеддинги ---
    print(f"[3/4] ResNet-50 (ImageNet) embeddings on {N} images, device={device}...")
    feats = _extract_features(X_sub, device=device, batch_size=embed_batch_size)
    print(f"    feats shape: {feats.shape}, mean L2-norm: {np.linalg.norm(feats, axis=1).mean():.4f}")

    # --- 4. Train/val/test split + KMeans на эмбеддингах ---
    indices = np.arange(N)
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0 and val_size + test_size < 1.0):
        raise ValueError(
            f"Некорректные доли split: val_size={val_size}, test_size={test_size} "
            f"(нужно val_size>0, test_size>0, val_size+test_size<1)"
        )
    trainval_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y_sub,
    )
    rel_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=rel_val, random_state=seed,
        stratify=y_sub[trainval_idx],
    )

    actual_n_clusters = min(n_clusters, len(train_idx))
    print(f"[4/4] KMeans {actual_n_clusters} clusters on {len(train_idx)} train embeddings "
          f"(val={len(val_idx)}, test={len(test_idx)} assigned by nearest centroid)...")
    km = KMeans(n_clusters=actual_n_clusters, random_state=seed, n_init=10).fit(
        feats[train_idx]
    )
    train_cluster_ids = km.labels_.astype(np.int64)
    val_cluster_ids = km.predict(feats[val_idx]).astype(np.int64)
    test_cluster_ids = km.predict(feats[test_idx]).astype(np.int64)
    cluster_centers = km.cluster_centers_.astype(np.float32)

    np.save(output_dir / "data_X.npy", X_img)
    np.save(output_dir / "data_y.npy", y_sub)
    np.save(output_dir / "train_indices.npy", train_idx)
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "test_indices.npy", test_idx)
    np.save(output_dir / "train_cluster_ids.npy", train_cluster_ids)
    np.save(output_dir / "val_cluster_ids.npy", val_cluster_ids)
    np.save(output_dir / "test_cluster_ids.npy", test_cluster_ids)
    np.save(output_dir / "cluster_centers.npy", cluster_centers)

    meta = {
        "num_classes": num_classes,
        "total_samples": N,
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
        "fraction": fraction,
        "seed": seed,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  data_X.npy:           {X_img.shape} uint8")
    print(f"  data_y.npy:           {y_sub.shape} int64 ({num_classes} classes)")
    print(f"  train/val/test:       {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"  clusters:             {actual_n_clusters}")
    print(f"  cluster_centers.npy:  {cluster_centers.shape}")

    train_sizes = np.bincount(train_cluster_ids, minlength=actual_n_clusters)
    val_sizes = np.bincount(val_cluster_ids, minlength=actual_n_clusters)
    test_sizes = np.bincount(test_cluster_ids, minlength=actual_n_clusters)
    print(f"  train cluster sizes:  min={train_sizes.min()} max={train_sizes.max()} "
          f"mean={train_sizes.mean():.1f}")
    print(f"  val cluster sizes:    min={val_sizes.min()} max={val_sizes.max()} "
          f"mean={val_sizes.mean():.1f}")
    print(f"  test cluster sizes:   min={test_sizes.min()} max={test_sizes.max()} "
          f"mean={test_sizes.mean():.1f}")
    return meta


def project_to_clusters(
    X_uint8_chw: np.ndarray,
    cluster_centers: np.ndarray,
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Маршрутизация произвольных CIFAR-изображений к ближайшему семантическому кластеру.

    Args:
        X_uint8_chw: [N, 3, 32, 32] uint8 (CHW, как в data_X.npy).
        cluster_centers: [M, 2048] float32 (из cluster_centers.npy).
        device: GPU/CPU для прогона ResNet-50.
        batch_size: размер батча эмбеддера.

    Returns:
        cluster_ids: [N] int64.
    """
    X_hwc = X_uint8_chw.transpose(0, 2, 3, 1)  # [N, 32, 32, 3]
    feats = _extract_features(X_hwc, device=device, batch_size=batch_size)
    d2 = (
        (feats ** 2).sum(axis=1, keepdims=True)
        - 2.0 * feats @ cluster_centers.T
        + (cluster_centers ** 2).sum(axis=1)
    )
    return d2.argmin(axis=1).astype(np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Download, subset and SEMANTICALLY cluster CIFAR-100 (ResNet-50 features)"
    )
    parser.add_argument("--output-dir", type=str,
                        default="./cifar100_data_semantic_testsplit")
    parser.add_argument("--fraction", type=float, default=0.7)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Доля val (от полного датасета)")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Доля отложенного test (от полного датасета)")
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    args = parser.parse_args()

    prepare_cifar100_semantic(
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        seed=args.seed,
        fraction=args.fraction,
        val_size=args.val_size,
        test_size=args.test_size,
        device=args.device,
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()

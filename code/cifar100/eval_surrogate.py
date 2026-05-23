"""Offline-оценка качества суррогата на собранных cifar100 obs.

Делает train/val split (80/20) на наблюдениях, перебирает несколько конфигов
гиперпараметров суррогата, для каждого:
  - обучает с нуля
  - на val считает R², Spearman ρ, MAE, RMSE
  - печатает результаты
  - сохраняет лучший по R²

Использование:
    # по одной директории
    python eval_surrogate.py --obs-dirs runs/cifar100_sgem_obs_lb005

    # объединить несколько (наблюдения сольются в один пул)
    python eval_surrogate.py --obs-dirs runs/cifar100_sgem_obs_lb005 \\
                                       runs/cifar100_sgem_obs_K3_v2 \\
                                       runs/cifar100_sgem_obs_K3_lb070
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Патч toy_graph.OPS под reduced 4-ops space до импорта collect_dataset
import cifar100_searchspace  # noqa: E402
cifar100_searchspace.patch_toy_graph_ops()

from cifar100_searchspace import OPS_NEW_SMALL  # noqa: E402
from toy_experiment.collect_dataset import (  # noqa: E402
    make_surrogate_loaders, train_surrogate, create_surrogate, SEED,
)


def evaluate_surrogate(surrogate, val_loader, device):
    surrogate.eval()
    true_all, pred_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = surrogate(
                batch.x, batch.edge_index, batch.batch, batch.bool_vector,
            )
            pred_all.append(out.cpu().numpy())
            true_all.append(batch.y.cpu().numpy())
    y_true = np.concatenate(true_all).flatten()
    y_pred = np.concatenate(pred_all).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sp, _ = spearmanr(y_true, y_pred)
    return {
        "r2": r2, "spearman": float(sp),
        "mae": mae, "rmse": rmse,
        "n_val": len(y_true),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline surrogate eval")
    parser.add_argument(
        "--obs-dirs", type=str, nargs="+", required=True,
        help="Список директорий с obs_*.json — все объединяются",
    )
    parser.add_argument("--data-dir", type=str,
                        default="./cifar100_data_semantic_testsplit",
                        help="Для cluster_centers.npy")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-best", type=str, default=None,
                        help="Куда сохранить лучший суррогат (по R²)")
    args = parser.parse_args()

    # --- Загрузить все obs ---
    obs_paths = []
    for d in args.obs_dirs:
        d_paths = sorted(Path(d).glob("obs_*.json"))
        print(f"  {d}: {len(d_paths)} obs")
        obs_paths.extend(d_paths)
    print(f"[total] {len(obs_paths)} observations")

    # --- Cluster centers (для data-aware GAT) ---
    centers_path = Path(args.data_dir) / "cluster_centers.npy"
    cluster_centers = np.load(centers_path)
    n_clusters = len(cluster_centers)
    print(f"[clusters] M = {n_clusters}, centers shape = {cluster_centers.shape}")

    n_features = len(OPS_NEW_SMALL)
    print(f"[features] {n_features} ops in reduced search space")

    # --- Гиперпараметры для перебора ---
    configs = [
        {"dropout": 0.3, "hidden_dim": 64,  "heads": 4, "lr": 3e-3,
         "epochs": 400, "patience": 50},
        {"dropout": 0.2, "hidden_dim": 64,  "heads": 4, "lr": 1e-3,
         "epochs": 400, "patience": 60},
        {"dropout": 0.1, "hidden_dim": 64,  "heads": 4, "lr": 1e-3,
         "epochs": 400, "patience": 60},
        {"dropout": 0.3, "hidden_dim": 128, "heads": 4, "lr": 1e-3,
         "epochs": 400, "patience": 60},
        {"dropout": 0.1, "hidden_dim": 32,  "heads": 4, "lr": 5e-4,
         "epochs": 500, "patience": 80},
        {"dropout": 0.1, "hidden_dim": 128, "heads": 4, "lr": 5e-4,
         "epochs": 500, "patience": 80},
    ]

    best_r2 = -float("inf")
    best_cfg_idx = -1
    best_state = None
    best_metrics = None

    for i, cfg in enumerate(configs):
        print(f"\n--- Config {i}: {cfg} ---")

        # Один и тот же seed во всех конфигах для одинакового train/val split
        train_loader, val_loader = make_surrogate_loaders(
            obs_paths, val_fraction=args.val_fraction, seed=args.seed,
        )
        print(f"  train={len(train_loader.dataset)}, "
              f"val={len(val_loader.dataset)}")

        surr = create_surrogate(
            n_features, n_clusters,
            dropout=cfg["dropout"],
            hidden_dim=cfg["hidden_dim"],
            heads=cfg["heads"],
            model_type="gat",
            nodes_per_graph=10,  # 9 ячейка + 1 input
            cluster_centers=cluster_centers,
        )

        hist = train_surrogate(
            surr, train_loader, val_loader,
            device=args.device, lr=cfg["lr"],
            epochs=cfg["epochs"], patience=cfg["patience"],
            verbose=False,
        )

        surr.to(args.device)
        m = evaluate_surrogate(surr, val_loader, args.device)
        print(f"  Epochs trained: {len(hist['train'])}")
        print(f"  R² = {m['r2']:.4f}   Spearman = {m['spearman']:.4f}")
        print(f"  MAE = {m['mae']:.4f}  RMSE = {m['rmse']:.4f}  "
              f"(n_val = {m['n_val']})")

        if m["r2"] > best_r2:
            best_r2 = m["r2"]
            best_cfg_idx = i
            best_metrics = m
            best_state = {k: v.cpu().clone() for k, v in surr.state_dict().items()}

    print(f"\n{'=' * 60}")
    print(f"BEST: config {best_cfg_idx}: {configs[best_cfg_idx]}")
    print(f"  R² = {best_metrics['r2']:.4f}")
    print(f"  Spearman = {best_metrics['spearman']:.4f}")
    print(f"  MAE = {best_metrics['mae']:.4f}")
    print(f"  RMSE = {best_metrics['rmse']:.4f}")
    print(f"{'=' * 60}")

    if args.save_best:
        save_path = Path(args.save_best)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_path)
        print(f"\n[save] best surrogate -> {save_path}")


if __name__ == "__main__":
    main()

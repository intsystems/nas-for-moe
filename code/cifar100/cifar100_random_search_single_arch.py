"""Random-search-single-arch baseline: одна архитектура → MoE из K её копий.

Этапы:
    1. Random search по архитектурам (без MoE): обучить N кандидатов на полных
       train-данных, выбрать лучшую по val_accuracy. Находим «единую
       универсальную» архитектуру, не учитывая разделение по кластерам.
    2. Построить MoE из K копий найденной архитектуры с learnable softmax-gating
       (cifar100_moe.CIFAR100MoE).
    3. Обучить MoE end-to-end на полных train-данных, оценить val accuracy.

Запуск:
    python cifar100_random_search_single_arch.py --device cuda:0 \\
        --data-dir ./cifar100_data --K 5 --n-arch-candidates 30 \\
        --search-epochs 10 --moe-epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# Патч toy_graph.OPS под уменьшенное search space
import cifar100_searchspace  # noqa: E402
cifar100_searchspace.patch_toy_graph_ops()

from cifar100_sgem import (  # noqa: E402
    CIFAR100DartsSearchSpace,
    CIFAR100Net,
    _train_cifar100_net,
    load_cifar100_meta,
)
from cifar100_moe import (  # noqa: E402
    CIFAR100MoE, load_cifar100_tensors, train_moe,
)
import toy_experiment.collect_dataset as collect_dataset  # noqa: E402


def search_best_architecture(
    ss: CIFAR100DartsSearchSpace,
    X_tr_np, y_tr_np, X_v_np, y_v_np,
    *,
    n_candidates: int,
    search_epochs: int,
    init_channels: int,
    num_classes: int,
    device: str,
    verbose: bool = True,
) -> tuple[dict, float]:
    """Random search: обучить N архитектур, вернуть лучшую."""
    best_acc = -1.0
    best_config = None
    for i in range(n_candidates):
        config = collect_dataset.sample_valid_config(ss)
        net = CIFAR100Net(config, C=init_channels, num_classes=num_classes)
        t0 = time.time()
        val_acc = _train_cifar100_net(
            net, X_tr_np, y_tr_np, X_v_np, y_v_np,
            epochs=search_epochs, lr=0.05, batch_size=128, device=device,
        )
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_config = config
        if verbose:
            ops = [config.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
            print(f"  [{i+1:3d}/{n_candidates}] ops={ops} "
                  f"val_acc={val_acc:.4f} time={time.time()-t0:.1f}s"
                  f"{' *** new best' if is_best else ''}")
    return best_config, best_acc


def main():
    parser = argparse.ArgumentParser(
        description="DARTS-baseline: search one architecture, MoE from K copies"
    )
    parser.add_argument("--data-dir", type=str, default="./cifar100_data")
    parser.add_argument("--save-results", type=str,
                        default="./runs/results_cifar100_random_search_single_arch.json")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n-arch-candidates", type=int, default=30)
    parser.add_argument("--search-epochs", type=int, default=10)
    parser.add_argument("--moe-epochs", type=int, default=30)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--gate-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)
    num_classes = meta["num_classes"]

    # Сырые numpy-данные для search (используют _train_cifar100_net)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    X_tr_np, y_tr_np = X[train_idx], y[train_idx]
    X_v_np, y_v_np = X[val_idx], y[val_idx]

    ss = CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    print(f"[search] num_classes={num_classes}, train={len(X_tr_np)}, "
          f"val={len(X_v_np)}, K={args.K}, "
          f"n_candidates={args.n_arch_candidates}, "
          f"search_epochs={args.search_epochs}")

    # --- 1. Random search по одиночной архитектуре ---
    best_config, best_search_acc = search_best_architecture(
        ss, X_tr_np, y_tr_np, X_v_np, y_v_np,
        n_candidates=args.n_arch_candidates,
        search_epochs=args.search_epochs,
        init_channels=args.init_channels,
        num_classes=num_classes,
        device=args.device,
    )
    print(f"\n[search] best single-arch val_acc = {best_search_acc:.4f}")
    ops = [best_config.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
    print(f"[search] best ops = {ops}")

    # --- 2. MoE из K копий найденной архитектуры ---
    configs = [dict(best_config) for _ in range(args.K)]

    moe = CIFAR100MoE(
        configs=configs,
        init_channels=args.init_channels,
        num_classes=num_classes,
        gate_channels=args.gate_channels,
    )
    n_params = sum(p.numel() for p in moe.parameters())
    print(f"\n[moe] K={args.K} (identical copies of best arch), "
          f"params={n_params:,}")

    # Тензоры для MoE-тренировки
    X_tr, y_tr, X_v, y_v = load_cifar100_tensors(data_dir)

    print(f"\n[moe] training MoE for {args.moe_epochs} epochs...")
    best_moe_acc = train_moe(
        moe, X_tr, y_tr, X_v, y_v,
        epochs=args.moe_epochs,
        batch_size=args.batch_size,
        lr=args.lr, wd=args.wd,
        device=args.device,
    )
    print(f"\n[moe] best MoE val_acc = {best_moe_acc:.4f}")
    print(f"[summary] single-arch={best_search_acc:.4f}  "
          f"MoE-K={args.K}={best_moe_acc:.4f}")

    # --- 3. Сохранить результаты ---
    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "cifar100_random_search_single_arch": {
            "K": args.K,
            "best_single_arch": best_config,
            "best_single_arch_val_acc": float(best_search_acc),
            "best_moe_val_acc": float(best_moe_acc),
            "n_arch_candidates": args.n_arch_candidates,
            "search_epochs": args.search_epochs,
            "moe_epochs": args.moe_epochs,
        }
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] results -> {save_path}")


if __name__ == "__main__":
    main()

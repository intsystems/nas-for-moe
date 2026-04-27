"""Random-MoE-baseline: сэмплирование N MoE-конфигов, выбор лучшего.

Этапы:
    1. Сэмплируется N MoE-конфигураций. Каждая = K случайных архитектур
       (по одной на эксперта).
    2. Каждая MoE обучается end-to-end (learnable softmax-gating, без кластеров).
    3. Выбирается лучшая по val accuracy.

Это baseline без архитектурной специализации по кластерам — для сравнения с
EM-методом (который такую специализацию ищет явно).

Запуск:
    python cifar100_random_moe_baseline.py --device cuda:0 \\
        --data-dir ./cifar100_data --K 5 --n-moe-candidates 10 --moe-epochs 30
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
    CIFAR100DartsSearchSpace, load_cifar100_meta,
)
from cifar100_moe import (  # noqa: E402
    CIFAR100MoE, load_cifar100_tensors, train_moe,
)
import toy_experiment.collect_dataset as collect_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Random-search baseline: sample N MoE configs, pick best"
    )
    parser.add_argument("--data-dir", type=str, default="./cifar100_data")
    parser.add_argument("--save-results", type=str,
                        default="./runs/results_cifar100_random_moe.json")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n-moe-candidates", type=int, default=10,
                        help="Сколько MoE-конфигураций перебрать")
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

    ss = CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    X_tr, y_tr, X_v, y_v = load_cifar100_tensors(data_dir)
    print(f"[data] num_classes={num_classes}, train={len(X_tr)}, val={len(X_v)}")
    print(f"[search] K={args.K}, n_moe_candidates={args.n_moe_candidates}, "
          f"moe_epochs={args.moe_epochs}")

    best_acc = -1.0
    best_configs = None
    best_index = -1
    history = []

    for cand_i in range(args.n_moe_candidates):
        configs = [
            collect_dataset.sample_valid_config(ss) for _ in range(args.K)
        ]
        moe = CIFAR100MoE(
            configs=configs,
            init_channels=args.init_channels,
            num_classes=num_classes,
            gate_channels=args.gate_channels,
        )

        if cand_i == 0:
            n_params = sum(p.numel() for p in moe.parameters())
            print(f"[moe] params per config: {n_params:,}\n")

        print(f"=== MoE candidate {cand_i+1}/{args.n_moe_candidates} ===")
        for k, cfg in enumerate(configs):
            ops = [cfg.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
            print(f"  expert {k}: ops={ops}")

        t0 = time.time()
        val_acc = train_moe(
            moe, X_tr, y_tr, X_v, y_v,
            epochs=args.moe_epochs,
            batch_size=args.batch_size,
            lr=args.lr, wd=args.wd,
            device=args.device,
            verbose=False,  # подавляем per-epoch вывод, печатаем только итог
        )
        print(f"  → val_acc={val_acc:.4f}  (time={time.time()-t0:.1f}s)")
        history.append({
            "index": cand_i,
            "configs": configs,
            "val_acc": float(val_acc),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_configs = configs
            best_index = cand_i
            print(f"  *** new best ***")
        print()

    print(f"[done] best MoE candidate {best_index+1}/{args.n_moe_candidates}: "
          f"val_acc={best_acc:.4f}")

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "cifar100_random_moe_baseline": {
            "K": args.K,
            "best_val_acc": float(best_acc),
            "best_index": best_index,
            "best_configs": best_configs,
            "n_moe_candidates": args.n_moe_candidates,
            "moe_epochs": args.moe_epochs,
            "history": history,
        }
    }
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] results -> {save_path}")


if __name__ == "__main__":
    main()

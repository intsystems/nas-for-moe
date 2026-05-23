"""Random-single-arch-baseline: сэмплирование N одиночных архитектур, выбор лучшей.

Этапы:
    1. Сэмплируется N архитектур (без MoE).
    2. Каждая обучается end-to-end на полном train-датасете (30 эпох).
    3. Выбирается лучшая по val accuracy.

Это нижний бейзлайн без MoE и без архитектурной специализации — для сравнения
с random_moe_learnable (где K архитектур объединяются через learnable gating).

Запуск:
    python cifar100_random_single_arch_baseline.py --device cuda:0 \\
        --data-dir ./cifar100_data_semantic_testsplit --n-arch-candidates 50 --epochs 30
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

import cifar100_searchspace  # noqa: E402
cifar100_searchspace.patch_toy_graph_ops()

from cifar100_sgem import (  # noqa: E402
    CIFAR100DartsSearchSpace, CIFAR100Net, _train_cifar100_net,
    load_cifar100_meta,
)
import toy_experiment.collect_dataset as collect_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Random-search baseline: sample N single architectures, pick best"
    )
    parser.add_argument("--data-dir", type=str,
                        default="./cifar100_data_semantic_testsplit")
    parser.add_argument("--save-results", type=str,
                        default="./runs_testsplit/results_cifar100_random_single_arch.json")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="JSON с предыдущей историей. Кандидаты дополняются "
                             "до --n-arch-candidates")
    parser.add_argument("--n-arch-candidates", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)
    num_classes = meta["num_classes"]

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_v, y_v = X[val_idx], y[val_idx]

    ss = CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    print(f"[data] num_classes={num_classes}, train={len(X_tr)}, val={len(X_v)}")
    print(f"[search] n_arch_candidates={args.n_arch_candidates}, "
          f"epochs={args.epochs}")

    best_acc = -1.0
    best_config = None
    best_index = -1
    history: list = []
    start_index = 0

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            prev_run = prev.get("cifar100_random_single_arch_baseline", {})
            history = list(prev_run.get("history", []))
            if history:
                start_index = max(h["index"] for h in history) + 1
                best_h = max(history, key=lambda h: h["val_acc"])
                best_acc = best_h["val_acc"]
                best_config = best_h["config"]
                best_index = best_h["index"]
                print(f"[resume] loaded {len(history)} prior candidates from "
                      f"{resume_path}, best so far = {best_acc:.4f} "
                      f"(index {best_index}). Continuing from index {start_index}.")
        else:
            print(f"[resume] {resume_path} does not exist, starting fresh.")

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_results():
        result = {
            "cifar100_random_single_arch_baseline": {
                "best_val_acc": float(best_acc),
                "best_index": best_index,
                "best_config": best_config,
                "n_arch_candidates": args.n_arch_candidates,
                "epochs": args.epochs,
                "n_done": len(history),
                "history": history,
            }
        }
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        tmp_path.replace(save_path)

    for cand_i in range(start_index, args.n_arch_candidates):
        config = collect_dataset.sample_valid_config(ss)
        net = CIFAR100Net(config, C=args.init_channels, num_classes=num_classes)

        if cand_i == start_index:
            n_params = sum(p.numel() for p in net.parameters())
            print(f"[net] params per arch: {n_params:,}\n")

        ops = [config.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
        print(f"=== arch candidate {cand_i+1}/{args.n_arch_candidates} ===")
        print(f"  ops={ops}")

        t0 = time.time()
        val_acc = _train_cifar100_net(
            net, X_tr, y_tr, X_v, y_v,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            device=args.device,
        )
        print(f"  → val_acc={val_acc:.4f}  (time={time.time()-t0:.1f}s)")
        history.append({
            "index": cand_i,
            "config": config,
            "val_acc": float(val_acc),
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config
            best_index = cand_i
            print(f"  *** new best ***")

        _write_results()
        print()

    print(f"[done] best arch index={best_index} "
          f"(out of {args.n_arch_candidates}): val_acc={best_acc:.4f}")
    _write_results()
    print(f"[save] results -> {save_path}")


if __name__ == "__main__":
    main()

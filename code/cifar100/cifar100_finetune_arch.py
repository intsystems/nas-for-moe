"""Дообучение одиночной архитектуры на CIFAR-100 (без MoE).

Принимает JSON с полем `config` (см. результат `cifar100_darts_search.py`
или `cifar100_random_single_arch_baseline.py`) и тренирует `CIFAR100Net`
с нуля как обычный классификатор: обучение на train ∪ val, единственная
оценка на отложенном test после последней эпохи (метрика — `test_acc`).

Запуск:
    python cifar100_finetune_arch.py \\
        --config-json ./runs_testsplit/results_cifar100_darts_search.json \\
        --data-dir ./cifar100_data_semantic_testsplit \\
        --epochs 100 --device cuda:1 \\
        --save-results ./runs_testsplit/results_cifar100_darts_finetune.json
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
    CIFAR100Net, _train_cifar100_net, load_cifar100_meta,
)


def _extract_config(doc: dict) -> dict:
    """Вытащить config из JSON. Поддерживает несколько ключей-обёрток."""
    for key in (
        "cifar100_darts_search",
        "cifar100_random_single_arch_baseline",
        "cifar100_sgem",
    ):
        if key in doc and isinstance(doc[key], dict):
            section = doc[key]
            if "config" in section:
                return section["config"]
            if "best_config" in section:
                return section["best_config"]
    if "config" in doc:
        return doc["config"]
    raise ValueError(f"Не нашёл 'config' в JSON. Ключи: {list(doc.keys())}")


def main():
    ap = argparse.ArgumentParser(
        description="Train a single-arch CIFAR-100 network from scratch",
    )
    ap.add_argument("--config-json", type=str, required=True,
                    help="JSON с полем config / best_config")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--save-results", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--init-channels", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tag", type=str, default="darts_finetune",
                    help="Тег для top-level ключа результата")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config_json) as f:
        doc = json.load(f)
    config = _extract_config(doc)

    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)
    num_classes = int(meta["num_classes"])

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    tr = np.load(data_dir / "train_indices.npy")
    v = np.load(data_dir / "val_indices.npy")
    te = np.load(data_dir / "test_indices.npy")
    trainval = np.concatenate([tr, v])
    X_trv, y_trv = X[trainval], y[trainval]
    X_te, y_te = X[te], y[te]
    print(f"[data] num_classes={num_classes}, train∪val={len(X_trv)}, test={len(X_te)}")

    ops = [config.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
    inputs = [config.get(f"input_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
    print(f"[arch] ops={ops}")
    print(f"[arch] inputs={inputs}")

    net = CIFAR100Net(config, C=args.init_channels, num_classes=num_classes)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[net] params: {n_params:,}")
    print(f"[train] epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, device={args.device}")

    t0 = time.time()
    test_acc = _train_cifar100_net(
        net, X_trv, y_trv, X_te, y_te,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        device=args.device,
    )
    elapsed = time.time() - t0
    print(f"[done] test_acc={test_acc:.4f}  time={elapsed/60:.2f} min")

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        args.tag: {
            "config": config,
            "test_acc": float(test_acc),
            "n_trainval": int(len(X_trv)),
            "n_test": int(len(X_te)),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "init_channels": args.init_channels,
            "n_params": int(n_params),
            "data_dir": str(data_dir),
            "config_source": str(args.config_json),
            "time_sec": float(elapsed),
            "seed": args.seed,
            "device": args.device,
        }
    }
    tmp = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    tmp.replace(save_path)
    print(f"[save] -> {save_path}")


if __name__ == "__main__":
    main()

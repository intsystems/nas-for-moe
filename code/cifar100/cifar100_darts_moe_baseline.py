"""DARTS×K baseline: MoE из K копий одной DARTS-архитектуры.

Берёт config из JSON-результата DARTS-finetune (или DARTS-search) и собирает
MoE с K одинаковыми экспертами. Обучает в режимах learnable и/или cluster
(те же гиперпараметры, что используются для retrain best в random/SGEM:
100 эпох, lr=0.05, wd=3e-4, batch=128, init_channels=16, gate_channels=16).

Цель — понять, нужен ли поиск гетерогенных архитектур экспертов, или
достаточно K раз положить одну DARTS-архитектуру.

Запуск:
    python cifar100_darts_moe_baseline.py \\
        --darts-results ./runs_testsplit/results_cifar100_darts_finetune.json \\
        --data-dir ./cifar100_data_semantic_testsplit \\
        --K 3 --epochs 100 --device cuda:0 \\
        --save-results ./runs_testsplit/results_cifar100_darts_moe_K3.json
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

from cifar100_sgem import load_cifar100_meta  # noqa: E402


def _json_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    return repr(v)
from cifar100_final_train import train_final_moe  # noqa: E402


def _load_darts_config(path: Path) -> dict:
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{path} не похож на JSON DARTS-результата")
    top_key = next(iter(doc))
    section = doc[top_key]
    cfg = section.get("config")
    if cfg is None:
        raise ValueError(
            f"В {path} нет поля '{top_key}.config'. Ключи: {list(section.keys())}"
        )
    return cfg


def _balanced_random_assignments(M: int, K: int, seed: int) -> list[int]:
    """Сбалансированное случайное разбиение M кластеров на K экспертов."""
    rng = np.random.RandomState(seed)
    base = np.array([m % K for m in range(M)], dtype=np.int64)
    rng.shuffle(base)
    return base.tolist()


def main():
    parser = argparse.ArgumentParser(
        description="DARTS×K baseline: MoE из K копий одной DARTS-архитектуры"
    )
    parser.add_argument("--darts-results", type=str, required=True,
                        help="JSON с DARTS-результатом (config будет извлечён)")
    parser.add_argument("--data-dir", type=str,
                        default="./cifar100_data_semantic_testsplit")
    parser.add_argument("--save-results", type=str,
                        default="./runs_testsplit/results_cifar100_darts_moe.json")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["learnable", "cluster", "both"])
    parser.add_argument("--epochs", type=int, default=100)
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
    M = meta["n_clusters"]

    darts_cfg = _load_darts_config(Path(args.darts_results))
    print(f"[darts] loaded config from {args.darts_results}")
    ops = [darts_cfg.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
    print(f"[darts] ops: {ops}")

    configs = [darts_cfg] * args.K
    hard_assignments = _balanced_random_assignments(M, args.K, args.seed)

    modes = ["learnable", "cluster"] if args.mode == "both" else [args.mode]
    final_moe: dict = {}
    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _write():
        result = {
            "cifar100_darts_moe_baseline": {
                "K": args.K,
                "M": M,
                "darts_config": darts_cfg,
                "darts_results_source": args.darts_results,
                "hard_assignments": hard_assignments,
                "final_moe": final_moe,
                "args": {k: _json_safe(v) for k, v in vars(args).items()},
            }
        }
        tmp = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        tmp.replace(save_path)

    for mode in modes:
        print(f"\n=== final training: mode={mode}, K={args.K}, "
              f"epochs={args.epochs} (train ∪ val → test) ===")
        t0 = time.time()
        info = train_final_moe(
            configs=configs,
            hard_assignments=hard_assignments,
            data_dir=data_dir,
            mode=mode,
            final=True,
            init_channels=args.init_channels,
            num_classes=num_classes,
            gate_channels=args.gate_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr, wd=args.wd,
            seed=args.seed,
            device=args.device,
            verbose=False,
        )
        print(f"[final-moe/{mode}] test_acc={info['test_acc']:.4f} "
              f"(time={time.time()-t0:.1f}s)")
        final_moe[mode] = info
        _write()

    print(f"\n[done] results -> {save_path}")
    for mode, info in final_moe.items():
        print(f"  {mode:>10s}: test_acc={info['test_acc']:.4f}")


if __name__ == "__main__":
    main()

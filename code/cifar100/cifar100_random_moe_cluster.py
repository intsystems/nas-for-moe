"""Random-MoE с cluster-gating.

Этапы (на одну итерацию):
    1. Случайно сэмплируется K архитектур экспертов.
    2. Случайно сэмплируется hard_assignments длины M (cluster_id → expert_id),
       причём у каждого эксперта ≥ `--min-clusters-per-expert` кластеров
       (по умолчанию авто = M//K − 3, но ≥ 1) — перевыбираем, пока условие
       не выполнено.
    3. MoE обучается на train с фиксированным cluster-gating'ом
       (ClusterGatedMoE), `--moe-epochs` эпох.
    4. Из истории выбирается лучшая конфигурация по val accuracy.

После завершения random-search'а лучший MoE-кандидат обучается заново на
train ∪ val на `--final-epochs` эпох (по умолчанию 100) и единственный раз
оценивается на отложенном test. Результат пишется в JSON под ключом
`final_moe` (метрика — `test_acc`). `--final-epochs 0` отключает этот этап.

Бейзлайн без архитектурной специализации по кластерам — для сравнения с SGEM
(который ищет совместно архитектуры + назначение кластеров).

Запуск:
    python cifar100_random_moe_cluster.py --device cuda:0 \\
        --data-dir ./cifar100_data_semantic_testsplit --K 3 --n-moe-candidates 200 \\
        --moe-epochs 30 --final-epochs 100 \\
        --save-results ./runs_testsplit/results_cifar100_random_moe_cluster.json
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

from cifar100_sgem import CIFAR100DartsSearchSpace, load_cifar100_meta, _json_safe  # noqa: E402
from cifar100_final_train import train_final_moe  # noqa: E402  (used inside loop and in final retrain)
import toy_experiment.collect_dataset as collect_dataset  # noqa: E402


def _detect_M(data_dir: Path) -> int:
    centers_path = data_dir / "cluster_centers.npy"
    if centers_path.exists():
        return int(np.load(centers_path).shape[0])
    cids = np.load(data_dir / "train_cluster_ids.npy")
    return int(cids.max()) + 1


def _auto_min_per_expert(M: int, K: int) -> int:
    """Эвристика «у каждого эксперта ≥ M/K − 3 кластеров» (минимум 1)."""
    return max(1, M // K - 3)


def _sample_hard_assignments(
    M: int, K: int, rng: random.Random, min_per_expert: int = 1,
) -> list[int]:
    """Случайное cluster→expert назначение, у каждого эксперта ≥ min_per_expert кластеров."""
    if K * min_per_expert > M:
        raise ValueError(
            f"K={K} · min_per_expert={min_per_expert} > M={M}: "
            f"невозможно выдать ≥{min_per_expert} кластеров каждому из K экспертов"
        )
    while True:
        ha = [rng.randrange(K) for _ in range(M)]
        counts = [ha.count(k) for k in range(K)]
        if min(counts) >= min_per_expert:
            return ha


def main():
    parser = argparse.ArgumentParser(
        description="Random-search baseline (cluster gating): "
                    "sample N (configs, hard_assignments) pairs, pick best",
    )
    parser.add_argument("--data-dir", type=str,
                        default="./cifar100_data_semantic_testsplit")
    parser.add_argument("--save-results", type=str,
                        default="./runs_testsplit/results_cifar100_random_moe_cluster.json")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="JSON с предыдущей историей. Кандидаты дополняются "
                             "до --n-moe-candidates")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n-moe-candidates", type=int, default=10,
                        help="Целевое суммарное число MoE-конфигураций "
                             "(включая загруженные через --resume-from)")
    parser.add_argument("--moe-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=100,
                        help="Эпохи финального retrain'а лучшего MoE-кандидата "
                             "(обучение на train ∪ val, оценка на test). "
                             "0 = пропустить.")
    parser.add_argument("--min-clusters-per-expert", type=int, default=-1,
                        help="Минимум кластеров на каждого эксперта при "
                             "случайном hard_assignment. -1 = авто (M//K − 3, "
                             "но ≥ 1).")
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)
    num_classes = meta["num_classes"]
    M = _detect_M(data_dir)
    min_per_expert = (
        args.min_clusters_per_expert if args.min_clusters_per_expert >= 1
        else _auto_min_per_expert(M, args.K)
    )
    print(f"[data] num_classes={num_classes}, M={M} clusters")
    print(f"[search] K={args.K}, n_moe_candidates={args.n_moe_candidates}, "
          f"moe_epochs={args.moe_epochs}, min_clusters_per_expert={min_per_expert}")

    ss = CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    best_acc = -1.0
    best_configs = None
    best_hard_assignments = None
    best_index = -1
    history: list = []
    start_index = 0

    if args.resume_from is not None:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            with open(resume_path) as f:
                prev = json.load(f)
            prev_run = prev.get("cifar100_random_moe_cluster", {})
            history = list(prev_run.get("history", []))
            if history:
                start_index = max(h["index"] for h in history) + 1
                best_h = max(history, key=lambda h: h["val_acc"])
                best_acc = best_h["val_acc"]
                best_configs = best_h["configs"]
                best_hard_assignments = best_h["hard_assignments"]
                best_index = best_h["index"]
                print(f"[resume] loaded {len(history)} prior candidates from "
                      f"{resume_path}, best so far = {best_acc:.4f} "
                      f"(index {best_index}). Continuing from index {start_index}.")
            # Прогреем rng так, чтобы прошлые сэмплы не повторялись
            for _ in range(start_index):
                _ = [collect_dataset.sample_valid_config(ss) for _ in range(args.K)]
                _ = _sample_hard_assignments(M, args.K, rng, min_per_expert)
        else:
            print(f"[resume] {resume_path} does not exist, starting fresh.")

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    final_moe_info: dict | None = None
    if args.resume_from is not None and Path(args.resume_from).exists():
        prev_run = json.load(open(args.resume_from)).get(
            "cifar100_random_moe_cluster", {}
        )
        final_moe_info = prev_run.get("final_moe")

    def _write_results():
        result = {
            "cifar100_random_moe_cluster": {
                "K": args.K,
                "M": M,
                "best_val_acc": float(best_acc),
                "best_index": best_index,
                "best_configs": best_configs,
                "best_hard_assignments": best_hard_assignments,
                "n_moe_candidates": args.n_moe_candidates,
                "moe_epochs": args.moe_epochs,
                "n_done": len(history),
                "history": history,
                "final_moe": final_moe_info,
                "args": {k: _json_safe(v) for k, v in vars(args).items()},
            }
        }
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=2)
        tmp_path.replace(save_path)

    for cand_i in range(start_index, args.n_moe_candidates):
        configs = [
            collect_dataset.sample_valid_config(ss) for _ in range(args.K)
        ]
        hard_assignments = _sample_hard_assignments(M, args.K, rng, min_per_expert)

        print(f"=== MoE candidate {cand_i+1}/{args.n_moe_candidates} ===")
        for k, cfg in enumerate(configs):
            ops = [cfg.get(f"op_{j}", "?") for j in [0, 1, 3, 4, 6, 7]]
            cs = [m for m, e in enumerate(hard_assignments) if e == k]
            print(f"  expert {k}: ops={ops} clusters={cs}")

        t0 = time.time()
        info = train_final_moe(
            configs=configs,
            hard_assignments=hard_assignments,
            data_dir=data_dir,
            mode="cluster",
            init_channels=args.init_channels,
            num_classes=num_classes,
            epochs=args.moe_epochs,
            batch_size=args.batch_size,
            lr=args.lr, wd=args.wd,
            seed=args.seed,
            device=args.device,
            verbose=False,
        )
        val_acc = float(info["val_acc"])
        print(f"  → val_acc={val_acc:.4f}  (time={time.time()-t0:.1f}s)")
        history.append({
            "index": cand_i,
            "configs": configs,
            "hard_assignments": hard_assignments,
            "val_acc": val_acc,
            "time_sec": info["time_sec"],
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_configs = configs
            best_hard_assignments = hard_assignments
            best_index = cand_i
            print(f"  *** new best ***")

        _write_results()
        print()

    print(f"[done] best MoE candidate index={best_index} "
          f"(out of {args.n_moe_candidates}): val_acc={best_acc:.4f}")
    _write_results()
    print(f"[save] results -> {save_path}")

    if args.final_epochs > 0 and best_configs is not None:
        print(f"\n=== Final retrain of best MoE (index {best_index}) "
              f"on train ∪ val for {args.final_epochs} epochs → test ===")
        t0 = time.time()
        info = train_final_moe(
            configs=best_configs,
            hard_assignments=best_hard_assignments,
            data_dir=data_dir,
            mode="cluster",
            final=True,
            init_channels=args.init_channels,
            num_classes=num_classes,
            epochs=args.final_epochs,
            batch_size=args.batch_size,
            lr=args.lr, wd=args.wd,
            seed=args.seed,
            device=args.device,
            verbose=False,
        )
        info["best_index"] = best_index
        info["search_val_acc"] = float(best_acc)
        final_moe_info = info
        print(f"[final-moe] test_acc={info['test_acc']:.4f} "
              f"(search val_acc={best_acc:.4f}, "
              f"time={time.time()-t0:.1f}s)")
        _write_results()
        print(f"[save] results (with final_moe) -> {save_path}")


if __name__ == "__main__":
    main()

"""Per-expert true val_acc для SGEM-результата.

Для каждого эксперта k:
  - Берёт его архитектуру α_k из result['configs'] и кластеры
    R_k = {m : hard_assignments[m] == k}.
  - Обучает α_k реально (CIFAR100Net) на train-точках, чьи кластеры ∈ R_k.
  - Считает val_acc на val-точках с кластерами ∈ R_k.

Это «оракульный» аналог суррогатного objective: тот же argmax по экспертам,
но без шума суррогата.

Агрегаты:
  - mean_val_acc = средний val_acc по экспертам
  - weighted_val_acc = Σ_k (|val_k| / |val|) · val_acc_k
        (≈ val_acc MoE с идеальным кластерным gating'ом)
  - true_objective = Σ_k |R_k| · log(val_acc_k)
        (несмещённая версия SGEM-objective Σ_m log u(α_{ha[m]}, R_{ha[m]}))

Запуск:
    python eval_sgem_per_expert.py \\
        --results-json ./runs_testsplit/results_cifar100_sgem.json \\
        --data-dir ./cifar100_data_semantic_testsplit --epochs 30 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
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


def evaluate_per_expert(
    configs: list,
    hard_assignments: list,
    data_dir: Path,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    init_channels: int = 16,
    device: str = "cuda:0",
    verbose: bool = True,
) -> dict:
    """Обучает каждую архитектуру α_k на её кластерах R_k, считает val_acc.

    Возвращает dict с per_expert/aggregates (true_objective, weighted_val_acc, ...).
    Используется и из CLI, и из колбека в cifar100_sgem.
    """
    K = len(configs)
    M = len(hard_assignments)

    meta = load_cifar100_meta(data_dir)
    num_classes = meta["num_classes"]

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    train_cluster_ids = np.load(data_dir / "train_cluster_ids.npy")
    val_cluster_ids = np.load(data_dir / "val_cluster_ids.npy")

    X_tr_all, y_tr_all = X[train_idx], y[train_idx]
    X_v_all, y_v_all = X[val_idx], y[val_idx]
    n_val_total = len(X_v_all)

    per_expert: list[dict] = []
    for k in range(K):
        clusters_k = sorted([m for m, a in enumerate(hard_assignments) if a == k])
        if not clusters_k:
            if verbose:
                print(f"  [eval] expert {k}: NO clusters, skip")
            per_expert.append({
                "expert": k, "clusters": [], "n_train": 0, "n_val": 0,
                "val_acc": None, "time_sec": 0.0,
            })
            continue

        tr_mask = np.isin(train_cluster_ids, clusters_k)
        v_mask = np.isin(val_cluster_ids, clusters_k)
        X_tr_k, y_tr_k = X_tr_all[tr_mask], y_tr_all[tr_mask]
        X_v_k, y_v_k = X_v_all[v_mask], y_v_all[v_mask]

        net = CIFAR100Net(configs[k], C=init_channels, num_classes=num_classes)
        t0 = time.time()
        val_acc = _train_cifar100_net(
            net, X_tr_k, y_tr_k, X_v_k, y_v_k,
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"  [eval] expert {k}: |R_k|={len(clusters_k)} "
                  f"train={len(X_tr_k)} val={len(X_v_k)} "
                  f"val_acc={val_acc:.4f} ({elapsed:.1f}s)")
        per_expert.append({
            "expert": k, "clusters": clusters_k,
            "n_train": int(len(X_tr_k)), "n_val": int(len(X_v_k)),
            "val_acc": float(val_acc), "time_sec": float(elapsed),
        })

    valid = [r for r in per_expert if r["val_acc"] is not None]
    accs = [r["val_acc"] for r in valid]
    mean_val_acc = sum(accs) / len(accs) if accs else 0.0
    weighted_val_acc = (
        sum(r["val_acc"] * r["n_val"] for r in valid) / n_val_total
        if n_val_total > 0 else 0.0
    )
    true_objective = sum(
        len(r["clusters"]) * math.log(max(r["val_acc"], 1e-12)) for r in valid
    )
    K_eff = (
        math.exp(-sum(
            (len(r["clusters"]) / M) * math.log(len(r["clusters"]) / M)
            for r in per_expert if len(r["clusters"]) > 0
        ))
        if M > 0 else 0.0
    )

    return {
        "per_expert": per_expert,
        "mean_val_acc": float(mean_val_acc),
        "weighted_val_acc": float(weighted_val_acc),
        "true_objective": float(true_objective),
        "K_used": len(valid),
        "K_total": int(K),
        "K_eff": float(K_eff),
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "init_channels": init_channels,
    }


def _load_sgem_section(path: Path) -> tuple[dict, dict, str]:
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{path} не похож на SGEM JSON")
    top_key = next(iter(doc))
    section = doc[top_key]
    for required in ("configs", "hard_assignments"):
        if required not in section:
            raise ValueError(
                f"В JSON нет '{top_key}.{required}'. Ключи: {list(section.keys())}"
            )
    return doc, section, top_key


def main():
    parser = argparse.ArgumentParser(description="Per-expert true val_acc для SGEM")
    parser.add_argument("--results-json", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--update-json", action="store_true",
                        help="Дописать результаты в --results-json под "
                             "ключом per_expert_eval")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_json = Path(args.results_json)
    data_dir = Path(args.data_dir)

    doc, section, top_key = _load_sgem_section(results_json)
    configs = section["configs"]
    hard = list(section["hard_assignments"])
    print(f"[load] {results_json.name} → top_key='{top_key}', "
          f"K={len(configs)}, M={len(hard)}")

    eval_block = evaluate_per_expert(
        configs=configs,
        hard_assignments=hard,
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        init_channels=args.init_channels,
        device=args.device,
        verbose=True,
    )
    eval_block["seed"] = args.seed

    print("\n=== summary ===")
    print(f"  K_used={eval_block['K_used']}/{eval_block['K_total']}  "
          f"K_eff={eval_block['K_eff']:.2f}")
    print(f"  mean_val_acc      = {eval_block['mean_val_acc']:.4f}")
    print(f"  weighted_val_acc  = {eval_block['weighted_val_acc']:.4f}")
    print(f"  true_objective    = {eval_block['true_objective']:.4f}  "
          f"(surrogate objective: {section.get('objective_value')})")

    if args.update_json:
        with open(results_json) as f:
            saved = json.load(f)
        saved[top_key]["per_expert_eval"] = eval_block
        tmp = results_json.with_suffix(results_json.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(saved, f, indent=2)
        tmp.replace(results_json)
        print(f"[save] обновлён {results_json}")


if __name__ == "__main__":
    main()

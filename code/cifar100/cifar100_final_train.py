"""Финальное обучение MoE на K архитектурах, найденных SGEM.

Поддерживается два варианта gating:

    1. learnable — обучаемый softmax-gating (CIFAR100MoE из cifar100_moe.py).
       Эквивалентно бейзлайну cifar100_random_moe_baseline.py — даёт прямое
       сравнение «архитектуры от SGEM vs случайные» при одинаковом gating.

    2. cluster   — жёсткий gating по hard_assignments из r_matrix:
       каждая точка маршрутизируется к одному эксперту согласно своему
       кластеру. Cluster IDs берутся из train_cluster_ids.npy /
       val_cluster_ids.npy (они уже посчитаны KMeans в PCA-пространстве
       prepare_cifar100.py).

Использование как библиотеки:
    from cifar100_final_train import train_final_moe
    res = train_final_moe(
        configs, hard_assignments, data_dir,
        mode="cluster", epochs=30, device="cuda:0",
    )

CLI (обучить MoE по уже сохранённому JSON-у SGEM):
    python cifar100_final_train.py \\
        --results-json ./runs/results_cifar100_sgem_K3_lb000_mix050.json \\
        --data-dir ./cifar100_data \\
        --mode both --epochs 30 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import cifar100_searchspace  # noqa: E402
cifar100_searchspace.patch_toy_graph_ops()

from cifar100_sgem import CIFAR100Net, CIFAR100_MEAN, CIFAR100_STD  # noqa: E402
from cifar100_moe import (  # noqa: E402
    CIFAR100MoE, load_cifar100_tensors, train_moe,
)


# ==========================================================================
# Cluster-gated MoE: жёсткое назначение эксперта по cluster_id
# ==========================================================================


class ClusterGatedMoE(nn.Module):
    """MoE с фиксированным gating'ом по hard_assignments из r_matrix.

    Args:
        configs: K dict-конфигов архитектур экспертов.
        hard_assignments: список длины M, hard_assignments[m] = индекс эксперта
            для кластера m (значения в [0, K)).
        init_channels: каналы в экспертах.
        num_classes: число классов.
    """

    def __init__(
        self,
        configs: List[dict],
        hard_assignments: List[int],
        init_channels: int = 16,
        num_classes: int = 100,
    ):
        super().__init__()
        self.K = len(configs)
        self.M = len(hard_assignments)
        self.configs = configs
        self.experts = nn.ModuleList([
            CIFAR100Net(cfg, C=init_channels, num_classes=num_classes)
            for cfg in configs
        ])
        # cluster_id → expert_id (long buffer, чтобы переезжал на device)
        self.register_buffer(
            "cluster_to_expert",
            torch.as_tensor(hard_assignments, dtype=torch.long),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 32, 32]; cluster_ids: [B] long → [B, num_classes]."""
        expert_ids = self.cluster_to_expert[cluster_ids]            # [B]
        out = x.new_zeros((x.size(0), self.num_classes))
        for k, expert in enumerate(self.experts):
            mask = (expert_ids == k)
            if not mask.any():
                continue
            out[mask] = expert(x[mask])
        return out


# ==========================================================================
# Загрузка данных + cluster_ids (для cluster-gated режима)
# ==========================================================================


def load_cifar100_tensors_with_clusters(data_dir: Path):
    """То же, что load_cifar100_tensors, но возвращает и cluster_ids."""
    data_dir = Path(data_dir)
    X = np.load(data_dir / "data_X.npy")               # uint8 [N, 3, 32, 32]
    y = np.load(data_dir / "data_y.npy").astype(np.int64)
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    train_cl = np.load(data_dir / "train_cluster_ids.npy").astype(np.int64)
    val_cl = np.load(data_dir / "val_cluster_ids.npy").astype(np.int64)

    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)

    def to_tensor(idx):
        xb = torch.from_numpy(X[idx]).float().div_(255.0).sub_(mean).div_(std)
        yb = torch.from_numpy(y[idx])
        return xb, yb

    X_tr, y_tr = to_tensor(train_idx)
    X_v, y_v = to_tensor(val_idx)
    c_tr = torch.from_numpy(train_cl)
    c_v = torch.from_numpy(val_cl)
    return X_tr, y_tr, c_tr, X_v, y_v, c_v


# ==========================================================================
# Тренировка cluster-gated MoE
# ==========================================================================


@torch.no_grad()
def _evaluate_cluster_moe(
    moe: ClusterGatedMoE, X, y, c, batch_size: int, device: str,
) -> tuple[float, np.ndarray]:
    moe.eval()
    correct = 0
    total = 0
    expert_count = torch.zeros(moe.K, device=device)
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        cb = c[i:i + batch_size].to(device)
        logits = moe(xb, cb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
        eids = moe.cluster_to_expert[cb]
        for k in range(moe.K):
            expert_count[k] += (eids == k).sum()
    return correct / max(1, total), (expert_count / max(1, total)).cpu().numpy()


def train_cluster_gated_moe(
    moe: ClusterGatedMoE,
    X_tr, y_tr, c_tr, X_v, y_v, c_v,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    wd: float = 3e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> float:
    """Обучить cluster-gated MoE; вернуть лучший val-accuracy."""
    moe.to(device)
    optimizer = torch.optim.SGD(
        moe.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr, c_tr),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        moe.train()
        t0 = time.time()
        running, n = 0.0, 0
        for xb, yb, cb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            cb = cb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = moe(xb, cb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(moe.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
        scheduler.step()

        val_acc, expert_share = _evaluate_cluster_moe(
            moe, X_v, y_v, c_v, batch_size, device,
        )
        if val_acc > best_val:
            best_val = val_acc
        if verbose:
            print(f"  [epoch {epoch:02d}/{epochs}] "
                  f"train_loss={running/n:.4f} val_acc={val_acc:.4f} "
                  f"expert_share={np.round(expert_share, 2).tolist()} "
                  f"time={time.time()-t0:.1f}s")
    return best_val


# ==========================================================================
# High-level: обучить финальный MoE в одном из режимов
# ==========================================================================


def train_final_moe(
    configs: List[dict],
    hard_assignments: Optional[List[int]],
    data_dir: Path,
    *,
    mode: str = "learnable",          # "learnable" | "cluster"
    init_channels: int = 16,
    num_classes: int = 100,
    gate_channels: int = 16,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    wd: float = 3e-4,
    seed: int = 322,
    device: str = "cuda:0",
    verbose: bool = True,
) -> dict:
    """Обучить финальный MoE и вернуть dict с метриками."""
    if mode not in ("learnable", "cluster"):
        raise ValueError(f"Unknown mode: {mode}")
    if mode == "cluster" and hard_assignments is None:
        raise ValueError("mode='cluster' требует hard_assignments")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dir = Path(data_dir)

    if mode == "learnable":
        X_tr, y_tr, X_v, y_v = load_cifar100_tensors(data_dir)
        moe = CIFAR100MoE(
            configs=configs,
            init_channels=init_channels,
            num_classes=num_classes,
            gate_channels=gate_channels,
        )
        n_params = sum(p.numel() for p in moe.parameters())
        print(f"[final-moe/{mode}] params: {n_params:,}")
        t0 = time.time()
        val_acc = train_moe(
            moe, X_tr, y_tr, X_v, y_v,
            epochs=epochs, batch_size=batch_size,
            lr=lr, wd=wd, device=device, verbose=verbose,
        )
    else:  # cluster
        X_tr, y_tr, c_tr, X_v, y_v, c_v = (
            load_cifar100_tensors_with_clusters(data_dir)
        )
        moe = ClusterGatedMoE(
            configs=configs,
            hard_assignments=hard_assignments,
            init_channels=init_channels,
            num_classes=num_classes,
        )
        n_params = sum(p.numel() for p in moe.parameters())
        print(f"[final-moe/{mode}] params: {n_params:,} "
              f"(M={moe.M}, K={moe.K})")
        from collections import Counter
        share = Counter(int(a) for a in hard_assignments)
        print(f"[final-moe/{mode}] cluster→expert counts: "
              f"{dict(sorted(share.items()))}")
        t0 = time.time()
        val_acc = train_cluster_gated_moe(
            moe, X_tr, y_tr, c_tr, X_v, y_v, c_v,
            epochs=epochs, batch_size=batch_size,
            lr=lr, wd=wd, device=device, verbose=verbose,
        )

    elapsed = time.time() - t0
    print(f"[final-moe/{mode}] best val_acc = {val_acc:.4f} "
          f"(time={elapsed:.1f}s)")
    return {
        "mode": mode,
        "val_acc": float(val_acc),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "wd": wd,
        "init_channels": init_channels,
        "gate_channels": gate_channels if mode == "learnable" else None,
        "K": len(configs),
        "seed": seed,
        "time_sec": elapsed,
    }


# ==========================================================================
# CLI: обучить MoE по уже сохранённому SGEM-результату
# ==========================================================================


def _load_sgem_result(results_json: Path) -> tuple[dict, dict, str]:
    """Прочитать JSON, вернуть (full_doc, sgem_section, top_key)."""
    with open(results_json) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{results_json} не похож на SGEM JSON")
    top_key = next(iter(doc))
    section = doc[top_key]
    if "configs" not in section:
        raise ValueError(
            f"В JSON нет поля '{top_key}.configs'. Ключи: {list(section.keys())}"
        )
    return doc, section, top_key


def main():
    parser = argparse.ArgumentParser(
        description="Финальное обучение MoE на конфигах из SGEM-результата"
    )
    parser.add_argument("--results-json", type=str, required=True,
                        help="Путь к JSON с результатом SGEM")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Подготовленная директория CIFAR-100")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["learnable", "cluster", "both"],
                        help="Какой gating обучать")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=3e-4)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--gate-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--update-json", action="store_true",
                        help="Дописать результаты обратно в --results-json "
                             "под ключом final_moe.{learnable,cluster}")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Если задано, сохранить отдельный JSON только "
                             "с финальными метриками")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    data_dir = Path(args.data_dir)

    doc, section, top_key = _load_sgem_result(results_json)
    configs = section["configs"]
    hard_assignments = section.get("hard_assignments")
    print(f"[load] {results_json} → top_key='{top_key}', K={len(configs)}, "
          f"M={len(hard_assignments) if hard_assignments else '?'}")

    # Число классов берём из meta.json данных
    with open(data_dir / "meta.json") as f:
        num_classes = json.load(f)["num_classes"]
    print(f"[data] num_classes={num_classes}")

    modes = ["learnable", "cluster"] if args.mode == "both" else [args.mode]
    if "cluster" in modes and hard_assignments is None:
        raise ValueError(
            "В JSON отсутствует hard_assignments — режим 'cluster' недоступен"
        )

    final_results: dict = section.get("final_moe", {}) or {}
    if not isinstance(final_results, dict):
        final_results = {}

    for mode in modes:
        print(f"\n=== final training: mode={mode} ===")
        r = train_final_moe(
            configs=configs,
            hard_assignments=hard_assignments,
            data_dir=data_dir,
            mode=mode,
            init_channels=args.init_channels,
            num_classes=num_classes,
            gate_channels=args.gate_channels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr, wd=args.wd,
            seed=args.seed, device=args.device,
            verbose=True,
        )
        final_results[mode] = r

    print("\n=== summary ===")
    for mode, r in final_results.items():
        print(f"  {mode:>10s}: val_acc = {r['val_acc']:.4f}")

    if args.update_json:
        section["final_moe"] = final_results
        doc[top_key] = section
        tmp = results_json.with_suffix(results_json.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, indent=2)
        tmp.replace(results_json)
        print(f"[save] обновлён {results_json} (final_moe.{{{','.join(final_results)}}})")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "source_json": str(results_json),
                "data_dir": str(data_dir),
                "final_moe": final_results,
            }, f, indent=2)
        print(f"[save] финальные метрики → {out_path}")


if __name__ == "__main__":
    main()

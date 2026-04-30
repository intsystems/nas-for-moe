"""
Surrogate-EM (S-шаг) алгоритм на подмножестве CIFAR-100
с упрощённым DARTS-подобным пространством.

Конфигурация (по умолчанию):
    M = 20 кластеров (PCA-50 + KMeans на выбранных классах CIFAR-100)
    K = 5 экспертов
    Search space: 3 промежуточных узла × 2 входа, 8 DARTS-операций
    Сеть: stem (3x3 conv, 3→C) → 2 normal cell → 1 fixed reduction cell → GAP → FC
    Бюджет:
        n-seed-obs = 50
        n-em-iterations = 50
        n-new-observations = 20 (на каждый EM-шаг)
        surrogate-retrain-every = 1
        cell-train-epochs = 30

Отличия от MNIST-версии:
    - 3-канальный вход (RGB, 32×32)
    - Нормализация CIFAR-100 (mean/std per channel)
    - Данные готовятся отдельно: prepare_cifar100.py

Запуск:
    # 1. Подготовить данные
    python prepare_cifar100.py --output-dir ./cifar100_data --n-classes 20 --fraction 0.5

    # 2. Запустить SGEM
    python cifar100_sgem.py --device cuda:0 --data-dir ./cifar100_data
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# --- Пути ---
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "mnist"))
sys.path.insert(0, str(SCRIPT_DIR))


# ==========================================================================
# 1. PATCH toy_graph.OPS ДО импорта pipeline-модулей
# ==========================================================================

import toy_experiment.toy_graph as toy_graph  # noqa: E402
from cifar100_searchspace import (  # noqa: E402
    DARTS_OPS_SMALL as DARTS_OPS,
    OPS_NEW_SMALL as OPS_NEW,
    make_op,
    patch_toy_graph_ops,
)

patch_toy_graph_ops()


# ==========================================================================
# 2. Импорт pipeline-модулей (используют пропатченный OPS)
# ==========================================================================

import toy_experiment.collect_dataset as collect_dataset  # noqa: E402
import toy_experiment.optimize_expert_assignments as optimize_expert_assignments  # noqa: E402
import optimize_surrogate_em as osem  # noqa: E402
from optimize_surrogate_em import optimize_surrogate_em  # noqa: E402
from toy_experiment.optimize_expert_assignments import print_result  # noqa: E402


# ==========================================================================
# 3. DARTS cell (идентична MNIST-версии)
# ==========================================================================

NODE_EDGE_IDX = {0: (0, 1), 1: (3, 4), 2: (6, 7)}
NODE_AGG_IDX = {0: 2, 1: 5, 2: 8}
NODE_INPUT_CHOICES = {
    0: [-1],
    1: [-1, 2],
    2: [-1, 2, 5],
}


class DartsCell(nn.Module):
    def __init__(self, config: dict, C: int):
        super().__init__()
        self.config = config
        self.op_modules = nn.ModuleDict()
        for edge_idx in [0, 1, 3, 4, 6, 7]:
            op_name = config[f"op_{edge_idx}"]
            self.op_modules[str(edge_idx)] = make_op(op_name, C)
        self.out_proj = nn.Sequential(
            nn.Conv2d(3 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        cache = {-1: x}
        for node_i in (0, 1, 2):
            a_idx, b_idx = NODE_EDGE_IDX[node_i]
            a_src = self.config[f"input_{a_idx}"][0]
            b_src = self.config[f"input_{b_idx}"][0]
            a_out = self.op_modules[str(a_idx)](cache[a_src])
            b_out = self.op_modules[str(b_idx)](cache[b_src])
            cache[NODE_AGG_IDX[node_i]] = a_out + b_out
        concat = torch.cat(
            [cache[NODE_AGG_IDX[0]], cache[NODE_AGG_IDX[1]], cache[NODE_AGG_IDX[2]]],
            dim=1,
        )
        return self.out_proj(concat)


class FixedReductionCell(nn.Module):
    """Фиксированная редукция: stride-2 conv (spatial /2, channels const)."""

    def __init__(self, C: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class CIFAR100Net(nn.Module):
    """Сеть для CIFAR-100: stem (3→C) → 2 normal cell → reduction → GAP → FC."""

    def __init__(self, config: dict, C: int = 16, num_classes: int = 100):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),  # 3 канала (RGB)
            nn.BatchNorm2d(C),
        )
        self.cell1 = DartsCell(config, C)
        self.cell2 = DartsCell(config, C)
        self.reduction = FixedReductionCell(C)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.cell1(x)
        x = self.cell2(x)
        x = self.reduction(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ==========================================================================
# 5. Search space
# ==========================================================================


class CIFAR100DartsSearchSpace:
    OPS = {op: None for op in DARTS_OPS}
    num_nodes_per_cell = 9  # 6 op-вершин + 3 add-вершины

    def __init__(self, init_channels: int = 16):
        self.init_channels = init_channels

    def create_random_config(self, num_nodes=None) -> dict:
        config = {}
        for node_i in (0, 1, 2):
            a_idx, b_idx = NODE_EDGE_IDX[node_i]
            choices = NODE_INPUT_CHOICES[node_i]
            config[f"op_{a_idx}"] = random.choice(DARTS_OPS)
            config[f"op_{b_idx}"] = random.choice(DARTS_OPS)
            config[f"input_{a_idx}"] = [random.choice(choices)]
            config[f"input_{b_idx}"] = [random.choice(choices)]
        for node_i, agg_idx in NODE_AGG_IDX.items():
            a_idx, b_idx = NODE_EDGE_IDX[node_i]
            config[f"op_{agg_idx}"] = "add"
            config[f"input_{agg_idx}"] = [a_idx, b_idx]
        return config

    def create_cell_from_config(self, config: dict) -> nn.Module:
        return DartsCell(config, self.init_channels)


# ==========================================================================
# 6. CIFAR-100: загрузка подготовленных данных
# ==========================================================================
#
# Данные готовятся отдельным скриптом prepare_cifar100.py.
# Здесь только загрузка и чтение meta.json.

# CIFAR-100 per-channel mean/std
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_cifar100_meta(data_dir: Path) -> dict:
    """Загрузить meta.json из подготовленной директории."""
    import json
    data_dir = Path(data_dir)
    meta_path = data_dir / "meta.json"
    required = [
        "data_X.npy", "data_y.npy",
        "train_indices.npy", "val_indices.npy",
        "train_cluster_ids.npy", "val_cluster_ids.npy",
        "cluster_centers.npy",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Data not prepared in {data_dir}. Missing: {missing}\n"
            f"Run prepare_cifar100.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"meta.json not found in {data_dir}. Run prepare_cifar100.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"[data] loaded from {data_dir}: "
          f"{meta['num_classes']} classes, {meta['total_samples']} samples, "
          f"{meta['n_clusters']} clusters")
    return meta


# ==========================================================================
# 7. CIFAR-100 оценка архитектуры на подмножестве кластеров
# ==========================================================================


def _train_cifar100_net(
    net: nn.Module,
    X_train: np.ndarray,  # uint8 [N, 3, 32, 32]
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> float:
    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)

    X_tr = torch.from_numpy(X_train).float().div_(255.0).sub_(mean).div_(std)
    y_tr = torch.from_numpy(y_train).long()
    X_v = torch.from_numpy(X_val).float().div_(255.0).sub_(mean).div_(std)
    y_v = torch.from_numpy(y_val).long()

    net.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=3e-4, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
    )

    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X_v), batch_size):
            xb = X_v[i:i + batch_size].to(device)
            yb = y_v[i:i + batch_size].to(device)
            out = net(xb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

    return correct / max(1, total)


# Глобальная переменная для числа классов (устанавливается в main)
_NUM_CLASSES = 100


def evaluate_architecture_on_subset_cifar100(
    config: dict,
    search_space,
    b: List[int],
    X_train_by_cluster: List[np.ndarray],
    y_train_by_cluster: List[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    val_cluster_ids: Optional[np.ndarray] = None,
) -> float:
    X_parts, y_parts = [], []
    for k, flag in enumerate(b):
        if flag == 1:
            X_parts.append(X_train_by_cluster[k])
            y_parts.append(y_train_by_cluster[k])
    if not X_parts:
        return 0.0
    X_sub_train = np.concatenate(X_parts, axis=0)
    y_sub_train = np.concatenate(y_parts, axis=0)

    if val_cluster_ids is not None:
        selected = [m for m, f in enumerate(b) if f == 1]
        mask = np.isin(val_cluster_ids, selected)
        if mask.sum() == 0:
            return 0.0
        X_v, y_v = X_val[mask], y_val[mask]
    else:
        X_v, y_v = X_val, y_val

    C_init = getattr(search_space, "init_channels", 16)
    net = CIFAR100Net(config, C=C_init, num_classes=_NUM_CLASSES)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_acc = _train_cifar100_net(
        net, X_sub_train, y_sub_train, X_v, y_v,
        epochs=epochs, lr=0.05, batch_size=128, device=device,
    )
    return val_acc


# ==========================================================================
# 8. Монки-патч pipeline-модулей под CIFAR-100
# ==========================================================================


def _install_patches():
    collect_dataset.evaluate_architecture_on_subset = (
        evaluate_architecture_on_subset_cifar100
    )
    osem.evaluate_architecture_on_subset = evaluate_architecture_on_subset_cifar100
    optimize_expert_assignments.evaluate_architecture_on_subset = (
        evaluate_architecture_on_subset_cifar100
    )


# ==========================================================================
# 9. MAIN
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Surrogate-EM (SGEM) on CIFAR-100 subset"
    )
    # --- Данные (подготовлены prepare_cifar100.py) ---
    parser.add_argument("--data-dir", type=str, default="./cifar100_data",
                        help="Директория с подготовленными данными (prepare_cifar100.py)")
    # --- Результаты ---
    parser.add_argument("--save-dir", type=str, default="./runs/cifar100_sgem_obs")
    parser.add_argument("--save-results", type=str,
                        default="./runs/results_cifar100_sgem.json")
    # --- MoE (M берётся из meta.json) ---
    parser.add_argument("--K", type=int, default=5)
    # --- Бюджет ---
    parser.add_argument("--n-seed-observations", type=int, default=50)
    parser.add_argument("--n-em-iterations", type=int, default=50)
    parser.add_argument("--n-new-observations", type=int, default=20)
    parser.add_argument("--surrogate-retrain-every", type=int, default=1)
    parser.add_argument("--cell-train-epochs", type=int, default=30)
    parser.add_argument("--n-arch-candidates", type=int, default=50)
    parser.add_argument("--n-candidates-s-step", type=int, default=50)
    parser.add_argument("--n-mc-forward", type=int, default=20)
    # --- Суррогат ---
    parser.add_argument("--surrogate-hidden-dim", type=int, default=64)
    parser.add_argument("--surrogate-heads", type=int, default=4)
    parser.add_argument("--surrogate-dropout", type=float, default=0.3)
    parser.add_argument("--surrogate-train-epochs", type=int, default=200)
    parser.add_argument("--surrogate-train-lr", type=float, default=3e-3)
    parser.add_argument("--surrogate-train-patience", type=int, default=30)
    # --- Сеть ---
    parser.add_argument("--init-channels", type=int, default=16)
    # --- Общее ---
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Устройство: cuda:0, cuda:1, cpu и т.д.")
    parser.add_argument("--per-cluster-eval", action="store_true",
                        help="Оценивать только на val-точках выбранных кластеров")
    parser.add_argument("--focused-ratio", type=float, default=0.5)
    parser.add_argument("--explore-flip-prob", type=float, default=0.1)
    parser.add_argument("--load-balance-weight", type=float, default=0.05,
                        help="Штраф за коллапс экспертов "
                             "(K·Σ_k (mean_m r_mk)^2). 0=выкл")
    parser.add_argument("--e-step-mc-samples", type=int, default=0,
                        help="MC-сэмплы для E-шага через Gumbel-Max. "
                             "0=argmax, >=1=усреднить u по T сэмплам R~r")
    parser.add_argument("--n-r-mc-samples", type=int, default=1,
                        help="MC-сэмплы для одного градиентного шага по r "
                             "на M-шаге. 1 = поведение по умолчанию")
    parser.add_argument("--n-r-gradient-steps", type=int, default=50,
                        help="Число градиентных шагов по r на M-шаге")
    parser.add_argument("--phase-c-uniform-mix", type=float, default=0.0,
                        help="Смесь Categorical(r) с равномерным в Phase C: "
                             "r_mix = (1-α)·r + α·(1/K). 0=чистая Categorical(r), "
                             "1=равномерное")
    parser.add_argument("--initial-surrogate-path", type=str, default=None,
                        help="Путь к предобученному суррогату (.pth)")
    parser.add_argument("--initial-obs-dir", type=str, default=None,
                        help="Директория с уже собранными obs_*.json")
    # --- Финальное обучение MoE на найденных архитектурах ---
    parser.add_argument("--final-moe-epochs", type=int, default=30,
                        help="Эпохи финального обучения MoE на конфигах из SGEM. "
                             "0 = отключить финальный этап")
    parser.add_argument("--final-moe-mode", type=str, default="both",
                        choices=["learnable", "cluster", "both"],
                        help="learnable=обучаемый softmax-gating; "
                             "cluster=жёсткий gating по hard_assignments; "
                             "both=оба варианта (по очереди)")
    parser.add_argument("--final-moe-batch-size", type=int, default=128)
    parser.add_argument("--final-moe-lr", type=float, default=0.05)
    parser.add_argument("--final-moe-wd", type=float, default=3e-4)
    parser.add_argument("--final-moe-gate-channels", type=int, default=16)
    args = parser.parse_args()

    # --- Воспроизводимость и патчи ---
    collect_dataset.set_seed(args.seed)
    _install_patches()

    # --- Загрузка подготовленных данных ---
    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)

    global _NUM_CLASSES
    _NUM_CLASSES = meta["num_classes"]
    M = meta["n_clusters"]

    X = np.load(data_dir / "data_X.npy")   # uint8 [N, 3, 32, 32]
    y = np.load(data_dir / "data_y.npy")   # int64 [N]

    ss = CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    print(f"[main] device = {args.device}")
    print(f"[main] num_classes={_NUM_CLASSES}, total_samples={len(X)}")
    print(f"[main] M={M}, K={args.K}, seed-obs={args.n_seed_observations}, "
          f"EM-iters={args.n_em_iterations}, new-obs/iter={args.n_new_observations}, "
          f"retrain-every={args.surrogate_retrain_every}, "
          f"cell-epochs={args.cell_train_epochs}")

    result = optimize_surrogate_em(
        X=X, y=y, cluster_dir=str(data_dir),
        search_space=ss, M=M, K=args.K,
        # EM
        n_em_iterations=args.n_em_iterations,
        n_arch_candidates=args.n_arch_candidates,
        n_r_gradient_steps=args.n_r_gradient_steps,
        r_lr=0.1,
        tau=0.5,
        entropy_weight=0.0,
        entropy_weight_end=None,
        max_logit_spread=0.0,
        load_balance_weight=args.load_balance_weight,
        e_step_mc_samples=args.e_step_mc_samples,
        n_r_mc_samples=args.n_r_mc_samples,
        phase_c_uniform_mix=args.phase_c_uniform_mix,
        # S-шаг
        surrogate_retrain_every=args.surrogate_retrain_every,
        n_new_observations=args.n_new_observations,
        n_mc_forward=args.n_mc_forward,
        cell_train_epochs=args.cell_train_epochs,
        n_candidates_s_step=args.n_candidates_s_step,
        save_dir=args.save_dir,
        # Суррогат
        surrogate_dropout=args.surrogate_dropout,
        surrogate_hidden_dim=args.surrogate_hidden_dim,
        surrogate_heads=args.surrogate_heads,
        surrogate_epochs=args.surrogate_train_epochs,
        surrogate_lr=args.surrogate_train_lr,
        surrogate_patience=args.surrogate_train_patience,
        # Инициализация
        initial_surrogate_path=args.initial_surrogate_path,
        initial_obs_dir=args.initial_obs_dir,
        n_seed_observations=args.n_seed_observations,
        init_assignment=None,
        per_cluster_eval=args.per_cluster_eval,
        focused_ratio=args.focused_ratio,
        explore_flip_prob=args.explore_flip_prob,
        # Суррогат тип
        model_type="gat",
        nodes_per_graph=ss.num_nodes_per_cell + 1,  # +1 на input
        # Post-EM refinement выключен
        refine_n_candidates=0,
        refine_n_top=0,
        refine_epochs=0,
        exhaustive_refine=False,
        device=args.device,
        verbose=True,
    )

    print_result(result)

    # --- Финальное обучение MoE на найденных архитектурах ---
    final_moe: dict = {}
    if args.final_moe_epochs > 0:
        configs = result.configs
        hard_assignments = result.hard_assignments
        if not configs:
            print("[final-moe] result.configs пуст — пропускаем финальное обучение")
        else:
            from cifar100_final_train import train_final_moe

            modes = (
                ["learnable", "cluster"]
                if args.final_moe_mode == "both"
                else [args.final_moe_mode]
            )
            if "cluster" in modes and hard_assignments is None:
                print("[final-moe] hard_assignments отсутствуют — "
                      "режим 'cluster' будет пропущен")
                modes = [m for m in modes if m != "cluster"]

            for mode in modes:
                print(f"\n[final-moe] mode={mode}, K={len(configs)}, "
                      f"epochs={args.final_moe_epochs}")
                final_moe[mode] = train_final_moe(
                    configs=configs,
                    hard_assignments=hard_assignments,
                    data_dir=data_dir,
                    mode=mode,
                    init_channels=args.init_channels,
                    num_classes=_NUM_CLASSES,
                    gate_channels=args.final_moe_gate_channels,
                    epochs=args.final_moe_epochs,
                    batch_size=args.final_moe_batch_size,
                    lr=args.final_moe_lr,
                    wd=args.final_moe_wd,
                    seed=args.seed,
                    device=args.device,
                    verbose=True,
                )
            print("\n[final-moe] summary:")
            for mode, r in final_moe.items():
                print(f"  {mode:>10s}: val_acc = {r['val_acc']:.4f}")

    if args.save_results:
        from toy_experiment.optimize_expert_assignments import save_results
        save_results({"cifar100_sgem": result}, args.save_results)
        if final_moe:
            import json
            with open(args.save_results, "r") as f:
                saved = json.load(f)
            saved["cifar100_sgem"]["final_moe"] = final_moe
            with open(args.save_results, "w") as f:
                json.dump(saved, f, indent=2)
        print(f"[main] saved results to {args.save_results}")


if __name__ == "__main__":
    main()

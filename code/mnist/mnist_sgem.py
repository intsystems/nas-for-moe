"""
Surrogate-EM (S-шаг) алгоритм на MNIST с упрощённым DARTS-подобным пространством.

Конфигурация:
    M = 20 кластеров (PCA-50 + KMeans на MNIST)
    K = 5 экспертов
    Search space: 3 промежуточных узла × 2 входа, 8 DARTS-операций
    Сеть: stem (3x3 conv) → 2 normal cell → 1 fixed reduction cell → GAP → FC
    Бюджет:
        n-seed-obs = 50
        n-em-iterations = 50
        n-new-observations = 20 (на каждый EM-шаг)
        surrogate-retrain-every = 1
        cell-train-epochs = 30

Запуск (внутри контейнера):
    docker exec nas-for-moe python \\
        /pbabkin/main/mipt/nas-for-moe/code/toy_experiment/mnist_sgem.py
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
sys.path.insert(0, str(SCRIPT_DIR))


# ==========================================================================
# 1. PATCH toy_graph.OPS ДО импорта pipeline-модулей
# ==========================================================================

import toy_experiment.toy_graph as toy_graph  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from darts_searchspace import DARTS_OPS, OPS_NEW, make_op  # noqa: E402

toy_graph.OPS = OPS_NEW
_enc = OneHotEncoder(handle_unknown="ignore")
toy_graph.OPS_ONE_HOT = _enc.fit_transform(
    np.array(OPS_NEW).reshape(-1, 1)
).toarray()


# ==========================================================================
# 2. Импорт pipeline-модулей (используют пропатченный OPS)
# ==========================================================================

import toy_experiment.collect_dataset as collect_dataset  # noqa: E402
import toy_experiment.optimize_expert_assignments as optimize_expert_assignments  # noqa: E402
import optimize_surrogate_em as osem  # noqa: E402
from optimize_surrogate_em import optimize_surrogate_em  # noqa: E402
from toy_experiment.optimize_expert_assignments import print_result  # noqa: E402


# ==========================================================================
# 3. DARTS cell: 3 промежуточных узла × 2 входа
# ==========================================================================
#
# Кодировка архитектуры в графе (совместимо с toy_graph.Graph):
#   idx  роль                         input choices
#   0    op_a для node 0              [-1]
#   1    op_b для node 0              [-1]
#   2    add (node 0 aggregator)      [0, 1]
#   3    op_a для node 1              [-1, 2]
#   4    op_b для node 1              [-1, 2]
#   5    add (node 1 aggregator)      [3, 4]
#   6    op_a для node 2              [-1, 2, 5]
#   7    op_b для node 2              [-1, 2, 5]
#   8    add (node 2 aggregator)      [6, 7]
#
# Выход cell = concat(node_0_out, node_1_out, node_2_out), далее 1x1 conv до C.

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


class MNISTNet(nn.Module):
    def __init__(self, config: dict, C: int = 16, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
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
# 5. Search space: генерация случайных конфигураций
# ==========================================================================


class MNISTDartsSearchSpace:
    OPS = {op: None for op in DARTS_OPS}  # совместимо с .OPS.keys()
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
        # Используется только для проверки в sample_valid_config
        # (cell должна иметь обучаемые параметры).
        return DartsCell(config, self.init_channels)


# ==========================================================================
# 6. MNIST: подготовка данных (PCA-50 → KMeans-20 + train/val split)
# ==========================================================================


def setup_mnist_data(
    data_dir: Path,
    n_clusters: int = 20,
    pca_dim: int = 50,
    seed: int = 322,
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "data_X.npy", "data_y.npy",
        "train_indices.npy", "val_indices.npy",
        "train_cluster_ids.npy", "val_cluster_ids.npy",
        "cluster_centers.npy",
    ]
    if all((data_dir / f).exists() for f in required):
        print(f"[data] already prepared in {data_dir}")
        return

    from torchvision import datasets
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    mnist_root = data_dir / "mnist_raw"
    train_ds = datasets.MNIST(str(mnist_root), train=True, download=True)
    test_ds = datasets.MNIST(str(mnist_root), train=False, download=True)

    X_full = torch.cat([train_ds.data, test_ds.data]).numpy()  # [70000, 28, 28] uint8
    y_full = torch.cat([train_ds.targets, test_ds.targets]).numpy().astype(np.int64)

    N = len(X_full)
    X_img = X_full[:, None, :, :]  # [N, 1, 28, 28] uint8
    X_flat = X_full.reshape(N, -1).astype(np.float32) / 255.0

    print(f"[data] PCA {pca_dim}-d on {N} MNIST images...")
    pca = PCA(n_components=pca_dim, random_state=seed).fit(X_flat)
    X_pca = pca.transform(X_flat)

    indices = np.arange(N)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=y_full,
    )

    print(f"[data] KMeans {n_clusters} clusters on PCA-train...")
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(
        X_pca[train_idx]
    )
    train_cluster_ids = km.labels_.astype(np.int64)
    val_cluster_ids = km.predict(X_pca[val_idx]).astype(np.int64)
    cluster_centers = km.cluster_centers_.astype(np.float32)  # [M, 50]

    np.save(data_dir / "data_X.npy", X_img)  # uint8 ~55 MB
    np.save(data_dir / "data_y.npy", y_full)
    np.save(data_dir / "train_indices.npy", train_idx)
    np.save(data_dir / "val_indices.npy", val_idx)
    np.save(data_dir / "train_cluster_ids.npy", train_cluster_ids)
    np.save(data_dir / "val_cluster_ids.npy", val_cluster_ids)
    np.save(data_dir / "cluster_centers.npy", cluster_centers)
    print(f"[data] saved to {data_dir}")


# ==========================================================================
# 7. MNIST-специфичная оценка архитектуры на подмножестве кластеров
# ==========================================================================


def _train_mnist_net(
    net: nn.Module,
    X_train: np.ndarray,  # uint8 [N, 1, 28, 28]
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
) -> float:
    mean = 0.1307
    std = 0.3081

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


def evaluate_architecture_on_subset_mnist(
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
    X_sub = np.concatenate(X_parts, axis=0)
    y_sub = np.concatenate(y_parts, axis=0)

    if val_cluster_ids is not None:
        selected = [m for m, f in enumerate(b) if f == 1]
        mask = np.isin(val_cluster_ids, selected)
        if mask.sum() == 0:
            return 0.0
        X_v, y_v = X_val[mask], y_val[mask]
    else:
        X_v, y_v = X_val, y_val

    C_init = getattr(search_space, "init_channels", 16)
    net = MNISTNet(config, C=C_init, num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_acc = _train_mnist_net(
        net, X_sub, y_sub, X_v, y_v,
        epochs=epochs, lr=0.05, batch_size=128, device=device,
    )
    return val_acc


# ==========================================================================
# 8. Монки-патч pipeline-модулей под MNIST
# ==========================================================================


def _install_patches():
    # Сам evaluator для реальной оценки архитектур
    collect_dataset.evaluate_architecture_on_subset = (
        evaluate_architecture_on_subset_mnist
    )
    osem.evaluate_architecture_on_subset = evaluate_architecture_on_subset_mnist
    optimize_expert_assignments.evaluate_architecture_on_subset = (
        evaluate_architecture_on_subset_mnist
    )


# ==========================================================================
# 9. MAIN
# ==========================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./mnist_data")
    parser.add_argument("--save-dir", type=str, default="./runs/mnist_sgem_obs")
    parser.add_argument("--save-results", type=str, default="./runs/results_mnist_sgem.json")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n-seed-observations", type=int, default=50)
    parser.add_argument("--n-em-iterations", type=int, default=50)
    parser.add_argument("--n-new-observations", type=int, default=20)
    parser.add_argument("--surrogate-retrain-every", type=int, default=1)
    parser.add_argument("--cell-train-epochs", type=int, default=30)
    parser.add_argument("--n-arch-candidates", type=int, default=50)
    parser.add_argument("--n-candidates-s-step", type=int, default=50)
    parser.add_argument("--n-mc-forward", type=int, default=20)
    parser.add_argument("--surrogate-hidden-dim", type=int, default=64)
    parser.add_argument("--surrogate-heads", type=int, default=4)
    parser.add_argument("--surrogate-dropout", type=float, default=0.3)
    parser.add_argument("--surrogate-train-epochs", type=int, default=200)
    parser.add_argument("--surrogate-train-lr", type=float, default=3e-3)
    parser.add_argument("--surrogate-train-patience", type=int, default=30)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--per-cluster-eval", action="store_true",
                        help="Оценивать только на val-точках выбранных кластеров")
    parser.add_argument("--focused-ratio", type=float, default=0.5,
                        help="Доля S-step бюджета на фазу B (focused)")
    parser.add_argument("--explore-flip-prob", type=float, default=0.1,
                        help="Bit-flip prob для b в фазе C (0→b совпадает с "
                             "expert-колонкой, 0.5→равномерно случайный b)")
    parser.add_argument("--initial-surrogate-path", type=str, default=None,
                        help="Путь к предобученному суррогату (.pth)")
    parser.add_argument("--initial-obs-dir", type=str, default=None,
                        help="Директория с уже собранными obs_*.json")
    args = parser.parse_args()

    # --- Воспроизводимость и патчи ---
    collect_dataset.set_seed(args.seed)
    _install_patches()

    # --- Подготовка данных ---
    data_dir = Path(args.data_dir)
    setup_mnist_data(data_dir, n_clusters=args.M, pca_dim=50, seed=args.seed)

    X = np.load(data_dir / "data_X.npy")   # uint8 [70000, 1, 28, 28]
    y = np.load(data_dir / "data_y.npy")   # int64 [70000]

    ss = MNISTDartsSearchSpace(init_channels=args.init_channels)

    print(f"[main] device = {args.device}")
    print(f"[main] M={args.M}, K={args.K}, seed-obs={args.n_seed_observations}, "
          f"EM-iters={args.n_em_iterations}, new-obs/iter={args.n_new_observations}, "
          f"retrain-every={args.surrogate_retrain_every}, "
          f"cell-epochs={args.cell_train_epochs}")

    result = optimize_surrogate_em(
        X=X, y=y, cluster_dir=str(data_dir),
        search_space=ss, M=args.M, K=args.K,
        # EM
        n_em_iterations=args.n_em_iterations,
        n_arch_candidates=args.n_arch_candidates,
        n_r_gradient_steps=50,
        r_lr=0.1,
        tau=0.5,
        entropy_weight=0.0,
        entropy_weight_end=None,
        max_logit_spread=0.0,
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
        # Post-EM refinement выключен (не финальный refinement)
        refine_n_candidates=0,
        refine_n_top=0,
        refine_epochs=0,
        exhaustive_refine=False,
        device=args.device,
        verbose=True,
    )

    print_result(result)

    if args.save_results:
        from toy_experiment.optimize_expert_assignments import save_results
        save_results({"mnist_sgem": result}, args.save_results)
        print(f"[main] saved results to {args.save_results}")


if __name__ == "__main__":
    main()

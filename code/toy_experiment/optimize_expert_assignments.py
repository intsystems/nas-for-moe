"""
optimize_expert_assignments.py

Shared utilities for MoE expert assignment optimization.

Contains:
    - OptimizationResult dataclass
    - Graph/surrogate helpers
    - Log-likelihood computation
    - Sampling and discretization utilities
    - Gumbel-Softmax
    - Architecture search
    - Surrogate loading and search space creation
    - Result printing
"""

import os
import sys
import json
import copy
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

# --- Настройка путей ---
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import toy_searchspace
from toy_experiment.toy_graph import Graph, OPS
from toy_experiment.collect_dataset import (
    sample_valid_config,
    set_seed,
    SEED,
)
import nas_moe.surrogate


# =========================================================================
# Контейнер для результатов
# =========================================================================

@dataclass
class OptimizationResult:
    """Результат оптимизации назначения экспертов."""

    configs: List[dict]           # Лучшие архитектуры α_1, ..., α_K
    r_matrix: np.ndarray          # Матрица назначений [M, K]
    hard_assignments: np.ndarray  # Дискретное назначение: argmax по строкам [M]
    objective_value: float        # Итоговое значение log-likelihood
    history: List[float] = field(default_factory=list)  # История оптимизации
    method: str = ""
    real_accuracies: Optional[List[float]] = None  # Реальные accuracy экспертов после переобучения


# =========================================================================
# 1. Утилиты: работа с графами и суррогатом
# =========================================================================

def build_graph_data(config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Конвертировать конфигурацию архитектуры в тензоры (x, edge_index)
    для PyG Data объекта.
    """
    graph = Graph(config, index=0)
    adj_matrix, _ops, features = graph.get_adjacency_matrix()
    edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
    x = torch.tensor(features, dtype=torch.float)
    return x, edge_index


def surrogate_eval_batch(
    surrogate: nn.Module,
    configs: List[dict],
    R_columns: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Вычислить u(α_k, R_k) для K пар (архитектура, R_k) одним batch-проходом.

    Args:
        surrogate: обученная суррогатная модель
        configs: список из K конфигураций архитектур
        R_columns: [K, M] — для каждого эксперта k его вектор R_k
        device: устройство

    Returns:
        tensor [K] предсказаний суррогата (дифференцируемо, если R_columns requires_grad)
    """
    data_list = []
    for config in configs:
        x, edge_index = build_graph_data(config)
        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list).to(device)
    bool_vec = R_columns.to(device)  # [K, M]

    out = surrogate(batch.x, batch.edge_index, batch.batch, bool_vec)
    return out.squeeze(-1)  # [K]


def surrogate_eval_with_prebatch(
    surrogate: nn.Module,
    batch: Batch,
    R_columns: torch.Tensor,
) -> torch.Tensor:
    """
    Вычислить u(α_k, R_k) используя заранее собранный batch графов.

    Args:
        surrogate: обученная модель
        batch: заранее собранный Batch графов (на нужном device)
        R_columns: [K, M] tensor (на нужном device)

    Returns:
        tensor [K]
    """
    out = surrogate(batch.x, batch.edge_index, batch.batch, R_columns)
    return out.squeeze(-1)


# =========================================================================
# 2. Утилиты: целевая функция
# =========================================================================

def compute_log_likelihood(
    r: torch.Tensor,
    u_values: torch.Tensor,
) -> torch.Tensor:
    """
    Вычислить log-likelihood: Σ_m log(Σ_k r_mk · u_k).

    Args:
        r: [M, K] — матрица назначений (строки суммируются в 1)
        u_values: [K] — предсказания суррогата для каждого эксперта

    Returns:
        scalar tensor (дифференцируемый)
    """
    inner_sum = torch.matmul(r, u_values)  # [M]
    inner_sum = torch.clamp(inner_sum, min=1e-10)
    return torch.log(inner_sum).sum()


def compute_log_likelihood_numpy(
    r: np.ndarray,
    u_values: np.ndarray,
) -> float:
    """Numpy-версия log-likelihood."""
    inner_sum = r @ u_values  # [M]
    inner_sum = np.maximum(inner_sum, 1e-10)
    return float(np.sum(np.log(inner_sum)))


# =========================================================================
# 3. Утилиты: дискретизация и сэмплирование
# =========================================================================

def discretize_assignments(r: np.ndarray) -> np.ndarray:
    """Дискретное назначение: argmax по строкам → [M] array."""
    return np.argmax(r, axis=1)


def r_to_hard_matrix(r: np.ndarray) -> np.ndarray:
    """
    Преобразовать мягкую матрицу r [M,K] в жёсткую (one-hot по строкам),
    используя argmax.
    """
    M, K = r.shape
    hard = np.zeros_like(r)
    assigns = np.argmax(r, axis=1)
    for m in range(M):
        hard[m, assigns[m]] = 1.0
    return hard


def sample_hard_assignment_matrix(M: int, K: int) -> np.ndarray:
    """
    Сэмплировать жёсткое назначение: каждый кластер m → ровно один эксперт k.
    Если M >= K, гарантирует что каждый эксперт получит хотя бы один кластер.

    Returns:
        [M, K] one-hot матрица
    """
    assignments = np.zeros((M, K))

    if M >= K:
        perm = np.random.permutation(M)
        for k in range(K):
            assignments[perm[k], k] = 1.0
        for idx in range(K, M):
            k = np.random.randint(0, K)
            assignments[perm[idx], k] = 1.0
    else:
        for m in range(M):
            k = np.random.randint(0, K)
            assignments[m, k] = 1.0

    return assignments


def sample_soft_assignment_matrix(M: int, K: int) -> np.ndarray:
    """Сэмплировать мягкое назначение через Dirichlet: строки суммируются в 1."""
    return np.random.dirichlet(np.ones(K), size=M)


# =========================================================================
# 4. Утилиты: Gumbel-Softmax
# =========================================================================

def gumbel_softmax_rows(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """
    Gumbel-Softmax для каждой строки матрицы logits [M, K].

    При hard=True (straight-through estimator):
        forward pass: дискретные {0,1} вектора → совместимо с суррогатом
        backward pass: непрерывные градиенты через softmax
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


# =========================================================================
# 5. Утилиты: поиск архитектур
# =========================================================================

def prebuild_graph_batch(
    configs: List[dict],
    device: str = "cpu",
) -> Batch:
    """Собрать PyG Batch для списка конфигураций."""
    data_list = []
    for config in configs:
        x, edge_index = build_graph_data(config)
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list).to(device)


def sample_architectures_for_experts(
    search_space: toy_searchspace.ToySearchSpace,
    K: int,
    n_candidates: int,
    surrogate: nn.Module,
    logits: torch.Tensor,
    device: str = "cpu",
    n_mc_forward: int = 10,
    tau: float = 1.0,
) -> List[dict]:
    """
    Для каждого эксперта k:
      1. Сгенерировать n_candidates случайных архитектур.
      2. Оценить каждую суррогатом, сэмплируя hard one-hot R через Gumbel-Softmax
         на каждом MC-прогоне (dropout ON). Это даёт E_R[u(α, R_k)] по
         дискретным бинарным векторам — распределение, на котором суррогат обучался.
      3. Выбрать лучшую по среднему предсказанию.

    Args:
        logits: [M, K] — logits распределения кластеров по экспертам.
                На каждом из n_mc_forward прогонов сэмплируется свежий hard one-hot R
                через F.gumbel_softmax(..., tau=tau, hard=True).
        tau: температура Gumbel-Softmax. tau=1.0 даёт честное категориальное сэмплирование.

    Returns:
        список из K конфигураций (по одной лучшей для каждого эксперта)
    """
    best_configs = []
    logits = logits.to(device)

    with torch.no_grad():
        for k in range(K):
            candidates = [sample_valid_config(search_space) for _ in range(n_candidates)]

            data_list = []
            for config in candidates:
                x, edge_index = build_graph_data(config)
                data_list.append(Data(x=x, edge_index=edge_index))

            batch = Batch.from_data_list(data_list).to(device)

            # MC ensemble: dropout ON + fresh Gumbel-Softmax hard one-hot per pass
            surrogate.train()  # enable dropout
            all_scores = []
            for _ in range(n_mc_forward):
                R_hard = gumbel_softmax_rows(logits, tau=tau, hard=True)  # [M, K]
                R_k = R_hard[:, k]  # [M] binary
                bool_vec = R_k.unsqueeze(0).expand(len(candidates), -1)
                scores = surrogate(
                    batch.x, batch.edge_index, batch.batch, bool_vec,
                )
                all_scores.append(scores.squeeze(-1).cpu().numpy())
            surrogate.eval()

            mean_scores = np.mean(all_scores, axis=0)

            best_idx = int(np.argmax(mean_scores))
            best_configs.append(candidates[best_idx])

    return best_configs


# =========================================================================
# Вывод результатов
# =========================================================================

def print_result(result: OptimizationResult):
    """Красивый вывод результатов оптимизации."""
    print(f"\n{'=' * 60}")
    print(f"Метод: {result.method}")
    print(f"{'=' * 60}")
    print(f"Objective (log-likelihood): {result.objective_value:.6f}")

    M, K = result.r_matrix.shape
    print(f"\nМатрица назначений r [{M}, {K}]:")
    print(np.round(result.r_matrix, 4))
    print(f"\nЖёсткие назначения (кластер → эксперт): {result.hard_assignments}")

    for k, config in enumerate(result.configs):
        print(f"\nЭксперт {k}: {config}")

    print(f"\nДлина истории: {len(result.history)}")
    if result.history:
        valid = [h for h in result.history if not np.isnan(h)]
        if valid:
            print(f"  Первое:  {valid[0]:.4f}")
            print(f"  Последнее: {valid[-1]:.4f}")
            print(f"  Лучшее: {max(valid):.4f}")


# =========================================================================
# Загрузка суррогата и создание пространства поиска
# =========================================================================

def load_surrogate(
    path: str,
    n_features: int = len(OPS),
    M: int = 2,
    dropout: float = 0.3,
    hidden_dim: int = 64,
    heads: int = 4,
    device: str = "cpu",
    model_type: str = "gat",
    nodes_per_graph: int = 4,
    cluster_centers: np.ndarray = None,
) -> nn.Module:
    """Загрузить обученную суррогатную модель."""
    if model_type == "hybrid":
        surrogate = nas_moe.surrogate.HybridSurrogate(
            input_dim=n_features,
            output_dim=1,
            dropout=dropout,
            hidden_dim=hidden_dim,
            heads=heads,
            bool_vec_dim=M,
            nodes_per_graph=nodes_per_graph,
        )
    else:
        surrogate = nas_moe.surrogate.GAT_Datafeature(
            input_dim=n_features,
            output_dim=1,
            dropout=dropout,
            hidden_dim=hidden_dim,
            heads=heads,
            bool_vec_dim=M,
            cluster_centers=cluster_centers,
        )
    state = torch.load(path, map_location=device)
    surrogate.load_state_dict(state)
    surrogate.to(device)
    surrogate.eval()
    return surrogate


def create_search_space(input_dim: int = 2, num_nodes_per_cell: int = 4):
    """Создать пространство поиска архитектур."""
    return toy_searchspace.ToySearchSpace(
        input_dim=input_dim, num_nodes_per_cell=num_nodes_per_cell,
    )


def add_common_args(parser: argparse.ArgumentParser):
    """Добавить общие CLI-аргументы для всех методов."""
    parser.add_argument(
        "--surrogate-path", type=str, default="./runs/surr_best.pth",
        help="Путь к суррогатной модели (.pth)",
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Директория с data_X.npy, data_y.npy")
    parser.add_argument("--cluster-dir", type=str, default=None,
                        help="Директория с кластерами (default: same as data-dir)")
    parser.add_argument("--M", type=int, default=2, help="Число кластеров данных")
    parser.add_argument("--K", type=int, default=2, help="Число экспертов")
    parser.add_argument("--input-dim", type=int, default=2,
                        help="Размерность входа ячейки")
    parser.add_argument("--n-features", type=int, default=len(OPS),
                        help="Число признаков графа")
    parser.add_argument("--surrogate-dropout", type=float, default=0.3)
    parser.add_argument("--surrogate-hidden-dim", type=int, default=64)
    parser.add_argument("--surrogate-heads", type=int, default=4)
    parser.add_argument("--surrogate-type", type=str, default="gat",
                        choices=["gat", "hybrid"],
                        help="Тип суррогатной модели: gat или hybrid")
    parser.add_argument("--n-nodes", type=int, default=4,
                        help="Число операций (узлов) в ячейке архитектуры")
    parser.add_argument("--n-arch-candidates", type=int, default=50)
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save-results", type=str, default=None,
                        help="Путь для сохранения результатов в JSON")


def setup_from_args(args) -> Tuple[nn.Module, toy_searchspace.ToySearchSpace]:
    """Загрузить суррогат и создать search space из parsed args."""
    set_seed(args.seed)
    surrogate = load_surrogate(
        args.surrogate_path,
        n_features=args.n_features,
        M=args.M,
        dropout=args.surrogate_dropout,
        hidden_dim=args.surrogate_hidden_dim,
        heads=args.surrogate_heads,
        device=args.device,
    )
    ss = create_search_space(input_dim=args.input_dim)
    return surrogate, ss


def evaluate_result_real(
    result: OptimizationResult,
    data_dir: str | Path,
    cluster_dir: str | Path | None = None,
    cell_train_epochs: int = 100,
    verbose: bool = True,
) -> List[float]:
    """
    Реальная оценка найденных архитектур: обучить каждого эксперта k
    на кластерах, назначенных ему (hard_assignments), и оценить на валидации.

    Возвращает список accuracy для каждого эксперта.
    Также записывает результат в result.real_accuracies.
    """
    from toy_experiment.collect_dataset import (
        prepare_data,
        evaluate_architecture_on_subset,
    )

    data_dir = Path(data_dir)
    if cluster_dir is None:
        cluster_dir = data_dir

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    data = prepare_data(X, y, cluster_dir=cluster_dir)
    n_clusters = data["n_clusters"]
    X_train_by_cluster = data["X_train_by_cluster"]
    y_train_by_cluster = data["y_train_by_cluster"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    K = len(result.configs)
    hard = result.hard_assignments  # [M] — кластер m → эксперт hard[m]

    ss = create_search_space()

    real_accs = []
    for k in range(K):
        # b_k: бинарный вектор — какие кластеры назначены эксперту k
        b_k = [1 if hard[m] == k else 0 for m in range(n_clusters)]

        if sum(b_k) == 0:
            if verbose:
                print(f"  Expert {k}: no clusters assigned, accuracy = 0.0")
            real_accs.append(0.0)
            continue

        val_acc = evaluate_architecture_on_subset(
            result.configs[k], ss, b_k,
            X_train_by_cluster, y_train_by_cluster,
            X_val, y_val,
            epochs=cell_train_epochs,
        )
        real_accs.append(val_acc)
        if verbose:
            print(f"  Expert {k}: clusters={[m for m, f in enumerate(b_k) if f]} "
                  f"-> val_accuracy = {val_acc:.4f}")

    result.real_accuracies = real_accs
    if verbose:
        print(f"  Mean real accuracy: {np.mean(real_accs):.4f}")
    return real_accs


def save_results(results: Dict[str, OptimizationResult], path: str):
    """Сохранить результаты в JSON."""
    save_data = {}
    for method_name, res in results.items():
        entry = {
            "configs": res.configs,
            "r_matrix": res.r_matrix.tolist(),
            "hard_assignments": res.hard_assignments.tolist(),
            "objective_value": res.objective_value,
            "history": res.history,
        }
        if res.real_accuracies is not None:
            entry["real_accuracies"] = res.real_accuracies
        save_data[method_name] = entry
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nРезультаты сохранены в {path}")

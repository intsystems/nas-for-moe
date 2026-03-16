"""
Сборка датасета обученных архитектур с active learning на основе MC Dropout.

Каждое наблюдение: (архитектура, бинарный вектор подмножества кластеров b) -> val accuracy.
Архитектура обучается только на train-данных из кластеров, где b[k]=1.
Валидация — на всех валидационных точках (кластер val-точки определяется по
ближайшему центроиду KMeans, обученному на train-данных).

Алгоритм active learning:
1. Собрать seed-датасет: случайные пары (архитектура, b).
2. Обучить суррогатную модель: (граф, b) -> val_accuracy.
3. Сгенерировать кандидатные пары (архитектура, b).
4. Оценить каждого кандидата через MC Dropout → mu, sigma.
5. Выбрать кандидата с наибольшим UCB = mu + sigma.
6. Реально обучить, сохранить наблюдение, добавить в датасет.
7. Повторить с шага 2.

Использование:
    python collect_dataset.py --data-dir ./data --save-dir ./active_dataset

Допущения:
    - Валидация проводится на ВСЕХ валидационных точках (не только из кластеров b).
      Обоснование: в MoE роутер может направить данные любого кластера к эксперту,
      поэтому важно знать обобщающую способность модели, обученной на подмножестве.
    - Для K кластеров существует 2^K - 1 ненулевых подмножеств. При малом K
      (например, K=2) мы перебираем все; при большом — сэмплируем случайные.
"""

import os
import sys
import json
import random
import argparse
import itertools
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Настройка путей ---
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import toy_searchspace
from toy_graph import Graph, OPS
from toy_dataset import ArchSubsetACCDataset
import nas_moe.surrogate

# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------
SEED = 322


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================
# 1. Подготовка данных: кластеризация, train/val split
# =========================================================================

def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    test_size: float = 0.2,
    seed: int = SEED,
) -> dict:
    """
    Глобальный train/val split → KMeans на train → присвоение val-точек кластерам.

    Возвращает словарь:
        X_train_by_cluster: list[np.ndarray]  — train-данные по кластерам
        y_train_by_cluster: list[np.ndarray]
        X_val: np.ndarray                      — все валидационные точки
        y_val: np.ndarray
        val_cluster_ids: np.ndarray            — кластер каждой val-точки (по ближайшему центру)
        cluster_centers: np.ndarray            — центроиды KMeans
    """
    # 1. Глобальный train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # 2. KMeans на train
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    # 3. Разбить train по кластерам
    X_train_by_cluster = []
    y_train_by_cluster = []
    for cid in range(n_clusters):
        mask = train_cluster_ids == cid
        X_train_by_cluster.append(X_train[mask])
        y_train_by_cluster.append(y_train[mask])

    # 4. Присвоить val-точки кластерам по ближайшему центроиду
    val_cluster_ids = assign_to_nearest_cluster(X_val, cluster_centers)

    return {
        "X_train_by_cluster": X_train_by_cluster,
        "y_train_by_cluster": y_train_by_cluster,
        "X_val": X_val,
        "y_val": y_val,
        "val_cluster_ids": val_cluster_ids,
        "cluster_centers": cluster_centers,
    }


def assign_to_nearest_cluster(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Присвоить каждую точку ближайшему центроиду. Возвращает массив cluster_id."""
    # distances: (N, K)
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


# =========================================================================
# 2. Формирование train-подвыборки по бинарному вектору b
# =========================================================================

def get_train_data_for_subset(
    b: List[int],
    X_train_by_cluster: List[np.ndarray],
    y_train_by_cluster: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Объединить train-данные из кластеров, где b[k]=1.
    Возвращает (X_subset, y_subset).
    """
    X_parts, y_parts = [], []
    for k, flag in enumerate(b):
        if flag == 1:
            X_parts.append(X_train_by_cluster[k])
            y_parts.append(y_train_by_cluster[k])

    if not X_parts:
        raise ValueError("Бинарный вектор b не должен быть полностью нулевым.")

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# =========================================================================
# 3. Обучение ячейки и оценка на валидации
# =========================================================================

def train_cell(
    cell: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    lr: float = 0.2,
    batch_size: int = 32,
) -> float:
    """Обучить ячейку на (X_train, y_train), вернуть accuracy на (X_val, y_val)."""
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    optimizer = torch.optim.SGD(cell.parameters(), lr=lr, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    n_batches = max(1, len(X_train_t) // batch_size)

    for epoch in range(epochs):
        indices = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[indices]
        y_shuffled = y_train_t[indices]

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_X = X_shuffled[start:end]
            batch_y = y_shuffled[start:end]

            optimizer.zero_grad()
            outputs = cell(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell.parameters(), max_norm=1.0)
            optimizer.step()

    with torch.no_grad():
        val_outputs = cell(X_val_t)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = (val_preds == y_val_t).float().mean().item()

    return val_acc


def evaluate_architecture_on_subset(
    config: dict,
    search_space: toy_searchspace.ToySearchSpace,
    b: List[int],
    X_train_by_cluster: List[np.ndarray],
    y_train_by_cluster: List[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
) -> float:
    """
    Обучить архитектуру на train-данных из кластеров, выбранных в b.
    Оценить на ВСЕХ валидационных данных.
    Вернуть val accuracy (скаляр).
    """
    X_sub, y_sub = get_train_data_for_subset(b, X_train_by_cluster, y_train_by_cluster)
    cell = search_space.create_cell_from_config(config)
    val_acc = train_cell(cell, X_sub, y_sub, X_val, y_val, epochs=epochs)
    return val_acc


# =========================================================================
# 4. Сохранение наблюдений
# =========================================================================

def save_observation(
    config: dict,
    b: List[int],
    val_acc: float,
    save_dir: str,
    index: int,
) -> Path:
    """Сохранить наблюдение {arch, subset_b, val_accuracy} в JSON."""
    data = {
        "arch": config,
        "subset_b": b,
        "val_accuracy": val_acc,
    }
    path = Path(save_dir) / f"obs_{index}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path


# =========================================================================
# 5. Сэмплирование
# =========================================================================

def sample_valid_config(search_space: toy_searchspace.ToySearchSpace) -> dict:
    """Сгенерировать случайную архитектуру с хотя бы одним обучаемым параметром."""
    while True:
        config = search_space.create_random_config()
        cell = search_space.create_cell_from_config(config)
        total_params = sum(p.numel() for p in cell.parameters())
        if total_params > 0:
            return config


def sample_random_b(n_clusters: int) -> List[int]:
    """
    Сгенерировать случайный ненулевой бинарный вектор длины n_clusters.
    Гарантирует, что хотя бы один элемент равен 1.
    """
    while True:
        b = [random.randint(0, 1) for _ in range(n_clusters)]
        if sum(b) > 0:
            return b


def enumerate_all_b(n_clusters: int) -> List[List[int]]:
    """
    Перечислить все ненулевые бинарные вектора длины n_clusters.
    Для K кластеров это 2^K - 1 вариантов.
    """
    all_b = []
    for bits in itertools.product([0, 1], repeat=n_clusters):
        if sum(bits) > 0:
            all_b.append(list(bits))
    return all_b


def sample_b_vectors(n_clusters: int, n_samples: int) -> List[List[int]]:
    """
    Если 2^K - 1 <= n_samples, вернуть все ненулевые b.
    Иначе сэмплировать n_samples случайных.
    """
    total_nonzero = 2 ** n_clusters - 1
    if total_nonzero <= n_samples:
        return enumerate_all_b(n_clusters)
    else:
        # Сэмплировать уникальные b
        seen = set()
        result = []
        while len(result) < n_samples:
            b = sample_random_b(n_clusters)
            b_key = tuple(b)
            if b_key not in seen:
                seen.add(b_key)
                result.append(b)
        return result


# =========================================================================
# 6. MC Dropout — оценка неопределённости
# =========================================================================

def enable_mc_dropout(model: nn.Module):
    """Перевести все Dropout-слои в train-режим (MC Dropout)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout_predict(
    surrogate: nn.Module,
    data: Data,
    n_forward: int = 30,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MC Dropout: n_forward стохастических forward-проходов.
    Возвращает (mean, std) предсказаний.
    """
    surrogate.eval()
    enable_mc_dropout(surrogate)

    data = data.to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(n_forward):
            out = surrogate(
                data.x, data.edge_index, data.batch, data.bool_vector
            )
            predictions.append(out.cpu())

    predictions = torch.stack(predictions, dim=0)  # (n_forward, batch, out_dim)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)

    return mean_pred, std_pred


def config_to_pyg_data(config: dict, b: List[int]) -> Data:
    """Конвертировать (архитектуру, b) в один PyG Data объект."""
    graph = Graph(config, index=0)
    adj_matrix, _ops, features = graph.get_adjacency_matrix()
    edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))

    bool_vector = torch.tensor(b, dtype=torch.float).unsqueeze(0)

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        bool_vector=bool_vector,
    )
    return data


def compute_ucb_score(
    surrogate: nn.Module,
    config: dict,
    b: List[int],
    n_forward: int = 30,
    device: str = "cpu",
) -> float:
    """
    UCB acquisition score = mu + sigma.
    Чем выше → тем интереснее кандидат (высокое ожидаемое качество или высокая
    неопределённость).
    """
    data = config_to_pyg_data(config, b)
    # Добавим batch-индексы для single-graph inference
    batch = Batch.from_data_list([data]).to(device)

    mean_pred, std_pred = mc_dropout_predict(
        surrogate, batch, n_forward=n_forward, device=device
    )
    # mu + sigma (скалярные)
    mu = mean_pred.item()
    sigma = std_pred.item()
    return mu + sigma


# =========================================================================
# 7. Обучение суррогатной модели
# =========================================================================

def collate_graphs(batch):
    return Batch.from_data_list(batch)


def train_surrogate(
    surrogate: nn.Module,
    train_loader: DataLoader,
    device: str = "cpu",
    lr: float = 3e-3,
    epochs: int = 120,
    weight_decay: float = 1e-5,
    verbose: bool = False,
) -> dict:
    """Обучить суррогатную модель. Вернуть историю loss."""
    optimizer = torch.optim.Adam(
        surrogate.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
    )

    surrogate.to(device)
    history = {"train": []}

    iterator = range(1, epochs + 1)
    if not verbose:
        iterator = tqdm(iterator, desc="  Обучение суррогата")

    for epoch in iterator:
        surrogate.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = surrogate(batch.x, batch.edge_index, batch.batch, batch.bool_vector)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        history["train"].append(avg_loss)
        scheduler.step()

        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch}/{epochs}  train_loss={avg_loss:.6f}")

    return history


# =========================================================================
# 8. Основной цикл active learning
# =========================================================================

def active_learning_loop(
    # Данные
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int = 2,
    # Пространство поиска
    input_dim: int = 2,
    num_nodes_per_cell: int = 4,
    # Active learning
    n_initial: int = 50,
    n_iterations: int = 200,
    n_candidates: int = 100,
    n_b_per_iter: int = 5,
    n_mc_forward: int = 30,
    # Суррогат
    surrogate_epochs: int = 120,
    surrogate_lr: float = 3e-3,
    # Обучение ячеек
    cell_train_epochs: int = 100,
    # IO
    save_dir: str = "./active_dataset",
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Active learning для сбора датасета (архитектура, подмножество кластеров) -> val_accuracy.

    Фаза 1 (seed):
        Сэмплируем n_initial случайных пар (архитектура, b), обучаем, сохраняем.

    Фаза 2 (active, n_iterations итераций):
        a. Обучить суррогат на текущем датасете.
        b. Сэмплировать набор b-векторов.
        c. Для каждого b сгенерировать n_candidates архитектур.
        d. Для каждого (архитектура, b) вычислить UCB = mu + sigma через MC Dropout.
        e. Выбрать пару (архитектура, b) с наибольшим UCB.
        f. Реально обучить, сохранить, добавить в датасет.

    Возвращает:
        observation_paths: List[Path] — пути ко всем сохранённым наблюдениям.
        final_surrogate: nn.Module    — последняя обученная суррогатная модель.
    """
    set_seed(SEED)
    os.makedirs(save_dir, exist_ok=True)

    # --- Подготовка данных ---
    data = prepare_data(X, y, n_clusters)
    X_train_by_cluster = data["X_train_by_cluster"]
    y_train_by_cluster = data["y_train_by_cluster"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    if verbose:
        for k in range(n_clusters):
            print(f"  Кластер {k}: {len(X_train_by_cluster[k])} train, "
                  f"{(data['val_cluster_ids'] == k).sum()} val точек")

    # --- Пространство поиска ---
    ss = toy_searchspace.ToySearchSpace(
        input_dim=input_dim, num_nodes_per_cell=num_nodes_per_cell
    )

    n_features = len(OPS)
    observation_paths: List[Path] = []
    obs_index = 0

    # =================================================================
    # ФАЗА 1: Seed-датасет — случайные (архитектура, b)
    # =================================================================
    if verbose:
        print(f"\n=== Фаза 1: Сбор {n_initial} начальных наблюдений ===")

    for i in tqdm(range(n_initial), desc="Seed", disable=not verbose):
        config = sample_valid_config(ss)
        b = sample_random_b(n_clusters)
        val_acc = evaluate_architecture_on_subset(
            config, ss, b,
            X_train_by_cluster, y_train_by_cluster,
            X_val, y_val,
            epochs=cell_train_epochs,
        )
        path = save_observation(config, b, val_acc, save_dir, obs_index)
        observation_paths.append(path)
        obs_index += 1

    if verbose:
        print(f"Seed-датасет: {len(observation_paths)} наблюдений\n")

    # =================================================================
    # ФАЗА 2: Active learning
    # =================================================================
    if verbose:
        print(f"=== Фаза 2: Active learning ({n_iterations} итераций) ===")

    surr = None  # последняя обученная модель

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n--- Итерация {iteration + 1}/{n_iterations} ---")

        # 2a. Построить датасет наблюдений и обучить суррогат
        dataset = ArchSubsetACCDataset(observation_paths)
        loader = DataLoader(
            dataset, batch_size=32, shuffle=True, collate_fn=collate_graphs
        )

        surr = nas_moe.surrogate.GAT_Datafeature(
            n_features, 1, 0.8,
            hidden_dim=8, heads=1, bool_vec_dim=n_clusters,
        )

        if verbose:
            print("  Обучаем суррогат...")

        train_surrogate(
            surr, loader,
            device=device,
            lr=surrogate_lr,
            epochs=surrogate_epochs,
            verbose=False,
        )

        # 2b. Сэмплировать b-вектора
        b_vectors = sample_b_vectors(n_clusters, n_b_per_iter)

        # 2c-d. Для каждого b сгенерировать кандидатов и оценить UCB
        if verbose:
            print(f"({len(b_vectors)} b-векторов × {n_candidates} архитектур, "
                  f"MC Dropout {n_mc_forward} проходов)...")

        surr.to(device)

        for b in b_vectors:
            best_config = None
            best_score = -float("inf")
            candidates = [sample_valid_config(ss) for _ in range(n_candidates)]

            for config in candidates:
                score = compute_ucb_score(
                    surr, config, b,
                    n_forward=n_mc_forward, device=device,
                )
                if score > best_score:
                    best_score = score
                    best_config = config
                    # best_b = b

            if verbose:
                print(f"  Лучший кандидат: UCB={best_score:.6f}, b={b}")

            # 2e. Реально обучить лучшего кандидата
            val_acc = evaluate_architecture_on_subset(
                best_config, ss, b,
                X_train_by_cluster, y_train_by_cluster,
                X_val, y_val,
                epochs=cell_train_epochs,
            )

            if verbose:
                print(f"  Реальная val accuracy: {val_acc:.4f}")

            # 2f. Сохранить и добавить в датасет
            path = save_observation(best_config, b, val_acc, save_dir, obs_index)
            observation_paths.append(path)
            obs_index += 1

            if verbose:
                print(f"  Всего наблюдений: {len(observation_paths)}")

    if verbose:
        print(f"\n=== Готово. Собрано {len(observation_paths)} наблюдений ===")

    dataset = ArchSubsetACCDataset(observation_paths)
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_graphs
    )

    surr = nas_moe.surrogate.GAT_Datafeature(
        n_features, 1, 0.8,
        hidden_dim=8, heads=1, bool_vec_dim=n_clusters,
    )

    return observation_paths, surr


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Active learning сборка датасета для суррогатной модели NAS"
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Директория с data_X.npy и data_y.npy")
    parser.add_argument("--save-dir", type=str, default="./active_dataset",
                        help="Директория для сохранения JSON-наблюдений")
    parser.add_argument("--n-clusters", type=int, default=2)
    parser.add_argument("--input-dim", type=int, default=2,
                        help="Размерность входа ячейки")
    parser.add_argument("--n-initial", type=int, default=50,
                        help="Число начальных случайных наблюдений")
    parser.add_argument("--n-iterations", type=int, default=200,
                        help="Число итераций active learning")
    
    parser.add_argument("--n-candidates", type=int, default=100,
                        help="Число кандидатных архитектур на b-вектор")
    parser.add_argument("--n-b-per-iter", type=int, default=2,
                        help="Число b-векторов на итерацию (при K>5)")
    parser.add_argument("--n-mc-forward", type=int, default=30,
                        help="Число MC Dropout forward-проходов")
    
    parser.add_argument("--surrogate-epochs", type=int, default=120)
    parser.add_argument("--surrogate-lr", type=float, default=3e-3)
    parser.add_argument("--checkpoint-path", type=str,
                        default="./surr.pth")

    parser.add_argument("--cell-train-epochs", type=int, default=100)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    observation_paths, final_surrogate = active_learning_loop(
        X=X,
        y=y,
        n_clusters=args.n_clusters,
        input_dim=args.input_dim,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        n_candidates=args.n_candidates,
        n_b_per_iter=args.n_b_per_iter,
        n_mc_forward=args.n_mc_forward,
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        cell_train_epochs=args.cell_train_epochs,
        save_dir=args.save_dir,
        device=args.device,
        verbose=True,
    )

    torch.save(final_surrogate.state_dict(), args.checkpoint_path)

    print(f"\nСобрано {len(observation_paths)} наблюдений в {args.save_dir}")


if __name__ == "__main__":
    main()

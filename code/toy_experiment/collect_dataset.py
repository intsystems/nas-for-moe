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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# 7. Обучение суррогатной модели (с train/val split)
# =========================================================================

def collate_graphs(batch):
    return Batch.from_data_list(batch)


def make_surrogate_loaders(
    observation_paths: List[Path],
    val_fraction: float = 0.2,
    batch_size: int = 32,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader]:
    """Split observation paths into train/val and return DataLoaders."""
    n = len(observation_paths)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_indices = set(indices[:n_val])
    train_paths = [observation_paths[i] for i in range(n) if i not in val_indices]
    val_paths = [observation_paths[i] for i in range(n) if i in val_indices]

    train_dataset = ArchSubsetACCDataset(train_paths)
    val_dataset = ArchSubsetACCDataset(val_paths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs
    )
    return train_loader, val_loader


def train_surrogate(
    surrogate: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    lr: float = 3e-3,
    epochs: int = 200,
    weight_decay: float = 1e-4,
    patience: int = 30,
    verbose: bool = False,
) -> dict:
    """Обучить суррогатную модель с optional early stopping. Вернуть историю loss."""
    optimizer = torch.optim.Adam(
        surrogate.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
    )

    surrogate.to(device)
    history = {"train": [], "val": []}

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    iterator = range(1, epochs + 1)
    if not verbose:
        iterator = tqdm(iterator, desc="  Surrogate training")

    for epoch in iterator:
        # --- Train ---
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

        avg_train_loss = total_loss / max(1, n_batches)
        history["train"].append(avg_train_loss)
        scheduler.step()

        # --- Val ---
        if val_loader is not None:
            surrogate.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = surrogate(
                        batch.x, batch.edge_index, batch.batch, batch.bool_vector
                    )
                    val_loss += criterion(out, batch.y).item()
                    n_val += 1
            avg_val_loss = val_loss / max(1, n_val)
            history["val"].append(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in surrogate.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch} "
                          f"(val_loss={avg_val_loss:.6f}, best={best_val_loss:.6f})")
                break

            if verbose and epoch % 20 == 0:
                print(f"    Epoch {epoch}/{epochs}  "
                      f"train={avg_train_loss:.6f}  val={avg_val_loss:.6f}")
        else:
            if verbose and epoch % 20 == 0:
                print(f"    Epoch {epoch}/{epochs}  train_loss={avg_train_loss:.6f}")

    # Restore best model if we have validation
    if best_state is not None:
        surrogate.load_state_dict(best_state)
        surrogate.to(device)

    return history


def create_surrogate(
    n_features: int,
    n_clusters: int,
    dropout: float = 0.3,
    hidden_dim: int = 16,
    heads: int = 2,
) -> nn.Module:
    """Create a GAT_Datafeature surrogate model."""
    return nas_moe.surrogate.GAT_Datafeature(
        n_features, 1, dropout,
        hidden_dim=hidden_dim, heads=heads, bool_vec_dim=n_clusters,
    )


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
    n_initial: int = 100,
    n_total: int = 1000,
    n_candidates: int = 50,
    n_b_per_iter: int = 5,
    n_mc_forward: int = 20,
    retrain_every: int = 15,
    # Суррогат
    surrogate_dropout: float = 0.3,
    surrogate_hidden_dim: int = 16,
    surrogate_heads: int = 2,
    surrogate_epochs: int = 200,
    surrogate_lr: float = 3e-3,
    surrogate_val_fraction: float = 0.2,
    surrogate_patience: int = 30,
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

    Фаза 2 (active):
        Собираем наблюдения до n_total. Суррогат переобучается каждые retrain_every
        новых наблюдений. Между переобучениями используем текущий суррогат для UCB.

    Возвращает:
        observation_paths: List[Path] — пути ко всем сохранённым наблюдениям.
        final_surrogate: nn.Module    — последняя обученная суррогатная модель.
        final_history: dict           — история финального обучения суррогата.
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
            print(f"  Cluster {k}: {len(X_train_by_cluster[k])} train, "
                  f"{(data['val_cluster_ids'] == k).sum()} val points")

    # --- Пространство поиска ---
    ss = toy_searchspace.ToySearchSpace(
        input_dim=input_dim, num_nodes_per_cell=num_nodes_per_cell
    )

    n_features = len(OPS)
    observation_paths: List[Path] = []
    obs_index = 0

    # --- Resume from existing observations if any ---
    existing = sorted(Path(save_dir).glob("obs_*.json"))
    if existing:
        observation_paths = list(existing)
        # Parse highest index
        obs_index = max(
            int(p.stem.split("_")[1]) for p in existing
        ) + 1
        if verbose:
            print(f"  Resuming from {len(observation_paths)} existing observations "
                  f"(next index: {obs_index})")

    # =================================================================
    # ФАЗА 1: Seed-датасет — случайные (архитектура, b)
    # =================================================================
    n_seed_needed = max(0, n_initial - len(observation_paths))
    if n_seed_needed > 0:
        if verbose:
            print(f"\n=== Phase 1: Collecting {n_seed_needed} seed observations ===")

        for i in tqdm(range(n_seed_needed), desc="Seed", disable=not verbose):
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
    else:
        if verbose:
            print(f"\n  Seed phase skipped ({len(observation_paths)} >= {n_initial})")

    if verbose:
        print(f"Seed dataset: {len(observation_paths)} observations\n")

    # =================================================================
    # ФАЗА 2: Active learning
    # =================================================================
    remaining = n_total - n_initial
    if verbose:
        print(f"=== Phase 2: Active learning ({remaining} more observations) ===")
        print(f"  Surrogate: dropout={surrogate_dropout}, hidden={surrogate_hidden_dim}, "
              f"heads={surrogate_heads}")
        print(f"  Retrain every {retrain_every} observations")

    surr = None
    new_since_retrain = retrain_every  # force initial training
    surr_train_count = 0

    pbar = tqdm(total=remaining, desc="Active", disable=not verbose)

    while len(observation_paths) < n_total:
        # --- Retrain surrogate if needed ---
        if new_since_retrain >= retrain_every:
            surr_train_count += 1
            if verbose:
                print(f"\n  [Surrogate retrain #{surr_train_count}] "
                      f"on {len(observation_paths)} observations...")

            train_loader, val_loader = make_surrogate_loaders(
                observation_paths,
                val_fraction=surrogate_val_fraction,
                seed=SEED + surr_train_count,
            )

            surr = create_surrogate(
                n_features, n_clusters,
                dropout=surrogate_dropout,
                hidden_dim=surrogate_hidden_dim,
                heads=surrogate_heads,
            )

            history = train_surrogate(
                surr, train_loader, val_loader,
                device=device,
                lr=surrogate_lr,
                epochs=surrogate_epochs,
                patience=surrogate_patience,
                verbose=False,
            )

            if verbose:
                final_train = history["train"][-1]
                final_val = history["val"][-1] if history["val"] else float("nan")
                print(f"  Surrogate trained for {len(history['train'])} epochs: "
                      f"train_loss={final_train:.6f}, val_loss={final_val:.6f}")

            new_since_retrain = 0
            surr.to(device)

        # --- Sample b-vectors ---
        b_vectors = sample_b_vectors(n_clusters, n_b_per_iter)

        # --- For each b, find best architecture by UCB ---
        for b in b_vectors:
            if len(observation_paths) >= n_total:
                break

            candidates = [sample_valid_config(ss) for _ in range(n_candidates)]

            best_config = None
            best_score = -float("inf")
            for config in candidates:
                score = compute_ucb_score(
                    surr, config, b,
                    n_forward=n_mc_forward, device=device,
                )
                if score > best_score:
                    best_score = score
                    best_config = config

            # Evaluate best candidate
            val_acc = evaluate_architecture_on_subset(
                best_config, ss, b,
                X_train_by_cluster, y_train_by_cluster,
                X_val, y_val,
                epochs=cell_train_epochs,
            )

            path = save_observation(best_config, b, val_acc, save_dir, obs_index)
            observation_paths.append(path)
            obs_index += 1
            new_since_retrain += 1
            pbar.update(1)

            if verbose and len(observation_paths) % 50 == 0:
                print(f"  [{len(observation_paths)}/{n_total}] "
                      f"UCB={best_score:.4f}, acc={val_acc:.4f}, b={b}")

    pbar.close()

    if verbose:
        print(f"\n=== Done. Collected {len(observation_paths)} observations ===")

    # =================================================================
    # Final surrogate training on full dataset
    # =================================================================
    if verbose:
        print("  Final surrogate training on all data...")

    train_loader, val_loader = make_surrogate_loaders(
        observation_paths,
        val_fraction=surrogate_val_fraction,
        seed=SEED,
    )

    surr = create_surrogate(
        n_features, n_clusters,
        dropout=surrogate_dropout,
        hidden_dim=surrogate_hidden_dim,
        heads=surrogate_heads,
    )

    final_history = train_surrogate(
        surr, train_loader, val_loader,
        device=device,
        lr=surrogate_lr,
        epochs=surrogate_epochs,
        patience=surrogate_patience,
        verbose=verbose,
    )

    return observation_paths, surr, final_history


# =========================================================================
# 9. Evaluation & Plotting
# =========================================================================

def evaluate_and_plot(
    surrogate: nn.Module,
    observation_paths: List[Path],
    n_clusters: int,
    device: str = "cpu",
    history: Optional[dict] = None,
    save_dir: str = "./active_dataset",
):
    """Evaluate surrogate on collected data and save diagnostic plots."""
    from scipy.stats import spearmanr

    os.makedirs(save_dir, exist_ok=True)

    # Use surrogate's val split for proper evaluation
    _, val_loader = make_surrogate_loaders(
        observation_paths, val_fraction=0.2, seed=SEED
    )
    # Also get full dataset for plotting
    full_dataset = ArchSubsetACCDataset(observation_paths)
    full_loader = DataLoader(
        full_dataset, batch_size=64, shuffle=False, collate_fn=collate_graphs
    )

    surrogate.to(device)
    surrogate.eval()

    def collect_preds(loader):
        true_all, pred_all = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = surrogate(
                    batch.x, batch.edge_index, batch.batch, batch.bool_vector
                )
                pred_all.append(out.cpu().numpy())
                true_all.append(batch.y.cpu().numpy())
        return np.concatenate(true_all).flatten(), np.concatenate(pred_all).flatten()

    val_true, val_pred = collect_preds(val_loader)
    full_true, full_pred = collect_preds(full_loader)

    def compute_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sp, _ = spearmanr(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": sp}

    val_metrics = compute_metrics(val_true, val_pred)
    full_metrics = compute_metrics(full_true, full_pred)

    print(f"\n=== Surrogate evaluation (VAL split, n={len(val_true)}) ===")
    for k, v in val_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    print(f"\n=== Surrogate evaluation (FULL dataset, n={len(full_true)}) ===")
    for k, v in full_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    # --- Plot 1: Training + Validation loss curve ---
    if history and "train" in history:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history["train"], linewidth=1.5, label="Train loss", alpha=0.8)
        if history.get("val"):
            ax.plot(history["val"], linewidth=1.5, label="Val loss", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Surrogate Training Loss (final)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Set y-limit to avoid outliers dominating
        if history.get("val"):
            median_val = np.median(history["val"])
            ax.set_ylim(0, min(median_val * 3, max(history["val"])))
        fig.tight_layout()
        path1 = os.path.join(save_dir, "surrogate_loss.png")
        fig.savefig(path1, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path1}")

    # --- Plot 2: Predicted vs True scatter (val split) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (y_true, y_pred, title) in zip(axes, [
        (val_true, val_pred, "Val split"),
        (full_true, full_pred, "Full dataset"),
    ]):
        metrics = compute_metrics(y_true, y_pred)
        ax.scatter(y_true, y_pred, alpha=0.4, s=12, edgecolors="none")
        lo = min(y_true.min(), y_pred.min()) - 0.02
        hi = max(y_true.max(), y_pred.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal")
        ax.set_xlabel("True val accuracy")
        ax.set_ylabel("Predicted val accuracy")
        ax.set_title(f"{title}  (R2={metrics['R2']:.3f}, Sp={metrics['Spearman']:.3f})")
        ax.legend(fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path2 = os.path.join(save_dir, "surrogate_pred_vs_true.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # --- Plot 3: Distribution of true accuracies ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(full_true, bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Val accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of collected val accuracies (n={len(full_true)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(save_dir, "accuracy_distribution.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # --- Plot 4: Residual plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = full_pred - full_true
    ax.scatter(full_true, residuals, alpha=0.4, s=12, edgecolors="none")
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("True val accuracy")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residual plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path4 = os.path.join(save_dir, "surrogate_residuals.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path4}")


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
    parser.add_argument("--n-initial", type=int, default=100,
                        help="Число начальных случайных наблюдений")
    parser.add_argument("--n-total", type=int, default=1000,
                        help="Общее число наблюдений для сбора")

    parser.add_argument("--n-candidates", type=int, default=50,
                        help="Число кандидатных архитектур на b-вектор")
    parser.add_argument("--n-b-per-iter", type=int, default=3,
                        help="Число b-векторов на итерацию (при K>5)")
    parser.add_argument("--n-mc-forward", type=int, default=20,
                        help="Число MC Dropout forward-проходов")
    parser.add_argument("--retrain-every", type=int, default=15,
                        help="Переобучать суррогат каждые N новых наблюдений")

    parser.add_argument("--surrogate-dropout", type=float, default=0.3)
    parser.add_argument("--surrogate-hidden-dim", type=int, default=16)
    parser.add_argument("--surrogate-heads", type=int, default=2)
    parser.add_argument("--surrogate-epochs", type=int, default=200)
    parser.add_argument("--surrogate-lr", type=float, default=3e-3)
    parser.add_argument("--surrogate-val-fraction", type=float, default=0.2)
    parser.add_argument("--surrogate-patience", type=int, default=30)
    parser.add_argument("--checkpoint-path", type=str,
                        default="./surr.pth")

    parser.add_argument("--cell-train-epochs", type=int, default=100)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    observation_paths, final_surrogate, final_history = active_learning_loop(
        X=X,
        y=y,
        n_clusters=args.n_clusters,
        input_dim=args.input_dim,
        n_initial=args.n_initial,
        n_total=args.n_total,
        n_candidates=args.n_candidates,
        n_b_per_iter=args.n_b_per_iter,
        n_mc_forward=args.n_mc_forward,
        retrain_every=args.retrain_every,
        surrogate_dropout=args.surrogate_dropout,
        surrogate_hidden_dim=args.surrogate_hidden_dim,
        surrogate_heads=args.surrogate_heads,
        surrogate_epochs=args.surrogate_epochs,
        surrogate_lr=args.surrogate_lr,
        surrogate_val_fraction=args.surrogate_val_fraction,
        surrogate_patience=args.surrogate_patience,
        cell_train_epochs=args.cell_train_epochs,
        save_dir=args.save_dir,
        device=args.device,
        verbose=True,
    )

    torch.save(final_surrogate.state_dict(), args.checkpoint_path)
    print(f"\nCollected {len(observation_paths)} observations in {args.save_dir}")
    print(f"Surrogate saved to {args.checkpoint_path}")

    # --- Evaluation & Plotting ---
    evaluate_and_plot(
        final_surrogate, observation_paths, args.n_clusters,
        device=args.device, history=final_history,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

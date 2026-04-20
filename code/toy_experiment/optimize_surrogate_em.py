"""
EM-алгоритм с одновременным обучением суррогатной функции (S-шаг).

На итерации t:
    E-шаг: q^{(t)} = argmax_{q} F_{u^{(t)}}(q, θ^{(t)})
        q_{mk} = r_{mk} · u(α_k, R_k) / Σ_j r_{mj} · u(α_j, R_j)

    M-шаг (GEM): найти θ^{(t+1)} такое, что
        F_{u^{(t)}}(q^{(t)}, θ^{(t+1)}) ≥ F_{u^{(t)}}(q^{(t)}, θ^{(t)})

    S-шаг: обновить суррогат u^{(t)} → u^{(t+1)}
        (реально обучить архитектуры, добавить точки, переобучить суррогат)

Использование:
    # С нуля (seed-датасет + EM + S-step)
    python optimize_surrogate_em.py --data-dir ./data_multi --M 20 --K 3

    # С существующими наблюдениями
    python optimize_surrogate_em.py --data-dir ./data_multi --M 20 --K 3 \\
        --initial-obs-dir ./model_dataset_balanced

    # С предобученным суррогатом
    python optimize_surrogate_em.py --data-dir ./data_multi --M 20 --K 3 \\
        --surrogate-path ./runs/surr_best.pth
"""

import copy
import json
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from toy_experiment.toy_graph import OPS
from toy_experiment.optimize_expert_assignments import (
    OptimizationResult,
    surrogate_eval_batch,
    surrogate_eval_with_prebatch,
    compute_log_likelihood_numpy,
    discretize_assignments,
    r_to_hard_matrix,
    gumbel_softmax_rows,
    prebuild_graph_batch,
    sample_architectures_for_experts,
    print_result,
    add_common_args,
    save_results,
    evaluate_result_real,
    load_surrogate,
    create_search_space,
)
from toy_experiment.collect_dataset import (
    prepare_data,
    evaluate_architecture_on_subset,
    save_observation,
    make_surrogate_loaders,
    train_surrogate,
    create_surrogate,
    sample_random_b,
    sample_valid_config,
    compute_ucb_score,
    set_seed,
    SEED,
)


# =========================================================================
# Post-EM architecture refinement (surrogate-guided + real eval)
# =========================================================================

def enumerate_all_architectures(search_space) -> List[dict]:
    """Enumerate all valid architectures (with trainable params) for 3-node cells."""
    ops = list(search_space.OPS.keys())
    num_nodes = search_space.num_nodes_per_cell
    all_configs = []
    for op0 in ops:
        for op1 in ops:
            for op2 in ops:
                possible_inputs_2 = list(range(num_nodes - 1))  # [0, 1] for 3 nodes
                for inp2 in possible_inputs_2:
                    config = {
                        "op_0": op0, "input_0": [-1],
                        "op_1": op1, "input_1": [0],
                        "op_2": op2, "input_2": [inp2],
                    }
                    cell = search_space.create_cell_from_config(config)
                    params = sum(p.numel() for p in cell.parameters())
                    if params > 0:
                        all_configs.append(config)
    return all_configs


def refine_architectures_real(
    hard_assignments: np.ndarray,
    K: int,
    n_clusters: int,
    search_space,
    surrogate: nn.Module,
    X_train_by_cluster: List[np.ndarray],
    y_train_by_cluster: List[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_candidates: int = 500,
    n_top: int = 20,
    cell_train_epochs: int = 200,
    n_mc_forward: int = 20,
    device: str = "cpu",
    val_cluster_ids: Optional[np.ndarray] = None,
    verbose: bool = True,
    exhaustive: bool = False,
) -> List[dict]:
    """
    Post-EM architecture refinement for each expert.

    If exhaustive=True: enumerate ALL valid architectures, train each one,
    pick the best by real accuracy (no surrogate filtering).

    Otherwise (default): sample n_candidates random architectures, filter by
    surrogate, train top-n_top, pick best.

    Returns list of K best configs.
    """
    from toy_experiment.optimize_expert_assignments import (
        build_graph_data,
        sample_valid_config,
    )
    from torch_geometric.data import Data, Batch

    best_configs = []

    for k in range(K):
        b_k = [1 if hard_assignments[m] == k else 0 for m in range(n_clusters)]
        if sum(b_k) == 0:
            best_configs.append(sample_valid_config(search_space))
            if verbose:
                print(f"  Expert {k}: no clusters, skipping refinement")
            continue

        clusters_k = [m for m in range(n_clusters) if b_k[m] == 1]

        if exhaustive:
            # Enumerate all architectures and train each one
            candidates = enumerate_all_architectures(search_space)
            if verbose:
                print(f"  Expert {k} (clusters={clusters_k}): "
                      f"exhaustive search over {len(candidates)} architectures")

            best_acc = -1.0
            best_config = candidates[0]
            for rank, config in enumerate(candidates):
                val_acc = evaluate_architecture_on_subset(
                    config, search_space, b_k,
                    X_train_by_cluster, y_train_by_cluster,
                    X_val, y_val,
                    epochs=cell_train_epochs,
                    val_cluster_ids=val_cluster_ids,
                )
                ops = [config.get(f"op_{i}", "?") for i in range(3)]
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc
                    best_config = config
                if verbose and is_best:
                    print(f"    [{rank+1}/{len(candidates)}] ops={ops} "
                          f"real_acc={val_acc:.4f} *** new best")

            best_configs.append(best_config)
            if verbose:
                ops = [best_config.get(f"op_{i}", "?") for i in range(3)]
                print(f"  Expert {k}: best real_acc = {best_acc:.4f}, ops={ops}")
        else:
            # Surrogate-guided: sample, filter, train top
            candidates = [sample_valid_config(search_space)
                          for _ in range(n_candidates)]
            R_k = torch.tensor(b_k, dtype=torch.float, device=device)

            data_list = []
            for config in candidates:
                x, edge_index = build_graph_data(config)
                data_list.append(Data(x=x, edge_index=edge_index))

            batch = Batch.from_data_list(data_list).to(device)
            bool_vec = R_k.unsqueeze(0).expand(len(candidates), -1)

            surrogate.train()  # enable dropout for MC
            all_scores = []
            with torch.no_grad():
                for _ in range(n_mc_forward):
                    scores = surrogate(
                        batch.x, batch.edge_index, batch.batch, bool_vec,
                    )
                    all_scores.append(scores.squeeze(-1).cpu().numpy())
            surrogate.eval()

            mean_scores = np.mean(all_scores, axis=0)

            # Take top-n_top
            top_indices = np.argsort(mean_scores)[-n_top:][::-1]

            if verbose:
                print(f"  Expert {k} (clusters={clusters_k}): "
                      f"surrogate top scores = "
                      f"[{', '.join(f'{mean_scores[i]:.4f}' for i in top_indices[:5])}...]")

            # Train each top candidate for real, pick best
            best_acc = -1.0
            best_config = candidates[top_indices[0]]
            for rank, idx in enumerate(top_indices):
                config = candidates[idx]
                val_acc = evaluate_architecture_on_subset(
                    config, search_space, b_k,
                    X_train_by_cluster, y_train_by_cluster,
                    X_val, y_val,
                    epochs=cell_train_epochs,
                    val_cluster_ids=val_cluster_ids,
                )
                if verbose:
                    ops = [config.get(f"op_{i}", "?")
                           for i in range(len([k2 for k2 in config
                                               if k2.startswith("op_")]))]
                    print(f"    [{rank+1}/{n_top}] ops={ops} "
                          f"surrogate={mean_scores[idx]:.4f} "
                          f"real_acc={val_acc:.4f}"
                          f"{' *** best' if val_acc > best_acc else ''}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_config = config

            best_configs.append(best_config)
            if verbose:
                print(f"  Expert {k}: best real_acc = {best_acc:.4f}")

    return best_configs


# =========================================================================
# Lookup best architectures from observations
# =========================================================================

def find_best_archs_from_observations(
    observation_paths: List[Path],
    hard_assignments: np.ndarray,
    K: int,
    n_clusters: int,
    verbose: bool = False,
) -> Optional[List[dict]]:
    """
    Find the best architecture for each expert's b-vector by looking up
    real accuracy from collected observations.

    Returns list of K configs, or None if no matching observations found.
    """
    # Build b-vectors for each expert
    expert_b = []
    for k in range(K):
        b_k = tuple(1 if hard_assignments[m] == k else 0 for m in range(n_clusters))
        expert_b.append(b_k)

    # Scan all observations, index by b-vector
    best_by_b: dict = {}  # b_tuple -> (best_acc, best_config)
    for path in observation_paths:
        try:
            with open(path) as f:
                obs = json.load(f)
            b_tuple = tuple(obs["subset_b"])
            acc = obs["val_accuracy"]
            if b_tuple not in best_by_b or acc > best_by_b[b_tuple][0]:
                best_by_b[b_tuple] = (acc, obs["arch"])
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    # Look up best config for each expert
    configs = []
    all_found = True
    for k in range(K):
        if expert_b[k] in best_by_b:
            acc, config = best_by_b[expert_b[k]]
            configs.append(config)
            if verbose:
                clusters = [m for m in range(n_clusters) if expert_b[k][m] == 1]
                tqdm.write(f"    Obs-lookup expert {k}: clusters={clusters}, "
                           f"best_acc={acc:.4f}")
        else:
            all_found = False
            configs.append(None)

    if not all_found:
        return None
    return configs


# =========================================================================
# S-шаг: сбор наблюдений
# =========================================================================

def sample_b_near_split(
    expert_b_vectors: List[List[int]],
    n_clusters: int,
    flip_prob: float,
) -> List[int]:
    """
    Сэмплировать b-вектор «около» текущего split:
    выбрать случайный expert-b и независимо инвертировать каждый бит с вероятностью
    flip_prob. Гарантируется непустой результат.
    """
    import random as _random
    while True:
        base = _random.choice(expert_b_vectors)
        b = [bit ^ int(_random.random() < flip_prob) for bit in base]
        if sum(b) > 0:
            return b


def collect_s_step_observations(
    configs: List[dict],
    hard_assignments: np.ndarray,
    M: int,
    K: int,
    search_space,
    surrogate: nn.Module,
    X_train_by_cluster: List[np.ndarray],
    y_train_by_cluster: List[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_clusters: int,
    n_new_observations: int,
    observation_paths: List[Path],
    save_dir: str,
    obs_index: int,
    cell_train_epochs: int = 100,
    n_mc_forward: int = 20,
    n_candidates: int = 50,
    device: str = "cpu",
    verbose: bool = True,
    val_cluster_ids: Optional[np.ndarray] = None,
    focused_ratio: float = 0.5,
    explore_flip_prob: float = 0.1,
) -> Tuple[List[Path], int, List[Optional[float]]]:
    """
    S-шаг: собрать новые наблюдения (реальное обучение архитектур).

    Фаза A: оценить текущих K экспертов на их кластерах.
    Фаза B: focused exploration — случайные архитектуры на b-векторах
            текущего split (улучшает ранжирование архитектур суррогатом).
    Фаза C: UCB-guided exploration по парам (α, b), где b — возмущение
            b-вектора одного из текущих экспертов (flip каждого бита с
            вероятностью explore_flip_prob). Если expert_b_vectors пуст —
            fallback на полностью случайные b.

    focused_ratio: доля бюджета (после фазы A) на фазу B (остальное — фаза C).
    explore_flip_prob: вероятность инверсии каждого бита b в фазе C.

    Returns:
        (updated observation_paths, updated obs_index, phase_a_accs)
        phase_a_accs — список длины K с реальными val-acc каждого эксперта
        (None, если эксперт не получил ни одного кластера или бюджет new_count
        был исчерпан до его оценки).
    """
    new_count = 0
    phase_a_accs: List[Optional[float]] = [None] * K

    # --- Фаза A: оценить текущих экспертов ---
    for k in range(K):
        if new_count >= n_new_observations:
            break

        b_k = [1 if hard_assignments[m] == k else 0 for m in range(n_clusters)]
        if sum(b_k) == 0:
            continue

        val_acc = evaluate_architecture_on_subset(
            configs[k], search_space, b_k,
            X_train_by_cluster, y_train_by_cluster,
            X_val, y_val,
            epochs=cell_train_epochs,
            val_cluster_ids=val_cluster_ids,
        )
        path = save_observation(configs[k], b_k, val_acc, save_dir, obs_index)
        observation_paths.append(path)
        obs_index += 1
        new_count += 1
        phase_a_accs[k] = float(val_acc)

        if verbose:
            clusters = [m for m in range(n_clusters) if b_k[m] == 1]
            print(f"    S-step [A] expert {k}: clusters={clusters}, "
                  f"acc={val_acc:.4f}")

    remaining = n_new_observations - new_count
    n_focused = int(remaining * focused_ratio)
    n_explore = remaining - n_focused

    # --- Фаза B: focused — разные архитектуры на b-векторах текущего split ---
    # Собираем b-вектора для каждого эксперта
    expert_b_vectors = []
    for k in range(K):
        b_k = [1 if hard_assignments[m] == k else 0 for m in range(n_clusters)]
        if sum(b_k) > 0:
            expert_b_vectors.append(b_k)

    for i in range(n_focused):
        # Циклически выбираем эксперта
        b = expert_b_vectors[i % len(expert_b_vectors)]

        # UCB по архитектурам с фиксированным b-вектором
        if surrogate is not None:
            candidates = [sample_valid_config(search_space)
                          for _ in range(n_candidates)]
            best_score = -float("inf")
            best_config = candidates[0]
            for cand in candidates:
                score = compute_ucb_score(
                    surrogate, cand, b,
                    n_forward=n_mc_forward, device=device,
                )
                if score > best_score:
                    best_score = score
                    best_config = cand
            config = best_config
        else:
            config = sample_valid_config(search_space)

        val_acc = evaluate_architecture_on_subset(
            config, search_space, b,
            X_train_by_cluster, y_train_by_cluster,
            X_val, y_val,
            epochs=cell_train_epochs,
            val_cluster_ids=val_cluster_ids,
        )
        path = save_observation(config, b, val_acc, save_dir, obs_index)
        observation_paths.append(path)
        obs_index += 1
        new_count += 1

        if verbose:
            clusters = [m for m in range(n_clusters) if b[m] == 1]
            print(f"    S-step [B] UCB-focused: clusters={clusters}, "
                  f"ucb={best_score:.4f}, acc={val_acc:.4f}")

    # --- Фаза C: UCB-guided exploration по (α, b), b ≈ текущий split ---
    def _sample_b_c() -> List[int]:
        if expert_b_vectors:
            return sample_b_near_split(
                expert_b_vectors, n_clusters, explore_flip_prob,
            )
        return sample_random_b(n_clusters)

    for _ in range(n_explore):
        config = sample_valid_config(search_space)
        b = _sample_b_c()

        if surrogate is not None:
            candidates = [sample_valid_config(search_space) for _ in range(n_candidates)]
            b_vectors = [_sample_b_c() for _ in range(n_candidates)]

            best_score = -float("inf")
            best_config, best_b = config, b
            for cand_config, cand_b in zip(candidates, b_vectors):
                score = compute_ucb_score(
                    surrogate, cand_config, cand_b,
                    n_forward=n_mc_forward, device=device,
                )
                if score > best_score:
                    best_score = score
                    best_config = cand_config
                    best_b = cand_b
            config, b = best_config, best_b

        val_acc = evaluate_architecture_on_subset(
            config, search_space, b,
            X_train_by_cluster, y_train_by_cluster,
            X_val, y_val,
            epochs=cell_train_epochs,
            val_cluster_ids=val_cluster_ids,
        )
        path = save_observation(config, b, val_acc, save_dir, obs_index)
        observation_paths.append(path)
        obs_index += 1
        new_count += 1

        if verbose:
            print(f"    S-step [C] explore: b={b}, acc={val_acc:.4f}")

    return observation_paths, obs_index, phase_a_accs


# =========================================================================
# S-шаг: переобучение суррогата
# =========================================================================

def retrain_surrogate_from_observations(
    observation_paths: List[Path],
    n_features: int,
    n_clusters: int,
    surrogate_dropout: float = 0.3,
    surrogate_hidden_dim: int = 32,
    surrogate_heads: int = 4,
    surrogate_epochs: int = 200,
    surrogate_lr: float = 3e-3,
    surrogate_patience: int = 30,
    device: str = "cpu",
    verbose: bool = True,
    retrain_count: int = 0,
    model_type: str = "gat",
    nodes_per_graph: int = 4,
    base_model: Optional[nn.Module] = None,
    cluster_centers: np.ndarray = None,
) -> nn.Module:
    """Обучить суррогат на всех наблюдениях.

    Если base_model задан — дообучить его (fine-tune).
    Иначе — создать новый и обучить с нуля.
    """
    train_loader, val_loader = make_surrogate_loaders(
        observation_paths,
        val_fraction=0.2,
        seed=SEED + retrain_count,
    )

    if base_model is not None:
        surr = copy.deepcopy(base_model)
        surr.train()
        for p in surr.parameters():
            p.requires_grad_(True)
        mode = "finetuned"
    else:
        surr = create_surrogate(
            n_features, n_clusters,
            dropout=surrogate_dropout,
            hidden_dim=surrogate_hidden_dim,
            heads=surrogate_heads,
            model_type=model_type,
            nodes_per_graph=nodes_per_graph,
            cluster_centers=cluster_centers,
        )
        mode = "from scratch"

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
        print(f"    Surrogate {mode} ({len(history['train'])} epochs): "
              f"train_loss={final_train:.6f}, val_loss={final_val:.6f}")

    surr.eval()
    surr.to(device)
    for p in surr.parameters():
        p.requires_grad_(False)

    return surr


# =========================================================================
# Основная функция
# =========================================================================

def optimize_surrogate_em(
    # Данные (для S-шага)
    X: np.ndarray,
    y: np.ndarray,
    cluster_dir: str,
    # Пространство поиска
    search_space,
    M: int,
    K: int,
    # EM параметры
    n_em_iterations: int = 30,
    n_arch_candidates: int = 50,
    n_r_gradient_steps: int = 50,
    r_lr: float = 0.1,
    tau: float = 0.5,
    entropy_weight: float = 0.0,
    entropy_weight_end: Optional[float] = None,
    max_logit_spread: float = 0.0,
    # S-шаг параметры
    surrogate_retrain_every: int = 5,
    n_new_observations: int = 10,
    n_mc_forward: int = 20,
    cell_train_epochs: int = 100,
    n_candidates_s_step: int = 50,
    save_dir: str = "./surrogate_em_obs",
    # Суррогат параметры
    surrogate_dropout: float = 0.3,
    surrogate_hidden_dim: int = 32,
    surrogate_heads: int = 4,
    surrogate_epochs: int = 200,
    surrogate_lr: float = 3e-3,
    surrogate_patience: int = 30,
    # Инициализация
    initial_surrogate_path: Optional[str] = None,
    initial_obs_dir: Optional[str] = None,
    n_seed_observations: int = 50,
    init_assignment: Optional[List[int]] = None,
    per_cluster_eval: bool = False,
    focused_ratio: float = 0.5,
    explore_flip_prob: float = 0.1,
    # Суррогат тип
    model_type: str = "gat",
    nodes_per_graph: int = 4,
    # Post-EM architecture refinement
    refine_n_candidates: int = 0,
    refine_n_top: int = 20,
    refine_epochs: int = 200,
    exhaustive_refine: bool = False,
    # Общее
    device: str = "cpu",
    verbose: bool = True,
) -> OptimizationResult:
    """
    EM-алгоритм с S-шагом (одновременное обучение суррогата).
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Подготовка данных ---
    data = prepare_data(X, y, cluster_dir=cluster_dir)
    n_clusters = data["n_clusters"]
    assert M == n_clusters, (
        f"M={M} не совпадает с числом кластеров={n_clusters} в {cluster_dir}"
    )

    X_train_by_cluster = data["X_train_by_cluster"]
    y_train_by_cluster = data["y_train_by_cluster"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    val_cluster_ids = data["val_cluster_ids"] if per_cluster_eval else None

    if verbose:
        eval_mode = "per-cluster" if per_cluster_eval else "global"
        print(f"Evaluation mode: {eval_mode}")

    n_features = len(OPS)
    observation_paths: List[Path] = []
    obs_index = 0
    retrain_count = 0

    # Load cluster centers for data-aware surrogate features
    centers_path = os.path.join(cluster_dir, "cluster_centers.npy")
    cluster_centers = np.load(centers_path) if os.path.exists(centers_path) else None

    # --- Инициализация наблюдений и суррогата ---
    if initial_obs_dir is not None:
        # Загрузить существующие наблюдения
        existing = sorted(Path(initial_obs_dir).glob("obs_*.json"))
        observation_paths = list(existing)
        if existing:
            obs_index = max(int(p.stem.split("_")[1]) for p in existing) + 1
        if verbose:
            print(f"Loaded {len(observation_paths)} existing observations "
                  f"from {initial_obs_dir}")

        if initial_surrogate_path is not None:
            # Режим 1a: obs + предобученный суррогат → fine-tune
            if verbose:
                print(f"Loading pre-trained surrogate from {initial_surrogate_path}")
            surrogate = load_surrogate(
                initial_surrogate_path,
                n_features=n_features, M=M,
                dropout=surrogate_dropout,
                hidden_dim=surrogate_hidden_dim,
                heads=surrogate_heads,
                device=device,
                model_type=model_type,
                nodes_per_graph=nodes_per_graph,
                cluster_centers=cluster_centers,
            )
            retrain_count += 1
            surrogate = retrain_surrogate_from_observations(
                observation_paths, n_features, n_clusters,
                surrogate_dropout=surrogate_dropout,
                surrogate_hidden_dim=surrogate_hidden_dim,
                surrogate_heads=surrogate_heads,
                surrogate_epochs=surrogate_epochs,
                surrogate_lr=surrogate_lr,
                surrogate_patience=surrogate_patience,
                device=device, verbose=verbose,
                retrain_count=retrain_count,
                model_type=model_type,
                nodes_per_graph=nodes_per_graph,
                base_model=surrogate,
                cluster_centers=cluster_centers,
            )
        else:
            # Режим 1b: obs → обучить суррогат с нуля
            retrain_count += 1
            surrogate = retrain_surrogate_from_observations(
                observation_paths, n_features, n_clusters,
                surrogate_dropout=surrogate_dropout,
                surrogate_hidden_dim=surrogate_hidden_dim,
                surrogate_heads=surrogate_heads,
                surrogate_epochs=surrogate_epochs,
                surrogate_lr=surrogate_lr,
                surrogate_patience=surrogate_patience,
                device=device, verbose=verbose,
                retrain_count=retrain_count,
                model_type=model_type,
                nodes_per_graph=nodes_per_graph,
                cluster_centers=cluster_centers,
            )

    elif initial_surrogate_path is not None:
        # Режим 2: только предобученный суррогат (без наблюдений)
        if verbose:
            print(f"Loading pre-trained surrogate from {initial_surrogate_path}")
        surrogate = load_surrogate(
            initial_surrogate_path,
            n_features=n_features, M=M,
            dropout=surrogate_dropout,
            hidden_dim=surrogate_hidden_dim,
            heads=surrogate_heads,
            device=device,
            model_type=model_type,
            nodes_per_graph=nodes_per_graph,
            cluster_centers=cluster_centers,
        )
        for p in surrogate.parameters():
            p.requires_grad_(False)

    else:
        # Режим 3: собрать seed-датасет с нуля
        if verbose:
            print(f"Collecting {n_seed_observations} seed observations...")

        for i in tqdm(range(n_seed_observations), desc="Seed",
                      disable=not verbose):
            config = sample_valid_config(search_space)
            b = sample_random_b(n_clusters)
            val_acc = evaluate_architecture_on_subset(
                config, search_space, b,
                X_train_by_cluster, y_train_by_cluster,
                X_val, y_val,
                epochs=cell_train_epochs,
                val_cluster_ids=val_cluster_ids,
            )
            path = save_observation(config, b, val_acc, save_dir, obs_index)
            observation_paths.append(path)
            obs_index += 1

        if verbose:
            print(f"Seed dataset: {len(observation_paths)} observations")

        retrain_count += 1
        surrogate = retrain_surrogate_from_observations(
            observation_paths, n_features, n_clusters,
            surrogate_dropout=surrogate_dropout,
            surrogate_hidden_dim=surrogate_hidden_dim,
            surrogate_heads=surrogate_heads,
            surrogate_epochs=surrogate_epochs,
            surrogate_lr=surrogate_lr,
            surrogate_patience=surrogate_patience,
            device=device, verbose=verbose,
            retrain_count=retrain_count,
            model_type=model_type,
            nodes_per_graph=nodes_per_graph,
            cluster_centers=cluster_centers,
        )

    # --- EM инициализация ---
    if init_assignment is not None:
        init_assignments = np.array(init_assignment)
    else:
        # Случайная сбалансированная инициализация: каждый кластер
        # случайно назначается одному из K экспертов
        init_assignments = np.random.randint(0, K, size=M)
        # Гарантируем, что каждый эксперт получит хотя бы один кластер
        for k in range(K):
            if k not in init_assignments:
                init_assignments[np.random.randint(0, M)] = k
    logits = torch.zeros(M, K, device=device)
    for m in range(M):
        logits[m, init_assignments[m]] = 2.0  # bias в сторону назначенного
    if verbose:
        print(f"Initial assignment: {init_assignments.tolist()}")
    configs = [sample_valid_config(search_space) for _ in range(K)]

    history: List[float] = []
    phase_a_history: List[List[Optional[float]]] = []
    best_log_lik = -float("inf")
    best_logits = logits.clone()
    best_configs = copy.deepcopy(configs)

    iterator = range(1, n_em_iterations + 1)
    if verbose:
        iterator = tqdm(iterator, desc="EM+S Algorithm")

    # Entropy annealing setup
    ew_start = entropy_weight
    ew_end = entropy_weight_end if entropy_weight_end is not None else entropy_weight
    if verbose and ew_start != ew_end:
        print(f"Entropy annealing: {ew_start} -> {ew_end}")

    for em_iter in iterator:
        # Linearly anneal entropy weight
        if n_em_iterations > 1:
            t = (em_iter - 1) / (n_em_iterations - 1)
            current_entropy_weight = ew_start + (ew_end - ew_start) * t
        else:
            current_entropy_weight = ew_end

        # =============================================================
        # E-step: q_{mk} ∝ r_{mk} · u(α_k, R_k)
        # =============================================================
        with torch.no_grad():
            r = F.softmax(logits, dim=-1)  # [M, K]
            r_np = r.cpu().numpy()

            R_hard = r_to_hard_matrix(r_np)  # [M, K]
            R_hard_columns = torch.tensor(
                R_hard.T, dtype=torch.float, device=device,
            )  # [K, M]

            u_values = surrogate_eval_batch(
                surrogate, configs, R_hard_columns, device,
            )
            u_vals_np = np.maximum(u_values.cpu().numpy(), 1e-10)

        # q_{mk} = r_{mk} * u_k, нормализация по k
        q = r_np * u_vals_np[None, :]  # [M, K]
        q_row_sums = np.maximum(q.sum(axis=1, keepdims=True), 1e-10)
        q = q / q_row_sums
        q_tensor = torch.tensor(q, dtype=torch.float, device=device)

        log_lik = compute_log_likelihood_numpy(r_np, u_vals_np)
        history.append(log_lik)

        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_logits = logits.clone()
            best_configs = copy.deepcopy(configs)

        if verbose:
            hard_current_log = discretize_assignments(r_np)
            counts = [int(np.sum(hard_current_log == k)) for k in range(K)]
            tqdm.write(f"  EM iter {em_iter}: log-lik = {log_lik:.4f} "
                       f"ew={current_entropy_weight:.3f} "
                       f"split={counts} "
                       f"(obs: {len(observation_paths)})")

        # =============================================================
        # M-step часть 1: оптимизация r (logits)
        # =============================================================
        logits = logits.detach().clone().requires_grad_(True)
        r_optimizer = torch.optim.Adam([logits], lr=r_lr)

        graph_batch = prebuild_graph_batch(configs, device)

        for _r_step in range(n_r_gradient_steps):
            r_optimizer.zero_grad()

            gumbel_r = gumbel_softmax_rows(logits, tau=tau, hard=True)
            R_cols = gumbel_r.t()  # [K, M]

            u_vals = surrogate_eval_with_prebatch(surrogate, graph_batch, R_cols)
            u_vals = torch.clamp(u_vals, min=1e-10)

            r_soft = F.softmax(logits, dim=-1)
            log_r = torch.log(torch.clamp(r_soft, min=1e-10))
            log_u = torch.log(u_vals)

            log_ru = log_r + log_u.unsqueeze(0)
            q_function = (q_tensor * log_ru).sum()

            # Entropy regularization: поощряем сбалансированное назначение
            # H(r) = -Σ_m Σ_k r_mk * log(r_mk) — максимизируем энтропию
            entropy = -(r_soft * log_r).sum()

            loss = -q_function - current_entropy_weight * entropy
            loss.backward()
            r_optimizer.step()

            # Logit clipping: prevent collapse by bounding logit spread per row
            if max_logit_spread > 0:
                with torch.no_grad():
                    row_mean = logits.mean(dim=-1, keepdim=True)
                    logits.data = row_mean + (logits.data - row_mean).clamp(
                        -max_logit_spread / 2, max_logit_spread / 2
                    )

        # =============================================================
        # M-step часть 2: поиск архитектур
        # =============================================================
        with torch.no_grad():
            r_current = F.softmax(logits, dim=-1)
            R_cols_current = r_current.t()
            r_np_current = r_current.cpu().numpy()

        hard_for_arch = discretize_assignments(r_np_current)

        # First try: lookup best architectures from observations
        obs_configs = find_best_archs_from_observations(
            observation_paths, hard_for_arch, K, n_clusters,
        )
        if obs_configs is not None:
            configs = obs_configs
        else:
            # Fallback: surrogate-based selection.
            # Pass logits directly — sampling of hard one-hot R happens inside
            # (fresh Gumbel-Softmax per MC forward pass; honours current soft r).
            configs = sample_architectures_for_experts(
                search_space, K, n_arch_candidates, surrogate,
                logits.detach(), device,
            )

        # =============================================================
        # S-step: обновить суррогат (каждые surrogate_retrain_every итераций)
        # =============================================================
        if em_iter % surrogate_retrain_every == 0:
            if verbose:
                tqdm.write(f"\n  === S-step at EM iter {em_iter} ===")

            # Текущие лучшие назначения для сбора наблюдений
            with torch.no_grad():
                r_best = F.softmax(best_logits, dim=-1).cpu().numpy()
            hard_current = discretize_assignments(r_best)

            observation_paths, obs_index, phase_a_accs = collect_s_step_observations(
                configs=best_configs,
                hard_assignments=hard_current,
                M=M, K=K,
                search_space=search_space,
                surrogate=surrogate,
                X_train_by_cluster=X_train_by_cluster,
                y_train_by_cluster=y_train_by_cluster,
                X_val=X_val, y_val=y_val,
                n_clusters=n_clusters,
                n_new_observations=n_new_observations,
                observation_paths=observation_paths,
                save_dir=save_dir,
                obs_index=obs_index,
                cell_train_epochs=cell_train_epochs,
                n_mc_forward=n_mc_forward,
                n_candidates=n_candidates_s_step,
                device=device,
                verbose=verbose,
                val_cluster_ids=val_cluster_ids,
                focused_ratio=focused_ratio,
                explore_flip_prob=explore_flip_prob,
            )
            phase_a_history.append(phase_a_accs)

            retrain_count += 1
            surrogate = retrain_surrogate_from_observations(
                observation_paths, n_features, n_clusters,
                surrogate_dropout=surrogate_dropout,
                surrogate_hidden_dim=surrogate_hidden_dim,
                surrogate_heads=surrogate_heads,
                surrogate_epochs=surrogate_epochs,
                surrogate_lr=surrogate_lr,
                surrogate_patience=surrogate_patience,
                device=device, verbose=verbose,
                retrain_count=retrain_count,
                model_type=model_type,
                nodes_per_graph=nodes_per_graph,
                base_model=surrogate,
                cluster_centers=cluster_centers,
            )

            # After S-step: lookup best architectures from observations
            obs_configs = find_best_archs_from_observations(
                observation_paths, hard_current, K, n_clusters,
                verbose=verbose,
            )
            if obs_configs is not None:
                configs = obs_configs

            if verbose:
                tqdm.write(f"  S-step done: {len(observation_paths)} total "
                           f"observations\n")

    # --- Post-EM architecture refinement ---
    with torch.no_grad():
        r_final = F.softmax(best_logits, dim=-1).cpu().numpy()
    final_hard = discretize_assignments(r_final)

    if exhaustive_refine or refine_n_candidates > 0:
        if verbose:
            if exhaustive_refine:
                print(f"\n=== Post-EM exhaustive architecture search ===")
                print(f"  Training epochs: {refine_epochs}")
            else:
                print(f"\n=== Post-EM architecture refinement ===")
                print(f"  Candidates: {refine_n_candidates}, "
                      f"top: {refine_n_top}, epochs: {refine_epochs}")

        refined_configs = refine_architectures_real(
            hard_assignments=final_hard,
            K=K,
            n_clusters=n_clusters,
            search_space=search_space,
            surrogate=surrogate,
            X_train_by_cluster=X_train_by_cluster,
            y_train_by_cluster=y_train_by_cluster,
            X_val=X_val,
            y_val=y_val,
            n_candidates=refine_n_candidates,
            n_top=refine_n_top,
            cell_train_epochs=refine_epochs,
            n_mc_forward=n_mc_forward,
            device=device,
            val_cluster_ids=val_cluster_ids,
            verbose=verbose,
            exhaustive=exhaustive_refine,
        )
        best_configs = refined_configs

    return OptimizationResult(
        configs=best_configs,
        r_matrix=r_final,
        hard_assignments=final_hard,
        objective_value=best_log_lik,
        history=history,
        method="surrogate_em",
        phase_a_history=phase_a_history,
    )


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EM-алгоритм с одновременным обучением суррогата (S-шаг)"
    )
    add_common_args(parser)

    # EM параметры
    parser.add_argument("--n-em-iterations", type=int, default=30)
    parser.add_argument("--n-r-gradient-steps", type=int, default=50)
    parser.add_argument("--r-lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.0,
                        help="Вес entropy regularization для r (0=выкл)")
    parser.add_argument("--entropy-weight-end", type=float, default=None,
                        help="Конечный вес entropy (для annealing). Если задан, "
                             "entropy-weight линейно уменьшается от start до end")
    parser.add_argument("--max-logit-spread", type=float, default=0.0,
                        help="Макс. разница logit от среднего в строке (0=выкл). "
                             "Пример: 2.0 -> min softmax ~0.12 для K=2")

    # S-шаг параметры
    parser.add_argument("--surrogate-retrain-every", type=int, default=5,
                        help="Переобучать суррогат каждые N EM-итераций")
    parser.add_argument("--n-new-observations", type=int, default=10,
                        help="Число новых наблюдений за один S-шаг")
    parser.add_argument("--n-mc-forward", type=int, default=20,
                        help="MC Dropout forward passes для UCB")
    parser.add_argument("--n-candidates-s-step", type=int, default=50,
                        help="Число кандидатов при UCB-exploration в S-шаге")
    parser.add_argument("--cell-train-epochs", type=int, default=100,
                        help="Эпохи обучения ячейки при реальной оценке")
    parser.add_argument("--obs-save-dir", type=str, default="./runs/surrogate_em_obs",
                        help="Директория для сохранения наблюдений S-шага")

    # Суррогат параметры (обучение)
    parser.add_argument("--surrogate-train-epochs", type=int, default=200,
                        help="Эпохи обучения суррогата")
    parser.add_argument("--surrogate-train-lr", type=float, default=3e-3,
                        help="Learning rate обучения суррогата")
    parser.add_argument("--surrogate-train-patience", type=int, default=30,
                        help="Early stopping patience суррогата")

    # Инициализация
    parser.add_argument("--initial-obs-dir", type=str, default=None,
                        help="Директория с существующими obs_*.json для warm start")
    parser.add_argument("--n-seed-observations", type=int, default=50,
                        help="Число seed-наблюдений при старте с нуля")
    parser.add_argument("--init-assignment", type=int, nargs="+", default=None,
                        help="Начальное назначение кластеров экспертам "
                             "(длина M, значения 0..K-1)")
    parser.add_argument("--per-cluster-eval", action="store_true",
                        help="S-step: оценивать accuracy только на val-точках из "
                             "выбранных кластеров (per-cluster eval)")
    parser.add_argument("--focused-ratio", type=float, default=0.5,
                        help="Доля S-step бюджета на focused exploration "
                             "(разные архитектуры на текущем split). "
                             "Остальное — UCB exploration вблизи текущего split")
    parser.add_argument("--explore-flip-prob", type=float, default=0.1,
                        help="Вероятность инверсии каждого бита b в фазе C: "
                             "b сэмплируется как возмущение столбца текущего "
                             "hard-назначения. 0.0 → b точно совпадает с "
                             "expert-колонкой, 0.5 → полностью случайный b")

    # Post-EM architecture refinement
    parser.add_argument("--refine-n-candidates", type=int, default=0,
                        help="Число кандидатов для post-EM refinement (0=выкл). "
                             "Сэмплируются случайно, ранжируются суррогатом, "
                             "top-N обучаются реально")
    parser.add_argument("--refine-n-top", type=int, default=20,
                        help="Сколько лучших по суррогату обучить реально")
    parser.add_argument("--refine-epochs", type=int, default=200,
                        help="Эпохи обучения при refinement")
    parser.add_argument("--exhaustive-refine", action="store_true",
                        help="Exhaustive architecture search: enumerate ALL valid "
                             "architectures and train each one (ignores "
                             "refine-n-candidates and refine-n-top)")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Загрузка данных ---
    data_dir = Path(args.data_dir)
    cluster_dir = args.cluster_dir if args.cluster_dir else str(data_dir)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    ss = create_search_space(input_dim=args.input_dim, num_nodes_per_cell=args.n_nodes)

    # Определить initial_surrogate_path
    initial_surrogate_path = None
    if os.path.exists(args.surrogate_path):
        initial_surrogate_path = args.surrogate_path

    result = optimize_surrogate_em(
        X=X, y=y, cluster_dir=cluster_dir,
        search_space=ss, M=args.M, K=args.K,
        # EM
        n_em_iterations=args.n_em_iterations,
        n_arch_candidates=args.n_arch_candidates,
        n_r_gradient_steps=args.n_r_gradient_steps,
        r_lr=args.r_lr,
        tau=args.tau,
        entropy_weight=args.entropy_weight,
        entropy_weight_end=args.entropy_weight_end,
        max_logit_spread=args.max_logit_spread,
        # S-step
        surrogate_retrain_every=args.surrogate_retrain_every,
        n_new_observations=args.n_new_observations,
        n_mc_forward=args.n_mc_forward,
        cell_train_epochs=args.cell_train_epochs,
        n_candidates_s_step=args.n_candidates_s_step,
        save_dir=args.obs_save_dir,
        # Surrogate training
        surrogate_dropout=args.surrogate_dropout,
        surrogate_hidden_dim=args.surrogate_hidden_dim,
        surrogate_heads=args.surrogate_heads,
        surrogate_epochs=args.surrogate_train_epochs,
        surrogate_lr=args.surrogate_train_lr,
        surrogate_patience=args.surrogate_train_patience,
        # Init
        initial_surrogate_path=initial_surrogate_path,
        initial_obs_dir=args.initial_obs_dir,
        n_seed_observations=args.n_seed_observations,
        init_assignment=args.init_assignment,
        per_cluster_eval=args.per_cluster_eval,
        focused_ratio=args.focused_ratio,
        explore_flip_prob=args.explore_flip_prob,
        # Surrogate type
        model_type=args.surrogate_type,
        nodes_per_graph=args.n_nodes + 1,  # +1 for input node
        # Post-EM refinement
        refine_n_candidates=args.refine_n_candidates,
        refine_n_top=args.refine_n_top,
        refine_epochs=args.refine_epochs,
        exhaustive_refine=args.exhaustive_refine,
        # General
        device=args.device,
        verbose=True,
    )
    print_result(result)

    # --- Реальная оценка ---
    print("\n=== Реальная оценка (переобучение на кластерах) ===")
    evaluate_result_real(result, data_dir=args.data_dir, cluster_dir=cluster_dir)

    if args.save_results:
        save_results({"surrogate_em": result}, args.save_results)


if __name__ == "__main__":
    main()

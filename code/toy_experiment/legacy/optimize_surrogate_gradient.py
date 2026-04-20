"""
Градиентный подъём с одновременным обучением суррогатной функции (S-шаг).

Комбинация gradient optimization (optimize_gradient.py) с S-шагом
из optimize_surrogate_em.py: каждые surrogate_retrain_every шагов
собираются новые наблюдения и суррогат переобучается.

На итерации t:
    1. Gradient step: logits → Gumbel-Softmax → surrogate → log-likelihood → backprop
    2. Arch re-sampling (каждые arch_search_every шагов)
    3. S-step (каждые surrogate_retrain_every шагов):
       - Собрать новые наблюдения (текущие эксперты + UCB exploration)
       - Переобучить суррогат с нуля

Использование:
    # С нуля (seed-датасет + gradient + S-step)
    python optimize_surrogate_gradient.py --data-dir ./data --M 20 --K 2

    # С существующими наблюдениями
    python optimize_surrogate_gradient.py --data-dir ./data --M 20 --K 2 \\
        --initial-obs-dir ./runs/surrogate_em_4class_v2
"""

import copy
import os
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from code.toy_experiment.legacy.toy_graph import OPS
from code.toy_experiment.legacy.optimize_expert_assignments import (
    OptimizationResult,
    surrogate_eval_with_prebatch,
    compute_log_likelihood,
    discretize_assignments,
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
from code.toy_experiment.legacy.collect_dataset import (
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
from optimize_surrogate_em import (
    collect_s_step_observations,
    retrain_surrogate_from_observations,
)


# =========================================================================
# Основная функция
# =========================================================================

def optimize_surrogate_gradient(
    # Данные (для S-шага)
    X: np.ndarray,
    y: np.ndarray,
    cluster_dir: str,
    # Пространство поиска
    search_space,
    M: int,
    K: int,
    # Gradient параметры
    n_steps: int = 300,
    lr: float = 0.1,
    tau_start: float = 2.0,
    tau_end: float = 0.1,
    n_arch_candidates: int = 50,
    arch_search_every: int = 20,
    entropy_weight: float = 0.0,
    max_logit_spread: float = 0.0,
    # S-шаг параметры
    surrogate_retrain_every: int = 50,
    n_new_observations: int = 10,
    n_mc_forward: int = 20,
    cell_train_epochs: int = 100,
    n_candidates_s_step: int = 50,
    save_dir: str = "./surrogate_gradient_obs",
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
    # Общее
    device: str = "cpu",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Градиентный подъём с S-шагом (одновременное обучение суррогата).
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

    # --- Инициализация наблюдений и суррогата ---
    if initial_obs_dir is not None:
        existing = sorted(Path(initial_obs_dir).glob("obs_*.json"))
        observation_paths = list(existing)
        if existing:
            obs_index = max(int(p.stem.split("_")[1]) for p in existing) + 1
        if verbose:
            print(f"Loaded {len(observation_paths)} existing observations "
                  f"from {initial_obs_dir}")

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
        )

    elif initial_surrogate_path is not None:
        if verbose:
            print(f"Loading pre-trained surrogate from {initial_surrogate_path}")
        surrogate = load_surrogate(
            initial_surrogate_path,
            n_features=n_features, M=M,
            dropout=surrogate_dropout,
            hidden_dim=surrogate_hidden_dim,
            heads=surrogate_heads,
            device=device,
        )
        for p in surrogate.parameters():
            p.requires_grad_(False)

    else:
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
        )

    # --- Инициализация logits ---
    if init_assignment is not None:
        init_assignments = np.array(init_assignment)
    else:
        init_assignments = np.random.randint(0, K, size=M)
        for k in range(K):
            if k not in init_assignments:
                init_assignments[np.random.randint(0, M)] = k

    logits = torch.zeros(M, K, device=device)
    for m in range(M):
        logits[m, init_assignments[m]] = 2.0
    logits.requires_grad_(True)

    if verbose:
        print(f"Initial assignment: {init_assignments.tolist()}")

    optimizer = torch.optim.Adam([logits], lr=lr)
    configs = [sample_valid_config(search_space) for _ in range(K)]
    graph_batch = prebuild_graph_batch(configs, device)

    history: List[float] = []
    best_log_lik = -float("inf")
    best_logits = logits.detach().clone()
    best_configs = copy.deepcopy(configs)

    iterator = range(1, n_steps + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Gradient+S Optimization")

    for step in iterator:
        progress = step / n_steps
        tau = tau_start * (tau_end / tau_start) ** progress

        optimizer.zero_grad()

        # Gumbel-Softmax: hard=True → straight-through estimator
        gumbel_r = gumbel_softmax_rows(logits, tau=tau, hard=True)  # [M, K]
        R_columns = gumbel_r.t()  # [K, M]

        # Суррогат: u(α_k, R_k)
        u_values = surrogate_eval_with_prebatch(surrogate, graph_batch, R_columns)
        u_values = torch.clamp(u_values, min=1e-10)  # [K]

        # Softmax r для целевой функции
        r_soft = F.softmax(logits, dim=-1)  # [M, K]

        # Log-likelihood
        log_lik = compute_log_likelihood(r_soft, u_values)

        # Entropy regularization
        if entropy_weight > 0:
            log_r = torch.log(torch.clamp(r_soft, min=1e-10))
            entropy = -(r_soft * log_r).sum()
            loss = -log_lik - entropy_weight * entropy
        else:
            loss = -log_lik

        loss.backward()
        optimizer.step()

        # Logit clipping
        if max_logit_spread > 0:
            with torch.no_grad():
                row_mean = logits.mean(dim=-1, keepdim=True)
                logits.data = row_mean + (logits.data - row_mean).clamp(
                    -max_logit_spread / 2, max_logit_spread / 2
                )

        log_lik_val = log_lik.item()
        history.append(log_lik_val)

        if log_lik_val > best_log_lik:
            best_log_lik = log_lik_val
            best_logits = logits.detach().clone()
            best_configs = copy.deepcopy(configs)

        # --- Arch re-sampling ---
        if step % arch_search_every == 0:
            with torch.no_grad():
                r_current = F.softmax(logits, dim=-1)
                R_cols_current = r_current.t()

            configs = sample_architectures_for_experts(
                search_space, K, n_arch_candidates, surrogate,
                R_cols_current, device,
            )
            graph_batch = prebuild_graph_batch(configs, device)

        # --- S-step ---
        if step % surrogate_retrain_every == 0:
            with torch.no_grad():
                r_best = F.softmax(best_logits, dim=-1).cpu().numpy()
            hard_current = discretize_assignments(r_best)

            counts = [int(np.sum(hard_current == k)) for k in range(K)]
            if verbose:
                tqdm.write(f"\n  === S-step at step {step} ===")
                tqdm.write(f"  log-lik={log_lik_val:.4f} tau={tau:.3f} "
                           f"split={counts} (obs: {len(observation_paths)})")

            observation_paths, obs_index = collect_s_step_observations(
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
            )

            # Rebuild graph batch with new surrogate
            graph_batch = prebuild_graph_batch(configs, device)

            if verbose:
                tqdm.write(f"  S-step done: {len(observation_paths)} total "
                           f"observations\n")

    # --- Финальная оценка ---
    with torch.no_grad():
        r_final = F.softmax(best_logits, dim=-1).cpu().numpy()

    return OptimizationResult(
        configs=best_configs,
        r_matrix=r_final,
        hard_assignments=discretize_assignments(r_final),
        objective_value=best_log_lik,
        history=history,
        method="surrogate_gradient",
    )


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Градиентный подъём с одновременным обучением суррогата (S-шаг)"
    )
    add_common_args(parser)

    # Gradient параметры
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--arch-search-every", type=int, default=20)
    parser.add_argument("--entropy-weight", type=float, default=0.0,
                        help="Вес entropy regularization для r (0=выкл)")
    parser.add_argument("--max-logit-spread", type=float, default=0.0,
                        help="Макс. разница logit от среднего в строке (0=выкл)")

    # S-шаг параметры
    parser.add_argument("--surrogate-retrain-every", type=int, default=50,
                        help="Переобучать суррогат каждые N gradient steps")
    parser.add_argument("--n-new-observations", type=int, default=10,
                        help="Число новых наблюдений за один S-шаг")
    parser.add_argument("--n-mc-forward", type=int, default=20,
                        help="MC Dropout forward passes для UCB")
    parser.add_argument("--n-candidates-s-step", type=int, default=50,
                        help="Число кандидатов при UCB-exploration в S-шаге")
    parser.add_argument("--cell-train-epochs", type=int, default=100,
                        help="Эпохи обучения ячейки при реальной оценке")
    parser.add_argument("--obs-save-dir", type=str,
                        default="./runs/surrogate_gradient_obs",
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
                             "(разные архитектуры на текущем split)")

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Загрузка данных ---
    data_dir = Path(args.data_dir)
    cluster_dir = args.cluster_dir if args.cluster_dir else str(data_dir)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    ss = create_search_space(
        input_dim=args.input_dim, num_nodes_per_cell=args.n_nodes,
    )

    # Определить initial_surrogate_path
    initial_surrogate_path = None
    if args.initial_obs_dir is None and os.path.exists(args.surrogate_path):
        initial_surrogate_path = args.surrogate_path

    result = optimize_surrogate_gradient(
        X=X, y=y, cluster_dir=cluster_dir,
        search_space=ss, M=args.M, K=args.K,
        # Gradient
        n_steps=args.n_steps,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        n_arch_candidates=args.n_arch_candidates,
        arch_search_every=args.arch_search_every,
        entropy_weight=args.entropy_weight,
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
        # General
        device=args.device,
        verbose=True,
    )
    print_result(result)

    # --- Реальная оценка ---
    print("\n=== Реальная оценка (переобучение на кластерах) ===")
    evaluate_result_real(result, data_dir=args.data_dir, cluster_dir=cluster_dir)

    if args.save_results:
        save_results({"surrogate_gradient": result}, args.save_results)


if __name__ == "__main__":
    main()

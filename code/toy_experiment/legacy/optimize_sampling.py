"""
Метод 1: Поиск через сэмплирование (Sampling Search).

Сэмплирование жёстких назначений кластеров → подбор архитектур → выбор лучшего.
Суррогат вызывается на дискретных бинарных R_k — без аппроксимации.

Использование:
    python optimize_sampling.py --surrogate-path ./runs/surr_best.pth --M 2 --K 2
"""

import copy
import argparse
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from code.toy_experiment.legacy.optimize_expert_assignments import (
    OptimizationResult,
    surrogate_eval_batch,
    compute_log_likelihood_numpy,
    discretize_assignments,
    sample_hard_assignment_matrix,
    sample_architectures_for_experts,
    print_result,
    add_common_args,
    setup_from_args,
    save_results,
    evaluate_result_real,
)


def optimize_sampling(
    surrogate,
    search_space,
    M: int,
    K: int,
    n_assignment_samples: int = 200,
    n_arch_candidates: int = 50,
    device: str = "cpu",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Метод 1: полный / частичный поиск через сэмплирование.

    Алгоритм:
        1. Сэмплировать n_assignment_samples жёстких назначений кластеров экспертам.
        2. Для каждого назначения:
           a. R_k = r[:, k] — бинарный вектор (какие кластеры у эксперта k).
           b. Подобрать лучшие архитектуры для каждого эксперта через сэмплирование.
           c. Вычислить log-likelihood.
        3. Вернуть назначение с максимальным log-likelihood.
    """
    surrogate.eval()
    surrogate.to(device)

    best_result = None
    best_log_lik = -float("inf")
    history: List[float] = []

    iterator = range(n_assignment_samples)
    if verbose:
        iterator = tqdm(iterator, desc="Sampling Search")

    for i in iterator:
        # 1. Сэмплировать жёсткое назначение
        r_hard = sample_hard_assignment_matrix(M, K)

        # 2. R_k — столбцы матрицы r
        R_columns = torch.tensor(r_hard.T, dtype=torch.float)  # [K, M]

        # Проверка: каждый R_k должен иметь хотя бы один ненулевой элемент
        has_empty = any(R_columns[k].sum() == 0 for k in range(K))
        if has_empty:
            history.append(best_log_lik if best_log_lik > -float("inf") else float("nan"))
            continue

        # 3. Подобрать лучшие архитектуры
        configs = sample_architectures_for_experts(
            search_space, K, n_arch_candidates, surrogate, R_columns, device,
        )

        # 4. Вычислить u-values
        with torch.no_grad():
            u_vals = surrogate_eval_batch(surrogate, configs, R_columns, device)
            u_vals_np = u_vals.cpu().numpy()

        # 5. Log-likelihood
        log_lik = compute_log_likelihood_numpy(r_hard, u_vals_np)
        history.append(log_lik)

        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_result = OptimizationResult(
                configs=copy.deepcopy(configs),
                r_matrix=r_hard.copy(),
                hard_assignments=discretize_assignments(r_hard),
                objective_value=log_lik,
                method="sampling",
            )

    if best_result is None:
        raise RuntimeError("Не удалось найти ни одного валидного назначения.")

    best_result.history = history
    return best_result


def main():
    parser = argparse.ArgumentParser(
        description="Метод 1: Sampling Search для назначения экспертов"
    )
    add_common_args(parser)
    parser.add_argument("--n-assignment-samples", type=int, default=200)
    args = parser.parse_args()

    surrogate, ss = setup_from_args(args)

    result = optimize_sampling(
        surrogate, ss, args.M, args.K,
        n_assignment_samples=args.n_assignment_samples,
        n_arch_candidates=args.n_arch_candidates,
        device=args.device,
    )
    print_result(result)

    # --- Реальная оценка архитектур ---
    print("\n=== Реальная оценка (переобучение на кластерах) ===")
    cluster_dir = args.cluster_dir if args.cluster_dir else args.data_dir
    evaluate_result_real(result, data_dir=args.data_dir, cluster_dir=cluster_dir)

    if args.save_results:
        save_results({"sampling": result}, args.save_results)


if __name__ == "__main__":
    main()

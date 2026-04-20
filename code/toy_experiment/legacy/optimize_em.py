"""
Метод 3: EM-алгоритм для оптимизации назначения экспертов.

E-step: q_{mk} ∝ r_{mk} · u(α_k, R_k), нормализация по k.
M-step: оптимизация r (gradient) и α (sampling).

ASSUMPTION: в E-step для вычисления u(α_k, R_k) R_k дискретизируется
через argmax жёсткого назначения, так как суррогат обучен на бинарных векторах.

Использование:
    python optimize_em.py --surrogate-path ./runs/surr_best.pth --M 2 --K 2
"""

import copy
import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from code.toy_experiment.legacy.optimize_expert_assignments import (
    OptimizationResult,
    surrogate_eval_batch,
    surrogate_eval_with_prebatch,
    compute_log_likelihood_numpy,
    discretize_assignments,
    r_to_hard_matrix,
    gumbel_softmax_rows,
    prebuild_graph_batch,
    sample_architectures_for_experts,
    sample_valid_config,
    print_result,
    add_common_args,
    setup_from_args,
    save_results,
    evaluate_result_real,
)


def optimize_em(
    surrogate,
    search_space,
    M: int,
    K: int,
    n_em_iterations: int = 30,
    n_arch_candidates: int = 50,
    n_r_gradient_steps: int = 50,
    r_lr: float = 0.1,
    tau: float = 0.5,
    device: str = "cpu",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Метод 3: EM-алгоритм.

    E-step:
        q_{mk} ∝ r_{mk} · u(α_k, R_k),  нормализация по k для каждого m.

    M-step:
        Максимизировать: Σ_m Σ_k q_{mk} · log(r_{mk} · u(α_k, R_k))
        По r / R_k: градиентная оптимизация logits (с Gumbel-Softmax).
        По α_k: сэмплирование / search по архитектурам.
    """
    surrogate.eval()
    surrogate.to(device)
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # --- Инициализация ---
    logits = torch.zeros(M, K, device=device)
    configs = [sample_valid_config(search_space) for _ in range(K)]

    history: List[float] = []
    best_log_lik = -float("inf")
    best_logits = logits.clone()
    best_configs = copy.deepcopy(configs)

    iterator = range(1, n_em_iterations + 1)
    if verbose:
        iterator = tqdm(iterator, desc="EM Algorithm")

    for em_iter in iterator:
        # =============================================================
        # E-step: q_{mk} ∝ r_{mk} · u(α_k, R_k)
        # =============================================================
        with torch.no_grad():
            r = F.softmax(logits, dim=-1)  # [M, K]
            r_np = r.cpu().numpy()

            # Дискретизировать R_k для суррогата
            R_hard = r_to_hard_matrix(r_np)  # [M, K], one-hot по строкам
            R_hard_columns = torch.tensor(
                R_hard.T, dtype=torch.float, device=device,
            )  # [K, M]

            # u(α_k, R_k) с дискретными R_k
            u_values = surrogate_eval_batch(
                surrogate, configs, R_hard_columns, device,
            )
            u_vals_np = np.maximum(u_values.cpu().numpy(), 1e-10)

        # q_{mk} = r_{mk} * u_k, нормализация по k
        q = r_np * u_vals_np[None, :]  # [M, K]
        q_row_sums = np.maximum(q.sum(axis=1, keepdims=True), 1e-10)
        q = q / q_row_sums  # [M, K], нормализованное
        q_tensor = torch.tensor(q, dtype=torch.float, device=device)

        # Текущий log-likelihood для мониторинга
        log_lik = compute_log_likelihood_numpy(r_np, u_vals_np)
        history.append(log_lik)

        if log_lik > best_log_lik:
            best_log_lik = log_lik
            best_logits = logits.clone()
            best_configs = copy.deepcopy(configs)

        if verbose:
            tqdm.write(f"  EM iter {em_iter}: log-lik = {log_lik:.4f}")

        # =============================================================
        # M-step (часть 1): оптимизация r (logits) при фиксированных q, α
        # =============================================================
        logits = logits.detach().clone().requires_grad_(True)
        r_optimizer = torch.optim.Adam([logits], lr=r_lr)

        graph_batch = prebuild_graph_batch(configs, device)

        for _r_step in range(n_r_gradient_steps):
            r_optimizer.zero_grad()

            # Gumbel-Softmax → дискретные R_k (straight-through)
            gumbel_r = gumbel_softmax_rows(logits, tau=tau, hard=True)  # [M, K]
            R_cols = gumbel_r.t()  # [K, M]

            # u(α_k, R_k) через суррогат
            u_vals = surrogate_eval_with_prebatch(surrogate, graph_batch, R_cols)
            u_vals = torch.clamp(u_vals, min=1e-10)  # [K]

            # Мягкие r для log(r_{mk})
            r_soft = F.softmax(logits, dim=-1)  # [M, K]

            # Q-функция: Σ_m Σ_k q_{mk} * (log r_{mk} + log u_k)
            log_r = torch.log(torch.clamp(r_soft, min=1e-10))
            log_u = torch.log(u_vals)  # [K]

            # log(r_{mk} * u_k) = log r_{mk} + log u_k
            log_ru = log_r + log_u.unsqueeze(0)  # [M, K] broadcast
            q_function = (q_tensor * log_ru).sum()

            loss = -q_function
            loss.backward()
            r_optimizer.step()

        # =============================================================
        # M-step (часть 2): поиск лучших архитектур при текущих r
        # =============================================================
        with torch.no_grad():
            r_current = F.softmax(logits, dim=-1)
            R_cols_current = r_current.t()  # [K, M]

        configs = sample_architectures_for_experts(
            search_space, K, n_arch_candidates, surrogate, R_cols_current, device,
        )

    # --- Финальная оценка ---
    with torch.no_grad():
        r_final = F.softmax(best_logits, dim=-1).cpu().numpy()

    return OptimizationResult(
        configs=best_configs,
        r_matrix=r_final,
        hard_assignments=discretize_assignments(r_final),
        objective_value=best_log_lik,
        history=history,
        method="em",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Метод 3: EM-алгоритм для назначения экспертов"
    )
    add_common_args(parser)
    parser.add_argument("--n-em-iterations", type=int, default=30)
    parser.add_argument("--n-r-gradient-steps", type=int, default=50)
    parser.add_argument("--r-lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()

    surrogate, ss = setup_from_args(args)

    result = optimize_em(
        surrogate, ss, args.M, args.K,
        n_em_iterations=args.n_em_iterations,
        n_arch_candidates=args.n_arch_candidates,
        n_r_gradient_steps=args.n_r_gradient_steps,
        r_lr=args.r_lr,
        tau=args.tau,
        device=args.device,
    )
    print_result(result)

    # --- Реальная оценка архитектур ---
    print("\n=== Реальная оценка (переобучение на кластерах) ===")
    cluster_dir = args.cluster_dir if args.cluster_dir else args.data_dir
    evaluate_result_real(result, data_dir=args.data_dir, cluster_dir=cluster_dir)

    if args.save_results:
        save_results({"em": result}, args.save_results)


if __name__ == "__main__":
    main()

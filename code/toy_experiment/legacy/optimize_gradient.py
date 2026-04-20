"""
Метод 2: Градиентная оптимизация по r + сэмплирование архитектур.

Параметризация: logits → softmax → r[m,:] = распределение.
Для передачи R_k в суррогат: Gumbel-Softmax (hard=True, straight-through).
Каждые arch_search_every шагов архитектуры пересэмплируются.

Использование:
    python optimize_gradient.py --surrogate-path ./runs/surr_best.pth --M 2 --K 2
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
    surrogate_eval_with_prebatch,
    compute_log_likelihood,
    discretize_assignments,
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


def optimize_gradient(
    surrogate,
    search_space,
    M: int,
    K: int,
    n_steps: int = 300,
    lr: float = 0.1,
    tau_start: float = 2.0,
    tau_end: float = 0.1,
    n_arch_candidates: int = 50,
    arch_search_every: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> OptimizationResult:
    """
    Метод 2: градиентная оптимизация по r + сэмплирование архитектур.

    ASSUMPTION: суррогат используется как фиксированная дифференцируемая функция.
    Его параметры заморожены, но computation graph строится через bool_vec input.
    """
    surrogate.eval()
    surrogate.to(device)
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # --- Инициализация ---
    logits = torch.zeros(M, K, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    configs = [sample_valid_config(search_space) for _ in range(K)]
    graph_batch = prebuild_graph_batch(configs, device)

    history: List[float] = []
    best_log_lik = -float("inf")
    best_logits = logits.detach().clone()
    best_configs = copy.deepcopy(configs)

    iterator = range(1, n_steps + 1)
    if verbose:
        iterator = tqdm(iterator, desc="Gradient Optimization")

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

        loss = -log_lik
        loss.backward()
        optimizer.step()

        log_lik_val = log_lik.item()
        history.append(log_lik_val)

        if log_lik_val > best_log_lik:
            best_log_lik = log_lik_val
            best_logits = logits.detach().clone()
            best_configs = copy.deepcopy(configs)

        # Пересэмплирование архитектур
        if step % arch_search_every == 0:
            with torch.no_grad():
                r_current = F.softmax(logits, dim=-1)
                R_cols_current = r_current.t()  # [K, M]

            configs = sample_architectures_for_experts(
                search_space, K, n_arch_candidates, surrogate, R_cols_current, device,
            )
            graph_batch = prebuild_graph_batch(configs, device)

            if verbose:
                tqdm.write(
                    f"  Step {step}: log-lik = {log_lik_val:.4f}, "
                    f"tau = {tau:.3f}, re-sampled architectures"
                )

    # --- Финальный результат ---
    with torch.no_grad():
        r_final = F.softmax(best_logits, dim=-1).cpu().numpy()

    return OptimizationResult(
        configs=best_configs,
        r_matrix=r_final,
        hard_assignments=discretize_assignments(r_final),
        objective_value=best_log_lik,
        history=history,
        method="gradient",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Метод 2: Gradient optimization для назначения экспертов"
    )
    add_common_args(parser)
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--arch-search-every", type=int, default=20)
    args = parser.parse_args()

    surrogate, ss = setup_from_args(args)

    result = optimize_gradient(
        surrogate, ss, args.M, args.K,
        n_steps=args.n_steps,
        lr=args.lr,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        n_arch_candidates=args.n_arch_candidates,
        arch_search_every=args.arch_search_every,
        device=args.device,
    )
    print_result(result)

    # --- Реальная оценка архитектур ---
    print("\n=== Реальная оценка (переобучение на кластерах) ===")
    cluster_dir = args.cluster_dir if args.cluster_dir else args.data_dir
    evaluate_result_real(result, data_dir=args.data_dir, cluster_dir=cluster_dir)

    if args.save_results:
        save_results({"gradient": result}, args.save_results)


if __name__ == "__main__":
    main()

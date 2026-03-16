"""
optimize_expert_assignments.py

Три метода оптимизации назначения экспертов в mixture-of-experts,
использующих обученную суррогатную функцию из collect_dataset.py.

Целевая функция (в логарифмической форме):

    L = Σ_{m=1}^{M} log( Σ_{k=1}^{K} r_{mk} · u(α_k, R_k) )  →  max

Где:
    M — число кластеров данных
    K — число экспертов
    r_{mk} — вероятность назначения кластера m эксперту k
    r[m, :] — распределение по экспертам для кластера m (сумма = 1)
    R_k = r[:, k] — столбец матрицы r, вектор длины M
    α_k — архитектура эксперта k
    u(α_k, R_k) — суррогатная функция: предсказание quality/accuracy

Важно: суррогат обучен на ДИСКРЕТНЫХ бинарных векторах {0,1}^M.
Для градиентных методов используется Gumbel-Softmax (straight-through).

Методы:
    1. Sampling Search — сэмплирование назначений и архитектур
    2. Gradient-Based — градиентная оптимизация logits r + сэмплирование архитектур
    3. EM — E-step: q_{mk} ∝ r_{mk}·u_k, M-step: оптимизация r и α

Использование:
    python optimize_expert_assignments.py --surrogate-path ./surr.pth --M 2 --K 2

    # Только один метод:
    python optimize_expert_assignments.py --surrogate-path ./surr.pth --method gradient

    # Сохранить результаты:
    python optimize_expert_assignments.py --surrogate-path ./surr.pth --save-results results.json
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
from toy_graph import Graph, OPS
from collect_dataset import (
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
    Позволяет не пересоздавать batch на каждой итерации внутренних циклов.

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
    # inner_sum[m] = Σ_k r_mk * u_k
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
        # Каждый эксперт получает хотя бы 1 кластер
        for k in range(K):
            assignments[perm[k], k] = 1.0
        # Оставшиеся кластеры — случайным экспертам
        for idx in range(K, M):
            k = np.random.randint(0, K)
            assignments[perm[idx], k] = 1.0
    else:
        # M < K: каждый кластер назначается случайному эксперту
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

    Args:
        logits: [M, K] — ненормализованные log-вероятности
        tau: температура (меньше → ближе к дискретному)
        hard: использовать straight-through

    Returns:
        [M, K] — (мягкие или жёсткие) назначения
    """
    return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)


# =========================================================================
# 5. Утилиты: поиск архитектур
# =========================================================================

def prebuild_graph_batch(
    configs: List[dict],
    device: str = "cpu",
) -> Batch:
    """
    Собрать PyG Batch для списка конфигураций.
    Можно переиспользовать во внутренних циклах.
    """
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
    R_columns: torch.Tensor,
    device: str = "cpu",
) -> List[dict]:
    """
    Для каждого эксперта k:
      1. Сгенерировать n_candidates случайных архитектур.
      2. Оценить каждую суррогатом с R_k.
      3. Выбрать лучшую.

    Args:
        search_space: пространство поиска
        K: число экспертов
        n_candidates: кандидатов на эксперта
        surrogate: обученная модель
        R_columns: [K, M] — R_k для каждого эксперта
        device: устройство

    Returns:
        список из K конфигураций (по одной лучшей для каждого эксперта)
    """
    best_configs = []
    surrogate.eval()

    with torch.no_grad():
        for k in range(K):
            R_k = R_columns[k]  # [M]
            candidates = [sample_valid_config(search_space) for _ in range(n_candidates)]

            # Batch evaluation
            data_list = []
            for config in candidates:
                x, edge_index = build_graph_data(config)
                data_list.append(Data(x=x, edge_index=edge_index))

            batch = Batch.from_data_list(data_list).to(device)
            # R_k одинаков для всех кандидатов эксперта k
            bool_vec = R_k.unsqueeze(0).expand(len(candidates), -1).to(device)

            scores = surrogate(batch.x, batch.edge_index, batch.batch, bool_vec)
            scores = scores.squeeze(-1).cpu().numpy()

            best_idx = int(np.argmax(scores))
            best_configs.append(candidates[best_idx])

    return best_configs


# =========================================================================
# Метод 1: Поиск через сэмплирование (Sampling Search)
# =========================================================================

def optimize_sampling(
    surrogate: nn.Module,
    search_space: toy_searchspace.ToySearchSpace,
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

    Суррогат вызывается на дискретных бинарных R_k — без аппроксимации.

    Args:
        surrogate: обученная суррогатная модель
        search_space: пространство поиска архитектур
        M: число кластеров данных
        K: число экспертов
        n_assignment_samples: число сэмплов назначений
        n_arch_candidates: кандидатных архитектур на эксперта
        device: устройство
        verbose: печатать прогресс

    Returns:
        OptimizationResult
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


# =========================================================================
# Метод 2: Градиентная оптимизация по r + сэмплирование архитектур
# =========================================================================

def optimize_gradient(
    surrogate: nn.Module,
    search_space: toy_searchspace.ToySearchSpace,
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

    Параметризация:
        logits ∈ R^{M×K}  →  r = softmax(logits, dim=-1)  →  r[m,:] = распределение

    Для передачи R_k в суррогат используется Gumbel-Softmax (hard=True):
        forward pass: дискретное {0,1} (совместимо с суррогатом)
        backward pass: градиенты через непрерывную relax.ацию (straight-through)

    ASSUMPTION: суррогат используется как фиксированная дифференцируемая функция.
    Его параметры заморожены, но computation graph строится через bool_vec input.

    Каждые arch_search_every шагов архитектуры пересэмплируются.

    Args:
        surrogate: обученная суррогатная модель
        search_space: пространство поиска
        M: число кластеров данных
        K: число экспертов
        n_steps: шагов градиентной оптимизации
        lr: learning rate для logits
        tau_start: начальная температура Gumbel-Softmax
        tau_end: конечная температура
        n_arch_candidates: кандидатов при search по архитектурам
        arch_search_every: частота пересэмплирования архитектур
        device: устройство
        verbose: печатать прогресс

    Returns:
        OptimizationResult
    """
    surrogate.eval()
    surrogate.to(device)
    # Заморозить параметры суррогата — градиенты нужны только по logits
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # --- Инициализация ---
    logits = torch.zeros(M, K, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    # Начальные архитектуры
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
        # Anneal температуры
        progress = step / n_steps
        tau = tau_start * (tau_end / tau_start) ** progress

        optimizer.zero_grad()

        # --- Gumbel-Softmax: hard=True → straight-through estimator ---
        # forward: дискретные R_k (бинарные столбцы)
        # backward: непрерывные градиенты
        gumbel_r = gumbel_softmax_rows(logits, tau=tau, hard=True)  # [M, K]
        R_columns = gumbel_r.t()  # [K, M]

        # --- Суррогат: u(α_k, R_k) для каждого k ---
        u_values = surrogate_eval_with_prebatch(surrogate, graph_batch, R_columns)
        u_values = torch.clamp(u_values, min=1e-10)  # [K]

        # --- Softmax r для целевой функции ---
        r_soft = F.softmax(logits, dim=-1)  # [M, K]

        # --- Log-likelihood ---
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

        # --- Пересэмплирование архитектур ---
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


# =========================================================================
# Метод 3: EM-алгоритм
# =========================================================================

def optimize_em(
    surrogate: nn.Module,
    search_space: toy_searchspace.ToySearchSpace,
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

    ASSUMPTION: в E-step для вычисления u(α_k, R_k) R_k дискретизируется
    через argmax жёсткого назначения, так как суррогат обучен на бинарных
    векторах.

    Args:
        surrogate: обученная суррогатная модель
        search_space: пространство поиска
        M: число кластеров данных
        K: число экспертов
        n_em_iterations: число EM-итераций
        n_arch_candidates: кандидатов при search архитектур
        n_r_gradient_steps: шагов градиентной оптимизации r на M-step
        r_lr: learning rate для logits на M-step
        tau: температура Gumbel-Softmax на M-step
        device: устройство
        verbose: печатать прогресс

    Returns:
        OptimizationResult
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
        # maximize Q = Σ_m Σ_k q_{mk} · log(r_{mk} · u(α_k, R_k))
        #            = Σ_m Σ_k q_{mk} · (log r_{mk} + log u(α_k, R_k))

        logits = logits.detach().clone().requires_grad_(True)
        r_optimizer = torch.optim.Adam([logits], lr=r_lr)

        # Pre-build graph batch для текущих архитектур
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


# =========================================================================
# Единый интерфейс
# =========================================================================

def optimize(
    method: str,
    surrogate: nn.Module,
    search_space: toy_searchspace.ToySearchSpace,
    M: int,
    K: int,
    device: str = "cpu",
    verbose: bool = True,
    **kwargs,
) -> OptimizationResult:
    """
    Единый интерфейс для запуска любого из трёх методов.

    Args:
        method: 'sampling', 'gradient' или 'em'
        surrogate: обученная суррогатная модель
        search_space: пространство поиска архитектур
        M: число кластеров данных
        K: число экспертов
        device: устройство
        verbose: печатать прогресс
        **kwargs: гиперпараметры конкретного метода

    Returns:
        OptimizationResult
    """
    dispatch = {
        "sampling": optimize_sampling,
        "gradient": optimize_gradient,
        "em": optimize_em,
    }
    if method not in dispatch:
        raise ValueError(
            f"Неизвестный метод: {method}. Доступны: {list(dispatch.keys())}"
        )
    return dispatch[method](
        surrogate, search_space, M, K,
        device=device, verbose=verbose, **kwargs,
    )


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
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Оптимизация назначения экспертов в mixture-of-experts"
    )

    parser.add_argument(
        "--method", type=str, default="all",
        choices=["sampling", "gradient", "em", "all"],
        help="Метод оптимизации (по умолчанию: все три)",
    )
    parser.add_argument(
        "--surrogate-path", type=str, required=True,
        help="Путь к сохранённой суррогатной модели (.pth)",
    )
    parser.add_argument("--M", type=int, default=2, help="Число кластеров данных")
    parser.add_argument("--K", type=int, default=2, help="Число экспертов")
    parser.add_argument(
        "--input-dim", type=int, default=2,
        help="Размерность входа ячейки (для ToySearchSpace)",
    )
    parser.add_argument(
        "--n-features", type=int, default=len(OPS),
        help="Число признаков графа (= len(OPS))",
    )

    # Параметры суррогата (нужны для правильной загрузки)
    parser.add_argument("--surrogate-dropout", type=float, default=0.8)
    parser.add_argument("--surrogate-hidden-dim", type=int, default=8)
    parser.add_argument("--surrogate-heads", type=int, default=1)

    # Sampling
    parser.add_argument("--n-assignment-samples", type=int, default=200)
    parser.add_argument("--n-arch-candidates", type=int, default=50)

    # Gradient
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tau-start", type=float, default=2.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--arch-search-every", type=int, default=20)

    # EM
    parser.add_argument("--n-em-iterations", type=int, default=30)
    parser.add_argument("--n-r-gradient-steps", type=int, default=50)
    parser.add_argument("--r-lr", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)

    # Общие
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--save-results", type=str, default=None,
        help="Путь для сохранения результатов в JSON",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # --- Загрузка суррогата ---
    print(f"Загрузка суррогата из {args.surrogate_path} ...")
    surrogate = nas_moe.surrogate.GAT_Datafeature(
        input_dim=args.n_features,
        output_dim=1,
        dropout=args.surrogate_dropout,
        hidden_dim=args.surrogate_hidden_dim,
        heads=args.surrogate_heads,
        bool_vec_dim=args.M,
    )
    state = torch.load(args.surrogate_path, map_location=args.device)
    surrogate.load_state_dict(state)
    surrogate.to(args.device)
    surrogate.eval()

    # --- Пространство поиска ---
    ss = toy_searchspace.ToySearchSpace(
        input_dim=args.input_dim, num_nodes_per_cell=4,
    )

    # --- Запуск оптимизации ---
    methods = (
        ["sampling", "gradient", "em"]
        if args.method == "all"
        else [args.method]
    )
    results: Dict[str, OptimizationResult] = {}

    for method in methods:
        print(f"\n{'#' * 60}")
        print(f"Запуск метода: {method}")
        print(f"{'#' * 60}")

        if method == "sampling":
            result = optimize_sampling(
                surrogate, ss, args.M, args.K,
                n_assignment_samples=args.n_assignment_samples,
                n_arch_candidates=args.n_arch_candidates,
                device=args.device,
            )
        elif method == "gradient":
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
        elif method == "em":
            result = optimize_em(
                surrogate, ss, args.M, args.K,
                n_em_iterations=args.n_em_iterations,
                n_arch_candidates=args.n_arch_candidates,
                n_r_gradient_steps=args.n_r_gradient_steps,
                r_lr=args.r_lr,
                tau=args.tau,
                device=args.device,
            )
        else:
            continue

        results[method] = result
        print_result(result)

    # --- Сохранение ---
    if args.save_results and results:
        save_data = {}
        for method_name, res in results.items():
            save_data[method_name] = {
                "configs": res.configs,
                "r_matrix": res.r_matrix.tolist(),
                "hard_assignments": res.hard_assignments.tolist(),
                "objective_value": res.objective_value,
                "history": res.history,
            }
        with open(args.save_results, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nРезультаты сохранены в {args.save_results}")

    # --- Сравнение методов ---
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("СРАВНЕНИЕ МЕТОДОВ")
        print(f"{'=' * 60}")
        for name, res in results.items():
            print(
                f"  {name:12s}: log-lik = {res.objective_value:.6f}, "
                f"assignments = {res.hard_assignments}"
            )


if __name__ == "__main__":
    main()

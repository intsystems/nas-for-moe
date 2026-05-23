"""Собрать MoE из результатов optimize_surrogate_em.

Минимальный вход — путь к JSON, сохранённому `save_results`
(например, `runs/results_mnist_sgem_k3_v2.json`). Структура:
    {
      "<run_name>": {
        "configs": [ arch_config_dict, ... ],   # K штук
        "r_matrix": [[...], ...],               # [M, K]
        "hard_assignments": [...],              # [M] int
        ...
      }
    }

Модуль строит `MoENet`:
    - K экспертов (каждый — MNISTNet из mnist_sgem.py)
    - Router: cluster_id → веса по экспертам
      * mode="hard" — one-hot на hard_assignments
      * mode="soft" — строка r_matrix[m]

Использование:
    from build_moe import MoENet
    moe = MoENet.from_results("runs/results_mnist_sgem_k3_v2.json")
    logits = moe(x_batch, cluster_ids)          # [B, num_classes]

Или из кода одной функцией:
    moe = load_moe_from_results(path, mode="soft", C=16)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mnist_sgem import MNISTNet


class ClusterRouter(nn.Module):
    """Определяет cluster_id для сырого изображения MNIST.

    Воспроизводит пайплайн из `mnist_sgem.setup_mnist_data`:
        uint8 [N,1,28,28] → flatten/255 → PCA-{pca_dim} → nearest centroid.

    PCA не сохраняется в data_dir, поэтому пересчитывается с тем же seed.
    Результат — тензорный буфер (PCA-матрица + центроиды), работает на GPU.
    """

    def __init__(
        self,
        data_dir: str | Path,
        pca_dim: int = 50,
        seed: int = 322,
    ):
        super().__init__()
        from sklearn.decomposition import PCA

        data_dir = Path(data_dir)
        X_img = np.load(data_dir / "data_X.npy")            # uint8 [N,1,28,28]
        centers = np.load(data_dir / "cluster_centers.npy")  # [M, pca_dim]

        N = len(X_img)
        X_flat = X_img.reshape(N, -1).astype(np.float32) / 255.0
        pca = PCA(n_components=pca_dim, random_state=seed).fit(X_flat)

        # PCA.transform(x) = (x - mean) @ components.T
        self.register_buffer(
            "pca_mean", torch.tensor(pca.mean_, dtype=torch.float32),
        )
        self.register_buffer(
            "pca_components",
            torch.tensor(pca.components_, dtype=torch.float32),   # [pca_dim, 784]
        )
        self.register_buffer(
            "centers", torch.tensor(centers, dtype=torch.float32),  # [M, pca_dim]
        )
        self.M = int(centers.shape[0])

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: uint8 или float [B,1,28,28] (значения в [0,255] или уже нормир.).

        Returns:
            cluster_ids: [B] long.
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.5:  # вероятно ещё в [0,255]
            x = x / 255.0
        flat = x.reshape(x.shape[0], -1)                         # [B, 784]
        proj = (flat - self.pca_mean) @ self.pca_components.T    # [B, pca_dim]
        # squared distance to each centroid
        d2 = torch.cdist(proj, self.centers, p=2.0)              # [B, M]
        return d2.argmin(dim=1)


class MoENet(nn.Module):
    """MoE поверх архитектур, найденных EM-алгоритмом.

    Args:
        configs: список из K dict-конфигов архитектур (формат toy_graph).
        r_matrix: массив [M, K] — soft routing (используется при mode="soft").
        hard_assignments: массив [M] int — жёсткое назначение кластер→эксперт
            (используется при mode="hard").
        C: число каналов в MNISTNet.
        num_classes: число классов на выходе.
        mode: "hard" или "soft".
    """

    def __init__(
        self,
        configs: list[dict],
        r_matrix: np.ndarray,
        hard_assignments: np.ndarray,
        C: int = 16,
        num_classes: int = 10,
        mode: str = "hard",
        cluster_router: Optional[ClusterRouter] = None,
    ):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode}")
        self.cluster_router = cluster_router

        self.K = len(configs)
        self.M = int(r_matrix.shape[0])
        self.num_classes = num_classes
        self.mode = mode

        if r_matrix.shape != (self.M, self.K):
            raise ValueError(
                f"r_matrix shape {r_matrix.shape} != (M={self.M}, K={self.K})"
            )
        if hard_assignments.shape != (self.M,):
            raise ValueError(
                f"hard_assignments shape {hard_assignments.shape} != (M={self.M},)"
            )

        self.experts = nn.ModuleList([
            MNISTNet(cfg, C=C, num_classes=num_classes) for cfg in configs
        ])

        # Таблицы роутинга как буферы (на device вместе с модулем, без grad).
        self.register_buffer(
            "r_matrix", torch.tensor(r_matrix, dtype=torch.float32),
        )
        self.register_buffer(
            "hard_assignments",
            torch.tensor(hard_assignments, dtype=torch.long),
        )

    def route(self, cluster_ids: torch.Tensor) -> torch.Tensor:
        """Вернуть веса экспертов для пакета.

        Args:
            cluster_ids: [B] int — id кластера для каждого примера.

        Returns:
            [B, K] float — веса (в сумме по K = 1).
        """
        if self.mode == "soft":
            return self.r_matrix[cluster_ids]                  # [B, K]
        assign = self.hard_assignments[cluster_ids]            # [B]
        return F.one_hot(assign, num_classes=self.K).float()   # [B, K]

    def forward(
        self,
        x: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Прямой проход.

        Args:
            x: [B, 1, 28, 28]
            cluster_ids: [B] long — id кластера для каждого примера.
                Если None, будет вычислен `self.cluster_router(x)`.

        Returns:
            [B, num_classes] — смесь логитов экспертов по роутингу.
        """
        if cluster_ids is None:
            if self.cluster_router is None:
                raise ValueError(
                    "cluster_ids не передан, и cluster_router не задан. "
                    "Либо передавай cluster_ids явно, либо собирай MoE с router."
                )
            cluster_ids = self.cluster_router(x)
        weights = self.route(cluster_ids)                      # [B, K]

        if self.mode == "hard":
            # Каждый пример идёт ровно в одного эксперта — считаем только нужных.
            out = x.new_zeros(x.shape[0], self.num_classes)
            assign = self.hard_assignments[cluster_ids]        # [B]
            for k, expert in enumerate(self.experts):
                mask = assign == k
                if mask.any():
                    out[mask] = expert(x[mask])
            return out

        # soft: прогоняем всех экспертов, смешиваем логиты по весам.
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # [B, K, C]
        return (stacked * weights.unsqueeze(-1)).sum(dim=1)         # [B, C]

    # ---------------------------------------------------------------
    # Convenience constructors
    # ---------------------------------------------------------------
    @classmethod
    def from_results(
        cls,
        results_path: str | Path,
        run_name: Optional[str] = None,
        C: int = 16,
        num_classes: int = 10,
        mode: str = "hard",
        data_dir: Optional[str | Path] = None,
        pca_dim: int = 50,
        seed: int = 322,
    ) -> "MoENet":
        with open(results_path, "r") as f:
            data = json.load(f)
        if run_name is None:
            if len(data) != 1:
                raise ValueError(
                    f"results JSON has {len(data)} runs, укажи run_name: "
                    f"{list(data.keys())}"
                )
            run_name = next(iter(data))
        run = data[run_name]

        configs = run["configs"]
        r_matrix = np.asarray(run["r_matrix"], dtype=np.float32)
        hard = np.asarray(run["hard_assignments"], dtype=np.int64)

        router = None
        if data_dir is not None:
            router = ClusterRouter(data_dir, pca_dim=pca_dim, seed=seed)

        return cls(
            configs=configs,
            r_matrix=r_matrix,
            hard_assignments=hard,
            C=C, num_classes=num_classes, mode=mode,
            cluster_router=router,
        )


# Функциональная обёртка для удобства.
def load_moe_from_results(
    results_path: str | Path,
    run_name: Optional[str] = None,
    C: int = 16,
    num_classes: int = 10,
    mode: str = "hard",
) -> MoENet:
    return MoENet.from_results(
        results_path, run_name=run_name, C=C,
        num_classes=num_classes, mode=mode,
    )


# ---------------------------------------------------------------
# CLI: собрать и напечатать сводку
# ---------------------------------------------------------------
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str,
                        help="Путь к results_*.json от optimize_surrogate_em")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--save-state-dict", type=str, default=None,
                        help="Если задан, сохранить инициализированный state_dict")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Директория с cluster_centers.npy/data_X.npy — "
                             "прикрепить ClusterRouter, MoE сможет работать "
                             "без явных cluster_ids")
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=322)
    args = parser.parse_args()

    moe = MoENet.from_results(
        args.results_path, run_name=args.run_name,
        C=args.init_channels, num_classes=args.num_classes, mode=args.mode,
        data_dir=args.data_dir, pca_dim=args.pca_dim, seed=args.seed,
    )

    n_params = sum(p.numel() for p in moe.parameters())
    per_expert = [sum(p.numel() for p in e.parameters()) for e in moe.experts]
    print(f"MoE: K={moe.K}, M={moe.M}, mode={moe.mode}")
    print(f"  total params: {n_params:,}")
    for k, n in enumerate(per_expert):
        print(f"  expert {k}: {n:,} params")
    print(f"  hard_assignments: {moe.hard_assignments.tolist()}")

    # Smoke-тест прямого прохода.
    B = 4
    x = torch.randn(B, 1, 28, 28)
    cids = torch.randint(0, moe.M, (B,))
    with torch.no_grad():
        y = moe(x, cids)
    print(f"  forward(explicit cids) OK: x{tuple(x.shape)} "
          f"cids={cids.tolist()} -> y{tuple(y.shape)}")

    if moe.cluster_router is not None:
        # Проверяем auto-routing на реальном MNIST-примере.
        data_dir = Path(args.data_dir)
        X_real = np.load(data_dir / "data_X.npy")[:B]  # uint8 [B,1,28,28]
        x_raw = torch.from_numpy(X_real)                      # uint8 — для router
        x_norm = (x_raw.float() / 255.0 - 0.1307) / 0.3081    # float — для экспертов
        with torch.no_grad():
            inferred = moe.cluster_router(x_raw)
            y = moe(x_norm, cluster_ids=inferred)
        print(f"  forward(auto-router) OK: inferred cids={inferred.tolist()} "
              f"-> y{tuple(y.shape)}")

    if args.save_state_dict:
        torch.save(moe.state_dict(), args.save_state_dict)
        print(f"  saved state_dict -> {args.save_state_dict}")


if __name__ == "__main__":
    _main()

"""MoE для CIFAR-100 с learnable softmax-gating.

Используется в бейзлайнах:
    - cifar100_random_search_single_arch.py — все K экспертов имеют ОДНУ И ТУ ЖЕ архитектуру
    - cifar100_random_moe_baseline.py — каждый эксперт случайный

Структура аналогична mnist/train_random_moe.py, но 3-канальный вход (RGB) и
gating-сеть глубже под 32×32.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cifar100_sgem import CIFAR100Net, CIFAR100_MEAN, CIFAR100_STD  # noqa: E402


class GatingNet(nn.Module):
    """Маленькая CNN-гейтинг-сеть: x [B,3,32,32] → K logits."""

    def __init__(self, K: int, C: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, C, 3, stride=2, padding=1, bias=False),    # 32→16
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),    # 16→8
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, K),
        )

    def forward(self, x):
        return self.net(x)


class CIFAR100MoE(nn.Module):
    """MoE с learnable softmax-gating.

    Args:
        configs: K dict-конфигов архитектур (могут совпадать → "DARTS-baseline")
        init_channels: каналы экспертов
        num_classes: число классов
        gate_channels: каналы gating-сети
    """

    def __init__(
        self,
        configs: List[dict],
        init_channels: int = 16,
        num_classes: int = 100,
        gate_channels: int = 16,
    ):
        super().__init__()
        self.K = len(configs)
        self.num_classes = num_classes
        self.configs = configs
        self.experts = nn.ModuleList([
            CIFAR100Net(cfg, C=init_channels, num_classes=num_classes)
            for cfg in configs
        ])
        self.gate = GatingNet(K=self.K, C=gate_channels)

    def forward(self, x: torch.Tensor):
        gate_logits = self.gate(x)                          # [B, K]
        gate_w = F.softmax(gate_logits, dim=-1)             # [B, K]
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # [B, K, C]
        mixed = (stacked * gate_w.unsqueeze(-1)).sum(dim=1)
        return mixed, gate_w


# ==========================================================================
# Загрузка данных (uint8 [N,3,32,32] → нормализованный float)
# ==========================================================================


def load_cifar100_tensors(data_dir: Path):
    X = np.load(data_dir / "data_X.npy")               # uint8 [N, 3, 32, 32]
    y = np.load(data_dir / "data_y.npy").astype(np.int64)
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")

    mean = torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(1, 3, 1, 1)

    def to_tensor(idx):
        xb = torch.from_numpy(X[idx]).float().div_(255.0).sub_(mean).div_(std)
        yb = torch.from_numpy(y[idx])
        return xb, yb

    X_tr, y_tr = to_tensor(train_idx)
    X_v, y_v = to_tensor(val_idx)
    return X_tr, y_tr, X_v, y_v


# ==========================================================================
# Тренировка / оценка одного MoE
# ==========================================================================


@torch.no_grad()
def evaluate_moe(moe: CIFAR100MoE, X, y, batch_size: int, device: str):
    moe.eval()
    correct = 0
    total = 0
    gate_sum = torch.zeros(moe.K, device=device)
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        logits, gw = moe(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
        gate_sum += gw.sum(dim=0)
    return correct / max(1, total), (gate_sum / total).cpu().numpy()


def train_moe(
    moe: CIFAR100MoE,
    X_tr, y_tr, X_v, y_v,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.05,
    wd: float = 3e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> float:
    """Обучить MoE и вернуть лучшую val-accuracy."""
    moe.to(device)
    optimizer = torch.optim.SGD(
        moe.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs),
    )
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        moe.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits, _ = moe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(moe.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
        scheduler.step()

        val_acc, val_gate = evaluate_moe(moe, X_v, y_v, batch_size, device)
        if val_acc > best_val:
            best_val = val_acc
        if verbose:
            print(f"  [epoch {epoch:02d}/{epochs}] "
                  f"train_loss={running/n:.4f} val_acc={val_acc:.4f} "
                  f"gate={np.round(val_gate, 2).tolist()} "
                  f"time={time.time()-t0:.1f}s")
    return best_val

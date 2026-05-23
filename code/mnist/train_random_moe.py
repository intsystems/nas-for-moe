"""Обучение случайного MoE на MNIST — baseline для сравнения с EM-версией.

Отличия от train_moe.py:
    * Архитектуры экспертов — случайные (MNISTDartsSearchSpace.create_random_config).
    * Роутинг — обычный learnable gating network: маленькая CNN по входу x,
      softmax даёт веса по K экспертам, soft-смешение логитов.
    * Кластеры MNIST не используются.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from mnist_sgem import MNISTNet, MNISTDartsSearchSpace  # noqa: E402


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class GatingNet(nn.Module):
    """Маленькая CNN-гейтинг-сеть: x → K logits."""

    def __init__(self, K: int, C: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, C, 3, stride=2, padding=1, bias=False),   # 28→14
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),   # 14→7
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, K),
        )

    def forward(self, x):
        return self.net(x)      # [B, K] logits


class RandomMoE(nn.Module):
    """MoE со случайными архитектурами и learnable softmax-gating."""

    def __init__(
        self,
        K: int,
        init_channels: int = 16,
        num_classes: int = 10,
        gate_channels: int = 8,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            random.seed(seed)
        ss = MNISTDartsSearchSpace(init_channels=init_channels)
        self.configs = [ss.create_random_config() for _ in range(K)]
        self.experts = nn.ModuleList([
            MNISTNet(cfg, C=init_channels, num_classes=num_classes)
            for cfg in self.configs
        ])
        self.gate = GatingNet(K=K, C=gate_channels)
        self.K = K
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        """Soft-mixture-of-experts.

        Returns:
            logits: [B, num_classes]
            gate_weights: [B, K] (для логирования/регуляризации)
        """
        gate_logits = self.gate(x)                          # [B, K]
        gate_w = F.softmax(gate_logits, dim=-1)             # [B, K]
        stacked = torch.stack([e(x) for e in self.experts], dim=1)  # [B, K, C]
        mixed = (stacked * gate_w.unsqueeze(-1)).sum(dim=1)          # [B, C]
        return mixed, gate_w


def load_mnist_tensors(data_dir: Path):
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy").astype(np.int64)
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")

    def to_tensor(idx):
        xb = torch.from_numpy(X[idx]).float().div_(255.0)
        xb.sub_(MNIST_MEAN).div_(MNIST_STD)
        return xb, torch.from_numpy(y[idx])

    X_tr, y_tr = to_tensor(train_idx)
    X_v, y_v = to_tensor(val_idx)
    return X_tr, y_tr, X_v, y_v


@torch.no_grad()
def evaluate(moe: RandomMoE, X, y, batch_size: int, device: str):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--init-channels", type=int, default=16)
    p.add_argument("--gate-channels", type=int, default=8)
    p.add_argument("--seed", type=int, default=322)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    print(f"[data] loading from {data_dir}")
    X_tr, y_tr, X_v, y_v = load_mnist_tensors(data_dir)
    print(f"[data] train={len(X_tr)}, val={len(X_v)}")

    moe = RandomMoE(
        K=args.K, init_channels=args.init_channels,
        gate_channels=args.gate_channels, seed=args.seed,
    ).to(args.device)
    n_params = sum(p.numel() for p in moe.parameters())
    n_gate = sum(p.numel() for p in moe.gate.parameters())
    n_experts = [sum(p.numel() for p in e.parameters()) for e in moe.experts]
    print(f"[moe] K={args.K}, total={n_params:,} params "
          f"(gate={n_gate:,}, experts={n_experts})")
    print("[moe] random architectures:")
    for k, cfg in enumerate(moe.configs):
        ops = [cfg[f"op_{i}"] for i in [0, 1, 3, 4, 6, 7]]
        print(f"  expert {k}: ops={ops}")

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        moe.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.wd, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs),
    )
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        moe.train()
        t0 = time.time()
        running = 0.0
        n = 0
        gate_accum = torch.zeros(args.K, device=args.device)
        for xb, yb in loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            logits, gw = moe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(moe.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
            gate_accum += gw.detach().sum(dim=0)
        scheduler.step()

        val_acc, val_gate = evaluate(
            moe, X_v, y_v, batch_size=args.batch_size, device=args.device,
        )
        dt = time.time() - t0
        tr_gate = (gate_accum / n).cpu().numpy()
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={running/n:.4f} val_acc={val_acc:.4f} "
              f"gate_train={np.round(tr_gate, 3).tolist()} "
              f"gate_val={np.round(val_gate, 3).tolist()} "
              f"lr={scheduler.get_last_lr()[0]:.4f} time={dt:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            if args.save_path:
                torch.save(
                    {
                        "state_dict": moe.state_dict(),
                        "configs": moe.configs,
                        "K": moe.K,
                    },
                    args.save_path,
                )

    print(f"\n[done] best val_acc = {best_val:.4f}")


if __name__ == "__main__":
    main()

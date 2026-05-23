"""Обучение MoE (build_moe.MoENet) на MNIST.

Грузит результаты EM (architectures + routing) и данные MNIST, прикрепляет
ClusterRouter по data_dir, обучает end-to-end в hard-routing режиме:
каждый эксперт получает градиенты только от своих примеров (по hard assignment).

Оценивает overall val accuracy + per-expert val accuracy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_moe import MoENet  # noqa: E402


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def load_mnist_tensors(data_dir: Path):
    X = np.load(data_dir / "data_X.npy")            # uint8 [N,1,28,28]
    y = np.load(data_dir / "data_y.npy").astype(np.int64)
    train_idx = np.load(data_dir / "train_indices.npy")
    val_idx = np.load(data_dir / "val_indices.npy")
    train_cids = np.load(data_dir / "train_cluster_ids.npy").astype(np.int64)
    val_cids = np.load(data_dir / "val_cluster_ids.npy").astype(np.int64)

    def to_tensor(idx):
        xb = torch.from_numpy(X[idx]).float().div_(255.0)
        xb.sub_(MNIST_MEAN).div_(MNIST_STD)
        yb = torch.from_numpy(y[idx])
        return xb, yb

    X_tr, y_tr = to_tensor(train_idx)
    X_v, y_v = to_tensor(val_idx)
    return (X_tr, y_tr, torch.from_numpy(train_cids),
            X_v, y_v, torch.from_numpy(val_cids))


def evaluate(moe: MoENet, X: torch.Tensor, y: torch.Tensor,
             cids: torch.Tensor, batch_size: int, device: str) -> dict:
    moe.eval()
    K, M = moe.K, moe.M
    total_correct = 0
    total = 0
    per_expert = [[0, 0] for _ in range(K)]     # [correct, total]
    per_cluster = [[0, 0] for _ in range(M)]

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size].to(device)
            yb = y[i:i + batch_size].to(device)
            cb = cids[i:i + batch_size].to(device)
            out = moe(xb, cluster_ids=cb)
            pred = out.argmax(dim=1)
            correct = (pred == yb)
            total_correct += correct.sum().item()
            total += len(yb)

            assign = moe.hard_assignments[cb]
            for k in range(K):
                mk = assign == k
                per_expert[k][0] += correct[mk].sum().item()
                per_expert[k][1] += mk.sum().item()
            for m in range(M):
                mm = cb == m
                per_cluster[m][0] += correct[mm].sum().item()
                per_cluster[m][1] += mm.sum().item()

    return {
        "overall": total_correct / max(1, total),
        "per_expert": [
            (c / t if t > 0 else float("nan"), t) for c, t in per_expert
        ],
        "per_cluster": [
            (c / t if t > 0 else float("nan"), t) for c, t in per_cluster
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-path", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--mode", choices=["hard", "soft"], default="hard")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=3e-4)
    p.add_argument("--init-channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=322)
    p.add_argument("--save-path", type=str, default=None)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    print(f"[data] loading from {data_dir}")
    X_tr, y_tr, c_tr, X_v, y_v, c_v = load_mnist_tensors(data_dir)
    print(f"[data] train={len(X_tr)}, val={len(X_v)}")

    print(f"[moe] building from {args.results_path}")
    moe = MoENet.from_results(
        args.results_path, C=args.init_channels, mode=args.mode,
        data_dir=args.data_dir, seed=args.seed,
    ).to(args.device)
    print(f"[moe] K={moe.K}, M={moe.M}, params={sum(p.numel() for p in moe.parameters()):,}")
    print(f"[moe] hard_assignments: {moe.hard_assignments.tolist()}")

    loader = DataLoader(
        TensorDataset(X_tr, y_tr, c_tr),
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
        for xb, yb, cb in loader:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            cb = cb.to(args.device, non_blocking=True)
            optimizer.zero_grad()
            out = moe(xb, cluster_ids=cb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(moe.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(yb)
            n += len(yb)
        scheduler.step()

        metrics = evaluate(moe, X_v, y_v, c_v,
                           batch_size=args.batch_size, device=args.device)
        dt = time.time() - t0
        exp_str = " ".join(
            f"E{k}={acc:.4f}(n={cnt})"
            for k, (acc, cnt) in enumerate(metrics["per_expert"])
        )
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"train_loss={running/n:.4f} "
              f"val_acc={metrics['overall']:.4f} "
              f"{exp_str} "
              f"lr={scheduler.get_last_lr()[0]:.4f} "
              f"time={dt:.1f}s")

        if metrics["overall"] > best_val:
            best_val = metrics["overall"]
            if args.save_path:
                torch.save(moe.state_dict(), args.save_path)

    print(f"\n[done] best val_acc = {best_val:.4f}")

    # Финальная оценка + per-cluster разбивка.
    final = evaluate(moe, X_v, y_v, c_v,
                     batch_size=args.batch_size, device=args.device)
    print("\nPer-cluster val accuracy:")
    for m, (acc, cnt) in enumerate(final["per_cluster"]):
        k = int(moe.hard_assignments[m].item())
        print(f"  cluster {m:2d} -> expert {k}: acc={acc:.4f} (n={cnt})")


if __name__ == "__main__":
    main()

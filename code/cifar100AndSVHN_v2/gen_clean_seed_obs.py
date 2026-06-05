#!/usr/bin/env python3
"""Generate ~100 domain-PURE seed observations (real training, real val_loss).

Fixes the coverage hole found in section 8.5: the surrogate never saw
single-domain subsets, so it was blind to 'pure SVHN -> low loss'. Here we
build b-vectors that live entirely within ONE domain (all clusters of the
chosen domain, or random subsets of them), train a randomly-sampled arch on
each for real, and save as obs_*.json (val_loss target) — ready for
--initial-obs-dir.

Composition (default n=100):
  - 8  full-domain anchors (full SVHN / full CIFAR) x several archs
  - ~46 pure-SVHN random subsets (varied size)
  - ~46 pure-CIFAR random subsets (varied size)
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cifar100"))
sys.path.insert(0, str(HERE.parent))  # code/ root for toy_experiment.*

import cifar100_sgem_v2 as v2                       # noqa: E402
from optimize_surrogate_em_v2 import prepare_data   # noqa: E402
from utils_v2 import save_observation_v2            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init-channels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    meta = v2.load_cifar100_meta(data_dir)
    v2._NUM_CLASSES = meta["num_classes"]
    M = int(meta["n_clusters"])
    cif = sorted(meta["ideal_split_by_source"]["cifar_clusters"])
    svh = sorted(meta["ideal_split_by_source"]["svhn_clusters"])

    X = np.load(data_dir / "data_X.npy"); y = np.load(data_dir / "data_y.npy")
    ss = v2.CIFAR100DartsSearchSpace(init_channels=args.init_channels)
    data = prepare_data(X, y, cluster_dir=str(data_dir))
    Xtr_c = data["X_train_by_cluster"]; ytr_c = data["y_train_by_cluster"]
    X_val = data["X_val"]; y_val = data["y_val"]
    vci = data["val_cluster_ids"]

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    def b_from_clusters(cl):
        b = [0] * M
        for c in cl:
            b[c] = 1
        return b

    # build the list of clean b-vectors
    specs = []  # (name, clusters)
    # full-domain anchors (repeat with different archs)
    for _ in range(4):
        specs.append(("full_svhn", list(svh)))
    for _ in range(4):
        specs.append(("full_cifar", list(cif)))
    # fill the rest with single-domain random subsets, alternating domains
    remaining = args.n - len(specs)
    half = remaining // 2
    for _ in range(half):  # pure SVHN subsets (size 3..len(svh))
        k = random.randint(3, len(svh))
        specs.append(("sub_svhn", sorted(random.sample(svh, k))))
    for _ in range(remaining - half):  # pure CIFAR subsets (size 3..len(cif))
        k = random.randint(3, len(cif))
        specs.append(("sub_cifar", sorted(random.sample(cif, k))))
    random.shuffle(specs)

    print(f"[gen] generating {len(specs)} clean obs into {save_dir}")
    for idx, (name, cl) in enumerate(specs):
        b = b_from_clusters(cl)
        cfg = ss.create_random_config()
        vloss = v2.evaluate_architecture_on_subset_cifar100_v2(
            cfg, ss, b, Xtr_c, ytr_c, X_val, y_val,
            epochs=args.epochs, val_cluster_ids=vci)
        save_observation_v2(cfg, b, float(vloss), str(save_dir), idx)
        if idx % 10 == 0:
            print(f"  [{idx}/{len(specs)}] {name:10s} |b|={sum(b):2d} val_loss={vloss:.3f}")

    print(f"[gen] done: {len(list(save_dir.glob('obs_*.json')))} obs written")


if __name__ == "__main__":
    main()

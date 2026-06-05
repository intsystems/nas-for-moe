#!/usr/bin/env python3
"""Collect a seed dataset with a DOMAIN-AGNOSTIC log-uniform b-sampler.

Motivation (sections 8.5-8.6): the default seed sampler is Bernoulli(0.5) per
cluster, which fixes |b|~M/2 (most-mixed regime) and makes pure single-domain
subsets essentially impossible (pure_svhn ~ 0). Clean subsets only appear at
SMALL |b|. The log-uniform sampler draws the subset SIZE k with log(k) uniform
on [log(1), log(M)] — equal mass per order-of-magnitude — so it heavily favors
small k (=> many near/fully pure subsets) WITHOUT using any domain labels,
while still covering large k.

Each sampled b is trained for real (random arch) and saved as obs_*.json with
val_loss target — ready for --initial-obs-dir.
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


def sample_b_loguniform(M, rng, kmin=3):
    """log-uniform subset size on [kmin, M], then k random clusters.

    Domain-agnostic. log(k) ~ U(ln kmin, ln M). Start at kmin=3 because tiny
    |b|<3 give high loss (too little data) and inject the wrong signal.
    """
    k = int(round(np.exp(rng.uniform(np.log(kmin), np.log(M)))))
    k = max(kmin, min(M, k))
    sel = set(rng.sample(range(M), k))
    return [1 if i in sel else 0 for i in range(M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init-channels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    meta = v2.load_cifar100_meta(data_dir)
    v2._NUM_CLASSES = meta["num_classes"]
    M = int(meta["n_clusters"])
    # domains used ONLY for reporting coverage, never for sampling
    cif = set(meta["ideal_split_by_source"]["cifar_clusters"])
    dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}

    X = np.load(data_dir / "data_X.npy"); y = np.load(data_dir / "data_y.npy")
    ss = v2.CIFAR100DartsSearchSpace(init_channels=args.init_channels)
    data = prepare_data(X, y, cluster_dir=str(data_dir))
    Xtr_c = data["X_train_by_cluster"]; ytr_c = data["y_train_by_cluster"]
    X_val = data["X_val"]; y_val = data["y_val"]; vci = data["val_cluster_ids"]

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen-loguniform] M={M}, n={args.n}, epochs={args.epochs}, seed={args.seed}")

    cov = {"pure_svhn": 0, "pure_cifar": 0, "mixed": 0}
    sizes = []
    for idx in range(args.n):
        b = sample_b_loguniform(M, rng)
        cfg = ss.create_random_config()
        vloss = v2.evaluate_architecture_on_subset_cifar100_v2(
            cfg, ss, b, Xtr_c, ytr_c, X_val, y_val,
            epochs=args.epochs, val_cluster_ids=vci)
        save_observation_v2(cfg, b, float(vloss), str(save_dir), idx)
        sel = [m for m, f in enumerate(b) if f]
        nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
        cat = "pure_svhn" if nc == 0 else ("pure_cifar" if ns == 0 else "mixed")
        cov[cat] += 1; sizes.append(sum(b))
        if idx % 25 == 0:
            print(f"  [{idx}/{args.n}] |b|={sum(b):2d} {cat:10s} val_loss={vloss:.3f}  "
                  f"(cov svhn={cov['pure_svhn']} cif={cov['pure_cifar']} mix={cov['mixed']})")

    print(f"[gen-loguniform] done: {len(list(save_dir.glob('obs_*.json')))} obs")
    print(f"  mean|b|={np.mean(sizes):.1f}  coverage: pure_svhn={cov['pure_svhn']} "
          f"pure_cifar={cov['pure_cifar']} mixed={cov['mixed']}")


if __name__ == "__main__":
    main()

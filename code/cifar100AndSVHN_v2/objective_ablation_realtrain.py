#!/usr/bin/env python3
"""Objective ablation WITHOUT the surrogate: real-trained u_k.

Takes ONE fixed DARTS architecture (sampled with seed 322, identical for
every expert and every split — so the comparison isolates the *split*, not
the architecture). For each candidate hard split it:
  1. builds each expert's b-vector,
  2. REALLY trains the fixed arch on that expert's cluster-subset
     (no surrogate) and measures u_k = mean per-sample val CE,
  3. computes the v2 objective L = sum_m |C_m| * log sum_k r_mk * exp(-u_k).

Splits compared:
  - sgem   : the final hard_assignments from the latest SGEM run (lbw2),
  - ideal  : ground-truth domain split (19 CIFAR / 11 SVHN) from meta.json,
  - random : a balanced 15/15 random split (context).

This answers: does the objective actually rank the ideal domain split above
the SGEM split, once u is HONEST (real training) rather than surrogate-predicted?
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CIFAR100_DIR = HERE.parent / "cifar100"
sys.path.insert(0, str(CIFAR100_DIR))

import cifar100_sgem_v2 as v2          # noqa: E402
from optimize_surrogate_em_v2 import prepare_data  # noqa: E402
from utils_v2 import compute_log_likelihood_loss    # noqa: E402


def b_from_assign(assign, M, k):
    return [1 if assign[m] == k else 0 for m in range(M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sgem-results", required=True,
                    help="results_*.json of the SGEM run to take the split from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init-channels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-results", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    meta = v2.load_cifar100_meta(data_dir)
    v2._NUM_CLASSES = meta["num_classes"]
    M = int(meta["n_clusters"])
    K = 2

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    ss = v2.CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    data = prepare_data(X, y, cluster_dir=str(data_dir))
    Xtr_c = data["X_train_by_cluster"]
    ytr_c = data["y_train_by_cluster"]
    X_val = data["X_val"]; y_val = data["y_val"]
    val_cluster_ids = data["val_cluster_ids"]

    train_cluster_ids = np.load(data_dir / "train_cluster_ids.npy")
    cluster_sizes = np.array(
        [int((train_cluster_ids == m).sum()) for m in range(M)], dtype=np.float64)

    # ground-truth domain split
    ideal = meta["ideal_split_by_source"]
    cif = set(ideal["cifar_clusters"])
    dom = np.array(["cifar" if m in cif else "svhn" for m in range(M)])

    # SGEM split
    sg = json.load(open(args.sgem_results))["cifar100_sgem"]
    sgem_assign = np.array(sg["hard_assignments"])

    # ideal assign: expert0=cifar, expert1=svhn
    ideal_assign = np.array([0 if dom[m] == "cifar" else 1 for m in range(M)])

    # random balanced 15/15
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(M)
    random_assign = np.zeros(M, dtype=int)
    random_assign[perm[15:]] = 1

    # ONE fixed architecture, identical everywhere (seed already set above)
    fixed_cfg = ss.create_random_config()
    assert isinstance(fixed_cfg, dict), type(fixed_cfg)
    print("[fixed arch] ops:",
          [fixed_cfg.get(f"op_{i}") for i in range(9)])

    def domain_match(assign):
        return max(sum(1 for m in range(M)
                       if dom[m] == ("cifar" if (int(assign[m]) ^ f) == 0 else "svhn"))
                   for f in (0, 1))

    splits = {"sgem": sgem_assign, "ideal": ideal_assign, "random": random_assign}
    out = {}
    for name, assign in splits.items():
        counts = [int((assign == k).sum()) for k in range(K)]
        print(f"\n===== split={name}  counts={counts}  "
              f"domain-match={domain_match(assign)}/{M} =====")
        u = np.zeros(K)
        per_expert = []
        for k in range(K):
            b = b_from_assign(assign, M, k)
            ncl = sum(b)
            nc = sum(1 for m in range(M) if b[m] and dom[m] == "cifar")
            nsv = sum(1 for m in range(M) if b[m] and dom[m] == "svhn")
            uk = v2.evaluate_architecture_on_subset_cifar100_v2(
                fixed_cfg, ss, b, Xtr_c, ytr_c, X_val, y_val,
                epochs=args.epochs, val_cluster_ids=val_cluster_ids)
            u[k] = uk
            per_expert.append(dict(expert=k, n_clusters=ncl, cifar=nc, svhn=nsv,
                                   u=float(uk)))
            print(f"  expert{k}: {ncl} clusters (CIFAR={nc} SVHN={nsv})  "
                  f"REAL u=val_loss={uk:.4f}")
        # objective with hard r
        r = np.zeros((M, K))
        r[np.arange(M), assign] = 1.0
        L = compute_log_likelihood_loss(r, u, cluster_sizes)
        print(f"  OBJECTIVE L = {L:.1f}   (u={u.round(4).tolist()})")
        out[name] = dict(counts=counts, domain_match=domain_match(assign),
                         u=u.tolist(), objective=float(L),
                         per_expert=per_expert,
                         assignment=[int(a) for a in assign])

    print("\n" + "=" * 60)
    print("SUMMARY (higher objective = better per v2 formula):")
    for name in ("ideal", "sgem", "random"):
        o = out[name]
        print(f"  {name:7s}: L={o['objective']:10.1f}  "
              f"domain-match={o['domain_match']}/{M}  "
              f"u={[round(x,3) for x in o['u']]}")
    best = max(out, key=lambda n: out[n]["objective"])
    print(f"  -> objective PREFERS: {best}")
    print(f"  -> is ideal the argmax? {'YES' if best=='ideal' else 'NO'}")

    if args.save_results:
        json.dump(dict(args=vars(args), fixed_arch=fixed_cfg, splits=out),
                  open(args.save_results, "w"), indent=2)
        print("saved ->", args.save_results)


if __name__ == "__main__":
    main()

import json, glob, os
import numpy as np
D = "/pbabkin/nas-for-moe/code/cifar100/runs_testsplit"

# Methods to include (K=2, CIFAR+SVHN). Exclude old sgem.
WANT_PREFIXES = [
    ("DARTS oracle MoE (ideal split, cheats)", "cifar100_darts_oracle_moe"),
    ("DARTS x K (same arch)",                  "cifar100_darts_moe_svhn_K2"),
    ("Random-MoE (cluster gate)",              "cifar100_random_moe_cluster_svhn_K2_n200"),
]

def find_cluster_acc(obj):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cluster" and isinstance(v, dict):
                for ak in ("test_acc", "acc", "val_acc"):
                    if ak in v:
                        hits.append(float(v[ak])); break
            hits += find_cluster_acc(v)
    elif isinstance(obj, list):
        for v in obj:
            hits += find_cluster_acc(v)
    return hits

rows = []
for label, prefix in WANT_PREFIXES:
    vals = []
    for f in glob.glob(D + f"/results_{prefix}*.json"):
        try:
            h = find_cluster_acc(json.load(open(f)))
            if h:
                vals.append(h[0])
        except Exception:
            pass
    vals = sorted(set(round(v, 4) for v in vals))
    if vals:
        rows.append((label, len(vals), np.mean(vals), vals))

# our result
rows.append(("SGEM honest (per-cluster, kmin3)  [OURS]", 1, 0.6100, [0.6100]))

print(f"{'method':45s} {'n':>2} {'mean':>7}   values")
for label, n, m, vals in sorted(rows, key=lambda r: -r[2]):
    print(f"{label:45s} {n:>2} {m:>7.4f}   " + " ".join(f"{v:.3f}" for v in vals))

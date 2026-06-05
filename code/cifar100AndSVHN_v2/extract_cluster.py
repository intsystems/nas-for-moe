import json, glob, os
import numpy as np
D = "/pbabkin/nas-for-moe/code/cifar100/runs_testsplit"

def find_cluster_acc(obj, path=""):
    """Recursively find any 'cluster' gating test_acc / acc in a result dict."""
    hits = []
    if isinstance(obj, dict):
        # common shapes: {"cluster": {"test_acc": x}} or {"final_moe":{"cluster":...}}
        for k, v in obj.items():
            if k == "cluster" and isinstance(v, dict):
                for ak in ("test_acc", "acc", "val_acc", "weighted_test_acc"):
                    if ak in v:
                        hits.append((path + "/cluster", ak, v[ak]))
            hits += find_cluster_acc(v, path + "/" + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_cluster_acc(v, f"{path}[{i}]")
    return hits

groups = {}
for f in sorted(glob.glob(D + "/results_*.json")):
    name = os.path.basename(f).replace("results_", "").replace(".json", "")
    try:
        obj = json.load(open(f))
    except Exception:
        continue
    hits = find_cluster_acc(obj)
    if not hits:
        continue
    # group by prefix (strip _seedNNN)
    base = name
    for tok in ("_seed322", "_seed1", "_seed2", "_seed3", "_seed4"):
        base = base.replace(tok, "")
    for p, ak, val in hits:
        try:
            val = float(val)
        except Exception:
            continue
        groups.setdefault((base, ak), []).append(val)

print("=== cluster-gating test_acc across baselines (CIFAR+SVHN testsplit) ===")
print(f"{'method (base)':45s} {'metric':>14s} {'n':>3} {'mean':>7} {'vals'}")
for (base, ak), vals in sorted(groups.items()):
    vals = sorted(vals)
    print(f"{base:45s} {ak:>14s} {len(vals):>3} {np.mean(vals):>7.4f}  "
          + " ".join(f"{v:.3f}" for v in vals))

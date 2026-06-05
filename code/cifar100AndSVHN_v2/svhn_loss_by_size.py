import json, glob
import numpy as np
DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
meta = json.load(open(DATA + "/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}

# scan BOTH log-uniform seed datasets (old [1,M] and kmin3 if present) + clean
DIRS = [
    "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/loguniform_seed_obs_500",
    "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/loguniform_seed_obs_500_kmin3",
    "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/clean_seed_obs_100",
]
rows = []  # (size, val_loss) for PURE SVHN obs
allrows = []
for d in DIRS:
    for p in glob.glob(d + "/obs_*.json"):
        o = json.load(open(p)); b = o["subset_b"]; vl = o.get("val_loss")
        if vl is None:
            continue
        sel = [m for m, f in enumerate(b) if f]
        nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
        if nc == 0 and ns > 0:           # PURE SVHN
            rows.append((ns, vl))
        allrows.append((len(sel), nc, ns, vl))

print(f"Total obs scanned: {len(allrows)}  |  PURE SVHN obs: {len(rows)}")
print("\n=== PURE SVHN: val_loss by #svhn clusters |b| ===")
print(f"{'|b|':>4} {'n':>4} {'loss_mean':>10} {'loss_min':>9} {'loss_max':>9}")
by = {}
for sz, vl in rows:
    by.setdefault(sz, []).append(vl)
for sz in sorted(by):
    v = np.array(by[sz])
    print(f"{sz:>4} {len(v):>4} {v.mean():>10.3f} {v.min():>9.3f} {v.max():>9.3f}")

print("\n=== individual smallest pure-SVHN samples (|b|=1,2,3) ===")
for sz, vl in sorted(rows):
    if sz <= 3:
        print(f"  |b|={sz}  val_loss={vl:.3f}")

# reference: full svhn from objective ablation
print("\nreference: ideal full-SVHN expert (|b|=11) real val_loss ≈ 0.51 (from objective-ablation)")

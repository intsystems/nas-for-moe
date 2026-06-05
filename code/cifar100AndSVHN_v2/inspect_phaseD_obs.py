import json, glob
import numpy as np
DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
OBS = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/sgem_v2_K2_seed322_phaseD_loguniform_e5x10_lbw1_obs"
meta = json.load(open(DATA + "/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}

rows = []
for p in sorted(glob.glob(OBS + "/obs_*.json"), key=lambda x: int(x.split("_")[-1].split(".")[0])):
    o = json.load(open(p)); b = o["subset_b"]; vl = o["val_loss"]
    sel = [m for m, f in enumerate(b) if f]
    nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
    tot = nc + ns
    cat = "pure_svhn" if nc == 0 else ("pure_cifar" if ns == 0 else "mixed")
    rows.append((sum(b), cat, vl, nc, ns))

print(f"Phase-D produced {len(rows)} obs")
for cat in ("pure_svhn", "pure_cifar", "mixed"):
    rr = [r for r in rows if r[1] == cat]
    if rr:
        vls = [r[2] for r in rr]
        print(f"  {cat:11s}: n={len(rr):2d}  val_loss mean={np.mean(vls):.3f} "
              f"min={np.min(vls):.3f} max={np.max(vls):.3f}")
    else:
        print(f"  {cat:11s}: n=0")

print("\n=== all PURE obs from phase D (|b|, domain, REAL val_loss) ===")
for sz, cat, vl, nc, ns in rows:
    if cat.startswith("pure"):
        print(f"  |b|={sz:2d}  {cat:10s}  val_loss={vl:.3f}")

print("\n=== smallest |b| obs (where clean lives) ===")
for sz, cat, vl, nc, ns in sorted(rows)[:12]:
    print(f"  |b|={sz:2d}  {cat:10s} (cif={nc},svhn={ns})  val_loss={vl:.3f}")

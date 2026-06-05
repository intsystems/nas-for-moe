import json, glob, numpy as np
DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
OBS = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/sgem_v2_K2_seed322_s500_e10x50_lbw2_obs"
meta = json.load(open(DATA + "/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
P = []; VL = []; MAJ = []; TOT = []
for p in glob.glob(OBS + "/obs_*.json"):
    o = json.load(open(p)); b = o["subset_b"]; vl = o.get("val_loss")
    if vl is None:
        continue
    sel = [m for m, f in enumerate(b) if f]
    nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
    tot = nc + ns
    if tot == 0:
        continue
    P.append(max(nc, ns) / tot); VL.append(vl); MAJ.append("cifar" if nc >= ns else "svhn"); TOT.append(tot)
P = np.array(P); VL = np.array(VL); MAJ = np.array(MAJ)
print("N obs:", len(P))
print("max purity:", round(float(P.max()), 3),
      " #purity>=0.8:", int((P >= 0.8).sum()),
      " >=0.9:", int((P >= 0.9).sum()),
      " ==1.0:", int((P >= 0.999).sum()))
print("=== purity bins -> real val_loss ===")
for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
    m = (P >= lo) & (P < hi)
    if m.sum():
        print(f"  purity[{lo:.1f},{hi:.1f}): n={int(m.sum()):4d}  loss_mean={VL[m].mean():.3f}  loss_min={VL[m].min():.3f}")
    else:
        print(f"  purity[{lo:.1f},{hi:.1f}): n=0")
print("=== top-12 purest obs ===")
order = np.argsort(-P)[:12]
for i in order:
    print(f"  purity={P[i]:.3f} major={MAJ[i]:5s} |b|={int(TOT[i]):2d} val_loss={VL[i]:.3f}")
print("=== corr(purity, loss) by major domain (negative=purer->lower) ===")
for d in ("svhn", "cifar"):
    m = MAJ == d
    if m.sum() > 5:
        print(f"  [{d}-major] n={int(m.sum())} corr={np.corrcoef(P[m], VL[m])[0,1]:+.3f}")

#!/usr/bin/env python3
"""Does adding clean obs let the surrogate RANK clean vs mixed correctly?

Train the surrogate on the UNION of given obs dirs (e.g. 1000 mixed seed obs
+ 100 clean domain-pure obs), then probe predicted u on:
  - pure_svhn / pure_cifar    (real u = 0.51 / 2.73 from real-train ablation)
  - the SGEM mixed split        (real u ~ 1.9)
  - several held-out clean obs and mixed obs (compare pred vs their real loss)

Reports whether pred(pure_svhn) < pred(mixed) — the ordering the EM needs.
Domains are used ONLY for evaluation/labelling, never for training.
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cifar100")); sys.path.insert(0, str(HERE.parent))

from optimize_surrogate_em_v2 import (        # noqa: E402
    retrain_surrogate_from_observations, _config_to_pyg_data, _mc_dropout_predict, OPS,
)


def purity_cat(b, dom):
    sel = [m for m, f in enumerate(b) if f == 1]
    if not sel:
        return "empty"
    nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
    if nc == 0:
        return "pure_svhn"
    if ns == 0:
        return "pure_cifar"
    return "mixed"


def predict_u(surr, cfg, b, nf, dev):
    data = _config_to_pyg_data(cfg, b)
    batch = Batch.from_data_list([data]).to(dev)
    mu, sigma = _mc_dropout_predict(surr, batch, n_forward=nf, device=dev)
    return float(mu.item()), float(sigma.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--obs-dirs", required=True, help="comma-separated obs dirs to UNION for training")
    ap.add_argument("--ablation-results", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--n-forward", type=int, default=30)
    ap.add_argument("--holdout-clean", type=int, default=15, help="hold out N clean obs from training for eval")
    ap.add_argument("--save-results", default=None)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    meta = json.load(open(Path(args.data_dir) / "meta.json"))
    cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = int(meta["n_clusters"])
    dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
    cc = np.load(Path(args.data_dir) / "cluster_centers.npy")

    dirs = [Path(d) for d in args.obs_dirs.split(",")]
    all_paths = []
    for d in dirs:
        ps = sorted(d.glob("obs_*.json"))
        all_paths.append((d.name, ps))
        print(f"[obs] {d.name}: {len(ps)} obs")

    # hold out some clean obs from the LAST dir (assumed clean) for eval
    clean_dir_name, clean_paths = all_paths[-1]
    rng = random.Random(args.seed)
    clean_shuf = clean_paths[:]
    rng.shuffle(clean_shuf)
    holdout = set(map(str, clean_shuf[:args.holdout_clean]))

    train_paths = []
    for name, ps in all_paths:
        for p in ps:
            if str(p) not in holdout:
                train_paths.append(p)
    print(f"[train] union = {len(train_paths)} obs  (held out {len(holdout)} clean for eval)")

    # training-signal coverage
    cov = {"pure_svhn": 0, "pure_cifar": 0, "mixed": 0}
    for p in train_paths:
        o = json.load(open(p))
        c = purity_cat(o["subset_b"], dom)
        if c in cov:
            cov[c] += 1
    print(f"[coverage in TRAIN] pure_svhn={cov['pure_svhn']} pure_cifar={cov['pure_cifar']} mixed={cov['mixed']}")

    dev = args.device
    surr = retrain_surrogate_from_observations(
        train_paths, len(OPS), M, device=dev, verbose=True,
        model_type="gat", nodes_per_graph=4, cluster_centers=cc)
    surr.eval()

    abl = json.load(open(args.ablation_results))
    fixed_arch = abl["fixed_arch"]; splits = abl.get("splits", {})

    def b_pure(d):
        return [1 if dom[m] == d else 0 for m in range(M)]

    print("\n" + "=" * 70)
    print("(1) CONTROLLED PROBES (fixed arch) — pred_u vs REAL_u")
    print("=" * 70)
    probes = {"pure_svhn (11cl)": (b_pure("svhn"), 0.5122),
              "pure_cifar (19cl)": (b_pure("cifar"), 2.7305)}
    if "sgem" in splits:
        a = splits["sgem"]["assignment"]
        probes["sgem_mix_e0 (15cl)"] = ([1 if a[m]==0 else 0 for m in range(M)], splits["sgem"]["u"][0])
        probes["sgem_mix_e1 (15cl)"] = ([1 if a[m]==1 else 0 for m in range(M)], splits["sgem"]["u"][1])
    print(f"{'probe':22s} {'pred_u':>9s} {'sigma':>7s} {'REAL_u':>8s} {'err':>8s}")
    out = {}
    for name, (b, ru) in probes.items():
        mu, sg = predict_u(surr, fixed_arch, b, args.n_forward, dev)
        print(f"{name:22s} {mu:9.3f} {sg:7.3f} {ru:8.3f} {mu-ru:+8.3f}")
        out[name] = dict(pred=mu, sigma=sg, real=ru)

    ps = out["pure_svhn (11cl)"]["pred"]; pc = out["pure_cifar (19cl)"]["pred"]
    mix = np.mean([out[k]["pred"] for k in out if k.startswith("sgem")]) if any(k.startswith("sgem") for k in out) else None
    print("\nRANKING CHECK (what EM needs):")
    print(f"  pred(pure_svhn)={ps:.3f}  pred(pure_cifar)={pc:.3f}  pred(mixed~sgem)={mix:.3f}" if mix else "")
    print(f"  pure_svhn < mixed ? {'YES ✓' if ps < mix else 'NO ✗'}  (real: 0.51 < 1.9, should be YES)")
    print(f"  domain gap: real=2.22, surrogate pred={pc-ps:+.3f} "
          f"({'PRESERVED' if (pc-ps)>1.0 else 'COLLAPSED'})")

    # (2) held-out clean obs: pred vs their own real loss
    print("\n" + "=" * 70)
    print("(2) HELD-OUT CLEAN OBS — pred vs real (these were NOT in training)")
    print("=" * 70)
    ho_rows = []
    for sp in sorted(holdout):
        o = json.load(open(sp))
        mu, sg = predict_u(surr, o["arch"], o["subset_b"], args.n_forward, dev)
        cat = purity_cat(o["subset_b"], dom)
        ho_rows.append((cat, sum(o["subset_b"]), o["val_loss"], mu))
    for cat in ("pure_svhn", "pure_cifar"):
        rr = [r for r in ho_rows if r[0] == cat]
        if rr:
            real = np.mean([r[2] for r in rr]); pred = np.mean([r[3] for r in rr])
            print(f"  {cat:11s}: n={len(rr):2d}  real_mean={real:.3f}  pred_mean={pred:.3f}  err={pred-real:+.3f}")
    if args.save_results:
        json.dump(dict(coverage=cov, probes=out,
                       holdout=[{"cat": r[0], "size": r[1], "real": r[2], "pred": r[3]} for r in ho_rows]),
                  open(args.save_results, "w"), indent=2)
        print("saved ->", args.save_results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose WHY the surrogate fails to drive domain specialization.

(A) Training-signal check: group collected obs by domain-purity of their
    b-vector, report mean REAL val_loss. Does the training data even contain
    'pure SVHN -> low loss'?
(B) Surrogate prediction check: retrain surrogate from obs (pipeline-identical),
    query predicted u (mu via MC-dropout) on controlled b-vectors with the SAME
    fixed arch used by the real-train ablation. Compare predicted vs real u
    (ideal-SVHN=0.51, ideal-CIFAR=2.73, sgem-e0=1.91, sgem-e1=1.90).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cifar100"))
sys.path.insert(0, str(HERE.parent))  # code/ root, for toy_experiment.*

from torch_geometric.data import Batch       # noqa: E402
from optimize_surrogate_em_v2 import (        # noqa: E402
    retrain_surrogate_from_observations,
    _config_to_pyg_data,
    _mc_dropout_predict,
    OPS,
)


def purity(b, dom):
    sel = [m for m, f in enumerate(b) if f == 1]
    nc = sum(dom[m] == "cifar" for m in sel); ns = sum(dom[m] == "svhn" for m in sel)
    if nc + ns == 0:
        return "empty"
    if nc == 0:
        return "pure_svhn"
    if ns == 0:
        return "pure_cifar"
    return "mixed"


def predict_u(surr, config, b, n_forward, device):
    data = _config_to_pyg_data(config, b)
    batch = Batch.from_data_list([data]).to(device)
    mu, sigma = _mc_dropout_predict(surr, batch, n_forward=n_forward, device=device)
    return float(mu.item()), float(sigma.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--ablation-results", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--n-forward", type=int, default=30)
    ap.add_argument("--save-results", default=None)
    args = ap.parse_args()

    meta = json.load(open(Path(args.data_dir) / "meta.json"))
    cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = int(meta["n_clusters"])
    dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
    cc = np.load(Path(args.data_dir) / "cluster_centers.npy")

    # ---- (A) training signal ----
    obs_paths = sorted(Path(args.obs_dir).glob("obs_*.json"))
    buckets = {"pure_svhn": [], "pure_cifar": [], "mixed": []}
    for p in obs_paths:
        o = json.load(open(p))
        vl = o.get("val_loss")
        if vl is None:
            continue
        c = purity(o["subset_b"], dom)
        if c in buckets:
            buckets[c].append(float(vl))
    print("=" * 66)
    print(f"(A) TRAINING SIGNAL — real val_loss across {len(obs_paths)} obs, by b-purity")
    print("=" * 66)
    for c in ("pure_svhn", "pure_cifar", "mixed"):
        v = np.array(buckets[c])
        if len(v):
            print(f"  {c:11s}: n={len(v):4d}  mean={v.mean():.3f}  min={v.min():.3f}  max={v.max():.3f}")
        else:
            print(f"  {c:11s}: n=0   <-- surrogate NEVER trained on this regime")

    # ---- (B) surrogate predictions ----
    print("\n" + "=" * 66)
    print("(B) SURROGATE PREDICTION — retrain from obs, query controlled b")
    print("=" * 66)
    dev = args.device
    surr = retrain_surrogate_from_observations(
        obs_paths, len(OPS), M,
        device=dev, verbose=True, model_type="gat",
        nodes_per_graph=4, cluster_centers=cc,
    )
    surr.eval()

    abl = json.load(open(args.ablation_results))
    fixed_arch = abl["fixed_arch"]; splits = abl.get("splits", {})

    def b_pure(d):
        return [1 if dom[m] == d else 0 for m in range(M)]

    probes = {
        "pure_svhn (11cl)":  (b_pure("svhn"),  0.5122),
        "pure_cifar (19cl)": (b_pure("cifar"), 2.7305),
    }
    if "sgem" in splits:
        a = splits["sgem"]["assignment"]
        probes["sgem_e0 (15cl)"] = ([1 if a[m] == 0 else 0 for m in range(M)], splits["sgem"]["u"][0])
        probes["sgem_e1 (15cl)"] = ([1 if a[m] == 1 else 0 for m in range(M)], splits["sgem"]["u"][1])

    print(f"\n{'probe':20s} {'pred_u(mu)':>11s} {'sigma':>7s} {'REAL_u':>8s} {'err':>8s}")
    out = {}
    for name, (b, real_u) in probes.items():
        mu, sigma = predict_u(surr, fixed_arch, b, args.n_forward, dev)
        print(f"{name:20s} {mu:11.3f} {sigma:7.3f} {real_u:8.3f} {mu-real_u:+8.3f}")
        out[name] = dict(pred_u=mu, sigma=sigma, real_u=real_u, err=mu - real_u)

    if "pure_svhn (11cl)" in out and "pure_cifar (19cl)" in out:
        ps = out["pure_svhn (11cl)"]["pred_u"]; pc = out["pure_cifar (19cl)"]["pred_u"]
        print("\nKEY CONTRAST — does surrogate SEE that pure-SVHN is easy?")
        print(f"  REAL:      pure_svhn=0.512  pure_cifar=2.731   gap=2.219")
        print(f"  SURROGATE: pure_svhn={ps:.3f}  pure_cifar={pc:.3f}   gap={pc-ps:+.3f}")
        print(f"  -> surrogate {'PRESERVES' if (pc-ps) > 1.0 else 'COLLAPSES'} the domain gap "
              f"({100*(pc-ps)/2.219:+.0f}% of real gap)")

    if args.save_results:
        json.dump(dict(training_signal={k: buckets[k] for k in buckets},
                       predictions=out), open(args.save_results, "w"), indent=2)
        print("saved ->", args.save_results)


if __name__ == "__main__":
    main()

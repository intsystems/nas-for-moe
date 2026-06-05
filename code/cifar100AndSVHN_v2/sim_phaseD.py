#!/usr/bin/env python3
"""Simulate Phase D selection: does a log-uniform-trained surrogate, when
picking subsets of each expert's split, ENRICH the chosen b-vectors toward a
single domain (i.e. pull toward the ideal SVHN/CIFAR partition)?

Setup:
  - surrogate retrained on a log-uniform dataset (it can rank splits),
  - a RANDOM split of M clusters into K experts,
  - good architectures (from cleaninit results),
  - run the Phase-D selection rule per expert: sample subsets of split_k of
    size ~U(3, |split_k|-3), score each FLAT by surrogate (no MC), pick best.
Measure: domain purity / SVHN-fraction of the SELECTED subsets vs the random
candidate pool. If the surrogate works, selected subsets should be more
domain-pure than random pool (enriched), demonstrating Phase D pulls toward
ideal partition.
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cifar100")); sys.path.insert(0, str(HERE.parent))
from optimize_surrogate_em_v2 import (        # noqa
    retrain_surrogate_from_observations, OPS,
)
from torch_geometric.data import Data
from toy_experiment.optimize_expert_assignments import build_graph_data


def flat_score(surr, cfg, b, device):
    """Flat surrogate value (dropout off, no MC). Lower = better."""
    surr.eval()
    x, edge_index = build_graph_data(cfg)
    batch = Batch.from_data_list([Data(x=x, edge_index=edge_index)]).to(device)
    bool_vec = torch.tensor(b, dtype=torch.float, device=device).unsqueeze(0)  # [1, M]
    with torch.no_grad():
        out = surr(batch.x, batch.edge_index, batch.batch, bool_vec)
    return float(out.squeeze(-1).item())


def sample_subset(clusters, rng):
    sk = len(clusters)
    if sk <= 3:
        size = sk
    else:
        size = rng.randint(3, max(3, sk - 3))
    sel = rng.sample(list(clusters), size)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--arch-results", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--pool", type=int, default=200, help="candidate subsets per expert")
    ap.add_argument("--topn", type=int, default=20, help="how many best selected")
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    meta = json.load(open(Path(args.data_dir) / "meta.json"))
    cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
    dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
    cc = np.load(Path(args.data_dir) / "cluster_centers.npy")

    obs = sorted(Path(args.obs_dir).glob("obs_*.json"))
    print(f"[surrogate] training on {len(obs)} obs from {Path(args.obs_dir).name}")
    surr = retrain_surrogate_from_observations(
        obs, len(OPS), M, device=args.device, verbose=True,
        model_type="gat", nodes_per_graph=4, cluster_centers=cc)

    archs = json.load(open(args.arch_results))["cifar100_sgem"]["configs"]
    K = len(archs)

    # RANDOM split of M clusters into K experts
    assign = np.array([rng.randrange(K) for _ in range(M)])
    print(f"\n[random split] " + " ".join(
        f"e{k}:{int((assign==k).sum())}cl(svhn={sum(1 for m in range(M) if assign[m]==k and dom[m]=='svhn')})"
        for k in range(K)))

    def svhn_frac(sub):
        return sum(1 for m in sub if dom[m] == "svhn") / len(sub)

    for k in range(K):
        split_k = [m for m in range(M) if assign[k] == assign[k] and assign[m] == k]
        if len(split_k) < 4:
            print(f"\n=== expert {k}: split too small ({len(split_k)}) — skip ==="); continue
        base_svhn = sum(1 for m in split_k if dom[m] == "svhn") / len(split_k)

        # candidate pool of subsets
        pool = [sample_subset(split_k, rng) for _ in range(args.pool)]
        scored = sorted(((flat_score(surr, archs[k], _b(sub, M), args.device), sub)
                         for sub in pool), key=lambda t: t[0])
        sel = scored[:args.topn]

        pool_sv = np.mean([svhn_frac(s) for s in pool])
        sel_sv = np.mean([svhn_frac(s) for _, s in sel])
        # also: purity = max(svhn_frac, 1-svhn_frac) averaged
        pool_pur = np.mean([max(svhn_frac(s), 1 - svhn_frac(s)) for s in pool])
        sel_pur = np.mean([max(svhn_frac(s), 1 - svhn_frac(s)) for _, s in sel])
        best_pred, best_sub = sel[0]
        print(f"\n=== expert {k}  (arch #{k}) split={len(split_k)}cl, "
              f"base svhn-frac={base_svhn:.2f} ===")
        print(f"  pool  : svhn-frac mean={pool_sv:.3f}  purity mean={pool_pur:.3f}")
        print(f"  TOP{args.topn}: svhn-frac mean={sel_sv:.3f}  purity mean={sel_pur:.3f}  "
              f"(enrichment {'SVHN +' if sel_sv>pool_sv else 'CIFAR +'}{abs(sel_sv-pool_sv):.3f})")
        bsv = svhn_frac(best_sub)
        print(f"  BEST  : pred={best_pred:.3f}  |b|={len(best_sub)}  "
              f"svhn-frac={bsv:.2f}  -> {'PURE '+('SVHN' if bsv==1 else 'CIFAR') if bsv in (0.0,1.0) else 'mixed'}")


def _b(sub, M):
    b = [0] * M
    for j in sub:
        b[j] = 1
    return b


if __name__ == "__main__":
    main()

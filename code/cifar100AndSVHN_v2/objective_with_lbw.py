#!/usr/bin/env python3
"""Compare splits by the PENALIZED objective the optimizer actually sees.

The M-step minimizes  loss = -q_function + lbw * lb_penalty  (entropy off),
i.e. it MAXIMIZES  (likelihood) - lbw * lb_penalty.

So to know which split optimization is pulled toward, we must rank splits by
    score(lbw) = L - lbw * lb_penalty
where
    L          = compute_log_likelihood_loss(r, u, |C_m|)      (the true objective)
    lb_penalty = (K * sum_k P_k^2) * sum_m|C_m|    with P_k = mean_m r_mk
                 (exact formula from optimize_surrogate_em_v2.py fix-B)

Uses the REAL trained u from results_objective_ablation_realtrain_lbw2split.json.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils_v2 import compute_log_likelihood_loss   # noqa: E402

DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
ABL = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/results_objective_ablation_realtrain_lbw2split.json"

meta = json.load(open(DATA + "/meta.json"))
M = int(meta["n_clusters"]); K = 2
tcid = np.load(DATA + "/train_cluster_ids.npy")
csize = np.array([(tcid == m).sum() for m in range(M)], dtype=np.float64)
N = csize.sum()

abl = json.load(open(ABL))["splits"]


def lb_penalty(assign):
    r = np.zeros((M, K)); r[np.arange(M), assign] = 1.0
    P = r.mean(axis=0)                 # [K]
    load_balance = K * (P * P).sum()   # in [1, K]
    return load_balance * N, load_balance


print(f"N(train)={int(N)}  K={K}\n")
LBWS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0]

rows = {}
for name in ("ideal", "sgem", "random"):
    s = abl[name]
    assign = np.array(s["assignment"]); u = np.array(s["u"], dtype=np.float64)
    r = np.zeros((M, K)); r[np.arange(M), assign] = 1.0
    L = compute_log_likelihood_loss(r, u, csize)
    pen, lbn = lb_penalty(assign)
    rows[name] = dict(assign=assign, u=u, L=L, pen=pen, lbn=lbn,
                      counts=[int((assign == k).sum()) for k in range(K)])
    print(f"{name:7s} split={rows[name]['counts']}  u={u.round(3).tolist()}  "
          f"L={L:10.1f}  LB_norm={lbn:.3f}  lb_penalty={pen:11.1f}")

print("\n=== score(lbw) = L - lbw * lb_penalty  (higher = optimizer prefers) ===")
hdr = f"{'lbw':>5} " + " ".join(f"{n:>13}" for n in ("ideal", "sgem", "random")) + "   winner"
print(hdr)
for lbw in LBWS:
    sc = {n: rows[n]["L"] - lbw * rows[n]["pen"] for n in rows}
    win = max(sc, key=sc.get)
    line = f"{lbw:>5.1f} " + " ".join(f"{sc[n]:13.0f}" for n in ("ideal", "sgem", "random"))
    print(line + f"   {win}{'  <-- ideal LOSES' if win!='ideal' else ''}")

# crossover lbw where sgem overtakes ideal
dL = rows["ideal"]["L"] - rows["sgem"]["L"]
dPen = rows["ideal"]["pen"] - rows["sgem"]["pen"]
if dPen > 0:
    cross = dL / dPen
    print(f"\nideal beats sgem only while lbw < {cross:.2f}")
    print(f"  (L advantage of ideal = {dL:.0f}; extra penalty per unit lbw = {dPen:.0f})")
else:
    print("\nideal has <= penalty than sgem; ideal wins for all lbw>=0")

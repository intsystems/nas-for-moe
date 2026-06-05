#!/usr/bin/env python3
"""Oracle ablation for SGEM-v2 routing (no surrogate, no architecture).

Replaces the surrogate u(alpha_k, R_k) with a domain-purity ORACLE that
depends only on which clusters are assigned to expert k:

    u_k = min(#cifar clusters in R_k, #svhn clusters in R_k)   [oracle-mode=min]

  - pure subset  (only one domain)  -> u = 0   (best / lowest loss)
  - mixed subset (both domains)     -> u > 0   (worse)

This isolates the EM routing optimisation (E-step + gradient M-step on r,
with the v2 |C_m|-weighted log-likelihood objective
    L = sum_m |C_m| * log sum_k r_mk * exp(-u_k)
and the fix-(B) load-balance penalty) from surrogate quality.
The S-step / architecture search are dropped because u does not depend on
alpha here.

Ground truth: SVHN/CIFAR cluster split from meta.json -> ideal_split_by_source.
NOTE the true split is 19 CIFAR / 11 SVHN (imbalanced), so a load-balance
penalty (which pulls toward 15/15) FIGHTS the correct answer; run with
--load-balance-weight 0 to test the oracle alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def gumbel_softmax_rows(logits: torch.Tensor, tau: float, hard: bool) -> torch.Tensor:
    """Per-row (per-cluster) Gumbel-Softmax over the K experts. [M,K] -> [M,K]."""
    u = torch.rand_like(logits).clamp_(1e-20, 1.0)
    g = -torch.log(-torch.log(u))
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        idx = y.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, idx, 1.0)
        y = (y_hard - y).detach() + y  # straight-through
    return y


def oracle_u(R: torch.Tensor, cif_mask: torch.Tensor, svhn_mask: torch.Tensor,
             mode: str, u_svhn: float = 0.3, u_cifar: float = 2.2,
             u_mixed_peak: float = 1.8) -> torch.Tensor:
    """u per expert from an [M,K] assignment matrix (soft or hard). Returns [K].

    Modes:
      min/max : count-based toy oracle (pure -> 0).
      realistic : surrogate-like CE oracle. Let f = cifar fraction of the
        expert's subset. Then
            u(f) = (1-f)*u_svhn + f*u_cifar + bump*4*f*(1-f)
        with bump chosen so u(0)=u_svhn, u(1)=u_cifar, u(0.5)=u_mixed_peak.
        => pure SVHN -> 0.3, pure CIFAR -> 2.2, balanced mix -> 1.8.
        Captures: SVHN intrinsically easy, CIFAR hard, mixing hurts
        (interference bump) — i.e. the asymmetric, biased signal a real
        NLL-surrogate would emit.
    """
    cif = (R * cif_mask.unsqueeze(1)).sum(dim=0)    # [K]
    svhn = (R * svhn_mask.unsqueeze(1)).sum(dim=0)  # [K]
    if mode == "min":
        return torch.minimum(cif, svhn)   # pure -> 0, mixed -> >0
    if mode == "max":
        return torch.maximum(cif, svhn)
    if mode == "realistic":
        tot = cif + svhn + 1e-8
        f = cif / tot                                  # cifar fraction [K]
        bump = u_mixed_peak - 0.5 * (u_svhn + u_cifar)
        return (1 - f) * u_svhn + f * u_cifar + bump * 4.0 * f * (1 - f)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n-em-iterations", type=int, default=15)
    ap.add_argument("--n-r-gradient-steps", type=int, default=200)
    ap.add_argument("--r-lr", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--n-r-mc-samples", type=int, default=8)
    ap.add_argument("--load-balance-weight", type=float, default=0.0)
    ap.add_argument("--entropy-weight", type=float, default=0.0)
    ap.add_argument("--u-scale", type=float, default=1.0,
                    help="multiply oracle u by this (controls exp(-u) sharpness)")
    ap.add_argument("--oracle-mode", choices=["min", "max", "realistic"],
                    default="min")
    ap.add_argument("--u-svhn", type=float, default=0.3)
    ap.add_argument("--u-cifar", type=float, default=2.2)
    ap.add_argument("--u-mixed-peak", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-results", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    D = Path(args.data_dir)

    meta = json.load(open(D / "meta.json"))
    cif = set(meta["ideal_split_by_source"]["cifar_clusters"])
    svh = set(meta["ideal_split_by_source"]["svhn_clusters"])
    M = int(meta["n_clusters"])
    dom = np.array(["cifar" if m in cif else "svhn" for m in range(M)])

    train_cid = np.load(D / "train_cluster_ids.npy")
    cluster_sizes = np.array([(train_cid == m).sum() for m in range(M)],
                             dtype=np.float32)

    cif_mask = torch.tensor([1.0 if m in cif else 0.0 for m in range(M)], device=dev)
    svhn_mask = torch.tensor([1.0 if m in svh else 0.0 for m in range(M)], device=dev)
    c_t = torch.tensor(cluster_sizes, device=dev)
    K = args.K

    print(f"[oracle-ablation] M={M}, K={K}, oracle-mode={args.oracle_mode}, "
          f"lbw={args.load_balance_weight}, u_scale={args.u_scale}, seed={args.seed}")
    print(f"[ground truth] CIFAR clusters={len(cif)}  SVHN clusters={len(svh)}  "
          f"(true split is {len(cif)}/{len(svh)}, NOT balanced)")

    def domain_match(assign):  # K==2 only
        best = 0
        for flip in (0, 1):
            ok = sum(1 for m in range(M)
                     if dom[m] == ("cifar" if (int(assign[m]) ^ flip) == 0 else "svhn"))
            best = max(best, ok)
        return best

    logits = (torch.randn(M, K, device=dev) * 0.01).requires_grad_(True)
    history = []

    for em in range(1, args.n_em_iterations + 1):
        # ---------- E-step: q_{mk} ∝ r_mk exp(-u_k), u from hard argmax R ----------
        with torch.no_grad():
            r_soft = F.softmax(logits, dim=-1)
            assign = r_soft.argmax(dim=-1)
            R_hard = torch.zeros(M, K, device=dev).scatter_(
                1, assign.unsqueeze(1), 1.0)
            u_e = oracle_u(R_hard, cif_mask, svhn_mask, args.oracle_mode,
                            args.u_svhn, args.u_cifar, args.u_mixed_peak) * args.u_scale
            log_q = torch.log(r_soft.clamp_min(1e-30)) + (-u_e).unsqueeze(0)
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)
            q = log_q.exp()  # [M,K]

        # ---------- M-step: gradient ascent on logits (mirrors v2) ----------
        opt = torch.optim.Adam([logits], lr=args.r_lr)
        for _ in range(args.n_r_gradient_steps):
            opt.zero_grad()
            r_soft = F.softmax(logits, dim=-1)
            log_r = torch.log(r_soft.clamp_min(1e-10))
            q_log_r = (c_t.unsqueeze(1) * q * log_r).sum()

            negu = torch.zeros(K, device=dev)
            for _ in range(args.n_r_mc_samples):
                R = gumbel_softmax_rows(logits, args.tau, hard=True)
                u_k = oracle_u(R, cif_mask, svhn_mask, args.oracle_mode,
                               args.u_svhn, args.u_cifar, args.u_mixed_peak) * args.u_scale
                negu = negu - u_k
            negu = negu / args.n_r_mc_samples
            q_log_u = (c_t.unsqueeze(1) * q * negu.unsqueeze(0)).sum()
            q_function = q_log_r + q_log_u

            entropy = -(r_soft * log_r).sum()
            P = r_soft.mean(dim=0)
            load_balance = K * (P * P).sum()              # normalized in [1,K]
            lb_penalty = load_balance * c_t.sum()         # fix (B): extensive

            loss = (-q_function
                    - args.entropy_weight * entropy
                    + args.load_balance_weight * lb_penalty)
            loss.backward()
            opt.step()

        # ---------- report ----------
        with torch.no_grad():
            r_soft = F.softmax(logits, dim=-1)
            assign = r_soft.argmax(dim=-1).cpu().numpy()
            R_hard = torch.zeros(M, K, device=dev).scatter_(
                1, torch.tensor(assign, device=dev).unsqueeze(1), 1.0)
            u_fin = oracle_u(R_hard, cif_mask, svhn_mask, args.oracle_mode,
                             args.u_svhn, args.u_cifar, args.u_mixed_peak).cpu().numpy()
            counts = [int((assign == k).sum()) for k in range(K)]
            P = r_soft.mean(dim=0).cpu().numpy()
            LBn = float(K * (P * P).sum())
            rec = dict(em=em, counts=counts, u=u_fin.tolist(), LB_norm=LBn)
            line = (f"iter {em:2d}: counts={counts} u_expert={u_fin.tolist()} "
                    f"LB_norm={LBn:.3f}")
            if K == 2:
                dm = domain_match(assign)
                rec["domain_match"] = dm
                line += f"  domain-match={dm}/{M}={100*dm/M:.0f}%"
            history.append(rec)
            print(line)

    print("\n=== FINAL ===")
    for k in range(K):
        cl = [m for m in range(M) if assign[m] == k]
        nc = sum(dom[m] == "cifar" for m in cl)
        ns = sum(dom[m] == "svhn" for m in cl)
        print(f"expert{k}: {len(cl):2d} clusters -> CIFAR={nc:2d} SVHN={ns:2d}  {cl}")
    print(f"ground-truth: CIFAR={len(cif)} SVHN={len(svh)}")
    if K == 2:
        dm = domain_match(assign)
        print(f"DOMAIN-MATCH: {dm}/{M} = {100*dm/M:.1f}%  "
              f"(30/30 = perfect SVHN vs CIFAR separation)")

    if args.save_results:
        json.dump(dict(args=vars(args), history=history,
                       final_assignment=[int(a) for a in assign],
                       domain=dom.tolist(),
                       cluster_sizes=cluster_sizes.tolist(),
                       cifar_clusters=sorted(cif), svhn_clusters=sorted(svh)),
                  open(args.save_results, "w"), indent=2)
        print("saved ->", args.save_results)


if __name__ == "__main__":
    main()

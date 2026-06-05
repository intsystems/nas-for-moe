import json, numpy as np
DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
R = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/results_sgem_v2_K2_seed322_cleaninit_e10x50_lbw2.json"
meta = json.load(open(DATA + "/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
res = json.load(open(R))["cifar100_sgem"]

def match(a):
    return max(sum(1 for m in range(M) if dom[m] == ("cifar" if (int(a[m]) ^ f) == 0 else "svhn")) for f in (0, 1))

snaps = res.get("iteration_snapshots", [])
print(f"{'iter':>4} {'n_obs':>6} {'e0':>3} {'e1':>3} {'CIFAR@e':>8} {'SVHN@e':>7} {'d-match':>8} {'log_lik':>11}")
first_ideal = None
for s in snaps:
    it = s["em_iter"]; ha = s["hard_assignments"]; nobs = s.get("n_obs", "?")
    dm = match(ha)
    e0 = sum(1 for a in ha if a == 0); e1 = M - e0
    # which expert is svhn (by majority)
    e0cl = [m for m in range(M) if ha[m] == 0]
    e0_sv = sum(dom[m] == "svhn" for m in e0cl)
    svhn_exp = 0 if e0_sv >= len(e0cl) - e0_sv else 1
    sv_clusters = [m for m in range(M) if ha[m] == svhn_exp]
    sv_pure = sum(dom[m] == "svhn" for m in sv_clusters)
    ll = s.get("log_lik", float("nan"))
    flag = ""
    if dm == M and first_ideal is None:
        first_ideal = it; flag = "  <-- FIRST 30/30"
    print(f"{it:>4} {str(nobs):>6} {e0:>3} {e1:>3} {'-':>8} {'-':>7} {dm:>5}/30 {ll:>11.1f}{flag}")

print(f"\nfirst reached ideal (30/30): EM iter {first_ideal}")
ha = res["hard_assignments"]
print(f"FINAL saved split domain-match = {match(ha)}/30")
print("history (log_lik):", [round(x) for x in res["history"]])

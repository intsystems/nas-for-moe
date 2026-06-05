import json, numpy as np
DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
R = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/results_sgem_v2_K2_seed322_cleaninit_e10x50_lbw2.json"
meta = json.load(open(DATA + "/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
res = json.load(open(R))["cifar100_sgem"]
ha = np.array(res["hard_assignments"])
def match(a):
    return max(sum(1 for m in range(M) if dom[m] == ("cifar" if (int(a[m]) ^ f) == 0 else "svhn")) for f in (0, 1))
print("split: e0=%d e1=%d  objective=%.1f" % ((ha == 0).sum(), (ha == 1).sum(), res["objective_value"]))
for e in (0, 1):
    cl = [m for m in range(M) if ha[m] == e]
    nc = sum(dom[m] == "cifar" for m in cl); ns = sum(dom[m] == "svhn" for m in cl)
    print("  expert%d: %2d clusters -> CIFAR=%2d SVHN=%2d" % (e, len(cl), nc, ns))
m = match(ha)
print("DOMAIN-MATCH: %d/30 = %.0f%%   (baseline lbw2 was 53%%)" % (m, 100 * m / M))
snaps = res.get("iteration_snapshots", [])
if snaps:
    print("e0-count per EM iter:", [sum(1 for a in s["hard_assignments"] if a == 0) for s in snaps])
    print("domain-match per EM iter:", [match(s["hard_assignments"]) for s in snaps])

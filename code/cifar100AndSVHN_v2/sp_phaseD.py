import json
meta = json.load(open("/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit/meta.json"))
cif = set(meta["ideal_split_by_source"]["cifar_clusters"]); M = meta["n_clusters"]
dom = {m: ("cifar" if m in cif else "svhn") for m in range(M)}
res = json.load(open("/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/results_sgem_v2_K2_seed322_phaseD_loguniform_e5x10_lbw1.json"))["cifar100_sgem"]
def match(a):
    return max(sum(1 for m in range(M) if dom[m] == ("cifar" if (int(a[m]) ^ f) == 0 else "svhn")) for f in (0, 1))
ha = res["hard_assignments"]
print("FINAL split e0=%d e1=%d  objective=%.0f" % (sum(1 for a in ha if a==0), sum(1 for a in ha if a==1), res["objective_value"]))
for e in (0, 1):
    cl = [m for m in range(M) if ha[m] == e]
    print("  expert%d: %2d cl -> CIFAR=%2d SVHN=%2d" % (e, len(cl), sum(dom[m]=="cifar" for m in cl), sum(dom[m]=="svhn" for m in cl)))
print("FINAL DOMAIN-MATCH: %d/30 = %d%%   (baseline 53, seed-only 67, cleaninit 100)" % (match(ha), round(100*match(ha)/M)))
print("per-EM-iter domain-match:", [match(s["hard_assignments"]) for s in res.get("iteration_snapshots", [])])
print("log_lik history:", [round(x) for x in res["history"]])

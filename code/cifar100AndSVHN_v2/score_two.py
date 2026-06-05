import json
meta=json.load(open("/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit/meta.json"))
cif=set(meta["ideal_split_by_source"]["cifar_clusters"]); M=meta["n_clusters"]
dom={m:("cifar" if m in cif else "svhn") for m in range(M)}
def match(a): return max(sum(1 for m in range(M) if dom[m]==("cifar" if (int(a[m])^f)==0 else "svhn")) for f in (0,1))
import glob
for s in (1,2):
    R=f"/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2/runs_v2/results_sgem_v2_K2_perclu_kmin3_e15x30_fr05_seed{s}.json"
    try:
        res=json.load(open(R))["cifar100_sgem"]
    except Exception as e:
        print(f"seed{s}: not ready ({e})"); continue
    ha=res["hard_assignments"]
    snaps=res.get("iteration_snapshots",[])
    dms=[match(x["hard_assignments"]) for x in snaps]
    print(f"=== seed{s}: FINAL d-match {match(ha)}/30 = {round(100*match(ha)/M)}%  split e0={ha.count(0)}")
    print(f"  per-EM d-match: {dms}")
    print(f"  peak d-match: {max(dms) if dms else '-'}/30")
    print(f"  objective: {round(res['objective_value'])}")

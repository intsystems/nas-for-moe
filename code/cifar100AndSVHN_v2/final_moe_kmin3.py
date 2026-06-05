"""Train the final MoE on the kmin3 SGEM result (found archs + found split).
final=True: train on train∪val, eval once on held-out test → test_acc.
Both gating modes: learnable softmax gate, cluster hard-assignment gate.
"""
import sys, json
from pathlib import Path

CONT = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
sys.path.insert(0, CONT)
sys.path.insert(0, "/pbabkin/nas-for-moe/code/cifar100")
sys.path.insert(0, "/pbabkin/nas-for-moe/code")

import cifar100_sgem_v2 as v2
from cifar100_final_train import train_final_moe

DATA = Path("/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit")
RES = CONT + "/runs_v2/results_sgem_v2_K2_seed322_perclu_kmin3_e10x10_lbw1.json"
meta = v2.load_cifar100_meta(DATA)
NUM_CLASSES = meta["num_classes"]

sg = json.load(open(RES))["cifar100_sgem"]
configs = sg["configs"]
hard = sg["hard_assignments"]
print(f"[final-moe] K={len(configs)} experts, split e0={hard.count(0)} e1={hard.count(1)}, "
      f"num_classes={NUM_CLASSES}")

# domain reference
cif = set(meta["ideal_split_by_source"]["cifar_clusters"])
dom = {m: ("cifar" if m in cif else "svhn") for m in range(meta["n_clusters"])}
for e in (0, 1):
    cl = [m for m in range(len(hard)) if hard[m] == e]
    print(f"  expert{e}: {len(cl)}cl  CIFAR={sum(dom[m]=='cifar' for m in cl)} "
          f"SVHN={sum(dom[m]=='svhn' for m in cl)}")

out = {}
for mode in ("learnable", "cluster"):
    print(f"\n===== final MoE mode={mode}, epochs=100 (train∪val → test) =====")
    r = train_final_moe(
        configs=configs, hard_assignments=hard, data_dir=DATA,
        mode=mode, final=True, init_channels=16, num_classes=NUM_CLASSES,
        gate_channels=16, epochs=100, batch_size=128, lr=0.05, wd=3e-4,
        seed=322, device="cuda:0", verbose=True)
    out[mode] = r
    print(f"  -> test_acc = {r.get('test_acc')}")

print("\n========== SUMMARY ==========")
for mode in ("learnable", "cluster"):
    r = out[mode]
    print(f"  {mode:>10s}: test_acc = {r.get('test_acc'):.4f}"
          + (f"  n_params={r.get('n_params')}" if r.get('n_params') else ""))

json.dump({k: {kk: vv for kk, vv in v.items() if not hasattr(vv, '__len__') or isinstance(vv, (str, list))}
           for k, v in out.items()},
          open(CONT + "/runs_v2/results_final_moe_kmin3.json", "w"), indent=2, default=str)
print("saved -> results_final_moe_kmin3.json")

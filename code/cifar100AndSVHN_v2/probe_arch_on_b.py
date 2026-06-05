"""Why does pure-SVHN |b|=3 cost 5.6 in Phase D but ~1.2 in the seed dataset?
Train expert0's topology vs several random topologies on the SAME b=[0,3,16].
If random archs give ~1 and configs[0] gives ~5, the expert topology is the cause.
"""
import json, sys
from pathlib import Path
import numpy as np

CONT = "/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
sys.path.insert(0, CONT)
sys.path.insert(0, "/pbabkin/nas-for-moe/code/cifar100")
sys.path.insert(0, "/pbabkin/nas-for-moe/code")

import cifar100_sgem_v2 as v2
from optimize_surrogate_em_v2 import prepare_data

DATA = "/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
meta = v2.load_cifar100_meta(Path(DATA)); v2._NUM_CLASSES = meta["num_classes"]
M = meta["n_clusters"]
X = np.load(DATA + "/data_X.npy"); y = np.load(DATA + "/data_y.npy")
ss = v2.CIFAR100DartsSearchSpace(init_channels=16)
data = prepare_data(X, y, cluster_dir=DATA)
Xtr = data["X_train_by_cluster"]; ytr = data["y_train_by_cluster"]
Xv = data["X_val"]; yv = data["y_val"]; vci = data["val_cluster_ids"]

b = [0] * M
for c in (0, 3, 16):
    b[c] = 1  # pure SVHN, |b|=3

res = json.load(open(CONT + "/runs_v2/results_sgem_v2_K2_seed322_phaseDpe_e5x10_lbw1_fr03.json"))["cifar100_sgem"]
expert0_cfg = res["configs"][0]

def ev(cfg, tag):
    vl = v2.evaluate_architecture_on_subset_cifar100_v2(
        cfg, ss, b, Xtr, ytr, Xv, yv, epochs=30, val_cluster_ids=vci)
    print(f"  {tag}: val_loss={vl:.3f}")
    return vl

print("b = pure SVHN clusters [0,3,16], |b|=3  (phase D reported val_loss=5.59)")
print("--- expert0 topology (configs[0]) ---")
ev(expert0_cfg, "expert0_cfg")
print("--- 4 random topologies ---")
import random
random.seed(0)
for i in range(4):
    ev(ss.create_random_config(), f"random_{i}")

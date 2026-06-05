"""Reproduce in-run conditions: call set_seed(322) like the run does, then
eval the same b=[0,3,16] with expert0 arch. If THIS gives ~5.6 (not 0.82),
the culprit is set_seed / global state, not phase D logic.
Also test: many torch RNG advances before eval (mimic mid-run state).
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
CONT="/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
sys.path.insert(0,CONT); sys.path.insert(0,"/pbabkin/nas-for-moe/code/cifar100"); sys.path.insert(0,"/pbabkin/nas-for-moe/code")
import cifar100_sgem_v2 as v2
from optimize_surrogate_em_v2 import prepare_data
import toy_experiment.collect_dataset as cd

DATA="/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
meta=v2.load_cifar100_meta(Path(DATA)); v2._NUM_CLASSES=meta["num_classes"]; M=meta["n_clusters"]
X=np.load(DATA+"/data_X.npy"); y=np.load(DATA+"/data_y.npy")
ss=v2.CIFAR100DartsSearchSpace(init_channels=16)
data=prepare_data(X,y,cluster_dir=DATA)
Xtr=data["X_train_by_cluster"]; ytr=data["y_train_by_cluster"]; Xv=data["X_val"]; yv=data["y_val"]; vci=data["val_cluster_ids"]
res=json.load(open(CONT+"/runs_v2/results_sgem_v2_K2_seed322_phaseDpe_e5x10_lbw1_fr03.json"))["cifar100_sgem"]
cfg0=res["configs"][0]
b=[0]*M
for c in (0,3,16): b[c]=1

def run(tag):
    vl=v2.evaluate_architecture_on_subset_cifar100_v2(cfg0,ss,b,Xtr,ytr,Xv,yv,epochs=30,val_cluster_ids=vci)
    print(f"  {tag}: val_loss={vl:.3f}")

print("b=[0,3,16] pure SVHN |b|=3. in-run=5.59, my earlier probe(no set_seed)=0.82")
print("--- A: after cd.set_seed(322) ---")
cd.set_seed(322); run("set_seed(322)")
print("--- B: cudnn flags ---")
print("  cudnn.deterministic=",torch.backends.cudnn.deterministic," benchmark=",torch.backends.cudnn.benchmark)
print("--- C: no seed at all (fresh) ---")
run("fresh")

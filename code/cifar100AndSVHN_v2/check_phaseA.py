# Phase A expert0 evaluated split=[0,3,5,12,16,22,25,26,28] (9 clusters, all SVHN)
# in-run gave val_loss=2.90/2.77/2.75 across EM iters. But |b|=9 pure SVHN
# should give ~0.76 (from loss-by-size table). Re-train it in isolation.
import sys
from pathlib import Path
import numpy as np
CONT="/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
sys.path.insert(0,CONT); sys.path.insert(0,"/pbabkin/nas-for-moe/code/cifar100"); sys.path.insert(0,"/pbabkin/nas-for-moe/code")
import cifar100_sgem_v2 as v2
from optimize_surrogate_em_v2 import prepare_data
DATA="/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
meta=v2.load_cifar100_meta(Path(DATA)); v2._NUM_CLASSES=meta["num_classes"]; M=meta["n_clusters"]
X=np.load(DATA+"/data_X.npy"); y=np.load(DATA+"/data_y.npy")
ss=v2.CIFAR100DartsSearchSpace(init_channels=16)
data=prepare_data(X,y,cluster_dir=DATA)
Xtr=data["X_train_by_cluster"]; ytr=data["y_train_by_cluster"]; Xv=data["X_val"]; yv=data["y_val"]; vci=data["val_cluster_ids"]
import json
res=json.load(open(CONT+"/runs_v2/results_sgem_v2_K2_seed322_phaseDpe_e5x10_lbw1_fr03.json"))["cifar100_sgem"]
cfg0=res["configs"][0]
b=[0]*M
for c in [0,3,5,12,16,22,25,26,28]: b[c]=1
print("phase-A expert0 split |b|=9 pure-SVHN. In-run reported ~2.90/2.77/2.75")
vl=v2.evaluate_architecture_on_subset_cifar100_v2(cfg0,ss,b,Xtr,ytr,Xv,yv,epochs=30,val_cluster_ids=vci)
print(f"ISOLATED re-train: val_loss={vl:.3f}")

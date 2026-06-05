"""Isolate: does running the FULL pipeline up to phase D corrupt something so
that the SAME b=[0,3,16] eval gives 5.6 instead of 0.84?

Strategy: replicate the exact in-run sequence cheaply:
  1) load 500 obs, retrain surrogate (as run does at start),
  2) THEN eval b=[0,3,16].
If step1 makes the eval jump to 5.6 -> the surrogate-retrain / dataloader
leaves global state (e.g. torch default dtype, grad mode, cudnn) broken.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
CONT="/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
sys.path.insert(0,CONT); sys.path.insert(0,"/pbabkin/nas-for-moe/code/cifar100"); sys.path.insert(0,"/pbabkin/nas-for-moe/code")
import cifar100_sgem_v2 as v2
from optimize_surrogate_em_v2 import prepare_data, retrain_surrogate_from_observations, OPS

DATA="/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit"
meta=v2.load_cifar100_meta(Path(DATA)); v2._NUM_CLASSES=meta["num_classes"]; M=meta["n_clusters"]
X=np.load(DATA+"/data_X.npy"); y=np.load(DATA+"/data_y.npy")
ss=v2.CIFAR100DartsSearchSpace(init_channels=16)
data=prepare_data(X,y,cluster_dir=DATA)
Xtr=data["X_train_by_cluster"]; ytr=data["y_train_by_cluster"]; Xv=data["X_val"]; yv=data["y_val"]; vci=data["val_cluster_ids"]
cfg0=json.load(open(CONT+"/runs_v2/results_sgem_v2_K2_seed322_phaseDpe_e5x10_lbw1_fr03.json"))["cifar100_sgem"]["configs"][0]
b=[0]*M
for c in (0,3,16): b[c]=1

def ev(tag):
    vl=v2.evaluate_architecture_on_subset_cifar100_v2(cfg0,ss,b,Xtr,ytr,Xv,yv,epochs=30,val_cluster_ids=vci)
    print(f"  {tag}: val_loss={vl:.3f}  | grad_enabled={torch.is_grad_enabled()} default_dtype={torch.get_default_dtype()}")

print("b=[0,3,16] pure SVHN. baseline expect ~0.84")
ev("BEFORE surrogate-retrain")
print("now retrain surrogate on 500 obs (as the run does at startup)...")
obs=sorted(Path(CONT+"/runs_v2/loguniform_seed_obs_500").glob("obs_*.json"))
cc=np.load(DATA+"/cluster_centers.npy")
surr=retrain_surrogate_from_observations(obs,len(OPS),M,device="cuda:0",verbose=False,model_type="gat",nodes_per_graph=4,cluster_centers=cc)
ev("AFTER surrogate-retrain")

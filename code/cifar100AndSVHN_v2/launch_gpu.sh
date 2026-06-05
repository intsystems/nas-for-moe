#!/bin/bash
# args: GPU_ID  GEN_SEED  RUN_SEED
GPU=$1; GENSEED=$2; RUNSEED=$3
CONT=/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2
DATA=/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit
SEEDDIR=$CONT/runs_v2/loguniform_kmin3_500_gen${GENSEED}
RUN=sgem_v2_K2_perclu_kmin3_e15x30_fr05_seed${RUNSEED}
# Stage 1: generate own kmin3 seed dataset (500 models)
python -u $CONT/gen_loguniform_seed_obs.py \
  --data-dir $DATA --save-dir $SEEDDIR --n 500 --epochs 30 \
  --seed ${GENSEED} --device cuda:0 \
  > $CONT/logs_v2/gen_kmin3_gen${GENSEED}.log 2>&1
echo "[gpu${GPU}] gen done: $(ls $SEEDDIR/obs_*.json 2>/dev/null | wc -l) obs"
# Stage 2: SGEM run on it
python -u $CONT/cifar100_sgem_v2.py \
  --data-dir $DATA --K 2 \
  --initial-obs-dir $SEEDDIR \
  --n-em-iterations 15 --n-new-observations 30 \
  --cell-train-epochs 30 --surrogate-retrain-every 1 \
  --focused-ratio 0.5 --load-balance-weight 1.0 \
  --save-dir $CONT/runs_v2/${RUN}_obs \
  --save-results $CONT/runs_v2/results_${RUN}.json \
  --device cuda:0 --seed ${RUNSEED} --final-moe-epochs 0 \
  > $CONT/logs_v2/${RUN}.log 2>&1
echo "[gpu${GPU}] SGEM done"

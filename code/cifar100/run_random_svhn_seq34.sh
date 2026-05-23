#!/bin/bash
C=/pbabkin/nas-for-moe/code/cifar100
DATA=$C/cifar100_svhn_data_semantic_testsplit
LOG=/home/pbabkin/nas-for-moe/code/cifar100/runs_testsplit/random_svhn_seq34.log
for s in 3 4; do
  echo "=== start random-moe svhn seed=$s $(date) ===" >> $LOG
  docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=1 nas-moe-container \
    python -u $C/cifar100_random_moe_cluster.py \
      --data-dir $DATA \
      --save-results $C/runs_testsplit/results_cifar100_random_moe_cluster_svhn_K2_n200_seed${s}.json \
      --K 2 --n-moe-candidates 200 --moe-epochs 30 --final-epochs 100 \
      --seed $s --device cuda:0 >> $LOG 2>&1
  echo "=== done random-moe svhn seed=$s $(date) ===" >> $LOG
done

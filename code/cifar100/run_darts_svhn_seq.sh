#!/bin/bash
RT=/pbabkin/nas-for-moe/code/cifar100/runs_testsplit
DD=/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit
LOG=/home/pbabkin/nas-for-moe/code/cifar100/runs_testsplit/darts_svhn_seq.log
for S in 322 1 2; do
  echo "=== search seed=$S $(date) ===" >> $LOG
  docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=0 nas-moe-container \
    python -u /pbabkin/nas-for-moe/code/cifar100/cifar100_darts_search.py \
      --data-dir $DD --save-results $RT/results_cifar100_darts_search_svhn_seed${S}.json \
      --seed $S --device cuda:0 >> $LOG 2>&1
  echo "=== darts-moe K=2 seed=$S $(date) ===" >> $LOG
  docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=0 nas-moe-container \
    python -u /pbabkin/nas-for-moe/code/cifar100/cifar100_darts_moe_baseline.py \
      --darts-results $RT/results_cifar100_darts_search_svhn_seed${S}.json \
      --data-dir $DD --save-results $RT/results_cifar100_darts_moe_svhn_K2_seed${S}.json \
      --K 2 --mode both --epochs 100 --seed $S --device cuda:0 >> $LOG 2>&1
  echo "=== done seed=$S $(date) ===" >> $LOG
done

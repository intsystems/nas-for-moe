#!/bin/bash
RT=/pbabkin/nas-for-moe/code/cifar100/runs_testsplit
DD=/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit
LOG=/home/pbabkin/nas-for-moe/code/cifar100/runs_testsplit/darts_finetune_svhn_seq34.log
for S in 3 4; do
  echo "=== finetune single-arch seed=$S $(date) ===" >> $LOG
  docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=1 nas-moe-container \
    python -u /pbabkin/nas-for-moe/code/cifar100/cifar100_finetune_arch.py \
      --config-json $RT/results_cifar100_darts_search_svhn_seed${S}.json \
      --data-dir $DD --save-results $RT/results_cifar100_darts_finetune_svhn_seed${S}.json \
      --epochs 100 --tag darts_finetune_svhn_seed${S} --seed $S --device cuda:0 >> $LOG 2>&1
  echo "=== done seed=$S $(date) ===" >> $LOG
done

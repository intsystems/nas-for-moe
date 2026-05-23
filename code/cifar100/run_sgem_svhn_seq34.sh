#!/bin/bash
RT=/pbabkin/nas-for-moe/code/cifar100/runs_testsplit
DD=/pbabkin/nas-for-moe/code/cifar100/cifar100_svhn_data_semantic_testsplit
LOG=/home/pbabkin/nas-for-moe/code/cifar100/runs_testsplit/sgem_svhn_seq34.log
for S in 3 4; do
  echo "=== start seed=$S $(date) ===" >> $LOG
  docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=1 nas-moe-container \
    python -u /pbabkin/nas-for-moe/code/cifar100/cifar100_sgem.py \
      --data-dir $DD \
      --save-dir $RT/cifar100_sgem_obs_K2_svhn_pc05_lbw50_seed${S} \
      --save-results $RT/results_cifar100_sgem_K2_svhn_pc05_lbw50_seed${S}.json \
      --K 2 --phase-c-uniform-mix 0.5 --load-balance-weight 50 \
      --n-seed-observations 500 --n-em-iterations 50 --n-new-observations 40 \
      --final-moe-epochs 100 --final-moe-mode both \
      --seed $S --device cuda:0 >> $LOG 2>&1
  echo "=== done seed=$S $(date) ===" >> $LOG
done

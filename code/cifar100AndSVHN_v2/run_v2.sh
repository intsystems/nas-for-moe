#!/bin/bash
# Smoke-test launcher for cifar100AndSVHN_v2.
#
# Usage:
#   ./run_v2.sh [seed]            # GPU=0, default seed=322
#   GPU=1 ./run_v2.sh 42          # GPU=1, seed=42
#
# Defaults are intentionally tiny so a smoke run finishes in minutes;
# override on the command line / via env for a full run.
set -e

V2_HOST="$HOME/nas-for-moe/code/cifar100AndSVHN_v2"
V2_CONT="/pbabkin/nas-for-moe/code/cifar100AndSVHN_v2"
LOG_DIR="$V2_HOST/logs_v2"
mkdir -p "$LOG_DIR" "$V2_HOST/runs_v2"

SEED="${1:-322}"
GPU_ID="${GPU:-0}"
TAG="sgem_v2_K2_seed${SEED}"

# Ensure container running
if ! docker ps --format '{{.Names}}' | grep -q '^nas-moe-run$'; then
    echo "[run_v2] starting container nas-moe-run..."
    docker start nas-moe-run
fi

echo "[run_v2] launching $TAG on GPU $GPU_ID; log -> $LOG_DIR/$TAG.log"
docker exec -e PYTHONUNBUFFERED=1 -e CUDA_VISIBLE_DEVICES=$GPU_ID nas-moe-run \
    python -u "$V2_CONT/cifar100_sgem_v2.py" \
        --data-dir "$V2_CONT/../cifar100/cifar100_svhn_data_semantic_testsplit" \
        --K 2 \
        --n-seed-observations 20 \
        --n-em-iterations 10 \
        --n-new-observations 5 \
        --cell-train-epochs 30 \
        --surrogate-retrain-every 1 \
        --save-dir "$V2_CONT/runs_v2/${TAG}_obs" \
        --save-results "$V2_CONT/runs_v2/results_${TAG}.json" \
        --device cuda:0 \
        --seed "$SEED" \
        --final-moe-epochs 0 \
    > "$LOG_DIR/${TAG}.log" 2>&1 &

echo "[run_v2] PID $!"
echo "[run_v2] tail -f $LOG_DIR/${TAG}.log"

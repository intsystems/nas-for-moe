"""Evaluate a single architecture on linear clusters with surrogate and real training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from code.toy_experiment.legacy.optimize_expert_assignments import (
    load_surrogate,
    surrogate_eval_batch,
    evaluate_result_real,
    OptimizationResult,
)
from code.toy_experiment.legacy.collect_dataset import set_seed, SEED
from code.toy_experiment.legacy.toy_graph import OPS

DATA_DIR = Path("./data")
SURROGATE_PATH = Path("./model_dataset_balanced/surrogate_k20_n3000_best.pth")

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

cluster_centers = np.load(DATA_DIR / "cluster_centers.npy")
M = len(cluster_centers)

# Ideal split
threshold = -2.0
linear_clusters = [m for m in range(M) if cluster_centers[m, 0] < threshold]
ring_clusters = [m for m in range(M) if cluster_centers[m, 0] >= threshold]

# b vector: only linear clusters
b_linear = np.zeros(M)
for m in linear_clusters:
    b_linear[m] = 1.0

# Architecture: shift, linear, skip_connect, skip_connect
config = {
    'op_0': 'shift', 'input_0': [-1],
    'op_1': 'skip_connect', 'input_1': [0],
    'op_2': 'rbf', 'input_2': [0],
    'op_3': 'linear', 'input_3': [0],
}

print(f"Architecture: {config}")
print(f"Linear clusters: {linear_clusters}")
print(f"b vector: {b_linear}")

# Surrogate evaluation
surrogate = load_surrogate(
    str(SURROGATE_PATH),
    n_features=len(OPS),
    M=M,
    dropout=0.1,
    hidden_dim=128,
    heads=4,
    device=device,
)

R_columns = torch.tensor(b_linear, dtype=torch.float32).unsqueeze(0)  # [1, M]
with torch.no_grad():
    u = surrogate_eval_batch(surrogate, [config], R_columns, device=device)
print(f"\nSurrogate prediction: {u.item():.4f}")

# Real evaluation: K=2, expert 0 = linear clusters, expert 1 = ring (dummy)
K = 2
r = np.zeros((M, K))
for m in linear_clusters:
    r[m, 0] = 1.0
for m in ring_clusters:
    r[m, 1] = 1.0
hard = np.argmax(r, axis=1)

# Dummy config for expert 1 (ring) — same arch, we only care about expert 0
result = OptimizationResult(
    configs=[config, config],
    r_matrix=r,
    hard_assignments=hard,
    objective_value=0.0,
    method="Single arch eval",
)

print(f"\nReal evaluation (training on linear clusters only):")
evaluate_result_real(result, data_dir=str(DATA_DIR))

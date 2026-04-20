"""Evaluate ideal cluster split: linear clusters vs ring clusters.

Determines ideal assignment based on cluster center positions,
then evaluates with surrogate and ground truth.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from code.toy_experiment.legacy.optimize_expert_assignments import (
    OptimizationResult,
    load_surrogate,
    create_search_space,
    sample_architectures_for_experts,
    surrogate_eval_batch,
    compute_log_likelihood_numpy,
    print_result,
    evaluate_result_real,
)
from code.toy_experiment.legacy.collect_dataset import set_seed, SEED
from code.toy_experiment.legacy.toy_graph import OPS

DATA_DIR = Path("./data")
SURROGATE_PATH = Path("./model_dataset_balanced/surrogate_k20_n3000_best.pth")
K = 2

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load cluster centers
cluster_centers = np.load(DATA_DIR / "cluster_centers.npy")
M = len(cluster_centers)
print(f"M = {M} clusters")

# Ideal split: linear data ~ x=-5, ring data ~ x=1
threshold = -2.0
linear_clusters = [m for m in range(M) if cluster_centers[m, 0] < threshold]
ring_clusters = [m for m in range(M) if cluster_centers[m, 0] >= threshold]

print(f"\nIdeal split (threshold x < {threshold}):")
print(f"  Linear clusters ({len(linear_clusters)}): {linear_clusters}")
print(f"  Ring clusters ({len(ring_clusters)}): {ring_clusters}")
print(f"  Linear centers x: {[f'{cluster_centers[m,0]:.2f}' for m in linear_clusters]}")
print(f"  Ring centers x: {[f'{cluster_centers[m,0]:.2f}' for m in ring_clusters]}")

# Build hard assignment: expert 0 = linear, expert 1 = ring
r = np.zeros((M, K))
for m in linear_clusters:
    r[m, 0] = 1.0
for m in ring_clusters:
    r[m, 1] = 1.0
hard_assignments = np.argmax(r, axis=1)
print(f"\nHard assignment: {hard_assignments}")

# Load surrogate and search space
surrogate = load_surrogate(
    str(SURROGATE_PATH),
    n_features=len(OPS),
    M=M,
    dropout=0.1,
    hidden_dim=128,
    heads=4,
    device=device,
)
ss = create_search_space(input_dim=2)

# Find best architectures for each expert using surrogate
R_columns = torch.tensor(r.T, dtype=torch.float32)  # [K, M]
n_candidates = 2000
print(f"\nSearching best architectures ({n_candidates} candidates per expert)...")
best_configs = sample_architectures_for_experts(
    ss, K, n_candidates, surrogate, R_columns, device=device,
)

# Evaluate surrogate objective
with torch.no_grad():
    u_values = surrogate_eval_batch(surrogate, best_configs, R_columns, device=device)
    u_np = u_values.cpu().numpy()

obj = compute_log_likelihood_numpy(r, u_np)

result = OptimizationResult(
    configs=best_configs,
    r_matrix=r,
    hard_assignments=hard_assignments,
    objective_value=obj,
    history=[obj],
    method="Ideal split (linear vs ring)",
)

print_result(result)

# Surrogate predictions per expert
print("\n=== Surrogate predictions ===")
for k in range(K):
    label = "linear" if k == 0 else "ring"
    print(f"  Expert {k} ({label}): u = {u_np[k]:.4f}")

# Real evaluation
print("\n=== Real evaluation (ground truth) ===")
evaluate_result_real(result, data_dir=str(DATA_DIR))

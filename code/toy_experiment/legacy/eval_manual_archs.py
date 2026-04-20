"""
Ручная оценка: линейная архитектура на линейных кластерах,
RBF архитектура на кольцевых кластерах.

Цель — показать, какой результат должен давать хороший алгоритм.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from code.toy_experiment.legacy.collect_dataset import prepare_data, evaluate_architecture_on_subset, set_seed, SEED
from code.toy_experiment.legacy.optimize_expert_assignments import create_search_space

DATA_DIR = Path("./data")
N_NODES = 3
N_RUNS = 5
EPOCHS = 200

set_seed(SEED)

# --- Данные ---
X = np.load(DATA_DIR / "data_X.npy")
y = np.load(DATA_DIR / "data_y.npy")
data = prepare_data(X, y, cluster_dir=str(DATA_DIR))

cluster_centers = np.load(DATA_DIR / "cluster_centers.npy")
M = len(cluster_centers)

# Идеальное разбиение
threshold = -2.0
linear_clusters = [m for m in range(M) if cluster_centers[m, 0] < threshold]
ring_clusters = [m for m in range(M) if cluster_centers[m, 0] >= threshold]

print(f"M = {M} clusters")
print(f"Linear clusters ({len(linear_clusters)}): {linear_clusters}")
print(f"Ring clusters ({len(ring_clusters)}): {ring_clusters}")

b_linear = [1 if m in linear_clusters else 0 for m in range(M)]
b_ring = [1 if m in ring_clusters else 0 for m in range(M)]
b_all = [1] * M

ss = create_search_space(input_dim=2, num_nodes_per_cell=N_NODES)

# --- Архитектуры для тестирования ---
archs = {
    "linear_only": {
        "op_0": "linear", "input_0": [-1],
        "op_1": "linear", "input_1": [0],
        "op_2": "linear", "input_2": [1],
    },
    "linear+relu": {
        "op_0": "linear", "input_0": [-1],
        "op_1": "relu", "input_1": [0],
        "op_2": "linear", "input_2": [1],
    },
    "rbf_only": {
        "op_0": "rbf", "input_0": [-1],
        "op_1": "rbf", "input_1": [0],
        "op_2": "linear", "input_2": [1],
    },
    "shift+rbf": {
        "op_0": "shift", "input_0": [-1],
        "op_1": "rbf", "input_1": [0],
        "op_2": "linear", "input_2": [1],
    },
    "linear+rbf": {
        "op_0": "linear", "input_0": [-1],
        "op_1": "rbf", "input_1": [0],
        "op_2": "linear", "input_2": [1],
    },
}

subsets = {
    "linear_clusters": b_linear,
    "ring_clusters": b_ring,
    "all_data": b_all,
}

print(f"\nОценка: {N_RUNS} прогонов x {EPOCHS} эпох каждый\n")
print(f"{'Архитектура':<20} {'Подмножество':<20} {'Accuracy (mean±std)'}")
print("-" * 65)

for arch_name, config in archs.items():
    for subset_name, b in subsets.items():
        accs = []
        for run in range(N_RUNS):
            acc = evaluate_architecture_on_subset(
                config, ss, b,
                data["X_train_by_cluster"],
                data["y_train_by_cluster"],
                data["X_val"], data["y_val"],
                epochs=EPOCHS,
            )
            accs.append(acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{arch_name:<20} {subset_name:<20} {mean_acc:.4f} ± {std_acc:.4f}")
    print()

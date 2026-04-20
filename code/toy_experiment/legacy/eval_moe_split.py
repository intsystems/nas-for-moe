"""Evaluate MoE splits with per-cluster validation accuracy.

Trains each expert on its assigned clusters, then evaluates:
  - Global val accuracy per expert (accuracy on ALL val data)
  - Per-cluster val accuracy per expert (accuracy on only assigned clusters' val data)
  - MoE accuracy: for each val point, use the expert assigned to that cluster

Compares ideal split vs all-in-one baseline vs random splits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from code.toy_experiment.legacy.collect_dataset import (
    prepare_data,
    get_train_data_for_subset,
    train_cell,
    set_seed,
    SEED,
    sample_valid_config,
)
from code.toy_experiment.legacy.optimize_expert_assignments import create_search_space
from code.toy_experiment.legacy.toy_graph import OPS


def train_and_eval_expert(
    config: dict,
    search_space,
    b: list,
    X_train_by_cluster: list,
    y_train_by_cluster: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_cluster_ids: np.ndarray,
    epochs: int = 100,
) -> dict:
    """Train expert on clusters in b, evaluate globally and per-cluster."""
    X_sub, y_sub = get_train_data_for_subset(b, X_train_by_cluster, y_train_by_cluster)
    cell = search_space.create_cell_from_config(config)

    # Train
    train_mean = X_sub.mean(axis=0)
    X_train_t = torch.FloatTensor(X_sub - train_mean)
    y_train_t = torch.LongTensor(y_sub)
    X_val_t = torch.FloatTensor(X_val - train_mean)
    y_val_t = torch.LongTensor(y_val)

    # Classification head for >2 classes
    num_classes = max(int(y_sub.max()), int(y_val.max())) + 1
    input_dim = X_sub.shape[1]
    if num_classes > input_dim:
        head = nn.Linear(input_dim, num_classes)
        model = nn.Sequential(cell, head)
    else:
        model = cell

    optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    n_batches = max(1, len(X_train_t) // batch_size)

    for epoch in range(epochs):
        indices = torch.randperm(len(X_train_t))
        X_shuffled = X_train_t[indices]
        y_shuffled = y_train_t[indices]
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            optimizer.zero_grad()
            outputs = model(X_shuffled[start:end])
            loss = criterion(outputs, y_shuffled[start:end])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Evaluate
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = torch.argmax(val_outputs, dim=1)
        correct = (val_preds == y_val_t).numpy()

    global_acc = correct.mean()

    # Per-cluster accuracy for assigned clusters
    assigned_clusters = [m for m, flag in enumerate(b) if flag == 1]
    assigned_mask = np.isin(val_cluster_ids, assigned_clusters)
    if assigned_mask.sum() > 0:
        per_cluster_acc = correct[assigned_mask].mean()
    else:
        per_cluster_acc = 0.0

    return {
        "global_acc": float(global_acc),
        "per_cluster_acc": float(per_cluster_acc),
        "n_train": len(X_sub),
        "n_val_assigned": int(assigned_mask.sum()),
        "correct_per_val": correct,
    }


def eval_split(
    assignment: np.ndarray,
    K: int,
    configs: list,
    search_space,
    data: dict,
    epochs: int = 100,
    n_repeats: int = 3,
    verbose: bool = True,
) -> dict:
    """Evaluate a cluster-to-expert assignment with multiple repeats."""
    n_clusters = data["n_clusters"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    val_cluster_ids = data["val_cluster_ids"]

    all_moe_accs = []
    all_expert_results = []

    for rep in range(n_repeats):
        set_seed(SEED + rep * 100)

        expert_results = []
        # Collect predictions from each expert
        expert_correct = [None] * K

        for k in range(K):
            b_k = [1 if assignment[m] == k else 0 for m in range(n_clusters)]
            if sum(b_k) == 0:
                expert_results.append({
                    "global_acc": 0.0,
                    "per_cluster_acc": 0.0,
                    "n_train": 0,
                    "n_val_assigned": 0,
                })
                expert_correct[k] = np.zeros(len(y_val), dtype=bool)
                continue

            result = train_and_eval_expert(
                configs[k], search_space, b_k,
                data["X_train_by_cluster"],
                data["y_train_by_cluster"],
                X_val, y_val, val_cluster_ids,
                epochs=epochs,
            )
            expert_results.append(result)
            expert_correct[k] = result["correct_per_val"]

        # MoE accuracy: each val point routed to its cluster's expert
        moe_correct = np.zeros(len(y_val), dtype=bool)
        for i in range(len(y_val)):
            cluster_id = val_cluster_ids[i]
            expert_id = assignment[cluster_id]
            moe_correct[i] = expert_correct[expert_id][i]

        moe_acc = moe_correct.mean()
        all_moe_accs.append(moe_acc)
        all_expert_results.append(expert_results)

        if verbose:
            print(f"  Repeat {rep}: MoE acc = {moe_acc:.4f}", end="")
            for k in range(K):
                clusters_k = [m for m in range(n_clusters) if assignment[m] == k]
                print(f" | Expert {k} ({len(clusters_k)} cls): "
                      f"global={expert_results[k]['global_acc']:.4f} "
                      f"per_cluster={expert_results[k]['per_cluster_acc']:.4f}", end="")
            print()

    return {
        "moe_acc_mean": float(np.mean(all_moe_accs)),
        "moe_acc_std": float(np.std(all_moe_accs)),
        "moe_accs": all_moe_accs,
        "expert_results": all_expert_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE splits")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--n-nodes", type=int, default=3)
    parser.add_argument("--n-arch-candidates", type=int, default=50,
                        help="Architecture candidates per eval (random search)")
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of training repeats for averaging")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-random-splits", type=int, default=5,
                        help="Number of random balanced splits to try")
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")
    data = prepare_data(X, y, cluster_dir=str(data_dir))
    M = data["n_clusters"]
    K = args.K

    ss = create_search_space(input_dim=2, num_nodes_per_cell=args.n_nodes)

    # Cluster center info
    cluster_centers = data["cluster_centers"]
    threshold = -2.0
    linear_clusters = [m for m in range(M) if cluster_centers[m, 0] < threshold]
    ring_clusters = [m for m in range(M) if cluster_centers[m, 0] >= threshold]
    print(f"M={M}, K={K}")
    print(f"Linear clusters ({len(linear_clusters)}): {linear_clusters}")
    print(f"Ring clusters ({len(ring_clusters)}): {ring_clusters}")

    # Find good architectures by random search
    print(f"\nSampling {args.n_arch_candidates} architecture candidates...")
    arch_candidates = [sample_valid_config(ss) for _ in range(args.n_arch_candidates)]

    results = {}

    # === 1. Ideal split ===
    print("\n" + "=" * 60)
    print("IDEAL SPLIT (linear vs ring)")
    print("=" * 60)
    ideal_assignment = np.array([1 if m in ring_clusters else 0 for m in range(M)])
    print(f"Assignment: {ideal_assignment.tolist()}")

    # For ideal split, try multiple architectures per expert and pick best combo
    best_moe_acc = -1
    best_configs_ideal = None
    for trial in range(min(args.n_arch_candidates, 20)):
        set_seed(args.seed + trial)
        configs = [sample_valid_config(ss) for _ in range(K)]
        res = eval_split(ideal_assignment, K, configs, ss, data,
                         epochs=args.epochs, n_repeats=1, verbose=False)
        if res["moe_acc_mean"] > best_moe_acc:
            best_moe_acc = res["moe_acc_mean"]
            best_configs_ideal = configs
    print(f"Best arch trial MoE acc: {best_moe_acc:.4f}")

    # Full eval with best configs
    print("Full evaluation with best architectures:")
    ideal_result = eval_split(ideal_assignment, K, best_configs_ideal, ss, data,
                              epochs=args.epochs, n_repeats=args.n_repeats)
    results["ideal_split"] = {
        "assignment": ideal_assignment.tolist(),
        "configs": best_configs_ideal,
        **ideal_result,
    }
    print(f"=> MoE accuracy: {ideal_result['moe_acc_mean']:.4f} "
          f"± {ideal_result['moe_acc_std']:.4f}")

    # === 2. All-in-one baseline (K=1 equivalent) ===
    print("\n" + "=" * 60)
    print("ALL-IN-ONE BASELINE (one expert gets all clusters)")
    print("=" * 60)
    all_in_one = np.zeros(M, dtype=int)  # all clusters -> expert 0
    print(f"Assignment: {all_in_one.tolist()}")

    best_moe_acc = -1
    best_configs_all = None
    for trial in range(min(args.n_arch_candidates, 20)):
        set_seed(args.seed + trial)
        configs = [sample_valid_config(ss) for _ in range(K)]
        configs[1] = configs[0]  # doesn't matter, expert 1 has no clusters
        res = eval_split(all_in_one, K, configs, ss, data,
                         epochs=args.epochs, n_repeats=1, verbose=False)
        if res["moe_acc_mean"] > best_moe_acc:
            best_moe_acc = res["moe_acc_mean"]
            best_configs_all = configs
    print(f"Best arch trial MoE acc: {best_moe_acc:.4f}")

    print("Full evaluation with best architecture:")
    all_result = eval_split(all_in_one, K, best_configs_all, ss, data,
                            epochs=args.epochs, n_repeats=args.n_repeats)
    results["all_in_one"] = {
        "assignment": all_in_one.tolist(),
        "configs": best_configs_all,
        **all_result,
    }
    print(f"=> MoE accuracy: {all_result['moe_acc_mean']:.4f} "
          f"± {all_result['moe_acc_std']:.4f}")

    # === 3. Random balanced splits ===
    print("\n" + "=" * 60)
    print(f"RANDOM BALANCED SPLITS ({args.n_random_splits} trials)")
    print("=" * 60)
    random_results = []
    for split_idx in range(args.n_random_splits):
        set_seed(args.seed + 1000 + split_idx)
        perm = np.random.permutation(M)
        half = M // K
        rand_assignment = np.zeros(M, dtype=int)
        for k in range(1, K):
            rand_assignment[perm[k * half:(k + 1) * half]] = k
        # Also assign remainder
        rand_assignment[perm[K * half:]] = K - 1

        configs = [sample_valid_config(ss) for _ in range(K)]
        print(f"\nRandom split {split_idx}: {rand_assignment.tolist()}")
        res = eval_split(rand_assignment, K, configs, ss, data,
                         epochs=args.epochs, n_repeats=1)
        random_results.append({
            "assignment": rand_assignment.tolist(),
            **res,
        })

    avg_random = np.mean([r["moe_acc_mean"] for r in random_results])
    results["random_splits"] = random_results
    print(f"\n=> Average random MoE accuracy: {avg_random:.4f}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Ideal split MoE accuracy:    {results['ideal_split']['moe_acc_mean']:.4f} "
          f"± {results['ideal_split']['moe_acc_std']:.4f}")
    print(f"All-in-one MoE accuracy:     {results['all_in_one']['moe_acc_mean']:.4f} "
          f"± {results['all_in_one']['moe_acc_std']:.4f}")
    print(f"Random splits avg accuracy:  {avg_random:.4f}")

    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.save_results}")


if __name__ == "__main__":
    main()

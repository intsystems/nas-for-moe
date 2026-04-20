"""KMeans clustering with train/val split. Saves cluster IDs, centers, and split indices as .npy."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


def assign_to_nearest_cluster(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest centroid. Returns array of cluster IDs."""
    dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def cluster_data(
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    test_size: float = 0.2,
    seed: int = 322,
) -> dict:
    """
    Train/val split -> KMeans on train -> assign val points by nearest centroid.

    Returns dict with:
        train_indices, val_indices: split indices into original X
        train_cluster_ids: cluster ID for each train point
        val_cluster_ids: cluster ID for each val point
        cluster_centers: KMeans centroids
    """
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=seed
    )

    X_train = X[train_idx]

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(X_train)
    cluster_centers = kmeans.cluster_centers_

    X_val = X[val_idx]
    val_cluster_ids = assign_to_nearest_cluster(X_val, cluster_centers)

    return {
        "train_indices": train_idx,
        "val_indices": val_idx,
        "train_cluster_ids": train_cluster_ids,
        "val_cluster_ids": val_cluster_ids,
        "cluster_centers": cluster_centers,
    }


def main():
    parser = argparse.ArgumentParser(description="KMeans clustering with train/val split")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with data_X.npy (default: data/ next to script)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as data-dir)")
    parser.add_argument("--n-clusters", type=int, default=2, help="Number of clusters")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    else:
        data_dir = Path(args.data_dir)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    np.random.seed(args.seed)
    result = cluster_data(X, y, args.n_clusters, args.test_size, args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train_indices", "val_indices", "train_cluster_ids",
                  "val_cluster_ids", "cluster_centers"]:
        np.save(output_dir / f"{name}.npy", result[name])

    print(f"Saved to {output_dir}")
    print(f"  n_clusters: {args.n_clusters}")
    print(f"  train: {len(result['train_indices'])} samples")
    print(f"  val:   {len(result['val_indices'])} samples")
    for k in range(args.n_clusters):
        n_train = (result["train_cluster_ids"] == k).sum()
        n_val = (result["val_cluster_ids"] == k).sum()
        print(f"  Cluster {k}: {n_train} train, {n_val} val")

    if not args.no_plot:
        plot_clusters(
            X, result["train_indices"], result["val_indices"],
            result["train_cluster_ids"], result["val_cluster_ids"],
            result["cluster_centers"], output_dir / "clusters.png",
        )


def plot_clusters(
    X: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_cluster_ids: np.ndarray,
    val_cluster_ids: np.ndarray,
    centers: np.ndarray,
    save_path: Path,
):
    n_clusters = len(centers)
    if n_clusters <= 10:
        # Use Set1 for small cluster counts — high contrast between colors
        cmap = plt.cm.Set1
        colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
    else:
        cmap = plt.cm.tab20
        colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(10, 8))

    X_train = X[train_idx]

    for k in range(n_clusters):
        mask = train_cluster_ids == k
        ax.scatter(X_train[mask, 0], X_train[mask, 1],
                   label=f"Cluster {k}", alpha=0.6, s=30, color=colors[k])
    ax.scatter(centers[:, 0], centers[:, 1],
               marker="X", s=300, color="black", edgecolors="white",
               linewidth=2, label="Centroids")
    ax.set_xlabel("Feature 0", fontsize=11)
    ax.set_ylabel("Feature 1", fontsize=11)
    ax.set_title(f"KMeans clustering (K={n_clusters}, {len(X_train)} train samples)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

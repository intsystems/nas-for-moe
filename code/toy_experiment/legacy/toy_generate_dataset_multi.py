"""Generate multi-cluster toy dataset: 10 linear clouds + 10 ring clouds, shifted apart."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_linear_cloud(
    n_samples: int, center: tuple[float, float], scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable cloud around `center`."""
    X = np.random.normal(loc=0, scale=scale, size=(n_samples, 2))
    # decision boundary: x0 > x1 relative to center
    y = np.where(X[:, 0] > X[:, 1], 0, 1).astype(float)
    X[:, 0] += center[0]
    X[:, 1] += center[1]
    return X, y


def generate_ring_cloud(
    n_samples: int,
    center: tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a ring (annulus) cloud around `center`."""
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    x_inner = inner_radius * np.cos(theta_inner)
    y_inner = inner_radius * np.sin(theta_inner)

    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    x_outer = outer_radius * np.cos(theta_outer)
    y_outer = outer_radius * np.sin(theta_outer)

    X = np.vstack([
        np.column_stack([x_inner, y_inner]),
        np.column_stack([x_outer, y_outer]),
    ])
    y = np.hstack([np.zeros(n_inner), np.ones(n_outer)])
    X += np.random.normal(loc=0, scale=noise, size=X.shape)

    X[:, 0] += center[0]
    X[:, 1] += center[1]

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def arrange_centers(
    n_linear: int,
    n_ring: int,
    spacing: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Place linear centers in one region and ring centers in another, shifted apart."""
    linear_centers = []
    cols = int(np.ceil(np.sqrt(n_linear)))
    for i in range(n_linear):
        row, col = divmod(i, cols)
        linear_centers.append((col * spacing, row * spacing))

    ring_cols = int(np.ceil(np.sqrt(n_ring)))
    x_offset = (cols + 1) * spacing  # shift rings to the right
    ring_centers = []
    for i in range(n_ring):
        row, col = divmod(i, ring_cols)
        ring_centers.append((x_offset + col * spacing, row * spacing))

    return linear_centers, ring_centers


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-cluster toy dataset (10 linear + 10 ring clouds)"
    )
    parser.add_argument("--n-linear", type=int, default=10, help="Number of linear clouds")
    parser.add_argument("--n-ring", type=int, default=10, help="Number of ring clouds")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per cloud")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data_multi/ next to script)")
    parser.add_argument("--spacing", type=float, default=12.0,
                        help="Distance between cloud centers")
    parser.add_argument("--linear-scale", type=float, default=1.5,
                        help="Scale for linear cloud normal distribution")
    parser.add_argument("--inner-radius", type=float, default=1.0, help="Inner ring radius")
    parser.add_argument("--outer-radius", type=float, default=2.5, help="Outer ring radius")
    parser.add_argument("--ring-noise", type=float, default=0.3, help="Gaussian noise for rings")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data_multi"
    else:
        output_dir = Path(args.output_dir)

    linear_centers, ring_centers = arrange_centers(
        args.n_linear, args.n_ring, args.spacing
    )

    all_X, all_y, all_dtype = [], [], []

    for center in linear_centers:
        X, y = generate_linear_cloud(args.n_samples, center, args.linear_scale)
        all_X.append(X)
        all_y.append(y)
        all_dtype.append(np.zeros(len(X)))  # 0 = linear

    for center in ring_centers:
        X, y = generate_ring_cloud(
            args.n_samples, center, args.inner_radius, args.outer_radius, args.ring_noise
        )
        all_X.append(X)
        all_y.append(y)
        all_dtype.append(np.ones(len(X)))  # 1 = ring

    X_combined = np.vstack(all_X)
    y_combined = np.hstack(all_y)
    data_type_combined = np.hstack(all_dtype)

    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    data_type_combined = data_type_combined[indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "data_X.npy", X_combined)
    np.save(output_dir / "data_y.npy", y_combined)

    print(f"Saved to {output_dir}")
    print(f"  X shape: {X_combined.shape}")
    print(f"  y shape: {y_combined.shape}")
    print(f"  Linear clouds: {args.n_linear}, Ring clouds: {args.n_ring}")
    print(f"  Samples per cloud: {args.n_samples}")
    print(f"  Total samples: {len(X_combined)}")
    print(f"  Classes: {dict(zip(*np.unique(y_combined, return_counts=True)))}")

    if not args.no_plot:
        plot_dataset(X_combined, y_combined, data_type_combined, output_dir / "dataset.png")


def plot_dataset(X: np.ndarray, y: np.ndarray, data_type: np.ndarray, save_path: Path):
    linear_mask = data_type == 0
    ring_mask = data_type == 1

    fig, ax = plt.subplots(figsize=(14, 10))

    ax.scatter(X[linear_mask & (y == 0), 0], X[linear_mask & (y == 0), 1],
               label="Linear, class 0", alpha=0.5, s=15, color="blue", marker="o")
    ax.scatter(X[linear_mask & (y == 1), 0], X[linear_mask & (y == 1), 1],
               label="Linear, class 1", alpha=0.5, s=15, color="red", marker="o")
    ax.scatter(X[ring_mask & (y == 0), 0], X[ring_mask & (y == 0), 1],
               label="Ring, class 0", alpha=0.5, s=15, color="cyan", marker="s")
    ax.scatter(X[ring_mask & (y == 1), 0], X[ring_mask & (y == 1), 1],
               label="Ring, class 1", alpha=0.5, s=15, color="orange", marker="s")

    ax.set_xlabel("x (feature 0)", fontsize=12)
    ax.set_ylabel("y (feature 1)", fontsize=12)
    ax.set_title("Multi-cluster dataset (10 linear + 10 ring clouds)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

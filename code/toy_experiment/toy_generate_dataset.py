"""Generate toy dataset: linear + ring data, save as .npy files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_linear_data(n_samples: int, scale: float, offset: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.random.normal(loc=[0, 0], scale=[scale, scale], size=(n_samples, 2))
    y = np.where(X[:, 0] > X[:, 1], 0, 1)
    X[:, 0] = X[:, 0] + offset
    return X, y


def generate_ring_data(n_samples: int, inner_radius: float, outer_radius: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    n_inner = n_samples // 2
    n_outer = n_samples - n_inner

    theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
    x_inner = inner_radius * np.cos(theta_inner) + 2
    y_inner = inner_radius * np.sin(theta_inner)

    theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
    x_outer = outer_radius * np.cos(theta_outer) + 2
    y_outer = outer_radius * np.sin(theta_outer)

    X = np.vstack([
        np.column_stack([x_inner, y_inner]),
        np.column_stack([x_outer, y_outer]),
    ])
    y = np.hstack([np.zeros(n_inner), np.ones(n_outer)])

    X += np.random.normal(loc=0, scale=noise, size=X.shape)

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def main():
    parser = argparse.ArgumentParser(description="Generate toy dataset (linear + ring data)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples per data type")
    parser.add_argument("--seed", type=int, default=322, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: data/ next to script)")
    parser.add_argument("--linear-scale", type=float, default=2.0, help="Scale for linear data normal distribution")
    parser.add_argument("--linear-offset", type=float, default=-6.0, help="X-axis offset for linear data")
    parser.add_argument("--inner-radius", type=float, default=1.0, help="Inner ring radius")
    parser.add_argument("--outer-radius", type=float, default=2.0, help="Outer ring radius")
    parser.add_argument("--ring-noise", type=float, default=0.3, help="Gaussian noise std for ring data")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "data"
    else:
        output_dir = Path(args.output_dir)

    X_linear, y_linear = generate_linear_data(args.n_samples, args.linear_scale, args.linear_offset)
    X_ring, y_ring = generate_ring_data(args.n_samples, args.inner_radius, args.outer_radius, args.ring_noise)

    data_type_linear = np.zeros(len(X_linear))
    data_type_ring = np.ones(len(X_ring))

    # 4 classes: linear → {0, 1}, ring → {2, 3}
    y_ring = y_ring + 2

    X_combined = np.vstack([X_linear, X_ring])
    y_combined = np.hstack([y_linear, y_ring])
    data_type_combined = np.hstack([data_type_linear, data_type_ring])

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
    print(f"  Classes: {dict(zip(*np.unique(y_combined, return_counts=True)))}")

    if not args.no_plot:
        plot_dataset(X_combined, y_combined, data_type_combined, output_dir / "dataset.png")


def plot_dataset(X: np.ndarray, y: np.ndarray, data_type: np.ndarray, save_path: Path):
    linear_mask = data_type == 0
    circle_mask = data_type == 1

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               label="Linear, class 0", alpha=0.6, s=50, color="blue", marker="o")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               label="Linear, class 1", alpha=0.6, s=50, color="red", marker="o")
    ax.scatter(X[y == 2, 0], X[y == 2, 1],
               label="Ring, class 2", alpha=0.6, s=50, color="cyan", marker="s")
    ax.scatter(X[y == 3, 0], X[y == 3, 1],
               label="Ring, class 3", alpha=0.6, s=50, color="orange", marker="s")

    ax.set_xlabel("x (feature 0)", fontsize=12)
    ax.set_ylabel("y (feature 1)", fontsize=12)
    ax.set_title("Generated dataset", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

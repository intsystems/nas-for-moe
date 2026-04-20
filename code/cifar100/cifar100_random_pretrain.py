"""
Сбор случайного датасета и pretraining суррогата на подмножестве CIFAR-100.

Этапы:
  1. Загрузка подготовленных CIFAR-100 данных из cifar100_data/.
  2. Сэмплирование N пар (random arch, random b) + реальное обучение.
  3. Сохранение наблюдений в cifar100_random_dataset/obs_*.json.
  4. Обучение суррогата (GAT_Datafeature) на собранном датасете.
  5. Сохранение весов в surr_cifar100_pretrained.pth.

Запуск:
    # 1. Подготовить данные
    python prepare_cifar100.py --output-dir ./cifar100_data --n-classes 20 --fraction 0.5

    # 2. Pretrain суррогата
    python cifar100_random_pretrain.py --device cuda:0 --data-dir ./cifar100_data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "mnist"))
sys.path.insert(0, str(SCRIPT_DIR))

# Патч OPS и CIFAR-100 evaluator
import cifar100_sgem  # noqa: E402  (делает toy_graph.OPS патч на импорте)

import toy_experiment.collect_dataset as collect_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Random dataset collection + surrogate pretraining on CIFAR-100 subset"
    )
    # --- Данные (подготовлены prepare_cifar100.py) ---
    parser.add_argument("--data-dir", type=str, default="./cifar100_data",
                        help="Директория с подготовленными данными (prepare_cifar100.py)")
    # --- Результаты ---
    parser.add_argument("--save-dir", type=str,
                        default="./runs/cifar100_random_dataset")
    parser.add_argument("--checkpoint-path", type=str,
                        default="./runs/surr_cifar100_pretrained.pth")
    # --- Бюджет ---
    parser.add_argument("--n-observations", type=int, default=200)
    parser.add_argument("--cell-train-epochs", type=int, default=30)
    parser.add_argument("--init-channels", type=int, default=16)
    # --- Суррогат ---
    parser.add_argument("--surrogate-hidden-dim", type=int, default=64)
    parser.add_argument("--surrogate-heads", type=int, default=4)
    parser.add_argument("--surrogate-dropout", type=float, default=0.3)
    parser.add_argument("--surrogate-epochs", type=int, default=400)
    parser.add_argument("--surrogate-lr", type=float, default=3e-3)
    parser.add_argument("--surrogate-patience", type=int, default=50)
    # --- Общее ---
    parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Устройство: cuda:0, cuda:1, cpu и т.д.")
    args = parser.parse_args()

    # Воспроизводимость + CIFAR-100 evaluator patch
    collect_dataset.set_seed(args.seed)
    cifar100_sgem._install_patches()

    # Данные (подготовлены prepare_cifar100.py)
    data_dir = Path(args.data_dir)
    meta = cifar100_sgem.load_cifar100_meta(data_dir)

    cifar100_sgem._NUM_CLASSES = meta["num_classes"]

    X = np.load(data_dir / "data_X.npy")
    y = np.load(data_dir / "data_y.npy")

    data = collect_dataset.prepare_data(X, y, cluster_dir=str(data_dir))
    n_clusters = data["n_clusters"]
    X_train_by_cluster = data["X_train_by_cluster"]
    y_train_by_cluster = data["y_train_by_cluster"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    cluster_centers = data["cluster_centers"]

    ss = cifar100_sgem.CIFAR100DartsSearchSpace(init_channels=args.init_channels)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] device={args.device}, M={n_clusters}, "
          f"num_classes={meta['num_classes']}, total_samples={meta['total_samples']}, "
          f"n_observations={args.n_observations}, "
          f"cell_epochs={args.cell_train_epochs}")

    # --- 1. Сбор случайных наблюдений ---
    existing = sorted(save_dir.glob("obs_*.json"))
    start_idx = 0
    if existing:
        start_idx = max(int(p.stem.split("_")[1]) for p in existing) + 1
        print(f"[collect] resume: {len(existing)} existing obs, start_idx={start_idx}")

    pbar = tqdm(range(start_idx, args.n_observations), desc="Random obs")
    for i in pbar:
        config = collect_dataset.sample_valid_config(ss)
        b = collect_dataset.sample_random_b(n_clusters)
        val_acc = cifar100_sgem.evaluate_architecture_on_subset_cifar100(
            config=config, search_space=ss, b=b,
            X_train_by_cluster=X_train_by_cluster,
            y_train_by_cluster=y_train_by_cluster,
            X_val=X_val, y_val=y_val,
            epochs=args.cell_train_epochs,
        )
        collect_dataset.save_observation(config, b, val_acc, str(save_dir), i)
        pbar.set_postfix(acc=f"{val_acc:.3f}")

    obs_paths = sorted(save_dir.glob("obs_*.json"))
    print(f"[collect] total observations: {len(obs_paths)}")

    # --- 2. Обучение суррогата ---
    from darts_searchspace import OPS_NEW
    n_features = len(OPS_NEW)
    surrogate = collect_dataset.create_surrogate(
        n_features=n_features,
        n_clusters=n_clusters,
        dropout=args.surrogate_dropout,
        hidden_dim=args.surrogate_hidden_dim,
        heads=args.surrogate_heads,
        model_type="gat",
        nodes_per_graph=ss.num_nodes_per_cell + 1,
        cluster_centers=cluster_centers,
    )

    train_loader, val_loader = collect_dataset.make_surrogate_loaders(
        obs_paths, val_fraction=0.2, batch_size=128, seed=args.seed,
    )

    print(f"[surrogate] train/val split: {len(train_loader.dataset)}/"
          f"{len(val_loader.dataset)}")

    history = collect_dataset.train_surrogate(
        surrogate, train_loader, val_loader,
        device=args.device,
        lr=args.surrogate_lr,
        epochs=args.surrogate_epochs,
        patience=args.surrogate_patience,
        verbose=True,
    )

    best_val = min(history["val"]) if history["val"] else float("nan")
    print(f"[surrogate] best val_loss={best_val:.6f}")

    # --- 3. Сохранение ---
    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surrogate.state_dict(), ckpt_path)
    print(f"[surrogate] saved to {ckpt_path}")


if __name__ == "__main__":
    main()

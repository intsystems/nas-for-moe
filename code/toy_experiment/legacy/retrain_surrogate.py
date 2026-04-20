"""Quick script to try different surrogate hyperparameters on existing observations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from scipy.stats import spearmanr

import nas_moe.surrogate
from code.toy_experiment.legacy.toy_graph import OPS
from code.toy_experiment.legacy.collect_dataset import (
    make_surrogate_loaders, train_surrogate, create_surrogate, SEED,
)

obs_dir = Path("./model_dataset_3nodes")
obs_paths = sorted(obs_dir.glob("obs_*.json"))
print(f"Total observations: {len(obs_paths)}")

n_features = len(OPS)
# Auto-detect n_clusters from cluster_centers.npy
cluster_centers = np.load(Path("./data") / "cluster_centers.npy")
n_clusters = len(cluster_centers)
print(f"n_clusters (auto-detected): {n_clusters}")
device = "cuda" if torch.cuda.is_available() else "cpu"

configs = [
    {"dropout": 0.15, "hidden_dim": 32, "heads": 4, "lr": 1e-3, "epochs": 400, "patience": 60},
    {"dropout": 0.1, "hidden_dim": 64, "heads": 4, "lr": 1e-3, "epochs": 400, "patience": 60},
    {"dropout": 0.2, "hidden_dim": 32, "heads": 2, "lr": 2e-3, "epochs": 400, "patience": 60},
    {"dropout": 0.1, "hidden_dim": 32, "heads": 4, "lr": 5e-4, "epochs": 400, "patience": 80},
    {"dropout": 0.05, "hidden_dim": 64, "heads": 4, "lr": 5e-4, "epochs": 500, "patience": 80},
    {"dropout": 0.1, "hidden_dim": 128, "heads": 4, "lr": 5e-4, "epochs": 500, "patience": 80},
]

best_r2 = -float("inf")
best_cfg_idx = -1

for i, cfg in enumerate(configs):
    print(f"\n--- Config {i}: {cfg} ---")
    train_loader, val_loader = make_surrogate_loaders(
        obs_paths, val_fraction=0.2, seed=SEED
    )
    surr = create_surrogate(
        n_features, n_clusters,
        dropout=cfg["dropout"],
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
    )
    hist = train_surrogate(
        surr, train_loader, val_loader,
        device=device, lr=cfg["lr"],
        epochs=cfg["epochs"], patience=cfg["patience"],
        verbose=False,
    )

    surr.eval()
    surr.to(device)
    true_all, pred_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = surr(batch.x, batch.edge_index, batch.batch, batch.bool_vector)
            pred_all.append(out.cpu().numpy())
            true_all.append(batch.y.cpu().numpy())
    y_true = np.concatenate(true_all).flatten()
    y_pred = np.concatenate(pred_all).flatten()

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sp, _ = spearmanr(y_true, y_pred)
    print(f"  Epochs: {len(hist['train'])}, Val R2={r2:.4f}, Sp={sp:.4f}, "
          f"MAE={mae:.4f}, RMSE={rmse:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_cfg_idx = i
        best_surr_state = {k: v.cpu().clone() for k, v in surr.state_dict().items()}

print(f"\n=== Best config: {best_cfg_idx} with R2={best_r2:.4f} ===")
print(f"  {configs[best_cfg_idx]}")

# Save best model
save_path = obs_dir / f"surrogate_k{n_clusters}_n{len(obs_paths)}_best.pth"
torch.save(best_surr_state, save_path)
print(f"Saved best model to {save_path}")

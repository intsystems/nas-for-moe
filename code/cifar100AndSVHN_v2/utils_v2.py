"""v2 helpers: per-sample CE loss surrogate target.

Differences from v1 (which used val_accuracy):
- save_observation_v2 writes 'val_loss' (and optionally keeps 'val_accuracy'
  for reference) instead of using val_accuracy as the regression target.
- read_obs_value returns 'val_loss' from an obs dict.
- compute_log_likelihood_loss implements the size-weighted MoE objective
  L(alpha, R) = sum_m |C_m| * log( sum_k r_mk * exp(-u_k) )
  where u_k is interpreted as mean per-sample CE (>= 0). Uses log-sum-exp
  for numerical stability.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def compute_log_likelihood_loss(
    r: np.ndarray,
    u_loss: np.ndarray,
    cluster_sizes: Optional[np.ndarray] = None,
) -> float:
    """Size-weighted log-likelihood for MoE with NLL-surrogate.

    L = sum_m |C_m| * log( sum_k r_mk * exp(-u_k) )

    Implemented via log-sum-exp on the matrix
        A[m, k] = log r_mk - u_k
    so that the inner term is logsumexp_k A[m, k].

    Parameters
    ----------
    r : np.ndarray [M, K]
        Routing matrix; rows assumed to sum to 1.
    u_loss : np.ndarray [K]
        Predicted per-sample CE losses (>= 0).
    cluster_sizes : np.ndarray [M] or None
        |C_m|. If None, all weights are 1.

    Returns
    -------
    float
    """
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u_loss, dtype=np.float64)
    M, K = r.shape
    log_r = np.log(np.maximum(r, 1e-30))
    A = log_r - u[None, :]  # [M, K]
    A_max = A.max(axis=1, keepdims=True)  # [M, 1]
    lse = A_max[:, 0] + np.log(np.exp(A - A_max).sum(axis=1))  # [M]
    if cluster_sizes is None:
        weights = np.ones(M, dtype=np.float64)
    else:
        weights = np.asarray(cluster_sizes, dtype=np.float64)
    return float(np.sum(weights * lse))


def save_observation_v2(
    config: dict,
    b,
    val_loss: float,
    save_dir: str,
    index: int,
    val_acc: Optional[float] = None,
) -> Path:
    """Save observation JSON with val_loss as the surrogate target.

    Layout:
        {"arch": config, "subset_b": b, "val_loss": float,
         "val_accuracy": float or null}
    """
    data = {
        "arch": config,
        "subset_b": list(b),
        "val_loss": float(val_loss),
        "val_accuracy": (None if val_acc is None else float(val_acc)),
    }
    path = Path(save_dir) / f"obs_{index}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path


def read_obs_value(obs: dict) -> float:
    """Return obs['val_loss']; raise KeyError if missing.

    v2 reads only loss; we do NOT auto-convert from val_accuracy.
    """
    if "val_loss" not in obs:
        raise KeyError(
            "v2 observation must contain 'val_loss'; got keys "
            f"{sorted(obs.keys())}"
        )
    return float(obs["val_loss"])

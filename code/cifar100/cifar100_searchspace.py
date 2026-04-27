"""Уменьшенное search space для cifar100 (4 операции).

Покрывает 4 категории, чтобы архитектурная специализация читалась глазами:
    - sep_conv_3x3   — стандартная свёртка (small kernel)
    - sep_conv_5x5   — свёртка с большим receptive field
    - max_pool_3x3   — пуллинг
    - skip_connect   — identity

Импортируется в cifar100_sgem.py и в бейзлайнах.
"""

import sys
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# darts_searchspace.py живёт в code/mnist/ — добавим в path до импорта
_MNIST_DIR = Path(__file__).resolve().parent.parent / "mnist"
if str(_MNIST_DIR) not in sys.path:
    sys.path.insert(0, str(_MNIST_DIR))

from darts_searchspace import make_op as _make_op_full  # noqa: E402

# 4 операции — выбраны так, чтобы покрыть conv-small / conv-large / pool / skip.
DARTS_OPS_SMALL = [
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "max_pool_3x3",
]

OPS_NEW_SMALL = ["input"] + DARTS_OPS_SMALL + ["add"]


def make_op(op_name: str, C: int):
    if op_name not in DARTS_OPS_SMALL:
        raise ValueError(
            f"Op '{op_name}' not in reduced search space {DARTS_OPS_SMALL}"
        )
    return _make_op_full(op_name, C)


def patch_toy_graph_ops():
    """Пропатчить toy_graph.OPS под маленькое search space.

    Должно быть вызвано до импорта pipeline-модулей (collect_dataset,
    optimize_expert_assignments, optimize_surrogate_em).
    """
    import toy_experiment.toy_graph as toy_graph

    toy_graph.OPS = OPS_NEW_SMALL
    enc = OneHotEncoder(handle_unknown="ignore")
    toy_graph.OPS_ONE_HOT = enc.fit_transform(
        np.array(OPS_NEW_SMALL).reshape(-1, 1)
    ).toarray()

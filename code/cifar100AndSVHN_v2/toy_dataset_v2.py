"""v2 PyG dataset for the surrogate.

Differences from toy_experiment/toy_dataset.py:
- Reads 'val_loss' instead of 'val_accuracy' as the regression target.
- The data attribute name `y` is unchanged (still a scalar tensor).
- Only `ArchSubsetLossDataset` is provided (it is the only one used by SGEM).
"""
from __future__ import annotations

import json

import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse

from toy_experiment.toy_graph import Graph


class ArchSubsetLossDataset(Dataset):
    """Dataset for active learning with cluster subsets, target = val_loss.

    Each JSON file format:
        {"arch": {...}, "subset_b": [1, 0, ...], "val_loss": float,
         "val_accuracy": float or null}

    One sample = (architecture graph, binary subset vector) -> scalar val_loss.
    """

    def __init__(self, observation_paths):
        super().__init__()
        self.observation_paths = list(observation_paths)
        with open(self.observation_paths[0], "r", encoding="utf-8") as f:
            obs = json.load(f)
        self.n_clusters = len(obs["subset_b"])

    def len(self):
        return len(self.observation_paths)

    def get(self, index):
        path = self.observation_paths[index]
        with open(path, "r", encoding="utf-8") as f:
            obs = json.load(f)

        graph = Graph(obs["arch"], index=index)
        adj_matrix, _ops, features = graph.get_adjacency_matrix()
        edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))

        bool_vector = torch.tensor(obs["subset_b"], dtype=torch.float).unsqueeze(0)
        val_loss = torch.tensor(obs["val_loss"], dtype=torch.float).reshape(1, 1)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=val_loss,
            bool_vector=bool_vector,
        )
        data.index = index
        return data

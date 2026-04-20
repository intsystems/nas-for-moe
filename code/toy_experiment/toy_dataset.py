import json
import torch
from pathlib import Path
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
from toy_experiment.toy_graph import Graph


class ArchClusterACCDataset(Dataset):
    """Датасет: граф архитектуры -> вектор точностей по всем кластерам."""

    def __init__(self, model_dicts_paths):
        super().__init__()
        self.model_dicts_paths = list(model_dicts_paths)

    def len(self):
        return len(self.model_dicts_paths)

    def get(self, index):
        path = self.model_dicts_paths[index]
        with open(path, "r", encoding="utf-8") as f:
            model_dict = json.load(f)

        graph = Graph(model_dict['arch'], index=index)
        adj_matrix, _ops, features = graph.get_adjacency_matrix()

        edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
        cluster_acc_vector = torch.tensor(
            model_dict['accuracies'], dtype=torch.float
        ).unsqueeze(0)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=cluster_acc_vector,
        )
        data.index = index
        return data


class ArchClusterACCDatasetBoolVector(Dataset):
    """
    Датасет: (архитектура, кластер) -> скалярная точность.
    Каждый сэмпл содержит one-hot bool_vector для индикации кластера.
    Размер датасета = число_архитектур * число_кластеров.
    """

    def __init__(self, model_dicts_paths):
        super().__init__()
        self.model_dicts_paths = list(model_dicts_paths)
        with open(self.model_dicts_paths[0], "r", encoding="utf-8") as f:
            model_dict = json.load(f)
        self.n_clusters = len(model_dict['accuracies'])

    def len(self):
        return len(self.model_dicts_paths) * self.n_clusters

    def get(self, index):
        n_cluster = index % self.n_clusters
        arch_index = index // self.n_clusters
        path = self.model_dicts_paths[arch_index]

        with open(path, "r", encoding="utf-8") as f:
            model_dict = json.load(f)

        graph = Graph(model_dict['arch'], index=index)
        adj_matrix, _ops, features = graph.get_adjacency_matrix()

        edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))

        cluster_acc = torch.tensor(
            model_dict['accuracies'][n_cluster], dtype=torch.float
        ).reshape(1, 1)

        bool_vector = torch.zeros(1, self.n_clusters)
        bool_vector[0, n_cluster] = 1.0

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=cluster_acc,
            bool_vector=bool_vector,
        )
        data.index = index
        return data


class ArchSubsetACCDataset(Dataset):
    """
    Датасет для active learning с подмножествами кластеров.

    Каждый JSON-файл имеет формат:
        {"arch": {...}, "subset_b": [1, 0, ...], "val_accuracy": float}

    Один сэмпл = (граф архитектуры, бинарный вектор подмножества) -> скалярная val accuracy.
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
        val_acc = torch.tensor(obs["val_accuracy"], dtype=torch.float).reshape(1, 1)

        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=edge_index,
            y=val_acc,
            bool_vector=bool_vector,
        )
        data.index = index
        return data

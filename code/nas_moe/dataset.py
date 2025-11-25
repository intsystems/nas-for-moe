import random
import torch
from torchvision import datasets, transforms
from .graph import Graph
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
import json
import numpy as np
from pathlib import Path

class DistortedMNIST(datasets.MNIST):

    def __init__(self, *args,
                 custom_transform=None,
                 distortions=None,
                 permutation=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.permutation = permutation
        self.distortions = distortions or []
        self.custom_transform = custom_transform
        self.input_size = 28 * 28
    
    def __len__(self):
        return super().__len__() * 2

    def __getitem__(self, index):
        if index >= super().__len__():
            img, target = super().__getitem__(index - super().__len__())
        else:
            img, target = super().__getitem__(index)

        if self.custom_transform is not None:
            img = self.custom_transform(img)

        if index >= super().__len__():
            distortion = random.choice(self.distortions)
            img = self.apply_distortion(img, distortion)

        return img, target

    def apply_distortion(self, img: torch.Tensor, distortion: str) -> torch.Tensor:
        """
        Apply the specified distortion to the image tensor.
        """
        c, h, w = img.shape
        img = img.clone()

        if distortion == 'zero_rows':
            # Choose a random row index to zero
            row = random.randrange(h)
            img[:, row, :] = 0.0

        elif distortion == 'zero_columns':
            # Choose a random column index to zero
            col = random.randrange(w)
            img[:, :, col] = 0.0

        elif distortion == 'permutation' and self.permutation is not None:
            img_flat = img.view(c, -1)
            
            # Apply permutation to each channel
            img_permuted = img_flat[:, self.permutation]
            
            # Reshape back to original shape: (C, H*W) -> (C, H, W)
            img = img_permuted.view(c, h, w)
        else:
            raise ValueError(f"Unsupported distortion: {distortion}")

        return img

class DistortedCIFAR10(datasets.CIFAR10):

    def __init__(self, *args, custom_transform=None, distortions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distortions = distortions or []
        self.custom_transform = custom_transform
        self.input_size = 32 * 32 * 3  # CIFAR-10 has 32x32 RGB images
    
    def __len__(self):
        return super().__len__() * 2

    def __getitem__(self, index):
        if index >= super().__len__():
            img, target = super().__getitem__(index - super().__len__())
        else:
            img, target = super().__getitem__(index)

        if self.custom_transform is not None:
            img = self.custom_transform(img)

        if index >= super().__len__():
            distortion = random.choice(self.distortions)
            img = self.apply_distortion(img, distortion)

        return img, target

    def apply_distortion(self, img: torch.Tensor, distortion: str) -> torch.Tensor:
        """
        Apply the specified distortion to the image tensor.
        """
        c, h, w = img.shape
        img = img.clone()

        if distortion == 'zero_rows':
            # Choose a random row index to zero
            row = random.randrange(h)
            img[:, row, :] = 0.0

        elif distortion == 'zero_columns':
            # Choose a random column index to zero
            col = random.randrange(w)
            img[:, :, col] = 0.0

        elif distortion == 'noise':
            # Add random noise
            noise = torch.randn_like(img) * 0.1
            img = torch.clamp(img + noise, 0.0, 1.0)

        else:
            raise ValueError(f"Unsupported distortion: {distortion}")

        return img



class ArchClusterACCDataset(Dataset):
    @staticmethod
    def preprocess(adj, features):
        adj = torch.tensor(adj, dtype=torch.float)
        features = torch.tensor(features, dtype=torch.float)
        return adj, features

    @staticmethod
    def process_graph(graph):
        adj, _, features = graph.get_adjacency_matrix()
        adj, features = ArchClusterACCDataset.preprocess(adj, features)
        return graph.index, adj, features

    def __init__(self, model_dicts_paths, cluster_val, labels_val):
        """
        Args:
            model_paths: список путей к JSON файлам моделей
            cluster_val: numpy array с метками кластеров для валидационных данных
            labels_val: numpy array с истинными метками для валидационных данных
        """
        self.model_dicts_paths = model_dicts_paths
        self.cluster_val = cluster_val
        self.labels_val = labels_val
        
        # Предвычисляем уникальные кластеры для эффективности
        self.unique_clusters = np.unique(cluster_val)
        
    def compute_cluster_accuracies(self, predictions):
        """
        Вычисляет точность модели на каждом кластере валидационных данных
        
        Args:
            predictions: numpy array с предсказаниями модели
            
        Returns:
            torch.Tensor: вектор точностей на кластерах
        """
        preds = np.array(predictions)
        is_correct = (preds == self.labels_val)
        
        clusters_acc_list = []
        for cluster_id in self.unique_clusters:
            cluster_indexes = (self.cluster_val == cluster_id)
            cluster_results = is_correct[cluster_indexes]
            
            if cluster_results.size > 0:
                cluster_acc = cluster_results.sum() / cluster_results.size
            else:
                cluster_acc = 0.0
                
            clusters_acc_list.append(cluster_acc)
        
        return torch.tensor(clusters_acc_list, dtype=torch.float)

    def __getitem__(self, index):
        path = self.model_dicts_paths[index]
        with path.open("r", encoding="utf-8") as f:
            model_dict = json.load(f)
        
        # Загружаем граф архитектуры
        graph = Graph(model_dict['architecture'], index=index)
        _, adj, features = self.process_graph(graph)
        
        edge_index, _ = dense_to_sparse(adj)
        
        # Вычисляем вектор точностей на кластерах
        valid_predictions = model_dict['valid_predictions']
        cluster_acc_vector = self.compute_cluster_accuracies(valid_predictions)
        cluster_acc_vector = cluster_acc_vector.unsqueeze(0)
        
        # Создаем объект Data
        data = Data(
            x=features, 
            edge_index=edge_index,
            y=cluster_acc_vector  # вектор точностей на кластерах
        )
        data.index = index
        
        return data

    def __len__(self):
        return len(self.model_dicts_paths)

    







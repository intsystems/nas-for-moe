
import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, List, Union, Tuple, Optional, Any
from sklearn.svm import SVC
import graphviz
from PIL import Image
import io


class Node(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class SkipConnection(Node):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x
    

class Dropout(Node):
    def __init__(self, input_dim: int, p: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(p)
    
    def forward(self, x):
        return self.dropout(x)
    
class ReLU(Node):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
    
    def forward(self, x):
        return torch.relu(x)


class Linear(Node):
    def __init__(self, input_dim: int, output_dim: int = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.layer = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        if x.dim() > 2:
            original_shape = x.shape
            x = x.view(-1, x.shape[1])
            out = self.layer(x)
            out = out.view(*original_shape[:-1], -1)
            return out
        return self.layer(x)


class Square(Node):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x):
        return x ** 2


class Shift(Node):
    def __init__(self, input_dim: int, shift_amount: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.shift = nn.Parameter(torch.randn(input_dim) * shift_amount)

    def forward(self, x):
        return x + self.shift


class Rbf(Node):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.centers = torch.zeros(1, input_dim)
        self.gamma = 1.
        self.linear = nn.Linear(1, input_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        distances = torch.cdist(x, self.centers)
        rbf_output = torch.exp(-self.gamma * (distances ** 2))
        output = self.linear(rbf_output)
        
        return output


class GraphCell(nn.Module):
    def __init__(self, nodes: Dict[int, Node], edges: Dict[int, List[int]], input_dim: int):
        super().__init__()
        self.nodes = nn.ModuleDict({str(k): v for k, v in nodes.items()})
        self.edges = edges
        self.input_dim = input_dim
        self.node_outputs = {}
        self.leaf_nodes = self._find_leaf_nodes()

    def _find_leaf_nodes(self) -> List[int]:
        all_nodes = set(self.edges.keys())
        used_as_input = set()
        
        for node_idx in self.edges.keys():
            input_indices = self.edges[node_idx]
            for input_idx in input_indices:
                if input_idx != -1:  # Исключаем начальный входной узел
                    used_as_input.add(input_idx)
        
        # Листовые узлы = все узлы минус те, что используются как входы
        leaf_nodes = sorted(list(all_nodes - used_as_input))
        return leaf_nodes

    def forward(self, x):
        self.node_outputs = {-1: x}
        
        for node_idx in sorted(self.edges.keys()):
            input_indices = self.edges[node_idx]
            if len(input_indices) == 1:
                node_input = self.node_outputs[input_indices[0]]
            else:
                node_input = torch.cat([self.node_outputs[i] for i in input_indices], dim=-1)
            
            node = self.nodes[str(node_idx)]
            self.node_outputs[node_idx] = node(node_input)
        
        if len(self.leaf_nodes) == 1:
            return self.node_outputs[self.leaf_nodes[0]]
        else:
            # Суммируем выходы всех листовых узлов
            leaf_outputs = [self.node_outputs[leaf_idx] for leaf_idx in self.leaf_nodes]
            return sum(leaf_outputs)


class ToySearchSpace:
    OPS = {
        'skip_connect': SkipConnection,
        'linear': Linear,
        'square': Square,
        'shift': Shift,
        'rbf': Rbf,
        'relu': ReLU,
        'dropout': Dropout,
    }

    OP_PARAMS = {
        'skip_connect': {},
        'linear': {},
        'square': {},
        'shift': {},
        'rbf': {},
        'relu': {},
        'dropout': {},
    }

    def __init__(self, input_dim: int, num_nodes_per_cell: int = 4):
        self.input_dim = input_dim
        self.num_nodes_per_cell = num_nodes_per_cell

    def create_cell_from_config(self, config: Dict[str, Any]) -> nn.Module:
        nodes = {}
        edges = {}

        for key, value in config.items():
            if 'op_' in key and 'input_' not in key:
                parts = key.split('_')
                node_idx = int(parts[1])
                
                op_name = value
                op_class = self.OPS[op_name]
                
                node = op_class(self.input_dim)
                nodes[node_idx] = node

        for key, value in config.items():
            if 'input_' in key and 'op_' not in key:
                parts = key.split('_')
                node_idx = int(parts[1])
                edges[node_idx] = value if isinstance(value, list) else [value]

        cell = GraphCell(nodes, edges, self.input_dim)
        return cell

    def load_from_json(self, json_path: str) -> nn.Module:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        return self.create_cell_from_config(config, self.input_dim)

    def create_random_config(self, num_nodes: int = None) -> Tuple[nn.Module, Dict[str, Any]]:
        if num_nodes is None:
            num_nodes = self.num_nodes_per_cell
        
        config = {}
        nodes = {}
        
        for node_idx in range(num_nodes):
            op_name = np.random.choice(list(self.OPS.keys()))
            config[f'op_{node_idx}'] = op_name
            
            if node_idx == 0:
                input_idx = [-1]
            else:
                input_idx = np.random.choice(range(node_idx), 1).tolist()
            
            config[f'input_{node_idx}'] = input_idx
            
            op_class = self.OPS[op_name]
            node = op_class(self.input_dim)
            nodes[node_idx] = node
        
        return config

def save_config_to_json(config: Dict[str, Any], output_path: str):
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)


def plot_single_cell(config: dict, title: str = "Random Cell"):
    g = graphviz.Digraph(
        node_attr=dict(style='filled', shape='rect', align='center'),
        format='png'
    )
    g.body.extend(['rankdir=LR'])
    
    # Input node
    g.node('-1', label='input', fillcolor='darkseagreen2')
    
    # Get all node indices
    node_indices = []
    for key in config.keys():
        if key.startswith('op_'):
            node_idx = int(key.split('_')[1])
            node_indices.append(node_idx)
    
    node_indices = sorted(node_indices)
    
    # Create nodes
    for node_idx in node_indices:
        g.node(str(node_idx), fillcolor='lightblue')
    
    # Create edges
    for node_idx in node_indices:
        op_name = config[f'op_{node_idx}']
        input_indices = config[f'input_{node_idx}']
        
        # Handle multiple inputs
        if not isinstance(input_indices, list):
            input_indices = [input_indices]
        
        for input_idx in input_indices:
            u = str(input_idx)
            v = str(node_idx)
            g.edge(u, v, label=op_name, fillcolor='gray')
    
    used_as_input = set()
    for node_idx in node_indices:
        input_indices = config[f'input_{node_idx}']
        
        if not isinstance(input_indices, list):
            input_indices = [input_indices]
        
        for input_idx in input_indices:
            if input_idx != -1:  # Exclude the initial input node
                used_as_input.add(input_idx)
    
    # Find leaf nodes (nodes that are not used as input to any other node)
    leaf_nodes = [node_idx for node_idx in node_indices if node_idx not in used_as_input]

    # Output node (last node in the graph)
    g.node('output', fillcolor='palegoldenrod')
    for leaf_node in leaf_nodes:
        g.edge(str(leaf_node), 'output', fillcolor='gray')
    
    g.attr(label=title)
    
    # Render to image
    image = Image.open(io.BytesIO(g.pipe()))
    image.show()

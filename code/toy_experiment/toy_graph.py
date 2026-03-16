import numpy as np
import re
import torch
from sklearn.preprocessing import OneHotEncoder
from graphviz import Digraph
from IPython.display import display

OPS = [
    'input',
    'skip_connect',
    'linear',
    'square',
    'shift',
    'rbf',
    'relu',
    'dropout',
]


encoder = OneHotEncoder(handle_unknown='ignore')
ops_array = np.array(OPS).reshape(-1, 1)
OPS_ONE_HOT = encoder.fit_transform(ops_array).toarray()


def extract_graph_structure(arch_dict):
    """
    Extract graph structure from architecture dictionary.
    Format: {'op_0': 'relu', 'input_0': [-1], 'op_1': 'shift', 'input_1': [0], ...}
    Returns list of tuples: [(node_id, op_name, input_nodes), ...]
    """
    nodes = []
    num_ops = len([k for k in arch_dict.keys() if k.startswith('op_')])

    for i in range(num_ops):
        op_key = f'op_{i}'
        input_key = f'input_{i}'
        
        if op_key in arch_dict and input_key in arch_dict:
            op = arch_dict[op_key]
            inputs = arch_dict[input_key]
            nodes.append((i, op, inputs))
    
    return nodes


class Vertex:
    def __init__(self, node_id, op, input_nodes):
        """
        node_id: unique identifier for this node
        op: operation name
        input_nodes: list of node IDs that are inputs to this node
        """
        self.node_id = node_id
        self.op = op
        self.input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
        
        # Get one-hot encoding
        if op in OPS:
            self.op_one_hot = OPS_ONE_HOT[OPS.index(op)]
        else:
            # For unknown ops, create zero vector
            self.op_one_hot = np.zeros(len(OPS))

    def __str__(self):
        return f"Node {self.node_id}: Op={self.op}, Inputs={self.input_nodes}"
    
    def __repr__(self):
        return self.__str__()


class Graph(torch.utils.data.Dataset):
    def __init__(self, model_dict, index=0):
        """
        model_dict: Dictionary with format {'op_0': 'relu', 'input_0': [-1], ...}
        """
        self.model_dict = model_dict
        self.index = index
        
        # Extract nodes from architecture
        nodes_data = extract_graph_structure(model_dict)
        
        # Create vertices
        self.vertices = []
        for node_id, op, inputs in nodes_data:
            self.vertices.append(Vertex(node_id, op, inputs))
        
        # Store number of nodes
        self.num_vertices = len(self.vertices)
        
        # print(f"Created graph with {self.num_vertices} nodes:")
        # for v in self.vertices:
        #     print(f"  {v}")

    def __len__(self):
        return self.num_vertices

    def __getitem__(self, idx):
        """Return vertex at index for Dataset compatibility"""
        return self.vertices[idx]

    def show_graph(self):
        """Visualize the computational graph"""
        adj_matrix, operations, _ = self.get_adjacency_matrix()
        graph_name = "Computational Graph"

        dot = Digraph(comment=graph_name, format="png")
        dot.attr(rankdir="TB")

        # Add input node
        dot.node("input", label="Input", shape="box", style="filled", fillcolor="lightblue")

        # Add all operation nodes
        for vertex in self.vertices:
            label = f"{{Node {vertex.node_id} | Op: {vertex.op}}}"
            dot.node(str(vertex.node_id), label=label, shape="record")

        # Add edges based on input connections
        for vertex in self.vertices:
            for input_node in vertex.input_nodes:
                if input_node == -1:
                    # Connection from input
                    dot.edge("input", str(vertex.node_id))
                else:
                    # Connection from another node
                    dot.edge(str(input_node), str(vertex.node_id))

        display(dot)

    def get_adjacency_matrix(self):
        """
        Returns:
            adj_matrix: NxN numpy array where N = num_vertices + 1 (including input node)
            operations: list of operation names
            operations_one_hot: numpy array of one-hot encoded operations
        """
        # Include input node (index -1 maps to 0)
        num_nodes = self.num_vertices + 1
        adj_matrix = np.zeros((num_nodes, num_nodes))
        
        # Input node operations (represented as 'none' or 'input')
        operations = ['input']
        operations_one_hot = [np.zeros(len(OPS))]  # or use a special encoding
        
        # Add all vertices
        for vertex in self.vertices:
            operations.append(vertex.op)
            operations_one_hot.append(vertex.op_one_hot)
        
        # Build adjacency matrix
        # Node indexing: input=-1 -> 0, node 0 -> 1, node 1 -> 2, etc.
        for vertex in self.vertices:
            target_idx = vertex.node_id + 1  # Shift by 1 to account for input node
            
            for input_node in vertex.input_nodes:
                source_idx = input_node + 1  # -1 becomes 0, 0 becomes 1, etc.
                adj_matrix[source_idx, target_idx] = 1
        
        adj_matrix = np.array(adj_matrix)
        operations_one_hot = np.array(operations_one_hot)
        
        return adj_matrix, operations, operations_one_hot

    def get_adjacency_list(self):
        """
        Alternative representation: adjacency list
        Returns dict: {node_id: [list of input node_ids]}
        """
        adj_list = {}
        for vertex in self.vertices:
            adj_list[vertex.node_id] = vertex.input_nodes
        return adj_list

    def get_features(self):
        """
        Get node feature matrix (one-hot encoded operations)
        Returns: numpy array of shape (num_vertices+1, num_ops)
        """
        features = [np.zeros(len(OPS))]  # Input node
        for vertex in self.vertices:
            features.append(vertex.op_one_hot)
        return np.array(features)


# Example usage:
if __name__ == "__main__":
    # Test with your architecture format
    test_arch = {
        'op_0': 'relu', 'input_0': [-1], 
        'op_1': 'shift', 'input_1': [0], 
        'op_2': 'shift', 'input_2': [1], 
        'op_3': 'relu', 'input_3': [0]
    }
    
    graph = Graph(test_arch)
    
    print("\n=== Adjacency Matrix ===")
    adj_matrix, ops, ops_one_hot = graph.get_adjacency_matrix()
    print(f"Shape: {adj_matrix.shape}")
    print(adj_matrix)
    
    print("\n=== Operations ===")
    print(ops)
    
    print("\n=== Adjacency List ===")
    print(graph.get_adjacency_list())
    
    print("\n=== Node Features ===")
    features = graph.get_features()
    print(f"Shape: {features.shape}")
    
    # Uncomment to visualize (requires graphviz and IPython)
    # graph.show_graph()

import torch
import numpy as np
import torch
from torch.nn.functional import kl_div
import networkx as nx

DISTANCE_METRICS = ['euclidean', 'cosine', 'kl']
GRAPH_FEATURES = ['n_degree', 'page_rank']

class GraphStorage:
    def __init__(self, max_size, choice='random', graph_feature = 'n_degree', distance_metric='euclidean'):
        self.adj_storage = [] # Fixed-size queue
        self.tensor_storage = []
        self.max_size = max_size
        self.measure = choice
        assert graph_feature in GRAPH_FEATURES
        self.graph_feature = graph_feature
        
        assert distance_metric in DISTANCE_METRICS
        self.distance_metric = distance_metric
        self.distance_matrix = torch.empty((0, 0))

        self.original_graph = None

    def add_tensor(self, adj, feat=None):
        """Adds a tensor, replacing the most redundant one if full."""

        new_tensor = self._calculate_tensor(adj, feat)

        if len(self.tensor_storage) < self.max_size:
            self.adj_storage.append(adj)
            self.tensor_storage.append(new_tensor)
            self._update_distance_matrix()
        else:
            new_distances = torch.tensor([self._distance(new_tensor, t) for t in self.tensor_storage])

            # Compute distances
            avg_distances = self.distance_matrix.mean(dim=1)
            new_tensor_avg_dist = new_distances.mean()

            # Find the most redundant tensor (smallest avg distance)
            min_dist_idx = avg_distances.argmin().item()

            # Replace if the new tensor has a higher avg distance
            if new_tensor_avg_dist > avg_distances[min_dist_idx]:
                print(f'Replaced old item with distance: {avg_distances[min_dist_idx]} to new item with distance: {new_tensor_avg_dist}')
                self.adj_storage[min_dist_idx] = adj
                self.tensor_storage[min_dist_idx] = new_tensor  # Replace tensor
                self.distance_matrix[min_dist_idx, :] = new_distances
                self.distance_matrix[:, min_dist_idx] = new_distances

    def get_random_tensor(self):
        """Returns a random tensor from storage."""
        if not self.adj_storage:
            raise ValueError("Storage is empty!")
        
        match self.measure:
            case 'random':
                choice = torch.randint(0, len(self.adj_storage), (1,)).item()
                return self.adj_storage[choice]
        raise Exception('Invalid measure')
    
    def _calculate_tensor(self, adj, features):
        adj = adj.cpu().numpy()
        NXG = nx.from_numpy_array(adj)

        # Structural information
        # Node degree
        if self.graph_feature == GRAPH_FEATURES[0]:
            degrees = [d for _, d in NXG.degree()]  
            degree_counts = np.bincount(degrees)  
            probabilities = degree_counts / sum(degree_counts)  
            return torch.tensor(probabilities, dtype=torch.float32).to(device=adj.device)
        
        # Page rank of known nodes
        elif self.graph_feature == GRAPH_FEATURES[1]:
            page_rank = list(nx.link_analysis.pagerank(NXG).values())[:self.original_graph.shape[0]]
            return torch.tensor(page_rank).to(device=adj.device)

        # Embedding information
        # Information maximisation betwee
        else:
            raise Exception('No feature selected')

    def _average_distance(self, tensor):
        """Computes the average distance of a tensor to all stored tensors."""
        if not self.tensor_storage:
            return float('inf')  # If storage is empty, return a high value
        distances = [self._distance(tensor, t) for t in self.tensor_storage]
        return sum(distances) / len(distances)

    def _update_distance_matrix(self):
        """Recomputes the full distance matrix when a new tensor is added."""
        num_tensors = len(self.tensor_storage)
        new_matrix = torch.zeros((num_tensors, num_tensors))

        for i in range(num_tensors):
            for j in range(i + 1, num_tensors):  # Compute only upper triangle
                dist = self._distance(self.tensor_storage[i], self.tensor_storage[j])
                new_matrix[i, j] = dist
                new_matrix[j, i] = dist  # Symmetric matrix

        self.distance_matrix = new_matrix

    def _distance(self, t1, t2):
        t1, t2 = self._pad_to_same_size(t1, t2)  # Ensure equal size

        """Computes distance between two tensors."""
        if self.distance_metric == "euclidean":
            return torch.norm(t1 - t2).item()
        elif self.distance_metric == "cosine":
            return 1 - torch.nn.functional.cosine_similarity(t1, t2, dim=0).mean().item()
        elif self.distance_metric == 'kl':
            return kl_div(t1, t2, reduction='mean')
        else:
            raise ValueError("Unsupported distance metric")
    
    def _pad_to_same_size(self, t1, t2):
        """Pads the smaller tensor with zeros to match the larger tensor's size."""
        size1, size2 = t1.shape[-1], t2.shape[-1]
        if size1 < size2:
            t1 = torch.cat([t1, torch.zeros(size2 - size1, device=t1.device)])
        elif size2 < size1:
            t2 = torch.cat([t2, torch.zeros(size1 - size2, device=t2.device)])
        return t1, t2
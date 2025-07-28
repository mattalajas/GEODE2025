import math
import networkx as nx
import numpy as np
from collections import deque
from tsl.datasets.prototypes import TabularDataset
from tsl.ops.imputation import to_missing_values_dataset


def add_missing_sensors(dataset: TabularDataset,
                       p_noise=0.05,
                       p_fault=0.01,
                       min_seq=1,
                       max_seq=10,
                       seed=None,
                       inplace=True,
                       masked_sensors = [],
                       connect = None,
                       spatial_shift = False, 
                       order = 0,
                       node_features = 'c_centrality',
                       mode='road'):
    if seed is None:
        seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random = np.random.default_rng(seed)

    # Compute evaluation mask
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    if masked_sensors is None:
        if spatial_shift:
            eval_mask = shift_mask(shape, feature=node_features, order=order, 
                                   adj=dataset.get_connectivity(**connect, layout='dense'))
            dataset.seed = seed
        else:
            eval_mask = sample_mask(shape,
                                p=p_fault,
                                p_noise=p_noise,
                                mode=mode,
                                adj=dataset.get_connectivity(**connect, layout='dense'))
            
            dataset.p_fault = p_fault
            dataset.p_noise = p_noise
            dataset.min_seq = min_seq
            dataset.max_seq = max_seq
            dataset.seed = seed
            dataset.random = random

        # mask = rearrange(eval_mask, "b n 1 -> b n")
        mask_sum = eval_mask.sum(0)  # n
        masked_sensors = (np.where(mask_sum > 0)[0]).tolist()
    else:
        masked_sensors = list(masked_sensors)
        eval_mask = np.zeros_like(dataset.mask)
        eval_mask[:, masked_sensors] = dataset.mask[:, masked_sensors]

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    test2 = np.sum(dataset.mask, axis=(0))
    test1 = np.sum(eval_mask, axis=(0))

    # Store evaluation mask params in dataset
    return dataset, masked_sensors

def shift_mask(shape, feature, order, adj):
    mask = np.zeros(shape).astype(bool)
    G = nx.from_numpy_array(adj)

    parts = math.ceil(adj.shape[0] / 4)

    if feature == 'c_centrality':
        # Compute closeness centrality
        closeness = nx.closeness_centrality(G)
        nonzero_c = {node: score for node, score in closeness.items() if score > 0}

        # Sort nodes by closeness centrality in descending order
        sorted_nodes = sorted(nonzero_c.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
        
    elif feature == 'degree':
        degree = dict(nx.degree(G))
        nonzero_d = {node: score for node, score in degree.items() if score > 0}

        # Sort nodes by node degree in descending order
        sorted_nodes = sorted(nonzero_d.items(), key=lambda x: x[1])
        ord_nodes = [x for x, _ in sorted_nodes]

        f_nodes = ord_nodes[parts*order:parts*(order+1)]
        f_nodes_mask = np.zeros(shape).astype(bool)
        f_nodes_mask[:, f_nodes] = True
        mask |= f_nodes_mask
    else:
        raise f"{feature} not implemented"
    
    return mask.astype('uint8')

def sample_mask(shape, p=0.002, p_noise=0., mode="random", adj=None):
    assert mode in ["random", "road", "mix", "region"], "The missing mode must be 'random' or 'road' or 'mix'."
    rand = np.random.random
    mask = np.zeros(shape).astype(bool)
    if mode == "random" or mode == "mix":
        mask = mask | (rand(mask.shape) < p)
    if mode == "road" or mode == "mix":
        road_shape = mask.shape[1]
        rand_mask = rand(road_shape) < p_noise
        road_mask = np.zeros(shape).astype(bool)
        road_mask[:, rand_mask] = True
        mask |= road_mask
    if mode == "region":
        num_vert = (rand(mask.shape[1]) < p_noise).sum().item()
        regions = region_masking(adj, num_vert)
        region_mask = np.zeros(shape).astype(bool)
        region_mask[:, regions] = True
        mask |= region_mask

    return mask.astype('uint8')

def region_masking(adj_matrix, n):
    num_nodes = adj_matrix.shape[0]
    if n >= num_nodes:
        return list(range(num_nodes))

    visited = set()
    all_nodes = set(range(num_nodes))

    while len(visited) < n:
        # Start from an unvisited random seed node
        seed = np.random.choice(list(all_nodes - visited))
        queue = deque([seed])
        visited.add(seed)

        while queue and len(visited) < n:
            current = queue.popleft()
            neighbors = np.nonzero(adj_matrix[current])[0]  # indices with edge

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor.item())
                    queue.append(neighbor)
                    if len(visited) == n:
                        break

    return list(visited)
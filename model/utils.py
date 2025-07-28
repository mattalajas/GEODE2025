import math

import networkx as nx
import numpy as np
import torch
from torch_scatter import scatter
from tsl.datasets.prototypes import TabularDataset
from tsl.metrics import numpy as numpy_metrics
from tsl.ops.imputation import to_missing_values_dataset

def closest_distances_unweighted(G, source_nodes, target_nodes):
    result = {}
    target_set = set(target_nodes)
    
    for source in source_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        distances = [lengths[t] for t in target_set if t in lengths]
        result[source] = min(distances) if distances else float('inf')
    
    return result

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    sum_of_diff_square = ((x1-x2)**2).sum(-1) + 1e-8
    return sum_of_diff_square.sqrt()

def moment_diff(sx1, sx2, k, og_batch, coarse_batch):
    """
    difference between moments
    """
    ss1 = scatter(sx1**k, og_batch, dim=0, dim_size=None, reduce='mean')
    ss2 = scatter(sx2**k, coarse_batch, dim=0, dim_size=None, reduce='mean')
    return l2diff(ss1,ss2)

def cmd(x1, x2, og_batch, coarse_batch, n_moments=2):
    """
    central moment discrepancy (cmd)
    - Zellinger, Werner et al. "Robust unsupervised domain adaptation
    for neural networks via moment alignment," arXiv preprint arXiv:1711.06114,
    2017.
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    #print("input shapes", x1.shape, x2.shape)
    mx1 = scatter(x1, og_batch, dim=0, dim_size=None, reduce='mean')
    mx2 = scatter(x2, coarse_batch, dim=0, dim_size=None, reduce='mean')
    #print("mx* shapes should be same (batch_szie, dim)", mx1.shape, mx2.shape)
    sx1 = x1 - mx1.repeat_interleave(torch.unique(og_batch, return_counts=True)[1], dim=0)
    sx2 = x2 - mx2.repeat_interleave(torch.unique(coarse_batch, return_counts=True)[1], dim=0)
    #print("sx1, sx2 should be same size as input", sx1.shape, sx2.shape)
    dm = l2diff(mx1, mx2)
    #print("dm should have shape (batch_size,)", dm.shape)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms = scms + moment_diff(sx1, sx2, i+2, og_batch, coarse_batch)
    return scms

def test_wise_eval(y_hat, y_true, mask, known_nodes, adj, mode, num_groups=4, alpha = 0.20):
    numpy_graph = nx.from_numpy_array(adj)
    k_nodes = np.array(known_nodes)
    u_nodes = np.array([i for i in range(adj.shape[0]) if i not in known_nodes])
    m_adj = (adj > 0).astype(float)
    group_size = u_nodes.shape[0] // num_groups

    # LPS
    n = adj.shape[-1]

    A_hat = m_adj
    idx = np.arange(n)
    A_hat[idx, idx] = 1
    D = np.diag(np.sum(A_hat, axis=1))

    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    I = np.eye(n)

    P = np.linalg.inv((I - (1-alpha)*A_norm))
    T = np.zeros((n, n))
    T[k_nodes, k_nodes] = 1
    ones = np.ones((n,))

    LPS = P @ T @ ones

    sorted_lps = sorted(u_nodes, key=lambda i: LPS[i])
    lps_gr = [sorted_lps[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_lps) % num_groups
    if remainder:
        lps_gr[-1].extend(sorted_lps[-remainder:])

    # CC
    closeness = nx.closeness_centrality(numpy_graph)
    closeness = {node: score for node, score in closeness.items() if score > 0}

    sorted_cls = sorted([i for i in u_nodes if i in closeness], key=lambda i: closeness[i])
    # sorted_cls = [x for x, _ in sorted_cls]
    cls_gr = [sorted_cls[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_cls) % num_groups
    if remainder:
        cls_gr[-1].extend(sorted_cls[-remainder:])

    # KHR
    khr_grouped = closest_distances_unweighted(numpy_graph, u_nodes, k_nodes.tolist())
    khr_grouped = {node: score for node, score in khr_grouped.items() if score < 1e9}
    khr_gr = [[] for _ in range(num_groups)]

    for key, pos in khr_grouped.items():
        value = pos-1
        if value < num_groups:
            khr_gr[value].append(key)
        else:
            khr_gr[num_groups-1].append(key)

    # Evaluate
    group_dict = {'LPS': lps_gr,
                'CC': cls_gr,
                'KHR': khr_gr}
    res = {f'{mode}_mae': numpy_metrics.mae(y_hat, y_true, mask),
               f'{mode}_mre': numpy_metrics.mre(y_hat, y_true, mask),
               f'{mode}_rmse': numpy_metrics.rmse(y_hat, y_true, mask)}

    for key, groups in group_dict.items():
        results = {'mae':[], 'mre':[], 'rmse':[]}
        for group in groups:
            node_mask = np.zeros_like(mask, dtype=bool)
            if len(node_mask.shape) == 4:
                node_mask[:, :, group] = True
            elif len(node_mask.shape) == 3:
                node_mask[:, group] = True
            else:
                raise 'node_mask dim only 3 or 2'
            
            masked_adj = mask * node_mask

            results['mae'].append(numpy_metrics.mae(y_hat, y_true, masked_adj))
            results['mre'].append(numpy_metrics.mre(y_hat, y_true, masked_adj))
            results['rmse'].append(numpy_metrics.rmse(y_hat, y_true, masked_adj))

        for metric, val in results.items():
            res[f'max_{metric}_{key}_{mode}'] = max(val)
            res[f'min_{metric}_{key}_{mode}'] = min(val)
    
    return res    

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
                       node_features = 'CC',
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

    if feature == 'CC':
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
        
    elif feature == 'ND':
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
    assert mode in ["random", "road", "mix"], "The missing mode must be 'random' or 'road' or 'mix'."
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
    return mask.astype('uint8')
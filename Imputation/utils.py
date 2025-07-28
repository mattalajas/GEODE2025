import math

import networkx as nx
import numpy as np
import torch
from einops import rearrange
from torch_scatter import scatter
from tsl.metrics import numpy as numpy_metrics

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

    # ND
    degrees = dict(nx.degree(numpy_graph))
    degrees = {node: score for node, score in degrees.items() if score > 0}

    sorted_nd = sorted([i for i in u_nodes if i in degrees], key=lambda i: degrees[i])
    # sorted_nd = [x for x, _ in sorted_nd]
    nd_gr = [sorted_nd[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_nd) % num_groups
    if remainder:
        nd_gr[-1].extend(sorted_nd[-remainder:])

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
                'ND': nd_gr,
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

            # nonzero_mask = masked_adj != 0  # shape: D x T x N
            # active_n = np.any(nonzero_mask, axis=(0, 1, 3))  # shape: N
            # nonzero_n_indices = np.where(active_n)[0]
            # print(nonzero_n_indices)

            results['mae'].append(numpy_metrics.mae(y_hat, y_true, masked_adj))
            results['mre'].append(numpy_metrics.mre(y_hat, y_true, masked_adj))
            results['rmse'].append(numpy_metrics.rmse(y_hat, y_true, masked_adj))

        for metric, val in results.items():
            wdp_num = 0
            wsd_num = 0
            ntotal = u_nodes.shape[0]

            for ind, a1 in enumerate(val):
                if math.isnan(a1) or a1 == 0:
                    a1 = 0

                n1 = len(groups[ind])
                wdp_num += (n1 * np.abs(a1 - res[f'{mode}_{metric}']))
                wsd_num += (n1 * (a1 - res[f'{mode}_{metric}'])**2)

            wdp = wdp_num / ntotal
            wsd = np.sqrt(wsd_num / ntotal)
            wcv = wsd/res[f'{mode}_{metric}']

            res[f'max_{metric}_{key}_{mode}'] = max(val)
            res[f'min_{metric}_{key}_{mode}'] = min(val)
            res[f'wdp_{metric}_{key}_{mode}'] = wdp
            res[f'wsd_{metric}_{key}_{mode}'] = wsd
            res[f'wcv_{metric}_{key}_{mode}'] = wcv
        
        if num_groups == 5:
            res[f'all_mae_{key}_{mode}'] = results['mae']
            res[f'all_mre_{key}_{mode}'] = results['mre']
            res[f'all_rmse_{key}_{mode}'] = results['rmse']
    
    return res    
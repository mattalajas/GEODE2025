import copy
import math

import networkx as nx
import numpy as np
import ot
import scipy
import torch
from einops import rearrange
from torch_scatter import scatter
from tsl.metrics import numpy as numpy_metrics

def kernel(X, mul_fac=2.0, n_ker=3, eps=1e-8):
    L2_distances = torch.cdist(X, X) ** 2
    bandwidth_multipliers = mul_fac ** (torch.arange(n_ker).to(X.device) - n_ker // 2)
    n_samples = L2_distances.shape[0]
    bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples + eps)

    return torch.exp(-L2_distances[None, ...] / (bandwidth * bandwidth_multipliers)[:, None, None]).sum(dim=0)

def mmd_loss(X_n, Y_n):
    mmd_losses = []

    for node in range(X_n.size(0)):
        K = kernel(torch.vstack([X_n[node], Y_n[node]]))

        X_size = X_n[node].shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        mmd_losses.append(XX - 2 * XY + YY)

    return torch.stack(mmd_losses).mean()

def mmd_loss_single(X_n, Y_n):
    K = kernel(torch.vstack([X_n, Y_n]))

    X_size = X_n.shape[0]
    XX = K[:X_size, :X_size].mean()
    XY = K[:X_size, X_size:].mean()
    YY = K[X_size:, X_size:].mean()
    return XX - 2 * XY + YY

def classical_mds_with_inf(D, d=2):
    """
    Classical MDS with support for infinite distances by computing shortest paths.
    D: (n x n) distance matrix with finite and inf values.
    d: target embedding dimension (e.g., 2 for x-y)
    Returns: (n x d) coordinate matrix
    """
    # Step 1: Replace inf by computing shortest paths (graph interpretation)
    D_filled = scipy.sparse.csgraph.floyd_warshall(D, directed=False)

    # Step 2: Classical MDS (same as before)
    n = D_filled.shape[0]
    D_squared = D_filled ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_squared @ J

    # Step 3: Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 4: Keep top d components
    L = np.diag(np.sqrt(np.maximum(eigvals[:d], 0)))
    V = eigvecs[:, :d]
    X = V @ L
    return X

def closest_distances_unweighted(G, source_nodes, target_nodes):
    result = {}
    target_set = set(target_nodes)
    
    for source in source_nodes:
        lengths = nx.single_source_shortest_path_length(G, source)
        distances = [lengths[t] for t in target_set if t in lengths]
        result[source] = min(distances) if distances else float('inf')
    
    return result

def batchwise_min_max_scale(x, eps=1e-8):
    x_min = x.view(x.size(0), -1).min(dim=1, keepdim=True).values  # [batch, 1]
    x_max = x.view(x.size(0), -1).max(dim=1, keepdim=True).values  # [batch, 1]

    x_min = x_min.view(-1, 1, 1)  # [batch, 1, 1]
    x_max = x_max.view(-1, 1, 1)  # [batch, 1, 1]

    return (x - x_min) / (x_max - x_min + eps)

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

def cmd_time(x1, x2, og_batch, coarse_batch, n_moments=2):
    # Shape T x B*N x D
    cmd_losses = []
    T, BN, D = x1.shape

    for t in range(T):
        cmd_losses.append(cmd(x1[t], x2[t], og_batch, coarse_batch, n_moments).mean())
    
    return torch.stack(cmd_losses).mean()


def degree_distribution(adj_matrix):
    """Computes the degree distribution from an adjacency matrix."""
    adj_matrix = torch.where(adj_matrix > 0, torch.ones_like(adj_matrix), torch.zeros_like(adj_matrix)).to(device=adj_matrix.device)
    degrees = adj_matrix.sum(dim=1)  # Sum rows for undirected graph
    
    max_degree = int(degrees.max().item()) # Find the highest degree

    # Count occurrences of each degree, filling missing degrees with zero counts
    degree_counts = torch.zeros(max_degree + 1, dtype=torch.long, device=adj_matrix.device)
    unique_degrees, counts = torch.unique(degrees, return_counts=True)
    unique_degrees = unique_degrees.to(dtype=torch.long, device=adj_matrix.device)
    counts = counts.to(device=adj_matrix.device)
    degree_counts[unique_degrees] = counts  # Assign counts to correct bins

    # Normalize to get probability distribution
    distribution = degree_counts / degree_counts.sum()

    return distribution

from typing import Sequence

from tsl.data.datamodule.splitters import Splitter, disjoint_months
from tsl.data.synch_mode import HORIZON, WINDOW
from tsl.utils.python_utils import ensure_list

def disjoint_year(dataset, years=None, synch_mode=WINDOW):
    idxs = np.arange(len(dataset))
    years = ensure_list(years)
    # divide indices according to window or horizon
    if synch_mode is WINDOW:
        start = 0
        end = dataset.window - 1
    elif synch_mode is HORIZON:
        start = dataset.horizon_offset
        end = dataset.horizon_offset + dataset.horizon - 1
    else:
        raise ValueError("synch_mode can only be one of "
                         f"{[WINDOW, HORIZON]}")
    # after idxs
    indices = np.asarray(dataset._indices)
    start_in_years = np.in1d(dataset.index[indices + start].year, years)
    end_in_years = np.in1d(dataset.index[indices + end].year, years)
    idxs_in_years = start_in_years & end_in_years
    after_idxs = idxs[idxs_in_years]
    # previous idxs
    min_y = dataset.index.min().year
    max_y = dataset.index.max().year

    years = np.setdiff1d(np.arange(min_y, max_y), years)
    start_in_years = np.in1d(dataset.index[indices + start].year, years)
    end_in_years = np.in1d(dataset.index[indices + end].year, years)
    idxs_in_years = start_in_years & end_in_years
    prev_idxs = idxs[idxs_in_years]
    return prev_idxs, after_idxs

class MonthYearSplitter(Splitter):
    def __init__(self, 
                 val_len: int = None,
                 gran: str = 'month',
                 test_times: Sequence = (3, 6, 9, 12)):
        
        super(MonthYearSplitter, self).__init__()
        self._val_len = val_len
        self.test_times = test_times

        assert gran in ['year', 'month']
        self.gran = gran
        
    def fit(self, dataset):
        if self.gran == 'month':
            nontest_idxs, test_idxs = disjoint_months(dataset,
                                                months=self.test_times,
                                                synch_mode=WINDOW)
        elif self.gran == 'year':
            nontest_idxs, test_idxs = disjoint_year(dataset,
                                                years=self.test_times,
                                                synch_mode=WINDOW)
        else:
            raise

        val_len = self._val_len
        if val_len < 1:
            val_len = int(val_len * len(nontest_idxs))
        val_len = val_len // len(self.test_times)

        delta = np.diff(test_idxs)
        delta_idxs = np.flatnonzero(delta > delta.min())
        end_month_idxs = test_idxs[1:][delta_idxs]
        if len(end_month_idxs) < len(self.test_times):
            end_month_idxs = np.insert(end_month_idxs, 0, test_idxs[0])

        month_val_idxs = [
            np.arange(v_idx - val_len, v_idx) - dataset.window
            for v_idx in end_month_idxs
        ]

        val_idxs = np.concatenate(month_val_idxs) % len(dataset)
        # remove overlapping indices from training set
        ovl_idxs, _ = dataset.overlapping_indices(nontest_idxs,
                                                  val_idxs,
                                                  synch_mode=HORIZON,
                                                  as_mask=True)
        train_idxs = nontest_idxs[~ovl_idxs]
        self.set_indices(train_idxs, val_idxs, test_idxs)

def month_splitter(val_len: int = None, gran = 'month', test_times: Sequence = (3, 6, 9, 12), *args, **kwargs):
    return MonthYearSplitter(test_times=test_times,
                             gran=gran,
                             val_len=val_len)

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

    sorted_cls = sorted(u_nodes, key=lambda i: closeness[i])
    # sorted_cls = [x for x, _ in sorted_cls]
    cls_gr = [sorted_cls[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_cls) % num_groups
    if remainder:
        cls_gr[-1].extend(sorted_cls[-remainder:])

    # ND
    degrees = dict(nx.degree(numpy_graph))

    sorted_nd = sorted(u_nodes, key=lambda i: degrees[i])
    # sorted_nd = [x for x, _ in sorted_nd]
    nd_gr = [sorted_nd[i*group_size : (i+1)*group_size] for i in range(num_groups)]
    remainder = len(sorted_nd) % num_groups
    if remainder:
        nd_gr[-1].extend(sorted_nd[-remainder:])

    # KHR
    khr_grouped = closest_distances_unweighted(numpy_graph, u_nodes, k_nodes)
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
            node_mask[:, :, group] = True
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
    
    return res    
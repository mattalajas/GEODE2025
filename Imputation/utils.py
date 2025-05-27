import numpy as np
import torch
import ot
import copy
import networkx as nx
import scipy
from einops import rearrange
from torch_scatter import scatter


"""
Tree Mover's Distance solver
"""
# Author: Ching-Yao Chuang <cychuang@mit.edu>
# License: MIT License
def get_neighbors(g):
    '''
    get neighbor indexes for each node

    Parameters
    ----------
    g : input torch_geometric graph


    Returns
    ----------
    adj: a dictionary that store the neighbor indexes

    '''
    adj = {}
    for i in range(len(g.edge_index[0])):
        node1 = g.edge_index[0][i].item()
        node2 = g.edge_index[1][i].item()
        if node1 in adj.keys():
            adj[node1].append(node2)
        else:
            adj[node1] = [node2]
    return adj

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

def cmd(x1, x2, og_batch, coarse_batch, n_moments=3):
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

def cmd_time(x1, x2, og_batch, coarse_batch, n_moments=3):
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
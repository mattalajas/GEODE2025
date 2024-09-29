from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
# from torch_geometric import nn as tgnn
from preprocessing import sparse_to_tuple
import scipy.sparse as sp
from torch_geometric.utils import *
import torch_scatter
import inspect
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import tqdm

from models import *

# utility functions

def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    assert name in ['sum', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name == 'max' else 0
    print(op(src, index, dim_size=dim_size))

    out = op(src, index, dim = 0, dim_size = dim_size, fill_value = fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out

def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))
    
    return out

# masking functions

def mask_edges_det(adjs_list, device):
    adj_train_l, train_edges_l, val_edges_l = [], [], []
    val_edges_false_l, test_edges_l, test_edges_false_l = [], [], []
    edges_list = []

    pbar = tqdm.tqdm(total=len(adjs_list))
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

        adj = adjs_list[i].to(device)
        # Remove diagonal elements
        adj = torch.abs(torch.eye(adj.shape[0]).to(device)*adj - adj)
        adj_all = adj + adj.T
        # Check that diag is zero:
        assert torch.diag(adj.to_dense()).sum() == 0
        
        # Return upper triangle part of array (assumes bidirectional)
        edges = torch.nonzero(adj.to_dense())

        # All edges
        edges_all = torch.nonzero(adj_all.to_dense())
        num_test = int(np.ceil(edges.shape[0]*.10))
        num_val = int(np.ceil(edges.shape[0]*.20))

        # Splits all edges to test val train
        all_edge_idx = torch.randperm(edges.shape[0])
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        train_edges = edges[~(torch.hstack([test_edge_idx, val_edge_idx]))]
        
        edges_list.append(edges)
        
        def ismember(a, b):
            rows_close = torch.all(torch.round(a - b[:, None]) == 0, dim = 1)
            return torch.any(rows_close)

        test_edges_false = torch.empty(0).to(device)
        # Get false test edges 1:1 ratio
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            coord1 = torch.Tensor([idx_i, idx_j]).to(device)
            coord2 = torch.Tensor([idx_j, idx_i]).to(device)
            if idx_i == idx_j:
                continue
            if ismember(coord1, edges_all):
                continue
            if test_edges_false.shape[0] > 0:
                if ismember(coord1, test_edges_false):
                    continue
                if ismember(coord2, test_edges_false):
                    continue
            test_edges_false = torch.cat((test_edges_false, coord1), 0)
        test_edges_false = test_edges_false.reshape(-1, 2)

        val_edges_false = torch.empty(0).to(device)
        # Get val edges 1:1 ratio
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            coord1 = torch.Tensor([idx_i, idx_j]).to(device)
            coord2 = torch.Tensor([idx_j, idx_i]).to(device)
            if idx_i == idx_j:
                continue
            if ismember(coord1, train_edges):
                continue
            if ismember(coord2, train_edges):
                continue
            if ismember(coord1, val_edges):
                continue
            if ismember(coord2, val_edges):
                continue
            if val_edges_false.shape[0] > 0:
                if ismember(coord1, val_edges_false):
                    continue
                if ismember(coord2, val_edges_false):
                    continue
            val_edges_false = torch.cat((val_edges_false, coord1), 0)
        val_edges_false = val_edges_false.reshape(-1, 2)

        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)

        train_edges_l.append(train_edges)
        val_edges_l.append(val_edges)
        val_edges_false_l.append(val_edges_false)
        test_edges_l.append(test_edges)
        test_edges_false_l.append(test_edges_false)
        pbar.update(1)
    pbar.close()

    # NOTE: these edge lists only contain single direction of edge!
    return train_edges_l, val_edges_l, val_edges_false_l, test_edges_l, test_edges_false_l

def mask_edges_prd(adjs_list, device):
    pos_edges_l , false_edges_l = [], []
    edges_list = []

    pbar = tqdm.tqdm(total=len(adjs_list))
    for i in range(0, len(adjs_list)):
        # Function to build test set with 10% positive links
        # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
        
        adj = adjs_list[i].to(device)
        # Remove diagonal elements
        adj = torch.abs(torch.eye(adj.shape[0]).to(device)*adj - adj)
        adj_all = adj + adj.T
        # Check that diag is zero:
        assert torch.diag(adj.to_dense()).sum() == 0
        
        # Return upper triangle part of array (assumes bidirectional)
        edges = torch.nonzero(adj.to_dense())

        # All edges
        edges_all = torch.nonzero(adj_all.to_dense())
        num_false = int(edges.shape[0])
        
        # Positive edges
        pos_edges_l.append(edges)
        
        def ismember(a, b):
            rows_close = torch.all(torch.round(a - b[:, None]) == 0, dim = 1)
            return torch.any(rows_close)
        
        # Retrieve negative edges
        edges_false = torch.empty(0).to(device)
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            coord1 = torch.Tensor([idx_i, idx_j]).to(device)
            coord2 = torch.Tensor([idx_j, idx_i]).to(device)
            if idx_i == idx_j:
                continue
            if ismember(coord1, edges_all):
                continue
            if edges_false.shape[0] > 0:
                if ismember(coord1, edges_false):
                    continue
                if ismember(coord2, edges_false):
                    continue
            edges_false = torch.cat((edges_false, coord1), 0)
        edges_false = edges_false.reshape(-1, 2)

        assert ~ismember(edges_false, edges_all)
        
        false_edges_l.append(edges_false)

        pbar.update(1)
    pbar.close()
    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l

def mask_edges_prd_new(adjs_list, device):
    pos_edges_l , false_edges_l = [], []
    edges_list = []
    
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    adj = adjs_list[0].to(device)
    # Remove diagonal elements
    adj = torch.abs(torch.eye(adj.shape[0]).to(device)*adj - adj)
    adj_all = adj + adj.T
    # Check that diag is zero:
    assert torch.diag(adj.to_dense()).sum() == 0

    # Return upper triangle part of array (assumes bidirectional)
    edges = torch.nonzero(adj.to_dense())

    # All edges
    edges_all = torch.nonzero(adj_all.to_dense())
    num_false = int(edges.shape[0])

    # Positive edges
    pos_edges_l.append(edges)

    def ismember(a, b):
        rows_close = torch.all(torch.round(a - b[:, None]) == 0, dim = 1)
        return torch.any(rows_close)

    # Generate negative edges
    edges_false = torch.empty(0).to(device)
    while len(edges_false) < num_false:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        coord1 = torch.Tensor([idx_i, idx_j]).to(device)
        coord2 = torch.Tensor([idx_j, idx_i]).to(device)
        if idx_i == idx_j:
            continue
        if ismember(coord1, edges_all):
            continue
        if edges_false.shape[0] > 0:
            if ismember(coord1, edges_false):
                continue
            if ismember(coord2, edges_false):
                continue
        edges_false = torch.cat((edges_false, coord1), 0)
    edges_false = edges_false.reshape(-1, 2)

    assert ~ismember(edges_false, edges_all)    
    false_edges_l.append(edges_false)
    
    pbar = tqdm.tqdm(total=len(adjs_list))
    for i in range(1, len(adjs_list)):
        # Get newly generated edges 
        cur = adjs_list[i].to(device).to_dense()
        prev = adjs_list[i-1].to(device).to_dense()
        edges_pos = torch.nonzero((cur - prev) > 0)
        num_false = int(edges_pos.shape[0])
        
        adj = adjs_list[i].to(device)
        # Remove diagonal elements
        adj = torch.abs(torch.eye(adj.shape[0]).to(device)*adj - adj)
        adj_all = adj + adj.T
        # Check that diag is zero:
        assert torch.diag(adj.to_dense()).sum() == 0
        
        # Get upper triangle part of array (assumes bidirectional)
        edges = torch.nonzero(adj.to_dense())

        edges_all = torch.nonzero(adj_all.to_dense())
        
        # Get negative edges for each newly generated edge
        edges_false = torch.empty(0).to(device)
        while len(edges_false) < num_false:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            coord1 = torch.Tensor([idx_i, idx_j]).to(device)
            coord2 = torch.Tensor([idx_j, idx_i]).to(device)
            if idx_i == idx_j:
                continue
            if ismember(coord1, edges_all):
                continue
            if edges_false.shape[0] > 0:
                if ismember(coord1, edges_false):
                    continue
                if ismember(coord2, edges_false):
                    continue
            edges_false = torch.cat((edges_false, coord1), 0)
        edges_false = edges_false.reshape(-1, 2)
        
        assert ~ismember(edges_false, edges_all)
        
        false_edges_l.append(edges_false)
        pos_edges_l.append(edges_pos)
    
        pbar.update(1)
    pbar.close()
    # NOTE: these edge lists only contain single direction of edge!
    return pos_edges_l, false_edges_l

# evaluation function

def get_roc_scores(edges_pos, edges_neg, adj_orig_dense_list, embs):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    auc_scores = []
    ap_scores = []
    
    for i in range(len(edges_pos)):
        # Predict on test set of edges
        emb = embs[i].detach().numpy()
        adj_rec = np.dot(emb, emb.T)
        adj_orig_t = adj_orig_dense_list[i].todense()
        preds = []
        pos = []
        for e in edges_pos[i]:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig_t[e[0], e[1]])
            
        preds_neg = []
        neg = []
        for e in edges_neg[i]:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig_t[e[0], e[1]])
        
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        auc_scores.append(roc_auc_score(labels_all, preds_all))
        ap_scores.append(average_precision_score(labels_all, preds_all))

    return auc_scores, ap_scores
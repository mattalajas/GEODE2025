from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
# from torch_geometric import nn as tgnn
from preprocessing import sparse_to_tuple, get_starboard_data
import scipy.sparse as sp
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import time
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.utils import *
from torch_geometric.nn import MessagePassing
import torch_scatter
import inspect
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import copy
import pickle
import tqdm

from models import *
from utils import *

seed = 123
np.random.seed(seed)

cuda_num = 5
device = torch.device('mps' if torch.backends.mps.is_available() else f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')

# loading data

# # Enron dataset
# with open('data/enron10/adj_time_list.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle)

# with open('data/enron10/adj_orig_dense_list.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle)


# # COLAB dataset
# with open('data/dblp/adj_time_list.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle)

# with open('data/dblp/adj_orig_dense_list.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle)


# Facebook dataset
# with open('GNNthesis/data/FB/adj_time_list.pickle', 'rb') as handle:
#     adj_time_list = pickle.load(handle, encoding='latin1')

# with open('GNNthesis/data/FB/adj_orig_dense_list.pickle', 'rb') as handle:
#     adj_orig_dense_list = pickle.load(handle, encoding='bytes')
    
# adj_time_list, adj_orig_dense_list = get_starboard_data('GNNthesis/data/Starboard', 5)

# Coo matrix format
with open('GNNthesis/data/Starboard/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle)

interval = -460
interval = -5
adj_time_list = adj_time_list[interval:]

adj_time_list_t = torch.stack(adj_time_list).to(device)

# masking edges

# Retrieves train_edges adjacency list
outs = mask_edges_det(adj_time_list_t, device)
train_edges_l = outs[0]

# Get negative and positive edges of adjacency list and returns a bidirectional adjacecny list
pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list_t, device)

# Get negative and positive edges that are newly generated (i has no edge but i+1 has one)
pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(adj_time_list_t, device)


# creating edge list

edge_idx_list = train_edges_l

# hyperparameters

h_dim = 32
z_dim = 16
n_layers =  1
clip = 10
learning_rate = 1e-2
seq_len = len(train_edges_l)
num_nodes = adj_time_list[seq_len-1].shape[0]
x_dim = num_nodes
eps = 1e-10
conv_type='GCN'
epochs = 50

# creating input tensors per sequence length

x_in_list = []

x_temp = torch.eye(num_nodes).to(device)
x_temp = x_temp.expand(1, num_nodes, num_nodes)
x_in = Variable(x_temp)


# building model

model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, device, conv=conv_type, bias=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# training

seq_start = 0
seq_end = seq_len - 3
tst_after = 0

for k in range(epochs):
    optimizer.zero_grad()
    start_time = time.time()
    kld_loss, nll_loss, _, _, hidden_st = model(x_in[seq_start:seq_end]
                                                , edge_idx_list[seq_start:seq_end]
                                                , adj_time_list[seq_start:seq_end])
    loss = kld_loss + nll_loss
    loss.backward()
    optimizer.step()
    
    nn.utils.clip_grad_norm(model.parameters(), clip)
    
    if k>tst_after:
        _, _, enc_means, pri_means, _ = model(x_in[seq_start:seq_end]
                                              , edge_idx_list[seq_end:seq_len]
                                              , adj_time_list[seq_end:seq_len]
                                              , hidden_st)
        
        auc_scores_prd, ap_scores_prd = get_roc_scores(pos_edges_l[seq_end:seq_len]
                                                        , false_edges_l[seq_end:seq_len]
                                                        , adj_time_list[seq_end:seq_len]
                                                        , pri_means)
        
        auc_scores_prd_new, ap_scores_prd_new = get_roc_scores(pos_edges_l_n[seq_end:seq_len]
                                                                , false_edges_l_n[seq_end:seq_len]
                                                                , adj_time_list[seq_end:seq_len]
                                                                , pri_means)
        
    
    print('epoch: ', k)
    print('kld_loss =', kld_loss.mean().item())
    print('nll_loss =', nll_loss.mean().item())
    print('loss =', loss.mean().item())
    if k>tst_after:
        print('----------------------------------')
        print('Link Prediction')
        print('link_prd_auc_mean', np.mean(np.array(auc_scores_prd)))
        print('link_prd_ap_mean', np.mean(np.array(ap_scores_prd)))
        print('----------------------------------')
        print('New Link Prediction')
        print('new_link_prd_auc_mean', np.mean(np.array(auc_scores_prd_new)))
        print('new_link_prd_ap_mean', np.mean(np.array(ap_scores_prd_new)))
        print('----------------------------------')
    print('----------------------------------')

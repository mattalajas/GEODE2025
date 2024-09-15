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
with open('GNNthesis/data/FB/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle, encoding='latin1')

with open('GNNthesis/data/FB/adj_orig_dense_list.pickle', 'rb') as handle:
    adj_orig_dense_list = pickle.load(handle, encoding='bytes')
    
# adj_time_list, adj_orig_dense_list = get_starboard_data('GNNthesis/data/Starboard', 5)

with open('GNNthesis/data/Starboard/adj_time_list.pickle', 'rb') as handle:
    adj_time_list = pickle.load(handle)

interval = -460
adj_time_list = adj_time_list[interval:]

# masking edges

outs = mask_edges_det(adj_time_list)
train_edges_l = outs[1]

pos_edges_l, false_edges_l = mask_edges_prd(adj_time_list)

pos_edges_l_n, false_edges_l_n = mask_edges_prd_new(adj_time_list, adj_orig_dense_list)


# creating edge list

edge_idx_list = []

for i in range(len(train_edges_l)):
    edge_idx_list.append(torch.tensor(np.transpose(train_edges_l[i]), dtype=torch.long))

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



# creating input tensors

x_in_list = []
for i in range(0, seq_len):
    x_temp = torch.tensor(np.eye(num_nodes).astype(np.float32))
    x_in_list.append(torch.tensor(x_temp))

x_in = Variable(torch.stack(x_in_list))



# building model

model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, conv=conv_type, bias=True)
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
        _, _, enc_means, pri_means, _ = model(x_in[seq_end:seq_len]
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

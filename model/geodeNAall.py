import random
import math

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import LayerNorm
from torch_geometric.utils import dense_to_sparse, softmax, scatter
from torch_geometric.nn.models import GCN
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.layers.graph_convs import DiffConv
from tsl.nn.models.base_model import BaseModel
from utils import closest_distances_unweighted

EPSILON = 1e-8

class GeodeNAall(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 adj,
                 gcn_layers,
                 psd_layers,
                 activation='tanh',
                 intervention_steps=2,
                 horizon=24,
                 cmd_sample_ratio=1.,
                 att_window=3,
                 k=5,
                 att_heads=8,
                 nbcd_layers=2):
        super(GeodeNAall, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.cmd_ratio = cmd_sample_ratio
        self.k = k
        self.att_heads = att_heads
        self.att_window = att_window

        self.init_emb = nn.Linear(input_size, hidden_size)

        self.layernorm0 = LayerNorm(hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        # self.layernorm3 = LayerNorm(hidden_size)

        # self.gcn1 = DiffConv(in_channels=hidden_size,
        #                     out_channels=hidden_size,
        #                     k=psd_layers,
        #                     root_weight=None,
        #                     activation=activation)
        
        self.gcn1 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=psd_layers,
                        out_channels=hidden_size,
                        norm='LayerNorm',
                        add_self_loops=None,
                        act=activation)
        # self.gcn3 = GCN(in_channels=hidden_size,
        #                 hidden_channels=hidden_size,
        #                 num_layers=psd_layers,
        #                 out_channels=hidden_size,
        #                 norm='LayerNorm',
        #                 add_self_loops=None,
        #                 act=activation)
        
        self.gcn2 = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=gcn_layers,
                            root_weight=True,
                            activation=activation)
        # self.gcn3 = DiffConv(in_channels=hidden_size,
        #                     out_channels=hidden_size,
        #                     k=gcn_layers,
        #                     root_weight=True,
        #                     activation=activation)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.adj = adj

    def forward(self,
                x,
                mask,
                known_set,
                masked_set=[],
                seened_set=[],
                sub_entry_num=0,
                edge_weight=None,
                training=False,
                reset=False,
                transform=None):
        # x: [batches steps nodes features]
        b, t, _, _ = x.size()
        device = x.device

        full_adj = torch.tensor(self.adj).to(device)
        if seened_set != []:
            o_adj = full_adj[seened_set, :]
            o_adj = o_adj[:, seened_set]
        else:
            o_adj = full_adj[known_set, :]
            o_adj = o_adj[:, known_set]

        edge_index, _ = dense_to_sparse(o_adj)
        x_fwd = self.init_emb(x)

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        output_invars = x_fwd

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        if seened_set != []:
            arrange = seened_set + masked_set

            o_adj = full_adj[arrange, :]
            o_adj = o_adj[:, arrange]

        if training:
            # inductive
            if reset:
                numpy_graph = nx.from_numpy_array(o_adj.cpu().numpy())
                target_nodes = list(range(o_adj.shape[0]))[:len(seened_set)]
                source_nodes = list(range(o_adj.shape[0]))[len(seened_set):]

                init_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)
                adj_aug, level_hops = self.get_new_adj(o_adj, self.k, n_add=sub_entry_num, init_hops=init_hops)
            else:
                adj_aug = o_adj

                numpy_graph = nx.from_numpy_array(adj_aug.cpu().numpy())
                target_nodes = list(range(adj_aug.shape[0]))[:len(known_set)]
                source_nodes = list(range(adj_aug.shape[0]))
                level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = adj_aug
        else:
            arrange = known_set + masked_set

            n_adj = full_adj[arrange, :]
            n_adj = n_adj[:, arrange]
            
            numpy_graph = nx.from_numpy_array(n_adj.cpu().numpy())
            target_nodes = list(range(n_adj.shape[0]))[:len(known_set)]
            source_nodes = list(range(n_adj.shape[0]))
            level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = n_adj

        b, t, _, d = output_invars.shape
        if seened_set != []:
            add_nodes = len(masked_set) + sub_entry_num
        else:
            add_nodes = sub_entry_num

        if add_nodes != 0:
            sub_entry = torch.zeros(b, t, add_nodes, d).to(device)

            xh_inv = torch.cat([output_invars, sub_entry], dim=2)  # b t n2 d
        else:
            xh_inv = output_invars

        # ========================================
        # Curriculum based pseudo-labelling
        # ========================================

        # Get the paritions of each index
        threshold = self.k
        grouped = {label: [] for label in list(range(self.k+1))}
        for key, value in level_hops.items():
            if value < threshold:
                if value in grouped:
                    grouped[value].append(key)
            else:
                grouped[threshold].append(key)

        ################# Curriculum learning ########################
        # Add loop here that goes at every khop
        # [batch, time, node, node]
        gcn_adj = dense_to_sparse(adj.to(torch.float32))
        
        xh_inv_2 = torch.zeros_like(xh_inv).to(device=device)

        cur_indices_tensor = torch.tensor(grouped[0], dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_inv.size(-1))
        
        xh_inv_2 = xh_inv_2.scatter(2, cur_ind_exp, xh_inv[:, :, grouped[0], :])

        for kh in range(1, self.k+1):
            # Pass if there are no k-hop reach nodes
            if grouped[kh] == []:
                continue

            # Organise the khop nodes 
            # Get the indices of vertices within k-hop reach
            rep_indices = []
            cur_indices = grouped[kh]
            for i in range(kh+1):
                rep_indices += grouped[i]

            alt_adj = adj.clone()

            if kh < self.k:
                rep_adj = alt_adj[:, rep_indices]
                rep_adj = rep_adj[rep_indices, :]

                rep_inv = xh_inv_2[:, :, rep_indices, :]
            else:
                rep_adj = alt_adj
                rep_inv = xh_inv_2

            rep_adj = dense_to_sparse(rep_adj.to(torch.float32))

            xh_inv_0 = self.gcn1(rep_inv, rep_adj[0], rep_adj[1])
            xh_inv_1 = self.layernorm1(xh_inv_0)
            # xh_inv_1 = self.gcn3(xh_inv_1, rep_adj[0], rep_adj[1])
            # xh_inv_1 = self.layernorm3(xh_inv_1)

            cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
            cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_inv_1.size(-1))

            xh_inv_2 = xh_inv_2.scatter(2, cur_ind_exp, xh_inv_1[:, :, -len(cur_indices):, :])

        # ========================================
        # Final Message Passing
        # ========================================
        xh_inv_3 = self.gcn2(xh_inv_2, gcn_adj[0], gcn_adj[1]) + xh_inv_2
        xh_inv_3 = self.layernorm2(xh_inv_3)
        # xh_inv_3 = self.gcn3(xh_inv_3, gcn_adj[0], gcn_adj[1]) + xh_inv_3
        # xh_inv_3 = self.layernorm3(xh_inv_3)

        finpreds = self.readout1(xh_inv_3)
        if not training:
            return finpreds
        
        # ========================================
        # CMD of embeddings
        # ========================================
        # Get embedding softmax
        finrecos = []

        # ========================================
        # Disentanglement module
        # ========================================
        # Predict the real nodes by propagating back using both variant and invariant features
        # Get IRM loss
        fin_irm_all = []

        return finpreds, finrecos, fin_irm_all

    def get_new_adj(self, adj, k, n_add, scale=1.0, init_hops={}):
        current_adj = adj.clone()
        n_current = current_adj.shape[0]
        prev_cur = 0

        # Get partitions
        partitions = np.random.exponential(scale, k)
        partitions = partitions / partitions.sum() * n_add
        partitions = np.round(partitions).astype(int)
        partitions[-1] += n_add - partitions.sum()
        partitions = np.sort(partitions)[::-1]

        if partitions[-1] < 0:
            partitions[0] += partitions[-1]
            partitions[-1] = 0

        levels = {i:0 for i in range(n_current+n_add)} | init_hops
        for _, part in enumerate(partitions):
            for _ in range(part):
                n = current_adj.shape[0]
                new_node_index = n

                # Initialize new (n+1)x(n+1) matrix
                expanded = torch.zeros(size=(n + 1, n + 1)).to(device=adj.device, dtype=torch.int)
                expanded[:n, :n] = current_adj

                # Select random anchor
                anchor = random.randint(prev_cur, n_current - 1)
                levels[n] = max(levels[anchor] + 1, 1)
                    
                # Connect to anchor
                expanded[anchor, new_node_index] = 1
                expanded[new_node_index, anchor] = 1

                # Optionally connect to anchor's neighbors
                neighbors = torch.nonzero(current_adj[anchor, :n_current]).squeeze(-1)
                # print(anchor, neighbors)
                for neighbor in neighbors:
                    connect_prob = np.random.rand(1)
                    if np.random.rand(1) < connect_prob and levels[n] >= levels[neighbor.item()]:
                        expanded[neighbor, new_node_index] = 1.
                        expanded[new_node_index, neighbor] = 1.

                        if levels[n] > levels[neighbor.item()]:
                            levels[n] = max(levels[neighbor.item()] + 1, 1)

                # Update current_adj
                current_adj = expanded
            prev_cur = n_current
            n_current += part

        # Create matrices for both n1 and n2
        adj_aug_n1 = torch.rand((current_adj.shape)).to(device = adj.device)  # n2, n2
        adj_aug_n1 = torch.triu(adj_aug_n1) + torch.triu(adj_aug_n1, 1).T

        # preserve original observed parts in newly-created adj
        adj_aug_n1 = adj_aug_n1.fill_diagonal_(0)
        adj_aug_n1 *= current_adj
        adj_aug_n1[:adj.shape[0], :adj.shape[0]] = adj

        if adj_aug_n1.shape[0] > adj.shape[0] + n_add:
            print('error')

        return adj_aug_n1, levels
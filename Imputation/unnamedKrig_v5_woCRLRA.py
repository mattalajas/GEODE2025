import copy
import random
import math

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# from Grin import get_dataset
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.utils import dense_to_sparse, softmax, scatter
from tsl.nn import utils
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.layers.graph_convs import DiffConv, GraphConv
from tsl.nn.models.base_model import BaseModel
from tsl.utils.casting import torch_to_numpy
from utils import closest_distances_unweighted

EPSILON = 1e-8

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''

    def __init__(self, n_hid, max_len=50, dropout=0.2):  # original max_len=240
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        texp = t[:, None].expand(-1, x.shape[1])
        temb = self.lin(self.emb(texp))
        return x + temb

class UnnamedKrigModelV5woCRLRA(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 adj,
                 exog_size,
                 gcn_layers,
                 psd_layers,
                 norm='LayerNorm',
                 activation='tanh',
                 dropout=0,
                 intervention_steps=2,
                 horizon=24,
                 mmd_sample_ratio=1.,
                 att_window=3,
                 k=5,
                 att_heads=8):
        super(UnnamedKrigModelV5woCRLRA, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.mmd_ratio = mmd_sample_ratio
        self.k = k
        self.att_heads = att_heads
        self.att_window = att_window
        input_size += exog_size

        self.time_emb = RelTemporalEncoding(hidden_size)
        self.init_emb = nn.Linear(input_size, hidden_size)

        self.layernorm0 = LayerNorm(hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)

        self.gcn1 = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=psd_layers,
                            root_weight=None,
                            activation=activation)
        
        self.gcn2 = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=gcn_layers,
                            root_weight=True,
                            activation=activation)
        
        self.squeeze1 = MLP(input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            activation=activation)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.readout2 = nn.Linear(hidden_size, output_size)
        self.readout3 = nn.Linear(hidden_size*2, output_size)


        self.adj = adj
        self.adj_n1 = None
        self.obs_neighbors = None

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
                u=None,
                transform=None):
        # x: [batches steps nodes features]
        # Unrandomised x, make sure this only has training nodes
        # x transferred here would be the imputed x
        # adj is the original 
        # mask is for nodes that need to get predicted, mask is also imputed 
        b, t, _, _ = x.size()
        x = utils.maybe_cat_exog(x, u)
        device = x.device

        full_adj = torch.tensor(self.adj).to(device)
        if seened_set != []:
            o_adj = full_adj[seened_set, :]
            o_adj = o_adj[:, seened_set]
        else:
            o_adj = full_adj[known_set, :]
            o_adj = o_adj[:, known_set]

        edge_index, _ = dense_to_sparse(o_adj)
        # Spatial encoding
        output_invars = self.init_emb(x)

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        # output_invars = self.scaled_dot_product_mhattention(x_fwd, edge_index,
        #                                                     self.att_window, self.att_heads)
        # output_invars = rearrange(output_invars, 'b t n d -> (b t) n d')
        # output_vars = rearrange(output_vars, 'b t n d -> (b t) n d')

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
                if self.adj_n1 is None:
                    adj_aug = o_adj
                else:
                    adj_aug = self.adj_n1

                numpy_graph = nx.from_numpy_array(adj_aug.cpu().numpy())
                target_nodes = list(range(adj_aug.shape[0]))[:len(known_set)]
                source_nodes = list(range(adj_aug.shape[0]))
                level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = adj_aug

            # transductive
            # unkn = [i for i in range(self.adj.shape[0]) if i not in arrange][:sub_entry_num]
            # full = arrange + unkn

            # n_adj = full_adj[full, :]
            # n_adj = n_adj[:, full]

            # adjs = [n_adj]
        else:
            arrange = known_set + masked_set

            n_adj = full_adj[arrange, :]
            n_adj = n_adj[:, arrange]
            
            numpy_graph = nx.from_numpy_array(n_adj.cpu().numpy())
            target_nodes = list(range(n_adj.shape[0]))[:len(known_set)]
            source_nodes = list(range(n_adj.shape[0]))
            level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = n_adj

        # for adj in adjs:
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

        ################# Standard Message passing ####################
        # gcn_adj = dense_to_sparse(adj.to(torch.float32))
        # t_mask = rearrange(mask, 'b t n d -> (b t) n d')

        # xh_inv_1 = self.gcn1(xh_inv, gcn_adj[0], gcn_adj[1]) * (1 - t_mask) + xh_inv
        # xh_inv_2 = self.layernorm1(xh_inv_1)

        # xh_var_1 = self.gcn1(xh_var, gcn_adj[0], gcn_adj[1]) * (1 - t_mask) + xh_var
        # xh_var_2 = self.layernorm1(xh_var_1)

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
                # if len(rep_indices) > xh_inv_2.shape[1]:
                #     breakpoint()
                # if max(rep_indices) >= xh_inv_2.shape[1]:
                #     breakpoint()

                rep_inv = xh_inv_2[:, :, rep_indices, :]
            else:
                rep_adj = alt_adj
                rep_inv = xh_inv_2

            # Concatenate t-1, t, t+1 into one vector then pass
            # rep_inv = self.expand_embs(b, rep_inv)
            # rep_var = self.expand_embs(b, rep_var)

            # rep_inv = self.squeeze2(rep_inv)
            # rep_var = self.squeeze2(rep_var)

            rep_adj = dense_to_sparse(rep_adj.to(torch.float32))

            xh_inv_0 = self.gcn1(rep_inv, rep_adj[0], rep_adj[1])
            xh_inv_1 = self.layernorm1(xh_inv_0)

            cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
            cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_inv_1.size(-1))

            xh_inv_2 = xh_inv_2.scatter(2, cur_ind_exp, xh_inv_1[:, :, -len(cur_indices):, :])

            # huh1 = torch.sum(xh_var_0, dim=(0, 2))
            # huh2 = torch.sum(xh_inv_0, dim=(0, 2))
            # hu31 = torch.sum(xh_var_1, dim=(0, 2))
            # hu41 = torch.sum(xh_inv_1, dim=(0, 2))
            # huh3 = torch.sum(xh_var_1[:, -len(cur_indices):, :], dim=(0, 2))
            # huh4 = torch.sum(xh_inv_1[:, -len(cur_indices):, :], dim=(0, 2))
            # huh5 = torch.sum(xh_var_2, dim=(0, 2))
            # huh6 = torch.sum(xh_inv_2, dim=(0, 2))
            # yes = 1

        # ========================================
        # Append the most similar embedding
        # ========================================
        # if seened_set != []:
        #     inv_sim = self.get_sim(xh_inv_2, seened_set, grouped)
        #     var_sim = self.get_sim(xh_var_2, seened_set, grouped)
        # else:
        #     inv_sim = self.get_sim(xh_inv_2, known_set, grouped)
        #     var_sim = self.get_sim(xh_var_2, known_set, grouped)

        xh_inv_3 = xh_inv_2

        # xh_inv_3 = torch.cat([xh_inv_2, inv_sim], dim=-1)
        # xh_var_3 = torch.cat([xh_var_2, var_sim], dim=-1)
        
        # xh_inv_3 = self.squeeze1(xh_inv_3)
        # xh_inv_3 = self.layernorm2(xh_inv_3)
        # xh_var_3 = self.squeeze1(xh_var_3)
        # xh_var_3 = self.layernorm2(xh_var_3)

        # xh_inv_3 = self.layernorm2(xh_inv_2 + inv_sim)
        # xh_var_3 = self.layernorm2(xh_var_2 + var_sim)

        # ========================================
        # Final Message Passing
        # ========================================
        xh_inv_3 = self.gcn2(xh_inv_3, gcn_adj[0], gcn_adj[1]) + xh_inv_3
        xh_inv_3 = self.layernorm3(xh_inv_3)

        # xh_inv_3 = rearrange(xh_inv_3, '(b t) n d -> b t n d', b=b, t=t)
        # xh_var_3 = rearrange(xh_var_3, '(b t) n d -> b t n d', b=b, t=t)

        finpreds = self.readout1(xh_inv_3)
        if not training:
            return finpreds
        
        # ========================================
        # MMD of embeddings
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
        """
        Add n_add new nodes to a subgraph adjacency matrix.
        
        Parameters:
        - subgraph_adj (np.ndarray): Initial adjacency matrix (n x n).
        - n_add (int): Number of nodes to add.
        - connect_prob (float): Probability of connecting to anchor's neighbors.

        Returns:
        - new_adj (np.ndarray): Updated adjacency matrix.
        - new_node_indices (list): Indices of added nodes.
        """
        current_adj = adj.clone()
        n_current = current_adj.shape[0]
        prev_cur = 0

        partitions = np.random.exponential(scale, k)
        partitions = partitions / partitions.sum() * n_add
        partitions = np.round(partitions).astype(int)
        partitions[-1] += n_add - partitions.sum()
        partitions = np.sort(partitions)[::-1]

        if partitions[-1] < 0:
            partitions[0] += partitions[-1]
            partitions[-1] = 0

        levels = {i:0 for i in range(n_current+n_add)} | init_hops
        for hops, part in enumerate(partitions):
            for i in range(part):
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
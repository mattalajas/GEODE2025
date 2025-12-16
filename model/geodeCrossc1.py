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
from tsl.nn.layers.graph_convs import DiffConv, GATConv
from tsl.nn import utils
from tsl.nn.blocks.encoders import TransformerLayer
from tsl.nn.layers.base import MultiHeadAttention
from tsl.nn.models.base_model import BaseModel
from utils import closest_distances_unweighted

EPSILON = 1e-8

class SpatioTemporalTransformerLayer(nn.Module):
    r"""A :class:`~tsl.nn.blocks.encoders.TransformerLayer` which attend both
    the spatial and temporal dimensions by stacking two
    :class:`~tsl.nn.layers.base.MultiHeadAttention` layers.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        ff_size (int): Units in the MLP after self attention.
        n_heads (int, optional): Number of parallel attention heads.
        causal (bool, optional): If :obj:`True`, then causally mask attention
            scores in temporal attention.
            (default: :obj:`True`)
        activation (str, optional): Activation function.
        dropout (float, optional): Dropout probability.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 ff_size=None,
                 n_heads=1,
                 causal=True,
                 activation='elu',
                 dropout=0.):
        super(SpatioTemporalTransformerLayer, self).__init__()
        self.temporal_att = MultiHeadAttention(embed_dim=hidden_size,
                                               qdim=input_size,
                                               kdim=input_size,
                                               vdim=input_size,
                                               heads=n_heads,
                                               axis='time',
                                               causal=causal)

        self.spatial_att = MultiHeadAttention(embed_dim=hidden_size,
                                              qdim=hidden_size,
                                              kdim=hidden_size,
                                              vdim=hidden_size,
                                              heads=n_heads,
                                              axis='nodes',
                                              causal=False)

        self.skip_conn = nn.Linear(input_size, hidden_size)

        self.norm1 = LayerNorm(input_size)
        self.norm2 = LayerNorm(hidden_size)

        self.mlp = nn.Sequential(LayerNorm(hidden_size),
                                 nn.Linear(hidden_size, ff_size),
                                 utils.get_layer_activation(activation)(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ff_size, hidden_size),
                                 nn.Dropout(dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        """"""
        # x: [batch, steps, nodes, features]
        x = self.skip_conn(x) + self.dropout(
            self.temporal_att(self.norm1(x))[0])
        x = x + self.dropout(
            self.spatial_att(self.norm2(x), attn_mask=mask)[0])
        x = x + self.mlp(x)
        return x

class GeodeCrossC1(BaseModel):
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
                 nbcd_layers=2,
                 dropout=0.2):
        super(GeodeCrossC1, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.cmd_ratio = cmd_sample_ratio
        self.k = k
        self.att_heads = att_heads
        self.att_window = att_window

        self.init_emb = nn.Linear(input_size, hidden_size)
        self.init_emb_tr = nn.Linear(input_size, hidden_size)
        # self.nbcds = nn.ModuleList(GeodeNBCD(hidden_size, att_window,
        #                                      att_heads, activation) for _ in range(nbcd_layers))

        self.layernorm0 = LayerNorm(hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)

        self.air_trans = SpatioTemporalTransformerLayer(input_size=hidden_size,
                                                       hidden_size=hidden_size,
                                                       ff_size=hidden_size,
                                                       n_heads=att_heads,
                                                       causal=False,
                                                       activation=activation,
                                                       dropout=dropout)
        
        self.temp_tra = TransformerLayer(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         ff_size=hidden_size,
                                         n_heads=att_heads,
                                         causal=False,
                                         axis='time',
                                         activation=activation,
                                         dropout=dropout)

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
                        act=activation,
                        dropout=dropout)
        
        self.gcn_tr = DiffConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            k=gcn_layers,
                            root_weight=True,
                            activation=activation)
        
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

        self.q_cross = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.k_cross = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.v_cross = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.out_proj = MLP(input_size=hidden_size,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            activation=activation)
        self.dist_emb = MLP(input_size=1,
                            hidden_size=hidden_size,
                            output_size=1,
                            activation=activation)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.readout2 = nn.Linear(hidden_size*2, output_size)

        self.adj = adj

    def forward(self,
                x,
                x_exog,
                mask,
                split,
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
        t_adj = full_adj[split:, :]
        t_adj = t_adj[:, split:]

        if seened_set != []:
            o_adj = full_adj[seened_set, :]
            o_adj = o_adj[:, seened_set]

            c_adj = full_adj[seened_set, split:]
        else:
            o_adj = full_adj[known_set, :]
            o_adj = o_adj[:, known_set]

            c_adj = full_adj[known_set, split:]

        # Check if nodes arent connected to anything, 
        # if so add self loops
        zero_inds = torch.where((o_adj.sum(0) + o_adj.sum(1)) == 0)[0]
        o_adj[zero_inds, zero_inds] = 1.

        edge_index, _ = dense_to_sparse(o_adj)
        x_fwd = self.init_emb(x)
        output_invars = self.air_trans(x_fwd, o_adj)

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        if seened_set != []:
            arrange = seened_set + masked_set

            o_adj = full_adj[arrange, :]
            o_adj = o_adj[:, arrange]

            c_adj = full_adj[arrange, split:]

        if training:
            # inductive
            if reset:
                numpy_graph = nx.from_numpy_array(o_adj.cpu().numpy())
                target_nodes = list(range(o_adj.shape[0]))[:len(seened_set)]
                source_nodes = list(range(o_adj.shape[0]))[len(seened_set):]

                init_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)
                adj_aug, level_hops, c_adj = self.get_new_adj(o_adj, self.k, n_add=sub_entry_num, cross_adj=c_adj, init_hops=init_hops)

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

            c_adj = full_adj[arrange, split:]
            
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
        # Traffic infusion
        # ========================================
        traf_adj = dense_to_sparse(t_adj)
        t_fwd = self.init_emb_tr(x_exog)
        t_fwd = self.temp_tra(t_fwd)
        
        tr_embs = self.gcn_tr(t_fwd, traf_adj[0], traf_adj[1])

        # TODO: Add layernorm and residuals and check if you can add distance component
        xh_inv_3 = self.scaled_dot_product_mhattention(xh_inv_2, tr_embs, c_adj, None, self.att_heads) + xh_inv_2
        # xh_var_3 = self.scaled_dot_product_mhattention(xh_var_2, tr_embs, c_adj, None, self.att_heads) + xh_var_2

        xh_inv_3 = self.layernorm3(xh_inv_3)
        # xh_var_3 = self.layernorm3(xh_var_3)

        # ========================================
        # Final Message Passing
        # ========================================
        xh_inv_4 = self.gcn2(xh_inv_3, gcn_adj[0], gcn_adj[1]) + xh_inv_3
        xh_inv_4 = self.layernorm2(xh_inv_4)
        # xh_inv_4 = self.gcn3(xh_inv_4, gcn_adj[0], gcn_adj[1]) + xh_inv_4
        # xh_inv_4 = self.layernorm3(xh_inv_4)

        finpreds = self.readout1(xh_inv_4)
        if not training:
            return finpreds

        # ========================================
        # CMD of embeddings
        # ========================================
        # Get embedding softmax
        N = xh_inv_4.shape[2]
        if self.cmd_ratio < 1.0:
            n_cmd = int(N*self.cmd_ratio)
            indx = torch.multinomial(torch.ones(N), n_cmd, replacement=False)
            indx = set(indx.tolist())
        else:
            indx = set(list(range(N)))
            n_cmd = N
        
        finrecos = []

        det_mask = torch.zeros_like(xh_inv_4).to(dtype=bool, device=device)
        det_mask[:, :, :len(known_set)] = 1
        xh_inv_4 = torch.where(det_mask, xh_inv_4.detach(), xh_inv_4) 

        for i in range(1, self.k+1):
            prev_group = []
            cur_group = []

            for j in range(i):
                prev_group.extend(grouped[j])
            for j in range(i+1):
                cur_group.extend(grouped[j])

            prev_group = list(set(prev_group) & indx)
            cur_group = list(set(cur_group) & indx)
            
            emb_com_inv = xh_inv_4[:, :, prev_group]
            emb_tru_inv = xh_inv_4[:, :, cur_group]

            emb_com_inv = rearrange(emb_com_inv, 'b t n d -> t b n d')
            emb_tru_inv = rearrange(emb_tru_inv, 'b t n d -> t b n d')

            if emb_tru_inv.numel() == 0 or emb_com_inv.numel() == 0:
                continue
            else:
                finrecos.append([emb_com_inv, emb_tru_inv])

        return finpreds, finrecos

    def get_new_adj(self, adj, k, n_add, cross_adj, scale=1.0, init_hops={}):
        current_adj = adj.clone()
        current_c_adj = cross_adj.clone()
        t_nodes = cross_adj.shape[1]

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

                # Initialize new (n+1)x(n+1) matrix
                expanded = torch.zeros(size=(n + 1, n + 1)).to(device=adj.device, dtype=torch.int)
                expanded[:n, :n] = current_adj

                # Select random anchor
                anchor = random.randint(prev_cur, n_current - 1)
                levels[n] = max(levels[anchor] + 1, 1)
                    
                # Connect to anchor
                expanded[anchor, n] = 1
                expanded[n, anchor] = 1

                # Optionally connect to anchor's neighbors
                neighbors = torch.nonzero(current_adj[anchor, :n_current]).squeeze(-1)
                # print(anchor, neighbors)
                for neighbor in neighbors:
                    connect_prob = np.random.rand(1)
                    if np.random.rand(1) < connect_prob and levels[n] >= levels[neighbor.item()]:
                        expanded[neighbor, n] = 1.
                        expanded[n, neighbor] = 1.

                        if levels[n] > levels[neighbor.item()]:
                            levels[n] = max(levels[neighbor.item()] + 1, 1)

                # Update current_adj
                current_adj = expanded
            prev_cur = n_current
            n_current += part

        # Create matrices for both n1 and n2
        adj_aug_n1 = torch.rand((current_adj.shape)).to(device = adj.device)  # n2, n2
        adj_aug_n1 = 0.9*adj_aug_n1 + 0.1
        adj_aug_n1 = torch.triu(adj_aug_n1) + torch.triu(adj_aug_n1, 1).T

        # preserve original observed parts in newly-created adj
        adj_aug_n1 = adj_aug_n1.fill_diagonal_(0)
        adj_aug_n1 *= current_adj
        adj_aug_n1[:adj.shape[0], :adj.shape[0]] = adj

        if adj_aug_n1.shape[0] > adj.shape[0] + n_add:
            print('error')
        
        # Create the new cross adjacency matrix using weighted average
        for _ in range(n_add):
            n, _ = current_c_adj.shape

            # Initialize new cross matrix
            c_expanded = torch.zeros(size=(n + 1, t_nodes)).to(device=adj.device)
            c_expanded[:n, :] = current_c_adj

            # Take weighted average of the distances 
            distances = adj_aug_n1[n, :n].unsqueeze(1)
            new_entry = distances * current_c_adj
            new_entry = new_entry.sum(0) / distances.sum()

            c_expanded[-1, :] = new_entry

            current_c_adj = c_expanded

        return adj_aug_n1, levels, current_c_adj

    def scaled_dot_product_mhattention(self, air_emb, traf_emb, cross_dist, mask, n_head):
        # Compute the dot products between Q and K, then scale by the square root of the key dimension
        _, t, na, d = air_emb.shape
        _, _, nt, _ = traf_emb.shape

        air_emb = rearrange(air_emb, 'b t n d -> (b t) n d')
        traf_emb = rearrange(traf_emb, 'b t n d -> (b t) n d')

        b = air_emb.shape[0]

        Q = self.q_cross(air_emb)
        K = self.k_cross(traf_emb)
        V = self.v_cross(traf_emb)

        assert d % n_head == 0

        Q = Q.view(b, na, n_head, d//n_head).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(b, nt, n_head, d//n_head).transpose(1, 2)
        V = V.view(b, nt, n_head, d//n_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d//n_head, dtype=torch.float32))

        # scores = scores + dists
        # test = torch.rand_like(scores).to(device=scores.device)
        # Apply mask if provided (useful for masked self-attention in transformers)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e16'))            

        # Softmax to normalize scores, producing attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Value should be aggregated using the attention weights as adjacency matrix weights
        output = torch.matmul(attention_weights, V)
        fin_output = self.out_proj(output.transpose(1, 2).contiguous().view(b, na, d))

        fin_output = rearrange(fin_output, '(b t) n d -> b t n d', t=t)
        return fin_output
    
class GraphProjector(nn.Module):
    def __init__(self, var_dim: int, embed_size):
        super(GraphProjector, self).__init__()
        scale = 0.00
        self.weight_key = nn.Parameter(scale*torch.randn(size=(embed_size, 1)))
        self.weight_query = nn.Parameter(scale*torch.randn(size=(embed_size, 1)))

    def graph_attention(self, input):
        bat, N, fea = input.shape
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        attention = query@key.transpose(2, 1)
        attention = torch.mean(attention, dim=0)
        attention = F.softmax(attention, dim=0)
        return attention
    
    def forward(self, x):
        x = x.flatten(1, 2)
        edge_weights = self.graph_attention(x)
        #print(torch.max(edge_weights, dim=0)[0])
        #exit()
        with torch.no_grad():
            # instead of masking by attention score, use the mask for the top K nodes, where
            # K is 0.8 * num_nodes
            min_attn_score = 0.9 # torch.quantile(edge_weights.flatten(), self.attn_thrs)
            mask = (edge_weights > min_attn_score)
            nconn = torch.sum(mask, dim=0)
            where_alone = torch.nonzero((nconn == 0))
            best_conn = torch.argmax(edge_weights, dim=0)
            mask[best_conn[where_alone], where_alone] = True
            edge_index = torch.nonzero(mask).transpose(0, 1)
            numeric_mask = mask.long()
            row_norm = numeric_mask*edge_weights

            row_norm = torch.sum(row_norm, dim=1)
            num_nonzero = torch.sum(numeric_mask, dim=1)
            row_norm = torch.repeat_interleave(row_norm, num_nonzero)
            
            edge_weights = edge_weights[mask] / row_norm
        #edge_index, mask = dropout_edge(edge_index, p=self.drop_edges)
        #edge_weights = edge_weights[mask]
        return edge_index, edge_weights
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
from tsl.nn.blocks.encoders import TransformerLayer
from tsl.nn.models.base_model import BaseModel
from utils import closest_distances_unweighted

EPSILON = 1e-8
ACTIVATIONS = {
    'relu': F.relu,
    'leakyrelu': F.leaky_relu,
    'elu': F.elu,
    'tanh': torch.tanh,
    'gelu': F.gelu,
    'selu': F.selu,
    'sigmoid': torch.sigmoid,
    'softplus': F.softplus
}

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''

    def __init__(self, n_hid, max_len=50):  # original max_len=240
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
    
class GeodeATT(nn.Module):
    def __init__(self,
                 hidden_size,
                 att_window,
                 att_heads,
                 activation='tanh'):
        super(GeodeATT, self).__init__()
        self.time_emb = RelTemporalEncoding(hidden_size)

        self.key = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.query = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.value = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.out_proj = MLP(input_size=hidden_size,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            activation=activation)
        self.dist_embedding = nn.Linear(1, hidden_size)

        self.layernorm = LayerNorm(hidden_size)
        self.att_window = att_window
        self.att_heads = att_heads
    
    def forward(self, a_fwd, edge_index, edge_weight=None):
        device = a_fwd.device
        srcs = edge_index[0]
        tars = edge_index[1]

        tar_nodes = a_fwd[:, :, tars]
        src_nodes = a_fwd[:, :, srcs]

        B, T, N, D = a_fwd.shape
        assert T >= 2 * self.att_window + 1
        assert D % self.att_heads == 0
        
        d_k = D // self.att_heads

        # Time encoding
        tar_nodes = rearrange(tar_nodes, 'b t e d -> t (b e) d')
        tar_nodes = self.time_emb(tar_nodes, torch.LongTensor(list(range(T))).to(device))
        tar_fwd = rearrange(tar_nodes, 't (b e) d -> t b e d', b=B, e=len(tars))

        src_nodes = rearrange(src_nodes, 'b t e d -> t (b e) d')
        src_nodes = self.time_emb(src_nodes, torch.LongTensor(list(range(T))).to(device))
        src_fwd = rearrange(src_nodes, 't (b e) d -> t b e d', b=B, e=len(srcs))

        # Distance encoding
        if edge_weight is not None:
            tar_fwd = rearrange(tar_fwd, 't b e d -> e (b t) d', b=B, e=len(srcs))
            src_fwd = rearrange(src_fwd, 't b e d -> e (b t) d', b=B, e=len(srcs))

            edge_weight = 2*edge_weight - 1 # scale to [-1, 1]
            dist_emb = self.dist_embedding(edge_weight[None, :].T)
            dist_emb = dist_emb.unsqueeze(1).repeat(1, B*T, 1)  # [E, B*T, D]

            tar_fwd = tar_fwd + dist_emb
            src_fwd = src_fwd + dist_emb

            tar_fwd = rearrange(tar_fwd, 'e (b t) d -> t b e d', b=B, t=T)
            src_fwd = rearrange(src_fwd, 'e (b t) d -> t b e d', b=B, t=T)

        # Get the Q, K, V
        q_mat = self.query(tar_fwd).view(T, B, len(tars), self.att_heads, d_k)
        k_mat = self.key(src_fwd).view(T, B, len(tars), self.att_heads, d_k) 
        v_mat = self.value(src_fwd).view(T, B, len(tars), self.att_heads, d_k)

        # Message and attention scores
        res_atts = (q_mat * k_mat).sum(dim=-1) / math.sqrt(d_k)
        res_msgs = v_mat

        padded_att = F.pad(res_atts, (0, 0, 0, 0, 0, 0, self.att_window, self.att_window))
        padded_msg = F.pad(res_msgs, (0, 0, 0, 0, 0, 0, 0, 0, self.att_window, self.att_window))
        
        context_att = []
        context_msg = []
        for offset in range(-self.att_window, self.att_window + 1):
            context_att.append(padded_att[self.att_window + offset : self.att_window + offset + T])  # [T, B, E, D]
            context_msg.append(padded_msg[self.att_window + offset : self.att_window + offset + T])  # [T, B, E, D]

        # [T, B, N*(att_window*2 + 1), D]
        res_att_og = torch.cat(context_att, dim=2)
        res_msg = torch.cat(context_msg, dim=2)

        ei_tar = tars.repeat(self.att_window*2 + 1)

        res_att = softmax(res_att_og, ei_tar, dim=2)
        res = res_msg * res_att.unsqueeze(-1)   
        res = res.view(T, B, -1, D)

        causal_hat = scatter(res, ei_tar, dim=2, dim_size=N, reduce='add')  # [N,F]
        causal_hat = rearrange(causal_hat, 't b n d -> b t n d')

        output_invars = self.layernorm(self.out_proj(causal_hat)+a_fwd)

        return output_invars

class GeodeCrossV11(BaseModel):
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
                 tra_sample_ratio=1.,
                 att_window=3,
                 k=5,
                 att_heads=8,
                 nbcd_layers=1):
        super(GeodeCrossV11, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.cmd_ratio = cmd_sample_ratio
        self.tra_ratio = tra_sample_ratio
        self.k = k
        self.att_heads = att_heads
        self.att_window = att_window
        
        self.activation = ACTIVATIONS[activation]

        self.init_emb = nn.Linear(input_size, hidden_size)
        self.init_emb_tr = nn.Linear(input_size, hidden_size)
        self.nbcds = nn.ModuleList(GeodeATT(hidden_size, att_window,
                                             att_heads, activation) for _ in range(nbcd_layers))

        self.layernorm0 = LayerNorm(hidden_size)
        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)
        
        self.gcn1 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=psd_layers,
                        out_channels=hidden_size,
                        norm='LayerNorm',
                        add_self_loops=None,
                        act=activation)

        # self.gcn_tr = DiffConv(in_channels=hidden_size,
        #                     out_channels=hidden_size,
        #                     k=gcn_layers,
        #                     root_weight=True,
        #                     activation=activation)
        self.temp_air = TransformerLayer(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         ff_size=hidden_size,
                                         n_heads=att_heads,
                                         axis='time',
                                         activation=activation)
        self.temp_tra = TransformerLayer(input_size=hidden_size,
                                         hidden_size=hidden_size,
                                         ff_size=hidden_size,
                                         n_heads=att_heads,
                                         axis='time',
                                         activation=activation)
        self.gcn_tr = nn.ModuleList(
                        GATConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            heads=att_heads,
                            edge_dim=1) for _ in range(gcn_layers))

        self.gcn_cross = nn.ModuleList(
                            GATConv(in_channels=hidden_size,
                                out_channels=hidden_size,
                                heads=att_heads,
                                edge_dim=1) for _ in range(gcn_layers))
        
        # self.gat_cross = GATConv(in_channels=hidden_size,
        #                          out_channels=hidden_size,
        #                          heads=att_heads,
        #                          edge_dim=1)
        
        # self.gcn3 = GCN(in_channels=hidden_size,
        #                 hidden_channels=hidden_size,
        #                 num_layers=psd_layers,
        #                 out_channels=hidden_size,
        #                 norm='LayerNorm',
        #                 add_self_loops=None,
        #                 act=activation)
        
        # self.gcn2 = DiffConv(in_channels=hidden_size,
        #                     out_channels=hidden_size,
        #                     k=gcn_layers,
        #                     root_weight=True,
        #                     activation=activation)
        
        self.gcn2 = nn.ModuleList(
                        GATConv(in_channels=hidden_size,
                            out_channels=hidden_size,
                            heads=att_heads,
                            edge_dim=1) for _ in range(gcn_layers))

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
        b, t, og_n, _ = x.size()
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

        edge_index, edge_weight = dense_to_sparse(o_adj)
        a_fwd = self.init_emb(x)
        a_fwd = self.temp_air(a_fwd)

        # ========================================
        # Get cross module embeddings
        # ========================================
        # Sample traffic nodes
        N_t = t_adj.shape[0]
        if self.tra_ratio < 1.0:
            n_tra = int(N_t*self.tra_ratio)
            tr_indx = torch.multinomial(torch.ones(N_t), n_tra, replacement=False).to(device)
        else:
            tr_indx = torch.arange(N_t).to(device)
            n_tra = N_t

        t_adj_sam = t_adj[tr_indx, :]
        t_adj_sam = t_adj_sam[:, tr_indx]
        x_exog = x_exog[:, :, tr_indx]

        # Get traffic embeddings
        traf_adj = dense_to_sparse(t_adj_sam)
        t_fwd = self.init_emb_tr(x_exog)
        t_fwd = self.temp_tra(t_fwd)

        for layer in self.gcn_tr:
            t_fwd_caus, _ = layer(t_fwd, traf_adj[0], traf_adj[1]) 
            t_fwd_caus = self.activation(t_fwd_caus)
            t_fwd = self.layernorm0(t_fwd_caus + t_fwd)

        # Intiialise cross embeddings
        cr_embs = torch.cat((a_fwd, t_fwd), dim=2)

        c_adj = c_adj[:, tr_indx]
        cr_edge_index, cr_edge_weight = dense_to_sparse(c_adj.T)
        cr_edge_index[0] += a_fwd.shape[2]

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        for layer in self.nbcds:
            a_fwd = layer(a_fwd, edge_index)
            cr_embs = layer(cr_embs, cr_edge_index)
        output_air = a_fwd
        output_cro = cr_embs[:, :, :a_fwd.shape[2]]

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        if seened_set != []:
            arrange = seened_set + masked_set

            o_adj = full_adj[arrange, :]
            o_adj = o_adj[:, arrange]

            c_adj = full_adj[arrange, split:]
            c_adj = c_adj[:, tr_indx]

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
            c_adj = c_adj[:, tr_indx]
            
            numpy_graph = nx.from_numpy_array(n_adj.cpu().numpy())
            target_nodes = list(range(n_adj.shape[0]))[:len(known_set)]
            source_nodes = list(range(n_adj.shape[0]))
            level_hops = closest_distances_unweighted(numpy_graph, source_nodes, target_nodes)

            adj = n_adj

        b, t, _, d = output_air.shape
        if seened_set != []:
            add_nodes = len(masked_set) + sub_entry_num
        else:
            add_nodes = sub_entry_num

        if add_nodes != 0:
            sub_entry = torch.zeros(b, t, add_nodes, d).to(device)

            xh_air = torch.cat([output_air, sub_entry], dim=2)  # b t n2 d
            xh_cro = torch.cat([output_cro, sub_entry], dim=2)
        else:
            xh_air = output_air
            xh_cro = output_cro

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

        ################# Curriculum learning #################
        # Add loop here that goes at every khop
        # [batch, time, node, node]
        gcn_adj = dense_to_sparse(adj.to(torch.float32))
        
        xh_air_2 = torch.zeros_like(xh_air).to(device=device)
        xh_cro_2 = torch.zeros_like(xh_cro).to(device=device)

        cur_indices_tensor = torch.tensor(grouped[0], dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_air.size(-1))
        
        xh_air_2 = xh_air_2.scatter(2, cur_ind_exp, xh_air[:, :, grouped[0], :])
        xh_cro_2 = xh_cro_2.scatter(2, cur_ind_exp, xh_cro[:, :, grouped[0], :])

        for kh in range(self.k, self.k+1):
            # Organise the khop nodes 
            # Get the indices of vertices within k-hop reach
            rep_indices = []
            cur_indices = []
            for i in range(kh+1):
                rep_indices += grouped[i]
            for i in range(1, kh+1):
                cur_indices += grouped[i]

            rep_air = xh_air_2
            rep_cro = xh_cro_2

            rep_adj = dense_to_sparse(adj.to(torch.float32))

            xh_air_0 = self.gcn1(rep_air, rep_adj[0], rep_adj[1])
            xh_air_1 = self.layernorm1(xh_air_0)
            # xh_air_1 = self.gcn3(xh_air_1, rep_adj[0], rep_adj[1])
            # xh_air_1 = self.layernorm3(xh_air_1)

            xh_cro_0 = self.gcn1(rep_cro, rep_adj[0], rep_adj[1])
            xh_cro_1 = self.layernorm1(xh_cro_0)
            # xh_cro_1 = self.gcn3(xh_cro_1, rep_adj[0], rep_adj[1])
            # xh_cro_1 = self.layernorm3(xh_cro_1)

            cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
            cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_air_1.size(-1))

            xh_air_2 = xh_air_2.scatter(2, cur_ind_exp, xh_air_1[:, :, -len(cur_indices):, :])
            xh_cro_2 = xh_cro_2.scatter(2, cur_ind_exp, xh_cro_1[:, :, -len(cur_indices):, :])

        # ========================================
        # Final Message Passing
        # ========================================
        for layer in self.gcn2:
            xh_air_2_tmp, _ = layer(xh_air_2, gcn_adj[0], gcn_adj[1])
            xh_air_2_tmp = self.activation(xh_air_2_tmp)
            xh_air_2 = self.layernorm3(xh_air_2_tmp + xh_air_2)
        
            xh_cro_2_tmp, _ = layer(xh_cro_2, gcn_adj[0], gcn_adj[1])
            xh_cro_2_tmp = self.activation(xh_cro_2_tmp)
            xh_cro_2 = self.layernorm3(xh_cro_2_tmp + xh_cro_2)

        # xh_air_2 = self.gcn2(xh_air_2, gcn_adj[0], gcn_adj[1]) + xh_air_2
        # xh_air_2 = self.layernorm3(xh_air_2)

        # xh_cro_2 = self.gcn2(xh_cro_2, gcn_adj[0], gcn_adj[1]) + xh_cro_2
        # xh_cro_2 = self.layernorm3(xh_cro_2)

        xh_air_3 = xh_air_2
        xh_cro_3 = xh_cro_2

        finpreds = self.readout1(xh_air_3)

        # ========================================
        # Get cosine similarity for each embedding and get mask 
        # ========================================
        N_a = xh_air_3.shape[2]
        if self.cmd_ratio < 1.0:
            n_air = int(N_a*self.cmd_ratio)
            ar_indx = torch.multinomial(torch.ones(N_a), n_air, replacement=False).to(device)
        else:
            ar_indx = torch.arange(N_a).to(device)
            n_air = N_a
        
        # det_mask = torch.zeros_like(xh_air_3).to(dtype=bool, device=device)
        # det_mask[:, :, :len(known_set)] = 1
        # xh_air_3 = torch.where(det_mask, xh_air_3.detach(), xh_air_3) 

        air_nodes = xh_air_3[:, :, ar_indx]
        air_nodes = rearrange(air_nodes, 'b t n d -> b (t n) d')
        air_nodes = F.normalize(air_nodes, dim=-1, eps=EPSILON)

        traf_nodes = xh_cro_3[:, :, ar_indx]
        traf_nodes = rearrange(traf_nodes, 'b t n d -> b (t n) d')
        traf_nodes = F.normalize(traf_nodes, dim=-1, eps=EPSILON)

        samp_mask = torch.diag(torch.ones(n_air)).to(device)
        samp_mask = samp_mask.unsqueeze(0).repeat(b*t, 1, 1)
        samp_mask = rearrange(samp_mask, '(b t) n d -> b (t n) d', b=b)

        sim_mat = torch.matmul(air_nodes, traf_nodes.transpose(-1, -2))
        # temp_mask = samp_mask.clone()
        # temp_mask[samp_mask == 0] = 1.
        # sim_mat = sim_mat * temp_mask

        finsim = [sim_mat, samp_mask]

        if not training:
            return finpreds, sim_mat

        fincross = self.readout1(xh_cro_3)
        return finpreds, fincross, finsim

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
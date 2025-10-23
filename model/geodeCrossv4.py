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
    
class GeodeNBCD(nn.Module):
    def __init__(self,
                 hidden_size,
                 att_window,
                 att_heads,
                 activation='tanh'):
        super(GeodeNBCD, self).__init__()
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
    
    def forward(self, x_fwd, edge_index, edge_weight=None):
        device = x_fwd.device
        srcs = edge_index[0]
        tars = edge_index[1]

        tar_nodes = x_fwd[:, :, tars]
        src_nodes = x_fwd[:, :, srcs]

        B, T, N, D = x_fwd.shape
        assert T >= 2 * self.att_window + 1
        assert D % self.att_heads == 0
        
        d_k = D // self.att_heads

        # Time encoding
        tar_nodes = rearrange(tar_nodes, 'b t e d -> t (b e) d')
        tar_nodes = self.time_emb(tar_nodes, torch.LongTensor(list(range(T))).to(device))
        tar_fwd = rearrange(tar_nodes, 't (b e) d -> t b e d', b=B, e=len(tars))

        src_nodes = rearrange(src_nodes, 'b t e d -> t (b e) d')
        src_nodes = self.time_emb(src_nodes, torch.LongTensor(list(range(T))).to(device))
        src_fwd = rearrange(src_nodes, 't (b e) d -> e (b t) d', b=B, e=len(srcs))

        # Distance encoding
        if edge_weight is not None:
            edge_weight = 2*edge_weight - 1 # scale to [-1, 1]
            dist_emb = self.dist_embedding(edge_weight[None, :].T)
            dist_emb = dist_emb.unsqueeze(1).repeat(1, B*T, 1)  # [E, B*T, D]
            src_fwd = src_fwd + dist_emb

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

        spu_att = softmax(-res_att_og, ei_tar, dim=2)
        spu = res_msg * spu_att.unsqueeze(-1)
        spu = spu.view(T, B, -1, D)

        causal_hat = scatter(res, ei_tar, dim=2, dim_size=N, reduce='add')  # [N,F]
        spurious_hat = scatter(spu, ei_tar, dim=2, dim_size=N, reduce='add')  # [N,F]

        causal_hat = rearrange(causal_hat, 't b n d -> b t n d')
        spurious_hat = rearrange(spurious_hat, 't b n d -> b t n d')

        output_invars = self.layernorm(self.out_proj(causal_hat)+x_fwd)
        output_vars = self.layernorm(self.out_proj(spurious_hat))

        return output_invars, output_vars

class GeodeCrossV4(BaseModel):
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
                 nbcd_layers=2):
        super(GeodeCrossV4, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.cmd_ratio = cmd_sample_ratio
        self.tra_ratio = tra_sample_ratio
        self.k = k
        self.att_heads = att_heads
        self.att_window = att_window

        self.init_emb = nn.Linear(input_size, hidden_size)
        self.init_emb_tr = nn.Linear(input_size, hidden_size)
        self.nbcds_air = GeodeNBCD(hidden_size, att_window, att_heads, activation)
        self.nbcds_tra = GeodeNBCD(hidden_size, att_window, att_heads, activation)

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
        x_fwd = self.init_emb(x)

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        output_invars_air, output_vars_air = self.nbcds_air(x_fwd, edge_index) #, edge_weight) 

        # ========================================
        # Traffic infusion
        # ========================================
        traf_adj = dense_to_sparse(t_adj)
        t_fwd = self.init_emb_tr(x_exog)
        tr_embs = self.gcn_tr(t_fwd, traf_adj[0], traf_adj[1])

        cr_embs = torch.cat((x_fwd, tr_embs), dim=2)
        cr_edge_index, cr_edge_weight = dense_to_sparse(c_adj.T)
        cr_edge_index[0] += x_fwd.shape[2]

        output_invars_tra, output_vars_tra = self.nbcds_tra(cr_embs, cr_edge_index) #, cr_edge_weight) 

        output_vars = self.layernorm0(output_vars_air + output_vars_tra[:, :, :x_fwd.shape[2]])
        output_invars = self.layernorm0(output_invars_air + output_invars_tra[:, :, :x_fwd.shape[2]])

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
            xh_var = torch.cat([output_vars, sub_entry], dim=2)
        else:
            xh_inv = output_invars
            xh_var = output_vars

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
        xh_var_2 = torch.zeros_like(xh_var).to(device=device)

        cur_indices_tensor = torch.tensor(grouped[0], dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_inv.size(-1))
        
        xh_inv_2 = xh_inv_2.scatter(2, cur_ind_exp, xh_inv[:, :, grouped[0], :])
        xh_var_2 = xh_var_2.scatter(2, cur_ind_exp, xh_var[:, :, grouped[0], :])

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
                rep_var = xh_var_2[:, :, rep_indices, :]
            else:
                rep_adj = alt_adj
                rep_inv = xh_inv_2
                rep_var = xh_var_2

            rep_adj = dense_to_sparse(rep_adj.to(torch.float32))

            xh_inv_0 = self.gcn1(rep_inv, rep_adj[0], rep_adj[1])
            xh_inv_1 = self.layernorm1(xh_inv_0)
            # xh_inv_1 = self.gcn3(xh_inv_1, rep_adj[0], rep_adj[1])
            # xh_inv_1 = self.layernorm3(xh_inv_1)

            xh_var_0 = self.gcn1(rep_var, rep_adj[0], rep_adj[1])
            xh_var_1 = self.layernorm1(xh_var_0)
            # xh_var_1 = self.gcn3(xh_var_1, rep_adj[0], rep_adj[1])
            # xh_var_1 = self.layernorm3(xh_var_1)

            cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
            cur_ind_exp = cur_indices_tensor[None, None, :, None].expand(b, t, -1, xh_inv_1.size(-1))

            xh_inv_2 = xh_inv_2.scatter(2, cur_ind_exp, xh_inv_1[:, :, -len(cur_indices):, :])
            xh_var_2 = xh_var_2.scatter(2, cur_ind_exp, xh_var_1[:, :, -len(cur_indices):, :])

        # ========================================
        # Final Message Passing
        # ========================================
        xh_inv_3 = self.gcn2(xh_inv_2, gcn_adj[0], gcn_adj[1]) + xh_inv_2
        xh_inv_3 = self.layernorm2(xh_inv_3)
        # xh_inv_4 = self.gcn3(xh_inv_4, gcn_adj[0], gcn_adj[1]) + xh_inv_4
        # xh_inv_4 = self.layernorm3(xh_inv_4)

        xh_var_3 = self.gcn2(xh_var_2, gcn_adj[0], gcn_adj[1]) + xh_var_2
        xh_var_3 = self.layernorm2(xh_var_3)
        # xh_var_4 = self.gcn3(xh_var_4, gcn_adj[0], gcn_adj[1]) + xh_var_4
        # xh_var_4 = self.layernorm3(xh_var_4)

        finpreds = self.readout1(xh_inv_3)
        if not training:
            return finpreds
        
        # ========================================
        # Disentanglement module
        # ========================================
        # Predict the real nodes by propagating back using both variant and invariant features
        # Get IRM loss
        fin_irm_all = []
        for _ in range(self.steps):
            seen_invr = xh_inv_3[:, :, :len(known_set)]
            seen_vars = xh_var_3[:, :, :len(known_set)]
            
            seen_vars_l = rearrange(seen_vars, 'b t n d -> b (t n) d', b=b, t=t)
            s_rands = torch.randperm(seen_vars_l.shape[1])
            rand_seen = seen_vars_l[:, s_rands, :].detach()

            rand_seen = rearrange(rand_seen, 'b (t n) d -> b t n d', t=t)
            fin_vars = torch.cat((seen_invr, rand_seen), dim=-1)
            fin_irm = self.readout2(fin_vars)

            fin_irm_all.append(fin_irm)

        fin_irm_all = torch.stack(fin_irm_all)

        # ========================================
        # Get cosine similarity for each embedding and get mask 
        # ========================================
        N_a = og_n
        if self.cmd_ratio < 1.0:
            n_air = int(N_a*self.cmd_ratio)
            ar_indx = torch.multinomial(torch.ones(N_a), n_air, replacement=False).to(device) + add_nodes
        else:
            ar_indx = torch.arange(N_a, add_nodes + N_a).to(device)
            n_air = N_a
        
        N_t = t_adj.shape[0]
        if self.cmd_ratio < 1.0:
            n_tra = int(N_t*self.cmd_ratio)
            tr_indx = torch.multinomial(torch.ones(N_t), n_tra, replacement=False).to(device)
        else:
            tr_indx = torch.arange(N_t).to(device)
            n_tra = N_t

        air_nodes = xh_inv_3[:, :, ar_indx]
        air_nodes = rearrange(air_nodes, 'b t n d -> (b t) n d')
        air_nodes = F.normalize(air_nodes, dim=-1, eps=EPSILON)

        traf_nodes = output_invars_tra[:, :, -t_adj.shape[0]:]
        traf_nodes = traf_nodes[:, :, tr_indx]
        traf_nodes = rearrange(traf_nodes, 'b t n d -> (b t) n d')
        traf_nodes = F.normalize(traf_nodes, dim=-1, eps=EPSILON)

        samp_mask = c_adj[ar_indx, :]
        samp_mask = samp_mask[:, tr_indx]
        samp_mask = samp_mask.unsqueeze(0).repeat(b*t, 1, 1)

        sim_mat = torch.matmul(air_nodes, traf_nodes.transpose(-1, -2))
        finsim = [sim_mat, samp_mask]

        return finpreds, fin_irm_all, finsim

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
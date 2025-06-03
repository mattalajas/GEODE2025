import copy
import random

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
from torch_geometric.utils import dense_to_sparse
from tsl.nn import utils
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.layers.graph_convs import DiffConv, GraphConv
from tsl.nn.models.base_model import BaseModel
from tsl.utils.casting import torch_to_numpy
from utils import closest_distances_unweighted

EPSILON = 1e-8

class UnnamedKrigModelV4(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 adj,
                 exog_size,
                 enc_layers,
                 gcn_layers,
                 psd_layers,
                 norm='LayerNorm',
                 activation='tanh',
                 dropout=0,
                 intervention_steps=2,
                 horizon=24,
                 mmd_sample_ratio=1.,
                 k=5,
                 att_heads=8):
        super(UnnamedKrigModelV4, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.mmd_ratio = mmd_sample_ratio
        self.k = k
        self.att_heads = att_heads
        input_size += exog_size

        self.gcn_layers_fwd = GCN(in_channels=input_size,
                                hidden_channels=hidden_size,
                                out_channels=hidden_size,
                                num_layers=enc_layers,
                                norm=norm,
                                act=activation)

        # self.gcn_layers_fwd = nn.ModuleList([GraphConv(input_size,
        #                                 hidden_size,
        #                                 root_weight=False,
        #                                 norm='mean',
        #                                 activation=activation)])
        
        # if enc_layers > 1:
        #     self.gcn_layers_fwd.extend([
        #                     GraphConv(hidden_size, 
        #                               hidden_size, 
        #                               root_weight=False, 
        #                               norm='mean', 
        #                               activation=activation)
        #                     for _ in range(gcn_layers - 1)
        #                 ])

        self.skip_con_fwd = nn.Linear(input_size, hidden_size)

        self.key_t = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.query_t = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.value_t = MLP(input_size=hidden_size,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)

        self.temporal_mult = nn.MultiheadAttention(embed_dim=hidden_size,
                                                   num_heads=self.att_heads,
                                                   dropout=dropout,
                                                   batch_first=True)

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

        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)

        self.gcn1 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=psd_layers, 
                        out_channels=hidden_size,
                        dropout=dropout,
                        norm=None,
                        add_self_loops=None,
                        act=activation)
        
        self.gcn2 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=gcn_layers, 
                        out_channels=hidden_size,
                        dropout=dropout,
                        norm=norm,
                        act=activation)
        
        self.squeeze1 = MLP(input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            output_size=hidden_size,
                            activation=activation)
        # self.squeeze2 = MLP(input_size=hidden_size*3,
        #                     hidden_size=hidden_size,
        #                     output_size=hidden_size,
        #                     activation=activation)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.readout2 = nn.Linear(hidden_size, output_size)
        self.readout3 = nn.Linear(hidden_size*2, output_size)


        self.adj = adj
        self.adj_n1 = None
        self.obs_neighbors = None

    # Experimental: Adding virtual and masking other nodes
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
            full = seened_set + masked_set

            o_adj = full_adj[full, :]
            o_adj = o_adj[:, full]
        else:
            o_adj = full_adj[known_set, :]
            o_adj = o_adj[:, known_set]

        edge_index, edge_weight = dense_to_sparse(o_adj)
        # ========================================
        # Simple GRU-GCN embedding
        # ========================================
        # Bigger encoders might exacerbate confounders 

        # Spatial encoding
        x_fwd = rearrange(x, 'b t n d -> (b t) n d')
        
        out_f = x_fwd
        out_f = self.gcn_layers_fwd(out_f, edge_index, edge_weight)
        out_f = out_f + self.skip_con_fwd(x_fwd)

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        # Query represents new matrix of n1
        out_f = rearrange(out_f, '(b t) n d -> (b n) t d', b=b)
        
        query_t = self.query_t(out_f)
        key_t = self.key_t(out_f)
        value_t = self.value_t(out_f)

        temp_f, _ = self.temporal_mult(query_t, key_t, value_t, need_weights=False)
        temp_f = rearrange(temp_f, '(b n) t d -> (b t) n d', b=b)

        # Calculate self attention between Q and K,V
        # Do this for the two Queries 
        query = self.query(temp_f)
        key = self.key(temp_f)
        value = self.value(temp_f)
        # o_adj.fill_diagonal_(1)
        output_invars, output_vars = self.scaled_dot_product_mhattention(query, 
                                                                        key, 
                                                                        value,
                                                                        mask = o_adj,
                                                                        n_head = self.att_heads)
        xh_var_3 = output_vars
        # o_adj.fill_diagonal_(0)

        # ========================================
        # Create new adjacency matrix 
        # ========================================
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
        bt, _, d = output_invars.shape

        if sub_entry_num != 0:
            sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)

            xh_inv = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
            # xh_var = torch.cat([output_vars, sub_entry], dim=1)
        else:
            xh_inv = output_invars
            # xh_var = output_vars

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
        # xh_var_2 = torch.zeros_like(xh_var).to(device=device)

        cur_indices_tensor = torch.tensor(grouped[0], dtype=torch.long, device=device)
        cur_ind_exp = cur_indices_tensor[None, :, None].expand(bt, -1, xh_inv.size(-1))
        
        xh_inv_2 = xh_inv_2.scatter(1, cur_ind_exp, xh_inv[:, grouped[0], :])
        # xh_var_2 = xh_var_2.scatter(1, cur_ind_exp, xh_var[:, grouped[0], :])

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

            # Remove the edges connecting to each unseen
            # rows_t = torch.tensor(grouped[kh], dtype=torch.long, device=device)
            # cols_t = torch.tensor(grouped[kh], dtype=torch.long, device=device)
            # rows, cols = torch.meshgrid(rows_t, cols_t, indexing='ij')
            # alt_adj[rows, cols] = 0

            if kh < self.k:
                rep_adj = alt_adj[:, rep_indices]
                rep_adj = rep_adj[rep_indices, :]
                # if len(rep_indices) > xh_inv_2.shape[1]:
                #     breakpoint()
                # if max(rep_indices) >= xh_inv_2.shape[1]:
                #     breakpoint()

                rep_inv = xh_inv_2[:, rep_indices, :].detach()
                # rep_var = xh_var_2[:, rep_indices, :].detach()
            else:
                rep_adj = alt_adj
                rep_inv = xh_inv_2.detach()
                # rep_var = xh_var_2.detach()

            # Concatenate t-1, t, t+1 into one vector then pass
            # rep_inv = self.expand_embs(b, rep_inv)
            # rep_var = self.expand_embs(b, rep_var)

            # rep_inv = self.squeeze2(rep_inv)
            # rep_var = self.squeeze2(rep_var)

            rep_adj = dense_to_sparse(rep_adj.to(torch.float32))

            xh_inv_0 = self.gcn1(rep_inv, rep_adj[0], rep_adj[1])
            xh_inv_1 = self.layernorm1(xh_inv_0)

            # xh_var_0 = self.gcn1(rep_var, rep_adj[0], rep_adj[1])
            # xh_var_1 = self.layernorm1(xh_var_0)

            cur_indices_tensor = torch.tensor(cur_indices, dtype=torch.long, device=device)
            cur_ind_exp = cur_indices_tensor[None, :, None].expand(bt, -1, xh_inv_1.size(-1))

            xh_inv_2 = xh_inv_2.scatter(1, cur_ind_exp, xh_inv_1[:, -len(cur_indices):, :])
            # xh_var_2 = xh_var_2.scatter(1, cur_ind_exp, xh_var_1[:, -len(cur_indices):, :])

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
        # else:
        inv_sim = self.get_sim(xh_inv_2, known_set, grouped)
        # var_sim = self.get_sim(xh_var_2, known_set, grouped, invs=False)

        # xh_inv_3 = xh_inv_2
        # xh_var_3 = xh_var_2

        xh_inv_3 = torch.cat([xh_inv_2, inv_sim], dim=-1)
        # xh_var_3 = xh_var_2
        # xh_var_3 = torch.cat([xh_var_2, var_sim], dim=-1)
        
        xh_inv_3 = self.squeeze1(xh_inv_3)
        xh_inv_3 = self.layernorm2(xh_inv_3)
        # xh_var_3 = self.squeeze1(xh_var_3)
        # xh_var_3 = self.layernorm2(xh_var_3)

        # xh_inv_3 = self.layernorm2(xh_inv_2 + inv_sim)
        # xh_var_3 = self.layernorm2(xh_var_2 + var_sim)

        # ========================================
        # Final Message Passing
        # ========================================
        xh_inv_3 = self.gcn2(xh_inv_3, gcn_adj[0], gcn_adj[1]) + xh_inv_3
        xh_inv_3 = self.layernorm3(xh_inv_3)

        # xh_var_3 = self.gcn2(xh_var_3, gcn_adj[0], gcn_adj[1]) + xh_var_3
        # xh_var_3 = self.layernorm3(xh_var_3)

        xh_inv_3 = rearrange(xh_inv_3, '(b t) n d -> b t n d', b=b, t=t)
        # xh_var_3 = rearrange(xh_var_3, '(b t) n d -> b t n d', b=b, t=t)

        finpreds = self.readout1(xh_inv_3)
        if not training:
            return finpreds
        
        # ========================================
        # MMD of embeddings
        # ========================================
        # Get embedding softmax
        s_batch = b
        if self.mmd_ratio < 1.0:
            s_batch = int(b*self.mmd_ratio)
            indx = torch.randperm(b, device=device)[:s_batch]

            # xh_inv_sim = rearrange(inv_sim, '(b t) n d -> b t n d', b=b, t=t)
            # xh_var_sim = rearrange(var_sim, '(b t) n d -> b t n d', b=b, t=t)
            # sml_inv = xh_inv_sim[indx]
            # sml_var = xh_var_sim[indx]

            vir_inv = xh_inv_3[indx]
            vir_var = xh_var_3[indx]
        
        finrecos = []
        for i in range(1, self.k+1):
            prev_group = []
            cur_group = []

            for j in range(i):
                prev_group.extend(grouped[j])
            for j in range(i+1):
                cur_group.extend(grouped[j])

            # emb_com_inv = sml_inv[:, :, grouped[i]]
            # emb_com_var = sml_var[:, :, grouped[i]]

            emb_com_inv = vir_inv[:, :, prev_group]
            # emb_com_var = vir_inv[:, :, prev_group]

            emb_tru_inv = vir_inv[:, :, grouped[i]]
            # emb_tru_var = vir_var[:, :, grouped[i]]

            emb_com_inv = rearrange(emb_com_inv, 'b t n d -> b (t n) d')
            # emb_com_var = rearrange(emb_com_var, 'b t n d -> b (t n) d')
            emb_tru_inv = rearrange(emb_tru_inv, 'b t n d -> b (t n) d')
            # emb_tru_var = rearrange(emb_tru_var, 'b t n d -> b (t n) d')

            emb_com_inv = emb_com_inv.detach()
            # emb_com_var = emb_com_var.detach()

            if emb_tru_inv.numel() == 0:
                continue
            else:
                finrecos.append([s_batch, emb_com_inv, emb_tru_inv])

        # ========================================
        # IRM model
        # ========================================
        # Predict the real nodes by propagating back using both variant and invariant features
        # Get IRM loss
        fin_irm_all = []
        for _ in range(self.steps):
            seen_invr = xh_inv_3[:, :, :len(known_set)]
            seen_vars = xh_var_3
            
            seen_vars_l = rearrange(seen_vars, '(b t) n d -> b (t n) d', b=b, t=t)
            s_rands = torch.randperm(seen_vars_l.shape[1])
            rand_seen = seen_vars_l[:, s_rands, :]

            rand_seen = rearrange(rand_seen, 'b (t n) d -> b t n d', t=t)
            fin_vars = torch.cat((seen_invr, rand_seen), dim=-1)
            fin_irm = self.readout3(fin_vars)
            fin_irm_all.append(fin_irm)

        fin_irm_all = torch.stack(fin_irm_all)

        return finpreds, finrecos, fin_irm_all
    
    def expand_embs(self, b, data):  
        data = rearrange(data, '(b t) n d -> b t n d', b=b)

        # Pad in the time dimension (dim=1) with zeros on both sides
        # Padding = (0, 0, 0, 0, 1, 1): pads nothing on dims/node, but 1 on time before and after
        pad = (0, 0, 0, 0, 1, 1)  # (D, N, T)
        padded = F.pad(data, pad, mode='replicate')
        
        # t-1, t, t+1 embeddings
        t_minus_1 = padded[:, :-2]  # shape: (B, T, N, D)
        t_current = padded[:, 1:-1]
        t_plus_1 = padded[:, 2:]
        
        # Concatenate on the last dimension (dim=3)
        expanded = torch.cat([t_minus_1, t_current, t_plus_1], dim=3)  # shape: (B, T, N, D*3)
        
        expanded = rearrange(expanded, 'b t n d -> (b t) n d')
        return expanded
    
    def get_sim(self, embs, knownset, grouped, invs=True, eps=1e-8):
        # B*T N D
        Bt, N, D = embs.shape
        q = embs.clone() 
        k = embs.clone().transpose(-2, -1)
        q_norm = torch.norm(q, 2, 2, True)
        k_norm = torch.norm(k, 2, 1, True)

        cos_sim = torch.bmm(q, k) / (torch.bmm(q_norm, k_norm) + eps) 
        cos_sim = (cos_sim + 1.) / 2.

        idx = torch.arange(N, device=embs.device)
        grouped_ks = []

        if invs:
            cos_sim[:, idx, idx] = 0
            cos_sim[:, :len(knownset), :len(knownset)] = 0
            cos_sim[:, len(knownset):, len(knownset):] = 0

            # for k in range(1, self.k+1):
            #     for ks in range(k, self.k+1):
            #         grouped_ks.extend(grouped[ks])

            #     if len(grouped[k]) != 0 or len(grouped_ks) != 0:
            #         rows_t = torch.tensor(grouped[k], dtype=torch.long, device=embs.device)
            #         cols_t = torch.tensor(grouped_ks, dtype=torch.long, device=embs.device)

            #         rows, cols = torch.meshgrid(rows_t, cols_t, indexing='ij')
            #         cos_sim[:, rows, cols] = 0
                
            #     grouped_ks = []
                
            cos_sim_val, cos_sim_ind = torch.max(cos_sim, dim=1)

        else:
            cos_sim[:, idx, idx] = torch.inf
            # cos_sim[:, :len(knownset), :len(knownset)] = torch.inf
            cos_sim[:, len(knownset):, len(knownset):] = torch.inf
            cos_sim_val, cos_sim_ind = torch.min(cos_sim, dim=1)

        cos_sim_ind = cos_sim_ind.unsqueeze(-1).expand(-1, -1, D).detach()
        cos_sim_val = cos_sim_val.unsqueeze(-1).expand(-1, -1, D).detach()

        sim_emb = torch.gather(embs, dim=1, index=cos_sim_ind).detach()
        # check1 = torch.min(cos_sim_ind[:, :len(knownset)])
        # if len(knownset) < cos_sim_ind.shape[1]:
        #     check2 = torch.max(cos_sim_ind[:, len(knownset):])

        return sim_emb * cos_sim_val

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
            breakpoint()

        return adj_aug_n1, levels
            
    def scaled_dot_product_mhattention(self, Q, K, V, mask, n_head):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
        B, T, D = Q.shape
        assert D % n_head == 0

        Q = Q.view(B, T, n_head, D//n_head).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, n_head, D//n_head).transpose(1, 2)
        V = V.view(B, T, n_head, D//n_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(D//n_head, dtype=torch.float32))
        # test = torch.rand_like(scores).to(device=scores.device)
        # Apply mask if provided (useful for masked self-attention in transformers)
        if mask is not None:
            # fully_masked = (mask.sum(dim=-1) == 0).unsqueeze(-1)
            scores_var = scores.masked_fill(mask == 0, float('1e16'))
            scores_invar = scores.masked_fill(mask == 0, float('-1e16'))
        
            # scores_var = scores.masked_fill(fully_masked, 0.0)
            # scores_invar = scores.masked_fill(fully_masked, 0.0)             
        else:
            scores_var = scores.clone()
            scores_invar = scores.clone()      

        # Softmax to normalize scores, producing attention weights
        attention_weights_var = F.softmax(-scores_var, dim=-1)
        attention_weights_invar = F.softmax(scores_invar, dim=-1)

        # Value should be aggregated using the attention weights as adjacency matrix weights
        output_invar = torch.matmul(attention_weights_invar, V)
        output_var = torch.matmul(attention_weights_var, V)

        output_invar = self.out_proj(output_invar.transpose(1, 2).contiguous().view(B, T, D))
        output_var = self.out_proj(output_var.transpose(1, 2).contiguous().view(B, T, D))

        return output_invar, output_var
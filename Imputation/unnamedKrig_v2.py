import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
# from Grin import get_dataset
from torch import Tensor, nn
from torch.nn import LayerNorm
from torch_geometric.nn import SimpleConv
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from torch_geometric.utils import dense_to_sparse
from tsl.nn import utils
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.layers.graph_convs import DiffConv, GraphConv
from tsl.nn.models.base_model import BaseModel
from tsl.utils.casting import torch_to_numpy
from utils import batchwise_min_max_scale

EPSILON = 1e-8

class UnnamedKrigModelV2(BaseModel):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 adj,
                 exog_size,
                 enc_layers,
                 gcn_layers,
                 norm='mean',
                 encode_edges=False,
                 activation='relu',
                 dropout=0,
                 intervention_steps=2,
                 horizon=24,
                 mmd_sample_ratio=1.):
        super(UnnamedKrigModelV2, self).__init__()

        self.steps = intervention_steps
        self.horizon = horizon
        self.mmd_ratio = mmd_sample_ratio
        input_size += exog_size
        self.input_encoder_fwd = RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 n_layers=enc_layers,
                                 return_only_last_state=False,
                                 cell='gru')
        
        self.input_encoder_bwd = RNN(input_size=input_size,
                                 hidden_size=hidden_size,
                                 n_layers=enc_layers,
                                 return_only_last_state=False,
                                 cell='gru')

        if encode_edges:
            self.edge_encoder = nn.Sequential(
                RNN(input_size=input_size,
                    hidden_size=hidden_size,
                    n_layers=enc_layers,
                    return_only_last_state=True,
                    cell='gru'),
                nn.Linear(hidden_size, 1),
                nn.Softplus(),
                Rearrange('e f -> (e f)', f=1),
            )
        else:
            self.register_parameter('edge_encoder', None)

        # # TODO: remove this eventually
        # self.temporal_enc = nn.Sequential(
        #     MLP(input_size=hidden_size*2,
        #         hidden_size=hidden_size,
        #         output_size=hidden_size//2)
        # )

        self.gcn_layers_fwd = nn.ModuleList([
            GraphConv(hidden_size,
                      hidden_size,
                      root_weight=False,
                      norm=norm,
                      activation=activation) for _ in range(enc_layers)
        ])

        self.gcn_layers_bwd = nn.ModuleList([
            GraphConv(hidden_size,
                      hidden_size,
                      root_weight=False,
                      norm=norm,
                      activation=activation) for _ in range(enc_layers)
        ])

        self.skip_con_fwd = nn.Linear(hidden_size, hidden_size)

        self.skip_con_bwd = nn.Linear(hidden_size, hidden_size)

        self.key = MLP(input_size=hidden_size*2,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.query = MLP(input_size=hidden_size*2,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        self.value = MLP(input_size=hidden_size*2,
                        hidden_size=hidden_size,
                        output_size=hidden_size,
                        activation=activation)
        # self.out_proj = MLP(input_size=hidden_size*horizon,
        #                     hidden_size=hidden_size,
        #                     output_size=hidden_size*horizon,
        #                     activation=activation)

        # self.init_pass = SimpleConv('max')

        self.layernorm1 = LayerNorm(hidden_size)
        self.layernorm2 = LayerNorm(hidden_size)
        self.layernorm3 = LayerNorm(hidden_size)

        self.gcn1 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=gcn_layers, 
                        out_channels=hidden_size,
                        dropout=dropout,
                        norm='LayerNorm')
        self.gcn2 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=max((gcn_layers-1), 1), 
                        out_channels=hidden_size,
                        dropout=dropout,
                        norm='LayerNorm')
        # self.gcn1 = DiffConv(in_channels=hidden_size, 
        #                     out_channels=hidden_size,
        #                     k=gcn_layers)
        # self.gcn2 = DiffConv(in_channels=hidden_size, 
        #                     out_channels=hidden_size,
        #                     k=gcn_layers)

        self.readout1 = nn.Linear(hidden_size, output_size)
        self.readout2 = nn.Linear(hidden_size, output_size)
        self.readout3 = nn.Linear(hidden_size*2, output_size)

        self.adj = adj
        self.adj_n1 = None
        self.adj_n2 = None
        self.obs_neighbors = None

    # Experimental: Adding virtual and masking other nodes
    def forward(self,
                x,
                mask,
                known_set,
                seened_set=None,
                masked_set=None, 
                sub_entry_num=0,
                edge_weight=None,
                edge_features=None,
                training=False,
                reset=False,
                u=None,
                predict=False,
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
        known_adj = full_adj[known_set, :]
        known_adj = known_adj[:, known_set]

        if seened_set is not None:
            o_adj = full_adj[seened_set, :]
            o_adj = o_adj[:, seened_set]
        else:
            o_adj = known_adj

        edge_index, edge_weight = dense_to_sparse(o_adj)
        # ========================================
        # Simple GRU-GCN embedding
        # ========================================
        # Bigger encoders might exacerbate confounders 

        # flat time dimension fwd and bwd
        x_fwd = self.input_encoder_fwd(x)
        x_bwd = self.input_encoder_bwd(torch.flip(x, (1,)))

        # Edge encoder
        if self.edge_encoder is not None:
            assert edge_weight is None
            edge_weight = self.edge_encoder(edge_features)

        # forward encoding 
        x_fwd = rearrange(x_fwd, 'b t n d -> (b t) n d')
        out_f = x_fwd
        for layer in self.gcn_layers_fwd:
            out_f = layer(out_f, edge_index, edge_weight)
        out_f = out_f + self.skip_con_fwd(x_fwd)

        # backward encoding
        x_bwd = rearrange(x_bwd, 'b t n d -> (b t) n d')
        out_b = x_bwd
        for layer in self.gcn_layers_bwd:
            out_b = layer(out_b, edge_index, edge_weight)
        out_b = out_b + self.skip_con_bwd(x_bwd)

        # Concatenate backward and forward processes
        sum_out = torch.cat([out_f, out_b], dim=-1)
        # sum_out = out_f

        # ========================================
        # Calculating variant and invariant features using self-attention 
        # across different nodes using their representations
        # ========================================
        # Query represents new matrix of n1
        query = self.query(sum_out)
        key = self.key(sum_out)
        value = self.value(sum_out)

        # query = rearrange(query, '(b t) n d -> b n (t d)', b=b)
        # key = rearrange(key, '(b t) n d -> b n (t d)', b=b)
        # value = rearrange(value, '(b t) n d -> b n (t d)', b=b)

        # # Calculate self attention between Q and K,V
        # # Do this for the two Queries 
        # adj_var, adj_invar, output_invars, \
        # output_vars, scores = self.scaled_dot_product_mhattention(query, 
        #                                                           key, 
        #                                                           value,
        #                                                           mask = o_adj,
        #                                                           n_head = t)
        
        # output_invars = rearrange(output_invars, 'b n (t d) -> (b t) n d', t=self.horizon)
        # output_vars = rearrange(output_vars, 'b n (t d) -> (b t) n d', t=self.horizon)
        # scores = rearrange(scores, 'b t n d -> (b t) n d')

        adj_var, adj_invar, output_invars, \
        output_vars, scores = self.scaled_dot_product_attention(query, 
                                                                key, 
                                                                value,
                                                                mask = o_adj)
        
        # adj_invar = rearrange(adj_invar, 'b t n d -> (b t) n d') + o_adj
        # adj_var = rearrange(adj_var, 'b t n d -> (b t) n d') + o_adj

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        # Get two graphs, one with one hop connections, another with two hop connections
        if seened_set is not None:
            arrange = seened_set + masked_set

            o_adj = full_adj[arrange, :]
            o_adj = o_adj[:, arrange]

            # Its a secret weapon we're using in the future
            masked_edges = full_adj[masked_set, :]
            masked_edges = masked_edges[:, masked_set]

        if training:
            # inductive
            if reset:
                adj_n1 = self.get_new_adj(o_adj, sub_entry_num)
            else:
                if self.adj_n1 is None and self.adj_n2 is None:
                    adj_n1 = o_adj
                else:
                    assert self.adj_n2 is not None and self.adj_n1 is not None
                    adj_n1 = self.adj_n1

            adjs = [adj_n1]

            # transductive
            # unkn = [i for i in range(self.adj.shape[0]) if i not in arrange][:sub_entry_num]
            # full = arrange + unkn

            # n_adj = full_adj[full, :]
            # n_adj = n_adj[:, full]

            # adjs = [n_adj]
        else:
            if masked_set is not None:
                arrange = known_set + masked_set
            else:
                arrange = known_set + [i for i in range(self.adj.shape[0]) if i not in known_set]

            n_adj = full_adj[arrange, :]
            n_adj = n_adj[:, arrange]

            # n_adj[len(known_set):, :] = 1
            # n_adj[:, len(known_set):] = 1

            adjs = [n_adj]

        finrecos = []
        finpreds = []
        fin_irm_all_s = []
        output_invars_s = []
        output_vars_s = []

        for adj in adjs:
            bt, n_og, d = output_invars.shape

            if seened_set is not None:
                virt = len(arrange)
                add_nodes = len(masked_edges) + sub_entry_num
            else:
                virt = n_og
                add_nodes = sub_entry_num
            # ========================================
            # Edge expansion
            # ========================================
            # Propagate variant and invariant features to new nodes features
            new_scores = self.expand_adj(adj, scores, add_nodes)

            if add_nodes != 0:
                sub_entry = torch.zeros(bt, add_nodes, d).to(device)

                xh_inv = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
                xh_var = torch.cat([output_vars, sub_entry], dim=1)
            else:
                xh_inv = output_invars
                xh_var = output_vars

            scores_var = new_scores.masked_fill(adj == 0, float('1e16'))
            scores_invar = new_scores.masked_fill(adj == 0, float('-1e16')) 

            # Softmax to normalize scores, producing attention weights
            new_var_adj = F.softmax(-scores_var, dim=-1) + adj
            # new_var_adj = batchwise_min_max_scale(new_var_adj)
            new_inv_adj = F.softmax(scores_invar, dim=-1) + adj
            # new_inv_adj = batchwise_min_max_scale(new_inv_adj)

            invar_adj = dense_to_sparse(new_inv_adj)
            var_adj = dense_to_sparse(new_var_adj)

            # adj_l = dense_to_sparse(adj)
            # sum_out = rearrange(sum_out, '(b t) n d -> b (t n) d', b=b, t=t)

            # ========================================
            # Final prediction
            # ========================================
            t_mask = rearrange(mask, 'b t n d -> (b t n) d')
            # Edge weights must be scaled down 
            # Need to propagate this too
            # Need to scale the new edges based on the probability of the softmax
            # [batch, time, node, node]

            n = xh_inv.shape[1]
            xh_inv = rearrange(xh_inv, 'b n d -> (b n) d')
            xh_var = rearrange(xh_var, 'b n d -> (b n) d')

            xh_inv_1 = self.gcn1(xh_inv, invar_adj[0], invar_adj[1]) * (1 - t_mask) + xh_inv
            xh_inv_1 = rearrange(xh_inv_1, '(b t n) d -> b t n d', b=b, t=t)
            xh_inv_1 = self.layernorm1(xh_inv_1)
            xh_inv_2 = rearrange(xh_inv_1, 'b t n d -> (b t n) d')

            xh_var_1 = self.gcn1(xh_var, var_adj[0], var_adj[1]) * (1 - t_mask) + xh_var
            xh_var_1 = rearrange(xh_var_1, '(b t n) d -> b t n d', b=b, t=t)
            xh_var_1 = self.layernorm1(xh_var_1)
            xh_var_2 = rearrange(xh_var_1, 'b t n d -> (b t n) d')

            xh_inv_2 = self.gcn2(xh_inv_2, invar_adj[0], invar_adj[1]) + xh_inv_2
            xh_inv_2 = rearrange(xh_inv_2, '(b t n) d -> b t n d', b=b, t=t)
            xh_inv_2 = self.layernorm2(xh_inv_2)

            xh_var_2 = self.gcn2(xh_var_2, var_adj[0], var_adj[1]) + xh_var_2
            xh_var_2 = rearrange(xh_var_2, '(b t n) d -> b t n d', b=b, t=t)
            xh_var_2 = self.layernorm2(xh_var_2)

            # With the final representations, predict the unknown nodes again, just using the invariant features
            # Get reconstruction loss with pseudo labels 

            # GCN
            # finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)

            finrp = self.readout1(xh_inv_2)
            finpreds.append(finrp)
            if not training:
                if predict:
                    return finpreds[0], new_inv_adj[0].unsqueeze(0), new_var_adj[0].unsqueeze(0)
                else:
                    return finpreds[0]

            # ========================================
            # MMD of embeddings
            # ========================================

            # Get embedding softmax
            if self.mmd_ratio < 1.0:
                s_batch = int(b*self.mmd_ratio)
                indx = torch.randperm(b, device=device)[:s_batch]

                vir_inv = xh_inv_2[indx]
                vir_var = xh_var_2[indx]

                mmd_scores = rearrange(new_scores, '(b t) n d -> b t n d', b=b)[indx]

            # Get embedding softmax
            sps_var_adj = F.softmax(-mmd_scores, dim=-1) #* (adj + EPSILON)
            sps_inv_adj = F.softmax(mmd_scores, dim=-1) #* (adj + EPSILON)

            # Get most similar embedding with virtual nodes and vice versa
            sps_var_max = torch.argmax((sps_var_adj[:, :, virt:, :virt]), dim=-1)
            sps_inv_max = torch.argmax((sps_inv_adj[:, :, virt:, :virt]), dim=-1)

            # sps_var_max = sps_var_max[:, n_og:]
            # sps_inv_max = sps_inv_max[:, n_og:]

            sps_var_max = sps_var_max.unsqueeze(-1).expand(-1, -1, -1, vir_var.size(-1))
            sps_inv_max = sps_inv_max.unsqueeze(-1).expand(-1, -1, -1, vir_inv.size(-1))

            sim_inv = torch.gather(vir_inv, dim=2, index=sps_inv_max)
            sim_var = torch.gather(vir_var, dim=2, index=sps_var_max)

            # # Sample batches
            # if self.mmd_ratio < 1.0:
            #     s_batch = int(b*self.mmd_ratio)
            #     indx = torch.randperm(b, device=device)[:s_batch]
            #     sim_inv = sim_inv[indx]
            #     sim_var = sim_var[indx]

            #     vir_inv = xh_inv_1[indx]
            #     vir_var = xh_var_1[indx]
            
            # Shape: b*t, n, n
            emb_tru_inv = sim_inv.view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)
            emb_tru_var = sim_var.view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)

            emb_vir_inv = vir_inv[:, :, virt:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)
            emb_vir_var = vir_var[:, :, virt:].view(s_batch*t, sub_entry_num, d).permute(1, 0, 2)

            finrecos.append([emb_tru_inv, emb_tru_var, emb_vir_inv, emb_vir_var])
            # ========================================
            # IRM model
            # ========================================
            # Predict the real nodes by propagating back using both variant and invariant features
            # Get IRM loss
            fin_irm_all = []
        
            for _ in range(self.steps):
                seen_vars = xh_var_2[:, :, :len(known_set)]
                seen_invr = xh_inv_2[:, :, :len(known_set)]
                # unsn_vars = xh_var_2[:, :, len(known_set):]
                
                seen_vars_l = rearrange(seen_vars, 'b t n d -> b (t n) d', b=b, t=t)
                # unsn_vars_l = rearrange(unsn_vars, 'b t n d -> b (t n) d', b=b, t=t)

                s_rands = torch.randperm(seen_vars_l.shape[1])
                # u_rands = torch.randperm(unsn_vars_l.shape[1])
                # rands = torch.randperm(xh_var_l.shape[1])

                rand_seen = seen_vars_l[:, s_rands, :]
                # rand_unsn = unsn_vars_l[:, u_rands, :]
                # rand_vars = xh_var_l[:, rands, :] #.detach()

                rand_seen = rearrange(rand_seen, 'b (t n) d -> b t n d', t=t)
                # rand_unsn = rearrange(rand_unsn, 'b (t n) d -> b t n d', t=t)
                
                # rand_vars = torch.cat((rand_seen, rand_unsn), dim=2)
                # fin_vars = self.layernorm2(xh_inv_2 + rand_vars)
                fin_vars = torch.cat((seen_invr, rand_seen), dim=-1)
                # fin_irm = self.gcn2(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
                fin_irm = self.readout3(fin_vars)
                fin_irm_all.append(fin_irm)

            fin_irm_all = torch.stack(fin_irm_all)
            fin_irm_all_s.append(fin_irm_all)
        # ========================================
        # Size regularisation
        # ========================================
        # Regularise both graphs using CMD using 
        # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

        return finpreds, finrecos, fin_irm_all_s, output_invars_s, output_vars_s

    def expand_adj(self, adj, scores, sub_entry_num):
        # Get immediate edge samples
        first_adj = adj[:scores.shape[-1], :scores.shape[-1]]
        mask = first_adj != 0
        # scores *= mask
        mask_scores = scores * mask
        means = mask_scores.sum(dim=-1)/mask.sum(dim=1).clamp(min=1)
        squared_diff = ((mask_scores - means.unsqueeze(-1)) ** 2) * mask
        variance = torch.sqrt(squared_diff.sum(dim=-1) / (mask.sum(dim=1) - 1).clamp(min=1))

        means_exp = means.unsqueeze(-1).expand(-1, -1, sub_entry_num)
        variance_exp = variance.unsqueeze(-1).expand(-1, -1, sub_entry_num)
        inc_edge = torch.normal(mean=means_exp, std=torch.sqrt(variance_exp)).to(device=scores.device)

        # Get virtual edge samples
        new_scores = torch.cat((scores, inc_edge), dim=-1)
        zeros = torch.zeros((new_scores.shape[0], sub_entry_num, new_scores.shape[-1])).to(device=adj.device)
        new_scores = torch.cat((new_scores, zeros), dim=1)

        upper = torch.triu(new_scores, diagonal=1)
        new_scores[:, scores.shape[-1]:, :] = ((new_scores + upper.transpose(-1, -2)))[:, scores.shape[-1]:, :]

        next_mask = adj[scores.shape[-1]:, :scores.shape[-1]]
        next_scores = new_scores[:, scores.shape[-1]:, :scores.shape[-1]] * next_mask

        means = next_scores.sum(dim=-1)/next_mask.sum(dim=1).clamp(min=1)
        squared_diff = ((next_scores - means.unsqueeze(-1)) ** 2) * next_mask
        variance = torch.sqrt(squared_diff.sum(dim=-1) / (next_mask.sum(dim=1) - 1).clamp(min=1))

        means_exp = means.unsqueeze(-1).expand(-1, -1, sub_entry_num)
        variance_exp = variance.unsqueeze(-1).expand(-1, -1, sub_entry_num)
        inc_edge = torch.normal(mean=means_exp, std=torch.sqrt(variance_exp)).to(device=scores.device)

        new_scores[:, scores.shape[-1]:, scores.shape[-1]:] = inc_edge
        # mask = adj != 0
        # new_scores *= mask
        return new_scores

    def get_new_adj(adj, sub_entry_num):
        n1 = adj.shape[0]
        
        # if self.obs_neighbors is None:
        neighbors_1h = {}

        for i in range(n1):
            row_n_1 = set(torch.nonzero(adj[i]).squeeze(1).tolist())
            col_n_1 = set(torch.nonzero(adj[:, i]).squeeze(1).tolist())
            all_n_1 = row_n_1.union(col_n_1)

            # 1-hop neighbors
            neighbors_1h[i] = list(all_n_1)

        n2 = n1 + sub_entry_num
        # Create matrices for both n1 and n2
        adj_aug_n1 = torch.rand((n2, n2)).to(device = adj.device)  # n2, n2
        adj_aug_n1 = torch.triu(adj_aug_n1) + torch.triu(adj_aug_n1, 1).T

        # preserve original observed parts in newly-created adj
        adj_aug_n1[:n1, :n1] = adj
        adj_aug_n1 = adj_aug_n1.fill_diagonal_(0)
        adj_aug_mask_n1 = torch.zeros_like(adj_aug_n1)  # n2, n2

        adj_aug_mask_n1[:n1, :n1] = 1
        neighbors_1 = copy.deepcopy(neighbors_1h)

        for i in range(n1, n2):
            n_current = range(len(neighbors_1.keys()))  # number of current entries (obs and already added virtual)
            rand_entry = random.sample(n_current, 1)[0] # randomly sample 1 entry (obs or already added virtual)
            rand_neighbors_1 = neighbors_1[rand_entry]  # get 1-hop neighbors of sampled entry

            p = np.random.rand(1)

            # randomly select 1 hop neighbors
            valid_neighbors_1 = (np.random.rand(len(rand_neighbors_1)) < p).astype(int)
            valid_neighbors_1 = np.where(valid_neighbors_1 == 1)[0].tolist()
            valid_neighbors_1 = [rand_neighbors_1[idx] for idx in valid_neighbors_1]

            all_entries = [rand_entry]
            all_entries.extend(valid_neighbors_1)

            # add current virtual entry to the 1-hop neighbors of selected entries
            for entry in all_entries:
                neighbors_1[entry].append(i)

            # add selected entries to the 1-hop neighbors of current virtual entry
            neighbors_1[i] = all_entries

            # Add to mask
            for j in range(len(all_entries)):
                entry = all_entries[j]
                adj_aug_mask_n1[entry, i] = 1
                adj_aug_mask_n1[i, entry] = 1

        adj_aug_n1 *= adj_aug_mask_n1

        return adj_aug_n1
    def scaled_dot_product_attention(self, Q, K, V, mask):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
        d_k = Q.size(-1)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # test = torch.rand_like(scores).to(device=scores.device)

        # Apply mask if provided (useful for masked self-attention in transformers)
        if mask is not None:
            scores_var = scores.masked_fill(mask == 0, float('1e16'))
            scores_invar = scores.masked_fill(mask == 0, float('-1e16')) 
        else:
            scores_var = scores.clone()
            scores_invar = scores.clone()    

        # Softmax to normalize scores, producing attention weights
        attention_weights_var = F.softmax(-scores_var, dim=-1)
        attention_weights_invar = F.softmax(scores_invar, dim=-1) 

        # Value should be aggregated using the attention weights as adjacency matrix weights
        output_invar = torch.bmm(attention_weights_invar, V)
        output_var = torch.bmm(attention_weights_var, V)

        return attention_weights_var.detach(), attention_weights_invar.detach(), output_invar, output_var, scores.detach()
            
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

        return attention_weights_var.detach(), attention_weights_invar.detach(), output_invar, output_var, scores.detach()

# from Grin import get_dataset
# from tsl.data import ImputationDataset, SpatioTemporalDataModule
# from tsl.data.preprocessing import StandardScaler
# from tsl.transforms import MaskInput

# if __name__ == '__main__':
#     mask_s = [5]
#     known_set = list(range(5))
#     dataset = get_dataset('air_auckland', p_noise=0.5, masked_s=mask_s)
#     # covariates = {'u': dataset.datetime_encoded('day').values}
#     adj = dataset.get_connectivity(method='distance', threshold=1, include_self=False, layout='dense', force_symmetric=True)
#     adj_list = dataset.get_connectivity(method='distance', threshold=1, include_self=False, force_symmetric=True)

#     adj_weights = torch.tensor(adj_list[1])
#     adj_list_t = torch.tensor(adj_list[0])
#     adj = torch.tensor(adj)

#     torch_dataset = ImputationDataset(target=dataset.dataframe(),
#                                         mask=dataset.training_mask,
#                                         eval_mask=dataset.eval_mask,
#                                     #   covariates=covariates,
#                                         transform=MaskInput(),
#                                         connectivity=adj,
#                                         window=12,
#                                         stride=1)

#     scalers = {'target': StandardScaler(axis=(0, 1))}

#     dm = SpatioTemporalDataModule(
#         dataset=torch_dataset,
#         scalers=scalers,
#         splitter=dataset.get_splitter(val_len= 0.1, test_len= 0.2),
#         batch_size=64,
#         workers=0)
#     dm.setup(stage='fit')

#     batch = next(iter(dm.train_dataloader()))

#     input_size = 1
#     hidden_size = 128
#     output_size = 1
#     horizon = 12
#     exog_size = 0
#     enc_layers = 2
#     gcn_layers = 2
#     dropout=0.4
#     S=3

#     model = UnnamedKrigModel(input_size=input_size,
#                         hidden_size=hidden_size,
#                         output_size=output_size,
#                         exog_size=exog_size,
#                         enc_layers=enc_layers,
#                         gcn_layers=gcn_layers,
#                         adj=adj,
#                         dropout=0.4,
#                         intervention_steps=S).to(device='cuda:3')
    
#     x = batch['x'][:, :, known_set, :].to(device='cuda:3')
#     mask = batch['mask'][:, :, known_set, :].to(device='cuda:3')
#     sub_entry_num = 1

#     b, s, n, d = mask.shape
#     sub_entry = torch.zeros(b, s, sub_entry_num, d).to(x.device)
#     mask = torch.cat([mask, sub_entry], dim=2).byte() 

#     finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s = model(x=x, edge_weight=None, sub_entry_num=sub_entry_num,
#                                               mask=mask, known_set=known_set, training=True, reset=True)
    
#     finrecos = torch.cat(finrecos)
#     finpreds = torch.cat(finpreds)
#     fin_irm_all_s = torch.cat(fin_irm_all_s)

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#     # Forward pass
#     target = torch.ones_like(fin_irm_all_s)
#     loss = F.mse_loss(fin_irm_all_s, target)
#     # loss = F.mse_loss(output_invars_s[0][:n], output_invars_s[1][:n])

#     # Backward pass
#     loss.backward()
#     optimizer.step()

#     # ✅ Check
#     print("Loss:", loss.item())
#     for name, param in model.named_parameters():
#         print(param.grad)
#         try:
#             if torch.isnan(param.grad).any():
#                 print(f"❌ NaNs in gradient for {name}")
#             else:
#                 print(f"✅ {name} gradient OK")
#         except:
#             print(f"{name} NoneType")
    

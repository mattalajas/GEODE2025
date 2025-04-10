import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
# from Grin import get_dataset
from torch import Tensor, nn
from torch.nn import MultiheadAttention
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import SimpleConv
from torch_geometric.nn.models import GCN
from tsl import logger
from tsl.data import ImputationDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.nn import utils
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.layers.graph_convs import GraphConv
from tsl.nn.models.base_model import BaseModel
from tsl.transforms import MaskInput
from tsl.utils.casting import torch_to_numpy


class UnnamedKrigModel(BaseModel):
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
                 activation='softplus',
                 n_heads=8,
                 dropout=0,
                 intervention_steps = 2):
        super(UnnamedKrigModel, self).__init__()

        self.steps = intervention_steps
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

        self.gcn_layers_fwd = nn.ModuleList([
            GraphConv(hidden_size,
                      hidden_size,
                      root_weight=False,
                      norm=norm,
                      activation=activation) for _ in range(gcn_layers)
        ])

        self.gcn_layers_bwd = nn.ModuleList([
            GraphConv(hidden_size,
                      hidden_size,
                      root_weight=False,
                      norm=norm,
                      activation=activation) for _ in range(gcn_layers)
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
        
        self.multihead = MultiheadAttention(hidden_size, n_heads, batch_first=True)

        self.init_pass = SimpleConv('mean', combine_root='sum')

        self.gcn1 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=gcn_layers, 
                        out_channels=output_size,
                        dropout=dropout,
                        norm='InstanceNorm')
        self.gcn2 = GCN(in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        num_layers=gcn_layers, 
                        out_channels=output_size,
                        dropout=dropout,
                        norm='InstanceNorm')

        self.adj = adj
        self.adj_n1 = None
        self.adj_n2 = None
        self.obs_neighbors = None

    def forward(self,
                x,
                mask=None,
                known_set=None,
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
        b, t, n, _ = x.size()
        x = utils.maybe_cat_exog(x, u)
        device = x.device

        o_adj = torch.tensor(self.adj).to(device)

        if training:
            o_adj = o_adj[known_set, :]
            o_adj = o_adj[:, known_set]

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

        # ========================================
        # Create new adjacency matrix 
        # ========================================
        # Get two graphs, one with one hop connections, another with two hop connections

        # TODO: need to put this in the filler
        if training:
            if reset:
                # d = mask.shape[-1]
                # sub_entry = torch.zeros(b, t, sub_entry_num, d).to(device)
                # mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
                # test = rearrange(mask, 'b t n d -> b n (t d)')
                # y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

                adj_n1, adj_n2 = self.get_new_adj(o_adj, sub_entry_num)
                adj_n1 = adj_n1
                adj_n2 = adj_n2
            else:
                if self.adj_n1 is None and self.adj_n2 is None:
                    adj_n1 = o_adj
                    adj_n2 = o_adj
                else:
                    assert self.adj_n2 is not None and self.adj_n1 is not None
                    adj_n1 = self.adj_n1
                    adj_n2 = self.adj_n2

            adjs = [adj_n1, adj_n2]
        else:
            adjs = [o_adj]

        finrecos = []
        finpreds = []
        fin_irm_all_s = []
        output_invars_s = []
        output_vars_s = []

        for adj in adjs:
            # ========================================
            # Calculating variant and invariant features using self-attention across different nodes using their representations
            # Adding the edge expansion here as well
            # ========================================
            bt, n, d = sum_out.shape
            if sub_entry_num != 0:
                sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
                full_sum_out = torch.cat([sum_out, sub_entry], dim=1)  # b*t n2 d
            else:
                full_sum_out = sum_out

            adj_l = dense_to_sparse(adj)

            t_mask = rearrange(mask, 'b t n d -> (b t) n d')
            full_sum_out = self.init_pass(full_sum_out, adj_l[0], adj_l[1]) * (1 - t_mask) + full_sum_out

            # Query represents new matrix of n1
            query = self.query(full_sum_out)
            key = self.key(full_sum_out)
            value = self.value(full_sum_out)

            # Calculate self attention between Q and K,V
            # Do this for the two Queries 
            adj_var, adj_invar, output_invars, output_vars = self.scaled_dot_product_attention(query, 
                                                                                               key, 
                                                                                               value,
                                                                                               mask = adj)

            output_invars_s.append(output_invars)
            output_vars_s.append(output_vars)
            # TODO: Check distributions after training

            # Edge weights must be scaled down 
            # Need to propagate this too
            # Need to scale the new edges based on the probability of the softmax
            # TODO: optimise this
            # [batch, time, node, node]

            # adj_invar_exp_n1 = torch.stack([adj_n1]*bt).to(device)
            # adj_invar_exp_n1[:, :n, :n] = adj_invar
            # adj_invar_exp_n2 = torch.stack([adj_n2]*bt).to(device)
            # adj_invar_exp_n2[:, :n, :n] = adj_invar
            # adj_var_exp_n1 = torch.stack([adj_n1]*bt).to(device)
            # adj_var_exp_n1[:, :n, :n] = adj_var
            # adj_var_exp_n2 = torch.stack([adj_n2]*bt).to(device)
            # adj_var_exp_n2[:, :n, :n] = adj_var

            # sub_entry = torch.zeros(bt, sub_entry_num, d).to(device)
            # rep_invars = torch.cat([output_invars, sub_entry], dim=1)  # b*t n2 d
            # rep_vars = torch.cat([output_vars, sub_entry], dim=1)  # b*t n2 d

            # Use this to get similarity between the known nodes 
            # rep_invars = (output_invars_n1 + output_invars_n2) / 2
            # rep_vars = (output_vars_n1 + output_vars_n2) / 2

            # Propagate variant and invariant features to new nodes features
            invar_adj = dense_to_sparse(adj_invar)
            var_adj = dense_to_sparse(adj_var)

            t_mask = rearrange(t_mask, 'b n d -> (b n) d')
            xh_inv = rearrange(output_invars, 'b n d -> (b n) d')
            xh_var = rearrange(output_vars, 'b n d -> (b n) d')

            # ========================================
            # Final prediction
            # ========================================
            # With the final representations, predict the unknown nodes again, just using the invariant features
            # Get reconstruction loss with pseudo labels 
            batches = torch.arange(0, b).to(device=device)
            batches = torch.repeat_interleave(batches, repeats=t*(n+sub_entry_num))

            finrp = self.gcn1(xh_inv, invar_adj[0], invar_adj[1], batch=batches)
            finrp = rearrange(finrp, '(b t n) d -> b t n d', b=b, t=t)
            finpreds.append(finrp)
            if not training and not predict:
                return finpreds[0]

            # ========================================
            # Reconstruction model
            # ========================================
            # Shape: b*t*n, d

            rec = xh_inv * (1 - t_mask)

            # Predict the real nodes by propagating back using just the invariant features
            # Get reconstruction loss
            finr = self.gcn2(rec, invar_adj[0], invar_adj[1], batch=batches)
            finr = rearrange(finr, '(b t n) d -> b t n d', b=b, t=t)
            finrecos.append(finr)

            # ========================================
            # IRM model
            # ========================================
            # Predict the real nodes by propagating back using both variant and invariant features
            # Get IRM loss
            fin_irm_all = []
        
            for _ in range(self.steps):
                rands = torch.randperm(xh_var.shape[0])
                rand_vars = xh_var[rands].detach()
                fin_irm = self.gcn1(xh_inv + rand_vars, invar_adj[0], invar_adj[1], batch=batches)
                fin_irm_all.append(rearrange(fin_irm, '(b t n) d -> b t n d', b=b, t=t))

            fin_irm_all = torch.cat(fin_irm_all)
            fin_irm_all_s.append(fin_irm_all)

        # ========================================
        # Size regularisation
        # ========================================
        # Regularise both graphs using CMD using 
        # <https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf>

        if training:
            return finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s
        elif predict:
            return finrecos[0], finpreds[0], fin_irm_all_s[0], output_invars_s[0], output_vars_s[0], adj_invar[0], adj_var[0]
        
    def get_new_adj(self, adj, sub_entry_num):
        n1 = adj.shape[0]
        
        if self.obs_neighbors is None:
            neighbors_1h = {}
            neighbors_2h = {}

            # Get two hop diagonal
            sp_adj = adj.to_sparse_coo()
            two_hops = torch.mm(sp_adj, sp_adj).to_dense()
            two_hops = two_hops.fill_diagonal_(0)

            for i in range(n1):
                row_n_1 = set(torch.nonzero(adj[i]).squeeze(1).tolist())
                col_n_1 = set(torch.nonzero(adj[:, i]).squeeze(1).tolist())
                all_n_1 = row_n_1.union(col_n_1)

                # 1-hop neighbors
                neighbors_1h[i] = list(all_n_1)

                row_n_2 = set(torch.nonzero(two_hops[i]).squeeze(1).tolist())
                col_n_2 = set(torch.nonzero(two_hops[:, i]).squeeze(1).tolist())
                all_n_2 = row_n_2.union(col_n_2)
                all_n = all_n_2.union(all_n_1)

                # 1 and 2 hop neighbors
                neighbors_2h[i] = list(all_n)
                
            self.obs_neighbors = (neighbors_1h, neighbors_2h)
        else:
            neighbors_1h, neighbors_2h = self.obs_neighbors  # n1, n1, note that cannot use copy!!!

        n2 = n1 + sub_entry_num
        # Create matrices for both n1 and n2
        adj_aug_n1 = torch.rand((n2, n2)).to(device = adj.device)  # n2, n2
        adj_aug_n2 = torch.rand((n2, n2)).to(device = adj.device)  # n2, n2

        # preserve original observed parts in newly-created adj
        adj_aug_n1[:n1, :n1] = adj
        adj_aug_n2[:n1, :n1] = adj

        adj_aug_n1 = adj_aug_n1.fill_diagonal_(0)
        adj_aug_n2 = adj_aug_n2.fill_diagonal_(0)

        adj_aug_mask_n1 = torch.zeros_like(adj_aug_n1)  # n2, n2
        adj_aug_mask_n2 = torch.zeros_like(adj_aug_n2)  # n2, n2

        adj_aug_mask_n1[:n1, :n1] = 1
        adj_aug_mask_n2[:n1, :n1] = 1

        neighbors_1 = copy.deepcopy(neighbors_1h)
        neighbors_2 = copy.deepcopy(neighbors_2h)

        for i in range(n1, n2):
            n_current = range(len(neighbors_1.keys()))  # number of current entries (obs and already added virtual)
            rand_entry = random.sample(n_current, 1)[0] # randomly sample 1 entry (obs or already added virtual)
            rand_neighbors_1 = neighbors_1[rand_entry]  # get 1-hop neighbors of sampled entry
            rand_neighbors_2 = neighbors_2[rand_entry]  # get 1 and 2-hop neighbors of sample entry

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

            # randomly select 1-2 hop neighbors
            valid_neighbors_2 = (np.random.rand(len(rand_neighbors_2)) < p).astype(int)
            valid_neighbors_2 = np.where(valid_neighbors_2 == 1)[0].tolist()
            valid_neighbors_2 = [rand_neighbors_2[idx] for idx in valid_neighbors_2]

            all_entries = [rand_entry]
            all_entries.extend(valid_neighbors_2)

            # add current virtual entry to the 1-2 hop neighbors of selected entries
            for entry in all_entries:
                neighbors_2[entry].append(i)

            # add selected entries to the 1-2 hop neighbors of current virtual entry
            neighbors_2[i] = all_entries

            # Add to mask
            for j in range(len(all_entries)):
                entry = all_entries[j]
                adj_aug_mask_n2[entry, i] = 1
                adj_aug_mask_n2[i, entry] = 1

        adj_aug_n1 *= adj_aug_mask_n1
        adj_aug_n2 *= adj_aug_mask_n2
    
        return adj_aug_n1, adj_aug_n2
            

    def scaled_dot_product_attention(self, Q, K, V, mask):
    # Compute the dot products between Q and K, then scale by the square root of the key dimension
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        # test = torch.rand_like(scores).to(device=scores.device)

        # Apply mask if provided (useful for masked self-attention in transformers)
        if mask is not None:
            scores_invar = scores.masked_fill(mask == 0, float('-inf'))
            scores_var = scores.masked_fill(mask == 0, float('inf'))

        # Softmax to normalize scores, producing attention weights
        attention_weights_var = torch.nan_to_num(F.softmax(-scores_var, dim=-1), nan=0.0)
        attention_weights_invar = torch.nan_to_num(F.softmax(scores_invar, dim=-1), nan=0.0)

        # Value should be aggregated using the attention weights as adjacency matrix weights
        output_invar = torch.matmul(attention_weights_invar, V)
        output_var = torch.matmul(attention_weights_var, V)

        # # ========================================
        # # New edge generation
        # # ========================================
        # # Maybe add the active learning here
        # # For now just randomly choose an entry from the 
        # n = mask.shape[0]
        # m = adj_mask.shape[0]
        # batch = Q.shape[0]
        # diff = m-n

        # exp_att_var = torch.ones((batch, m, m)).to(device=Q.device)
        # exp_att_inv = torch.ones((batch, m, m)).to(device=Q.device)
        # exp_list = [exp_att_var, exp_att_inv]

        # # Randomly choose from the existing edges to impute the immediate missing edge weights
        # for i in range(n):
        #     for ind, scores in enumerate([scores_var, scores_invar]):
        #         original = scores[:, i]
        #         valid_mask = original != float('-inf')
                
        #         safe_tensor = torch.where(valid_mask, original, torch.tensor(0.0))

        #         valid_counts = valid_mask.sum(dim=1)
        #         sorted_indices = torch.argsort(~valid_mask.cpu(), dim=1).to(device=Q.device)
        #         sorted_values = torch.gather(safe_tensor, dim=1, index=sorted_indices)

        #         max_valid = valid_counts.max()
        #         rand_idx = torch.randint(0, max_valid, (batch, m)).to(device=Q.device)
    
        #         clamped_idx = torch.minimum(rand_idx, valid_counts.unsqueeze(1) - 1)

        #         # Now gather using clamped indices
        #         batch_idx = torch.arange(batch).unsqueeze(1).expand(-1, m)
        #         exp_list[ind][:, i, :] = sorted_values[batch_idx, clamped_idx]

        # exp_att_var[:, :n, :n] = scores_var.detach()
        # exp_att_inv[:, :n, :n] = scores_invar.detach()


        return attention_weights_var.detach(), attention_weights_invar.detach(), output_invar, output_var

# if __name__ == '__main__':
#     mask_s = list(range(0, 69))
#     known_set = list(range(69, 207))
#     dataset = get_dataset('metrla', p_noise=0.5, masked_s=mask_s)
#     # covariates = {'u': dataset.datetime_encoded('day').values}
#     adj = dataset.get_connectivity(method='distance', threshold=0.1, include_self=False, layout='dense', force_symmetric=True)
#     adj_list = dataset.get_connectivity(method='distance', threshold=0.1, include_self=False, force_symmetric=True)

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
#                         S=3,
#                         device='cuda:3')
    
#     x = batch['x'][:, :, known_set, :].to(device='cuda:3')
#     mask = batch['mask'][:, :, known_set, :].to(device='cuda:3')
#     sub_entry_num = 69

#     finrecos, finpreds, fin_irm_all_s, output_invars_s, output_vars_s = model(x=x, edge_weight=None, sub_entry_num=sub_entry_num,
#                                               mask=mask, known_set=known_set, training=True, reset=True)
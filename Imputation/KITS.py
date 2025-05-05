import sys
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from tsl.nn.models.base_model import BaseModel

EPSILON = 1e-8

class SpatialConvOrderK(nn.Module):
    """
    Spatial convolution of order K with possibly different diffusion matrices (useful for directed graphs)

    Efficient implementation inspired from graph-wavenet codebase
    """

    def __init__(self, c_in, c_out, support_len=3, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order
        self.c_out = c_out

    @staticmethod
    def compute_support(adj):
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + EPSILON)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + EPSILON)
        support = [adj_fwd, adj_bwd]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj)
        supp_k = []
        for a in support:
            ak = a
            for i in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support, support_diag=None, pattern=None):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if (type(support) is not list):
            support = [support]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            # x1 = torch.einsum('ncvl,nwvl->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                # x2 = torch.einsum('ncvl,nwvl->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)

        if support_diag is not None:
            out_diag = [x] if self.include_self else []
            if (type(support_diag) is not list):
                support_diag = [support_diag]
            for a in support_diag:
                x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
                # x1 = torch.einsum('ncvl,nwvl->ncwl', (x, a)).contiguous()
                out_diag.append(x1)
                for k in range(2, self.order + 1):
                    x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                    # x2 = torch.einsum('ncvl,nwvl->ncwl', (x1, a)).contiguous()
                    out_diag.append(x2)
                    x1 = x2
            out_diag = torch.cat(out_diag, dim=1)

        # out => b t*d n s, w/o self-loop and temporal
        # out_diag => b t*d n s, w/ self-loop and temporal
        # suppose t=0,1 (current),2

        if pattern is not None:
            t = out.size(1) // self.c_out
            mid = t // 2

            if pattern == "wo_self_loop_w_temporal":
                # if w/o self-loop, but w/ temporal
                out[:, :mid, :, :] = out_diag[:, :mid, :, :]
                out[:, mid+1:, :, :] = out_diag[:, mid+1:, :, :]
            elif pattern == "w_self_loop_wo_temporal":
                # if w/ self-loop, but w/o temporal
                out[:, mid, :, :] = out_diag[:, mid, :, :]
            elif pattern == "wo_self_loop_wo_temporal":
                out = out_diag

        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out
    
class KITS(BaseModel):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 args
                 ):
        super(KITS, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dataset_name = args.name

        self.t_dim = 3
        # self.register_buffer('adj', torch.tensor(adj).float())
        self.adj = torch.tensor(adj).float()
        self.fc_1 = nn.Linear(1, d_hidden)

        self.gcn_1 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_2 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_3 = SpatialConvOrderK(c_in=self.t_dim * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)

        self.smooth = nn.Linear(2 * d_hidden, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, 1)

        self.relu = nn.ReLU(inplace=True)
        self.supp = None
        self.adj_aug = None

        self.obs_neighbors = None

        if args.use_adj_drop:
            print("use adj dropout...")
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Identity()

        if args.use_init:
            print("use init...")
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)

    def adj_drop(self, supp, mask):
        # supp: list, fwd and bwd adj - (n, n)
        # mask: b, s, n, 1
        mask = rearrange(mask, 'b s n 1 -> (b s) n')
        mask = mask.sum(0)  # n
        obs_index = mask > 0  # n
        # unobs_index = mask == 0  # n
        supp_update = []
        for i in range(len(supp)):
            s = supp[i].clone().detach()

            s_hor = s[obs_index, :]  # n1, n
            s_ver = s[:, obs_index]  # n, n1

            s_hor = self.dropout(s_hor)
            s_ver = self.dropout(s_ver)

            s[obs_index, :] = s_hor
            s[:, obs_index] = s_ver

            supp_update.append(s)
        return supp_update

    def forward(self, x, mask=None, known_set=None, sub_entry_num=None, reset=False, training=False,
                edge_weight = None, edge_index = None, u = None, transform=None):
        adj = self.adj.clone().to(device=x.device)  # adjacency matrix

        # # Do residual
        # original_x = x.clone()
        # gain = torch.zeros_like(original_x, device=original_x.device)
        # gain[:, 1:] = (x[:, 1:] - x[:, :-1]) 
        # x = gain

        # resid = torch.roll(original_x, shifts=1, dims=1)
        # resid[:, 0] = resid[:, 1]

        # testing = resid+gain

        # print(torch.equal(original_x, testing))

        if training:
            if reset:
                # ========================================
                # Obtain 1-hop neighbors of each observed entry
                # ========================================
                # preserve observed entries
                adj = adj[known_set, :]
                adj = adj[:, known_set]  # n1, n1
                n1 = adj.size(0)

                # get the 1-hop neighbors of each observed entry.
                # if self.obs_neighbors is None:
                obs_neighbors = {}
                neighbors_2h = {}

                # Get two hop diagonal
                sp_adj = adj.to_sparse_coo()
                two_hops = torch.mm(sp_adj, sp_adj).to_dense()
                two_hops = two_hops.fill_diagonal_(0)

                for i in range(n1):
                    row_nonzero = set(torch.where(adj[i, :] > 0)[0].detach().cpu().numpy().tolist())
                    col_nonzero = set(torch.where(adj[:, i] > 0)[0].detach().cpu().numpy().tolist())
                    nonzero = row_nonzero.union(col_nonzero)

                    row_n_2 = set(torch.nonzero(two_hops[i]).squeeze(1).tolist())
                    col_n_2 = set(torch.nonzero(two_hops[:, i]).squeeze(1).tolist())
                    all_n_2 = row_n_2.union(col_n_2)
                    all_n = all_n_2.union(nonzero)

                    obs_neighbors[i] = list(nonzero)  # 1-hop neighbors
                self.obs_neighbors = obs_neighbors
                # else:
                #     obs_neighbors = self.obs_neighbors  # n1, n1, note that cannot use copy!!!

                # ========================================
                # Create dynamic adjacency matrix
                # ========================================
                # initialize dynamic adjacency matrix
                n2 = n1 + sub_entry_num
                adj_aug = torch.rand(n2, n2).to(adj.device)  # n2, n2

                # preserve original observed parts in newly-created adj
                adj_aug[:n1, :n1] = adj

                # remove self-loop
                adj_diag = 1. - torch.eye(n2).to(adj.device)  # n2, n2
                adj_aug = adj_aug * adj_diag  # n2, n2

                # for each newly-created virtual entry, randomly connect it to one observed entry
                neighbors = copy.deepcopy(obs_neighbors)  # initially has n1 entries' 1-hop neighbors
                adj_aug_mask = torch.zeros_like(adj_aug)  # n2, n2
                adj_aug_mask[:n1, :n1] = 1
                for i in range(n1, n2):
                    n_current = range(len(neighbors.keys()))  # number of current entries (obs and already added virtual)
                    rand_entry = random.sample(n_current, 1)[0]  # randomly sample 1 entry (obs or already added virtual)
                    rand_neighbors = neighbors[rand_entry]  # get 1-hop neighbors of sampled entry
                    p = np.random.rand(1)  # randomly generate a probability

                    # randomly select neighbors
                    valid_neighbors = (np.random.rand(len(rand_neighbors)) < p).astype(int)
                    valid_neighbors = np.where(valid_neighbors == 1)[0].tolist()
                    valid_neighbors = [rand_neighbors[idx] for idx in valid_neighbors]
                    all_entries = [rand_entry]
                    all_entries.extend(valid_neighbors)

                    # add current virtual entry to the 1-hop neighbors of selected entries
                    for entry in all_entries:
                        neighbors[entry].append(i)

                    # add selected entries to the 1-hop neighbors of current virtual entry
                    neighbors[i] = all_entries

                    options = [0, 1, 2]  # 0: forward; 1: backward; 2: bi-direction
                    connected_conditions = [random.choice(options) for _ in range(len(all_entries))]
                    for j in range(len(all_entries)):
                        entry = all_entries[j]
                        condition = connected_conditions[j]

                        if condition == 0 or condition == 2:
                            adj_aug_mask[entry, i] = 1
                        if condition == 1 or condition == 2:
                            adj_aug_mask[i, entry] = 1

                adj_aug = adj_aug * adj_aug_mask

                if self.dataset_name in ["sea_loop_point"]:
                    adj_aug[adj_aug > 0] = 1  # only for sea-loop, because their adj are binary

                self.adj_aug = adj_aug
            else:
                if self.adj_aug is not None:
                    adj_aug = self.adj_aug
                else:
                    adj_aug = adj
            adj = adj_aug.detach()
        else:
            if known_set is not None:
                adj = adj[known_set, :]
                adj = adj[:, known_set] 

        supp = SpatialConvOrderK.compute_support(adj)

        imputation = self.impute(x, mask, supp)
        if not training:
            imputation = torch.where(mask, x, imputation)
            # imputation += resid
            return imputation
        else:
            y = torch.where(mask, x, imputation)
            x = imputation * (1 - mask)
            imputation_cyc = self.impute(x, 1 - mask, supp)

            # imputation += resid
            # imputation_cyc += resid
            return imputation, imputation_cyc, y

    def impute(self, x, mask, supp):
        b, s, n, c = x.size()
        imputation = self.relu(self.fc_1(x))  # bs, s, n, dim
        imputation = rearrange(imputation, 'b s n d -> b d n s')
        d = imputation.size(1)

        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)
        
        # for ind in range(len(supp_drop)):
        #     suppx = supp_drop[ind].expand(b, -1, -1)
        #     suppx = suppx.expand(s, -1, -1, -1)
        #     supp_drop[ind] = rearrange(suppx, 's b n m -> b n m s')

        imputation = self.relu(self.gcn_1(imputation, supp_drop))

        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)

        # for ind in range(len(supp_drop)):
        #     suppx = supp_drop[ind].expand(b, -1, -1)
        #     suppx = suppx.expand(s, -1, -1, -1)
        #     supp_drop[ind] = rearrange(suppx, 's b n m -> b n m s')
        
        imputation = self.relu(self.gcn_2(imputation, supp_drop))

        # cross reference
        # b d n s
        feat = imputation.clone()
        feat = rearrange(feat, 'b d n s -> (b s) n d')
        feat_mask = rearrange(mask, 'b s n d -> (b s) n d')

        cosine_eps = 1e-7
        q = feat.clone()  # b n d
        k = feat.clone().transpose(-2, -1)  # b n d
        q_norm = torch.norm(q, 2, 2, True)
        k_norm = torch.norm(k, 2, 1, True)

        # ========================================
        # Hard Transfer
        # Align each obs road with the most similar unobs road
        # Align each unobs road with the most similar obs road
        # ========================================
        cos_sim = torch.bmm(q, k) / (torch.bmm(q_norm, k_norm) + cosine_eps)  # b n n, [-1, 1]
        cos_sim = (cos_sim + 1.) / 2.  # [0, 1]

        cos_sim_max = cos_sim * feat_mask  # observed positions
        cos_sim_max_score, cos_sim_max_index = torch.max(cos_sim_max, dim=1)  # b n
        cos_sim_min = cos_sim * (1. - feat_mask)  # unobserved positions
        cos_sim_min_score, cos_sim_min_index = torch.max(cos_sim_min, dim=1)  # b n

        v = feat.clone().transpose(-2, -1)  # b d n
        v_unobs = self.bis(v, 2, cos_sim_max_index)  # find the most similar observed road for each unobserved road
        v_obs = self.bis(v, 2, cos_sim_min_index)  # find the most dissimilar unobserved road for each observed road
        v_unobs = v_unobs * cos_sim_max_score.unsqueeze(1)  # b d n
        v_obs = v_obs * cos_sim_min_score.unsqueeze(1)  # b d n

        v_unobs = rearrange(v_unobs, '(b s) d n -> b d n s', b=b, s=s)
        v_obs = rearrange(v_obs, '(b s) d n -> b d n s', b=b, s=s)

        feat_mask = rearrange(feat_mask, '(b s) n d -> b d n s', b=b, s=s)
        feat_transfer = v_unobs * (1. - feat_mask) + v_obs * feat_mask  # b d n s

        imputation = torch.cat([imputation, feat_transfer], dim=1)
        imputation = rearrange(imputation, 'b d n s -> b s n d')
        imputation = self.relu(self.smooth(imputation))
        imputation = rearrange(imputation, 'b s n d -> b d n s')

        # ========================================
        # Output
        # ========================================
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(self.t_dim, n), padding=(self.t_dim // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, self.t_dim * d, -1, s)  # b d n' s
        supp_drop = self.adj_drop(supp, mask)
        
        # for ind in range(len(supp_drop)):
        #     suppx = supp_drop[ind].expand(b, -1, -1)
        #     suppx = suppx.expand(s, -1, -1, -1)
        #     supp_drop[ind] = rearrange(suppx, 's b n m -> b n m s')
        
        imputation = self.relu(self.gcn_3(imputation, supp_drop))

        imputation = rearrange(imputation, 'b d n s -> b s n d')
        imputation = self.fc_2(imputation)  # b s n d
        return imputation

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)
    
    def predict(self, x, mask=None, known_set=None, sub_entry_num=None, reset=False, 
                edge_weight = None, edge_index = None, u = None, transform =None):
        """"""
        imputation = self.forward(x=x,
                                  mask=mask,
                                  known_set=known_set,
                                  sub_entry_num=sub_entry_num,
                                  reset=reset,
                                  training=False, 
                                  edge_weight = None, 
                                  edge_index = None, 
                                  u = None, 
                                  transform =None)
        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        return parser

from torch_geometric.nn.inits import glorot
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from torch_scatter import scatter

import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
import math
import time

from einops import rearrange

from tsl.nn.models.base_model import BaseModel


class RelTemporalEncoding(nn.Module):
    def __init__(self, n_hid, max_len=50, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()

        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = torch.cat([x_i, x_j], dim=1)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class EAConv(nn.Module):
    def __init__(self, dim, n_factors, agg_param, use_RTE=False):
        super(EAConv, self).__init__()

        assert dim % n_factors == 0
        self.d = dim
        self.k = n_factors
        self.delta_d = self.d // self.k
        self.dk = self.d - self.delta_d
        self._o_cache_zero_d = torch.zeros(1, self.d)
        self._o_cache_zero_dk = torch.zeros(1, self.dk)
        self._o_cache_zero_k = torch.zeros(1, self.k)
        self._o_cache_zero_kk = torch.zeros(1, self.k - 1)

        self._cache_zero_d = None
        self._cache_zero_dk = None
        self._cache_zero_k = None
        self._cache_zero_kk = None
        self.use_RTE = use_RTE
        self.rte = RelTemporalEncoding(self.d)
        self.agg_param = agg_param

    def time_encoding(self, x_all):
        if self.use_RTE:
            times = len(x_all)
            for t in range(times):
                x_all[t] = self.rte(x_all[t], torch.LongTensor([t]).to(x_all[t].device))
        return x_all

    def aggregate_former_v2(self, x, neighbors, max_iter):
        b, n, m = x.size(0), x.size(1), neighbors.size(0) // x.size(1)
        d, k, delta_d = self.d, self.k, self.delta_d
        x = F.normalize(x.view(b, n, k, delta_d), dim=3).view(b, n, d)
        z = torch.cat([x, self._cache_zero_d], dim=1)
        z = z[:, neighbors].view(b, n, m, k, delta_d)
        u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(b, n * m, k).view(b, n, m, k)
            else:
                p = torch.sum(z * u.view(b, n, 1, k, delta_d), dim=4)
            p = F.softmax(p, dim=3)
            u = torch.sum(z * p.view(b, n, m, k, 1), dim=2)
            u += x.view(b, n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = F.normalize(u, dim=3)
        u = u.view(b, n, k * delta_d)
        return u

    def forward(self, x_all, neighbors_all, max_iter):
        dev = x_all[0].device
        b = x_all[0].shape[0]
        n = x_all[0].shape[1]
        m = len(neighbors_all[0][0])
        self._cache_zero_d = (self._o_cache_zero_d.to(dev)).unsqueeze(0).repeat(b, 1, 1)
        self._cache_zero_dk = (self._o_cache_zero_dk.to(dev)).unsqueeze(0).repeat(b, 1, 1)
        self._cache_zero_k = (self._o_cache_zero_k.to(dev)).unsqueeze(0).repeat(b, 1, 1)
        self._cache_zero_kk = (self._o_cache_zero_kk.to(dev)).unsqueeze(0).repeat(b, 1, 1)
        times = len(x_all)  # nums of time slices
        emb = torch.zeros((times, b, n, self.d)).to(dev)

        for t in range(times):
            x_temp = self.aggregate_former_v2(
                x_all[t], neighbors_all[t].view(-1), max_iter
            )
            if t > 0:
                weights = F.sigmoid(
                    torch.tensor(list(range(t))).view(t, 1, 1, 1).to(x_all[0].device)
                )
                emb[t] = (
                    torch.sum(weights * emb[:t], dim=0) / t
                ) * self.agg_param + x_temp * (1 - self.agg_param)
            else:
                emb[t] = x_temp
            emb[t] = emb[t].view(b, n, self.d)

        return emb.to(dev)


class EAGLE(BaseModel):
    def __init__(self, 
                 n_layers,
                 n_factors,
                 delta_d,
                 nfeat,
                 maxiter,
                 use_RTE,
                 agg_param,
                 num_nodes, 
                 dropout,
                 d_for_cvae,
                 interv_size_ratio,
                 gen_ratio,
                 output_size):
        super(EAGLE, self).__init__()

        self.n_layers = n_layers
        self.n_factors = n_factors
        self.delta_d = delta_d
        self.in_dim = nfeat
        self.hid_dim = self.n_factors * self.delta_d
        self.maxiter = maxiter
        self.use_RTE = use_RTE
        self.agg_param = agg_param
        self.interv_size_ratio = interv_size_ratio
        self.gen_ratio = gen_ratio
        self.d_for_cvae = d_for_cvae
        self.feat = Parameter(
            (torch.ones(num_nodes, nfeat)), requires_grad=True
        )
        self.linear = nn.Linear(self.in_dim, self.hid_dim)
        self.layers = nn.ModuleList(
            EAConv(self.hid_dim, self.n_factors, self.agg_param, self.use_RTE)
            for i in range(self.n_layers)
        )
        self.relu = F.relu
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = dropout
        self.reset_parameter()
        self.ecvae = ECVAE(delta_d, n_factors, d_for_cvae)
        self.decoder = nn.Linear(self.hid_dim, output_size)

    def reset_parameter(self):
        glorot(self.feat)

    def vae_model(self, x, y):
        return self.ecvae(x, y)
    
    def loss_cvae(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss
    
    def cal_fact_var(self, x_all):
        b = x_all.shape[0]
        n = x_all.shape[2]
        k = self.n_factors
        x_all_trans = x_all.permute(0, 2, 3, 1, 4)
        points = torch.var(x_all_trans, dim=[3, 4]).view(b, n, k)
        return points
    
    def cal_mask_faster(self, x_m):
        var = self.cal_fact_var(x_m)

        def max_avg_diff_index(sorted_tensor):
            ds, n, k = sorted_tensor.shape
            # Cumulative sum along the last dimension
            cumsum = torch.cumsum(sorted_tensor, dim=2)
            # Create index tensors for dividing by j and (k - j)
            j = torch.arange(1, k, device=sorted_tensor.device).view(1, 1, -1)  # shape (1, 1, k-1)
            total = cumsum[:, :, -1].unsqueeze(-1)  # shape (ds, n, 1)
            # avg1 = sum(row[:j]) / j
            avg1 = cumsum[:, :, :-1] / j
            # avg2 = (total - cumsum[:, :, :-1]) / (k - j)
            avg2 = (total - cumsum[:, :, :-1]) / (k - j)
            # abs difference
            diff = torch.abs(avg1 - avg2)
            # Find index of maximum diff
            max_diff, max_index = torch.max(diff, dim=2)
            # Adjust as in original: subtract 1 to match `max_index - 1`
            result = max_index - 1
            return result.float()

        var_sorted = torch.sort(var, dim=2)
        var_sorted_index = var_sorted.indices
        indices = max_avg_diff_index(var_sorted.values).to(device=var.device, dtype=int)

        B, N, D = var.shape
        range_tensor = torch.arange(D, device=var.device).view(1, 1, D)  # shape (1, 1, D)
        indices_expanded = indices.unsqueeze(-1)                         # shape (B, N, 1)
        mask = (range_tensor <= indices_expanded).to(var.dtype)  

        var = var.scatter(dim=2, index=var_sorted_index, src=mask)  
        # for i in range(var.shape[1]):
        #     sort_indices = var_sorted_index[:, i]
        #     values = var[:, i, sort_indices]
        #     mask = torch.zeros_like(values)
        #     mask[:, :indices[i] + 1] = 1
        #     var[:, i, sort_indices] = mask

        return var

    def saved_env(self, x_all, mask):
        batch = x_all.shape[0]
        times = x_all.shape[1]
        n = x_all.shape[2]
        k = self.n_factors
        d = self.hid_dim
        delta_d = self.delta_d
        m = mask.shape[1]
        x_all = x_all.view(batch, times, n, d)
        mask_env = 1 - mask
        mask_expand = torch.repeat_interleave(mask_env, delta_d, dim=2).view(
            batch, m, k * delta_d
        )
        mask_expand = torch.stack([mask_expand] * times)
        mask_expand = rearrange(mask_expand, 't b n d -> b t n d')

        extract_env = (
            (x_all * mask_expand).view(batch, times, n, k, delta_d).permute(0, 3, 1, 2, 4)
        )
        extract_env = rearrange(extract_env, 'a b c d e -> a b (c d) e')
        extract_env = extract_env[:, :, torch.randperm(times * n), :]
        for i in range(k):
            zero_rows = (extract_env[:, i].sum(dim=2) == 0).nonzero(as_tuple=True)[0]
            non_zero_rows = (extract_env[:, i].sum(dim=2) != 0).nonzero(as_tuple=True)[0]
            if non_zero_rows.shape[0] > 0:
                replacement_rows = non_zero_rows[
                    torch.randint(0, non_zero_rows.shape[0], (zero_rows.shape[0],))
                ]
                extract_env[:, i, zero_rows] = extract_env[:, i, replacement_rows]

        return extract_env.view(batch, times, n, d)

    def gen_env(self, extract_env):
        batch = extract_env.shape[0]
        times = extract_env.shape[1]
        n = extract_env.shape[2]
        k = self.n_factors
        d = self.hid_dim
        delta_d = self.delta_d
        n_gen = max(int(self.gen_ratio * n), 1)

        z = torch.randn(batch, n_gen * k, self.d_for_cvae).to(extract_env.device)
        y = torch.ones(batch, n_gen, k)
        for i in range(k):
            y[:, :, i : i + 1] = y[:, :, i : i + 1] * i
        y_T = y.transpose(1, 2)
        y = (F.one_hot(y_T.reshape(batch, -1).to(torch.int64))).to(extract_env.device)
        gen_env = self.ecvae.decode(z, y).view(batch, n_gen, k * delta_d)

        random_indices = torch.randperm(n)[:n_gen]
        extract_env = rearrange(extract_env, 'b t n d -> t b n d')
        extract_env[:, :, random_indices] = gen_env
        extract_env = rearrange(extract_env, 't b n d -> b t n d')
        return extract_env.view(batch, times, n, d)

    def intervention_final(self, x_all_original):
        x_all = x_all_original.clone()
        batch = x_all.shape[0]
        times = x_all.shape[1]
        n = x_all.shape[2]
        k = self.n_factors
        d = self.hid_dim
        delta_d = self.delta_d
        m = int(self.interv_size_ratio * n)
        indices = torch.randperm(n)[:m]
        x_m = x_all[:, :, indices, :].view(batch, times, m, k, delta_d)
        mask = self.cal_mask_faster(x_m)
        mask_expand = torch.repeat_interleave(mask, delta_d, dim=2).view(-1, m, k * delta_d)
        mask_expand = torch.stack([mask_expand] * times)
        mask_expand = rearrange(mask_expand, 't b n d -> b t n d')

        saved_env = self.saved_env(x_m, mask)
        sampled_env = self.gen_env(saved_env)

        x_m = x_m.view(batch, times, m, k * delta_d)
        embeddings_interv = x_m * mask_expand + sampled_env * (1 - mask_expand)
        x_all[:, :, indices, :] = embeddings_interv.to(torch.float32)
        return x_all, indices

    def forward(self, edge_index_list, x_list, neighbors_all, edge_index, transform):
        times = len(edge_index_list)
        if x_list is None:
            x_list = [self.linear(self.feat) for i in range(len(edge_index_list))]
        else:
            x_list = [self.linear(x) for x in x_list]

        for i, layer in enumerate(self.layers):
            x_list = layer(x_list, neighbors_all, self.maxiter)
            if i != len(self.layers) - 1:
                x_list = x_list.view(
                    len(x_list), x_list[0].shape[0], x_list[0].shape[1], self.n_factors, self.delta_d
                )
                x_list = self.LeakyReLU(x_list)
                x_list = [
                    F.dropout(
                        input=F.normalize(x, dim=3),
                        p=self.dropout,
                        training=self.training,
                    )
                    for x in x_list
                ]

        # x_list = torch.stack(x_list)
        embeddings = rearrange(x_list, 't b n d -> b t n d')
        batch = embeddings.shape[0]
        # fin_x_list = self.decoder(F.relu(x_list))

        y = torch.ones(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2], self.n_factors)
        for i in range(self.n_factors):
            y[:, :, i : i + 1] = y[:, :, i : i + 1] * i
        y_T = y.transpose(1, 2)
        y = (F.one_hot(y_T.reshape(batch, -1).to(torch.int64))).to(x_list.device)

        embeddings_view = embeddings.view(
            batch, embeddings.shape[1], embeddings.shape[2], self.n_factors, self.delta_d
        )
        embeddings_trans = embeddings_view.permute(0, 3, 1, 2, 4)
        x_flatten = torch.flatten(embeddings_trans, start_dim=1, end_dim=3)
        recon, mu, log_std = self.ecvae(x_flatten, y)
        cvae_loss = self.loss_cvae(recon, x_flatten, mu, log_std) / (
            len(embeddings[0] * len(embeddings[0]))
        )
        fin_x_list = self.decoder(F.relu(embeddings))
        return fin_x_list, embeddings, cvae_loss


class ECVAE(nn.Module):
    def __init__(self, delta_d, n_factors, d_for_cvae):
        super(ECVAE, self).__init__()

        delta_d = delta_d
        n_factors = n_factors
        latent_size = d_for_cvae
        self.fc1_mu = nn.Linear(delta_d + n_factors, latent_size)
        self.fc1_log_std = nn.Linear(delta_d + n_factors, latent_size)
        self.fc2 = nn.Linear(latent_size + n_factors, delta_d)

    def encode(self, x, y):
        h1 = F.relu(torch.cat([x, y], dim=2))
        mu = self.fc1_mu(h1)
        log_std = self.fc1_log_std(h1)
        return mu, log_std

    def decode(self, z, y):
        h3 = F.relu(torch.cat([z, y], dim=2))
        recon = self.fc2(h3)
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, y):
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        return recon, mu, log_std

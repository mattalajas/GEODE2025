from torch_geometric.nn.inits import glorot
from torch import nn
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_scatter import scatter
import torch.nn.functional as F
import torch
import math

from einops import rearrange

from tsl.nn.models.base_model import BaseModel


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
        return x + self.lin(self.emb(t))


class DGNNLayer(nn.Module):
    """Our proposed Disentangled Spatio-temporal Graph Attention Layer.
    """

    def __init__(self, in_dim, hid_dim, n_heads, norm=True, dropout=0, 
                 skip=False, use_RTE=False, sample_r=0.1, use_fmask=False, 
                 only_causal=False):
        super(DGNNLayer, self).__init__()
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.node_dim = 1
        self.norm = norm
        self.skip = skip
        self.use_RTE = use_RTE
        self.use_fmask = use_fmask
        self.in_dim, self.hid_dim = in_dim, hid_dim
        self.aggr = 'add'

        self.q_linear = nn.Linear(in_dim, hid_dim)
        self.k_linear = nn.Linear(in_dim, hid_dim)
        self.v_linear = nn.Linear(in_dim, hid_dim)

        self.update_norm = nn.LayerNorm(hid_dim)
        self.update_linear = nn.Linear(hid_dim, hid_dim)
        self.update_drop = nn.Dropout(dropout)
        self.update_skip = nn.Parameter(torch.ones(1))

        self.time_emb = RelTemporalEncoding(hid_dim)
        self.cs_mlp = nn.Sequential(nn.Linear(hid_dim, 2*hid_dim), nn.GELU(), nn.Linear(2*hid_dim, hid_dim))

        self.fmask = nn.Parameter(torch.ones(hid_dim))

        self.only_causal = only_causal

    def collect_neighbors(self, edge_index_list, x_list):
        """
        for each time t : collect edges of 1:t for aggregation
        return [x_tar,topos]
        """
        # x_list: list of x embeddings over time (twin horizon)
        twin = len(x_list)
        for t_tar in range(twin):
            topos = []
            x_tar = x_list[t_tar]                                                       # features in t_tar
            for t_src in range(t_tar+1): # Goes thru past times
                # aggregate from tsrc -> ttar
                ei_src = edge_index_list[t_src]                                         # edges in t_src
                x_tar_e = x_list[t_tar].index_select(self.node_dim, ei_src[1, :])     # features for edge target node in time t_tar
                x_src_e = x_list[t_src].index_select(self.node_dim, ei_src[0, :])     # features for edge source node in time t_src
                ei_tar = ei_src[1, :].T                                                 # edge target nodes id
                topo = [x_tar_e, x_src_e, t_tar, t_src, ei_tar] # Tar and src: node embeddings of the 
                topos.append(topo)
            yield x_tar, topos

    def DAttn(self, x_tar, x_src, t_tar, t_src):
        """ do attention
        return attention,message
        """
        B, N, D = x_tar.shape

        # to sparse
        target_node_vec = x_tar
        source_node_vec = x_src

        # add RTE
        if self.use_RTE:
            device = x_tar.device
            target_node_vec = self.time_emb(target_node_vec, torch.LongTensor([t_tar]).to(device))
            source_node_vec = self.time_emb(source_node_vec, torch.LongTensor([t_src]).to(device))

        # get weights
        q_linear, k_linear, v_linear = self.q_linear, self.k_linear, self.v_linear
        # calculation
        q_mat = q_linear(target_node_vec).view(B, N, self.n_heads, self.d_k)
        k_mat = k_linear(source_node_vec).view(B, N, self.n_heads, self.d_k) 
        v_mat = v_linear(source_node_vec).view(B, N, self.n_heads, self.d_k)   # [E,h,F/h]

        res_att = (q_mat * k_mat).sum(dim=-1) / self.sqrt_dk  # [E,h]
        res_msg = v_mat  # [E,h,F/h]
        return res_att, res_msg

    def DAttnMulti(self, x_tar, topos):
        """x_tar attends to st neighbors in topos
        return updated x_tar
        """
        # Message Passing
        res_atts = []
        res_msgs = []
        ei_tars = []
        for (x_tar_e, x_src_e, t_tar, t_src, ei_tar) in topos:
            att, msg = self.DAttn(x_tar_e, x_src_e, t_tar, t_src)
            res_atts.append(att)
            res_msgs.append(msg)
            ei_tars.append(ei_tar)
        res_att = torch.cat(res_atts, dim=1)
        res_msg = torch.cat(res_msgs, dim=1)
        ei_tar = torch.cat(ei_tars)

        res_att = softmax(res_att, ei_tar, dim=1)                     # [ET,h]
        b, t = res_att.shape[0], res_att.shape[1]
        res = res_msg * res_att.view(b, t, self.n_heads, 1)    # [E,h,F/h]
        res = res.view(b, t, self.hid_dim)  # [E,F]

        spu_att = softmax(-res_att, ei_tar, dim=1)
        b, t = spu_att.shape[0], spu_att.shape[1]
        spu = res_msg * spu_att.view(b, t, self.n_heads, 1)    # [E,h,F/h]
        spu = spu.view(b, t, self.hid_dim)  # [E,F]

        # disentangle
        causal_hat = scatter(res, ei_tar, dim=self.node_dim, dim_size=x_tar.shape[1], reduce=self.aggr)  # [N,F]
        spurious_hat = scatter(spu, ei_tar, dim=self.node_dim, dim_size=x_tar.shape[1], reduce=self.aggr)  # [N,F]

        if self.use_fmask:
            fmask_c = F.softmax(self.fmask, dim=0)
            causal_hat = causal_hat * fmask_c

        def ffn(x):

            if self.norm:
                res = self.cs_mlp(self.update_norm(x))
            else:
                res = self.cs_mlp(x)

            res = self.update_drop(res)

            if self.skip:
                alpha = torch.sigmoid(self.update_skip)
                res = (1-alpha)*x + alpha*res
            else:
                res = x + res

            return res

        causal = ffn(causal_hat+x_tar)
        spurious = ffn(spurious_hat)

        if self.only_causal:
            res = causal
        else:
            res = causal+spurious

        return res, causal, spurious

    def forward(self, edge_index_list, x_list):
        xs = []
        cs = []
        ss = []
        for t_tar, (x_tar, topos) in enumerate(self.collect_neighbors(edge_index_list, x_list)): # Do this for each time step
            x, c, s = self.DAttnMulti(x_tar, topos)
            xs.append(x)
            cs.append(c)
            ss.append(s)
        return xs, cs, ss


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
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
        x = torch.concat([x_i, x_j], dim=1)
        # x = x_i*x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x).squeeze()


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()
        # self.dum=Parameter(torch.ones(1), requires_grad=True)
    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class DGNN(BaseModel):
    """Our proposed Disentangled Dynamic Graph Attention Networks
    """

    def __init__(self, 
                 n_layers, 
                 heads, 
                 norm, 
                 num_nodes,
                 nfeat,
                 hidden_size,
                 lin_bias,
                 dropout, 
                 skip, 
                 use_RTE,
                 fmask,
                 output_size,
                 only_causal):
        
        super(DGNN, self).__init__()
        n_layers = n_layers
        n_heads = heads
        norm = norm

        self.feat = Parameter((torch.ones(num_nodes, nfeat)), requires_grad=True)
        in_dim = nfeat
        self.linear = nn.Linear(in_dim, hidden_size, bias=bool(lin_bias))
        self.layers = nn.ModuleList(DGNNLayer(hidden_size, hidden_size, n_heads=n_heads, norm=norm, dropout=dropout, 
                                              skip=skip, use_RTE=use_RTE, sample_r=0.1, use_fmask=fmask, 
                                              only_causal=only_causal) for i in range(n_layers))
        self.act = F.relu
        self.reset_parameter()

        self.cs_decoder = nn.Linear(hidden_size, output_size)
        self.ss_decoder = nn.Linear(hidden_size, output_size)

        self.fmask = nn.Parameter(torch.ones(in_dim))

    def reset_parameter(self):
        glorot(self.feat)

    def forward(self, x_list, edge_index_list, edge_index, transform):
        if x_list is None:
            x_list = [self.linear(self.feat) for i in range(len(edge_index_list))]
        else:
            x_list = [self.linear(x) for x in x_list]
        for i, layer in enumerate(self.layers):
            x_list, cs, ss = layer(edge_index_list, x_list)
            if i != len(self.layers)-1:
                x_list = [self.act(x) for x in x_list]

        cs = torch.stack(cs)
        ss = torch.stack(ss)

        cs = rearrange(cs, 't b n d -> b t n d')
        ss = rearrange(ss, 't b n d -> b t n d')

        cs = self.cs_decoder(F.relu(cs))
        ss = self.ss_decoder(F.relu(ss))

        ss = ss.detach()
        return cs, ss

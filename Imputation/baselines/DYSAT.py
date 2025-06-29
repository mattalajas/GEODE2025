# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Time    :   2021/02/18 14:30:13
@Author  :   Fei gao 
@Contact :   feig@mail.bnu.edu.cn
BNU, Beijing, China
'''
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch.nn.modules.loss import BCEWithLogitsLoss
from einops import rearrange
from tsl.nn.models.base_model import BaseModel

class DySAT(BaseModel):
    def __init__(self, structural_head_config, structural_layer_config, 
                 temporal_head_config, temporal_layer_config, 
                 spatial_drop, temporal_drop, num_features, time_length, residual=True):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.num_time_steps = time_length
        self.num_features = num_features
        self.residual = residual

        self.structural_head_config = list(map(int, structural_head_config.split(",")))
        self.structural_layer_config = list(map(int, structural_layer_config.split(",")))
        self.temporal_head_config = list(map(int, temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, temporal_layer_config.split(",")))
        self.spatial_drop = spatial_drop
        self.temporal_drop = temporal_drop

        self.structural_attn, self.temporal_attn = self.build_model()
        self.bceloss = BCEWithLogitsLoss()

    def forward(self, x_list, edge_index, edge_weights, transform):
        # B T N F
        # Structural Attention forward
        structural_out = []
        for t in range(0, self.num_time_steps):
            structural_out.append(self.structural_attn([x_list[t], edge_index, edge_weights]))
        # structural_outputs = [g.x[:,None,:] for g in structural_out] # list of [Ni, 1, F]
        structural_outputs = torch.stack(structural_out)
        structural_outputs_fin = rearrange(structural_outputs, 't b n d -> b n t d')

        # # padding outputs along with Ni
        # maximum_node_num = structural_outputs[-1].shape[0]
        # out_dim = structural_outputs[-1].shape[-1]
        # structural_outputs_padded = []
        # for out in structural_outputs:
        #     zero_padding = torch.zeros(maximum_node_num-out.shape[0], 1, out_dim).to(out.device)
        #     padded = torch.cat((out, zero_padding), dim=0)
        #     structural_outputs_padded.append(padded)
        # structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1) # [N, T, F]
        
        # Temporal Attention forward
        temporal_out = self.temporal_attn(structural_outputs_fin) # B N T F
        out = rearrange(temporal_out, 'b n t d -> b t n d')
        out = self.output(out)
    
        return out 

    def build_model(self):
        input_dim = self.num_features

        # 1: Structural Attention Layers
        structural_attention_layers = nn.Sequential()
        for i in range(len(self.structural_layer_config)):
            layer = StructuralAttentionLayer(input_dim=input_dim,
                                             output_dim=self.structural_layer_config[i],
                                             n_heads=self.structural_head_config[i],
                                             attn_drop=self.spatial_drop,
                                             ffd_drop=self.spatial_drop,
                                             residual=self.residual)
            structural_attention_layers.add_module(name="structural_layer_{}".format(i), module=layer)
            input_dim = self.structural_layer_config[i]
        
        # 2: Temporal Attention Layers
        input_dim = self.structural_layer_config[-1]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]

        self.output = nn.Linear(input_dim, self.num_features)
        return structural_attention_layers, temporal_attention_layers

class StructuralAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                output_dim, 
                n_heads, 
                attn_drop, 
                ffd_drop,
                residual):
        super(StructuralAttentionLayer, self).__init__()
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.ffd_drop = nn.Dropout(ffd_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, batch):
        og_x, edge_index, edge_weight = batch
        edge_weight = edge_weight.unsqueeze(-1)
        H, C = self.n_heads, self.out_dim
        x = self.lin(og_x).view(-1, H, C) # [N, heads, out_dim]

        # attention
        alpha_l = (x * self.att_l).sum(dim=-1) # [N, heads]
        alpha_r = (x * self.att_r).sum(dim=-1)
        alpha_l = alpha_l[edge_index[0]] # [num_edges, heads]
        alpha_r = alpha_r[edge_index[1]]
        alpha = alpha_r + alpha_l
        alpha = edge_weight * alpha
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1], num_nodes=og_x.shape[1]) # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x = self.ffd_drop(x)
        x_j = x[edge_index[0]]  # [num_edges, heads, out_dim]

        # output
        out = self.act(scatter(x_j * coefficients.unsqueeze(-1), edge_index[1], dim=0, dim_size=og_x.shape[1], reduce="sum"))
        out = out.reshape(-1, H * C) #[num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(og_x)
        fin_o = out
        return fin_o

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

        
class TemporalAttentionLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        B, N, T, D = inputs.shape
        H = self.n_heads
        D_head = D // H
        
        # 1: Add position embeddings to input
        position_ids = torch.arange(T, device=inputs.device).unsqueeze(0).unsqueeze(0)
        pos_embed = self.position_embeddings[position_ids]
        temporal_inputs = inputs + pos_embed # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([3],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([3],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([3],[0])) # [N, T, F]

        # 3: Split heads -> [B*N, H, T, D_head]
        def split_heads(x):
            x = x.view(B * N, T, H, D_head).transpose(1, 2)  # [B*N, H, T, D_head]
            return x.reshape(B * N * H, T, D_head)           # [B*N*H, T, D_head]

        q_ = split_heads(q)
        k_ = split_heads(k)
        v_ = split_heads(v)
        
        attn_scores = torch.matmul(q_, k_.transpose(1, 2)) / (D_head ** 0.5)  # [B*N*H, T, T]

        # 4: Masked (causal) softmax to compute attention weights.
        mask = torch.tril(torch.ones(T, T, device=inputs.device)).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*N*H, T, T]
        self.attn_wts_all = attn_weights # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        if self.training:
            attn_weights = self.attn_dp(attn_weights)

        attn_output = torch.matmul(attn_weights, v_)  # [B*N*H, T, D_head]

        attn_output = attn_output.view(B * N, H, T, D_head).transpose(1, 2)  # [B*N, T, H, D_head]
        attn_output = attn_output.reshape(B, N, T, D)  # [B, N, T, F]
        
        # 7: Feedforward + Residual
        out = self.feedforward(attn_output)
        if self.residual:
            out = out + temporal_inputs  # broadcasting okay: both [B, N, T, F]

        return out  # [B, N, T, F]

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)

from torch import Tensor, nn

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import EvolveGCN, MLP
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog

import torch
import torch.nn.functional as F
from einops import rearrange


class INCREASE(BaseModel):
    return_type = Tensor

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon,
                 K):
        super(INCREASE, self).__init__()
        self.horizon = horizon
        self.fc1 = MLP(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')

        self.fc11 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        self.fc12 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        
        self.fc21 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        
        self.fc22 = nn.Linear(in_features=hidden_size,
                              out_features=hidden_size)
        
        self.fc31 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        self.fc32 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        
        self.fc4 = MLP(input_size=horizon,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')

        self.fc5 = MLP(input_size=hidden_size,
                       hidden_size=hidden_size,
                       output_size=hidden_size,
                       activation='relu')
        
        self.gru = nn.GRUCell(input_size=hidden_size*2,
                              hidden_size=hidden_size*2)
        self.gru_ff = MLP(input_size=hidden_size*2,
                          hidden_size=hidden_size*2,
                          output_size=hidden_size*2,
                          activation='relu')

        self.fc61 = MLP(input_size=hidden_size*2,
                       hidden_size=hidden_size*2,
                       output_size=hidden_size*2,
                       activation='tanh')
        self.fc62 = nn.Linear(in_features=hidden_size*2,
                              out_features=output_size,
                              bias=False)
        
        self.fc71 = MLP(input_size=hidden_size*2,
                       hidden_size=hidden_size*2,
                       output_size=hidden_size*2,
                       activation='relu')
        self.fc72 = nn.Linear(in_features=hidden_size*2,
                              out_features=output_size)
    def forward(self, x_gp, gp, TE, T, transform):
        """"""
        N_target = x_gp.shape[0]
        h = x_gp.shape[1]
        K = gp.shape[-1]
        # input
        x_gp = self.fc1(x_gp)
        # spatial aggregation
        gp = gp.repeat(1, h, 1, 1) 
        y_gp = torch.matmul(gp, x_gp)
        x_gp = self.fc11(x_gp)
        y_gp = self.fc12(y_gp)

        x_gp = torch.abs(y_gp - x_gp)
        x_gp = torch.matmul(gp, x_gp)

        x_gp = F.tanh(self.fc21(x_gp))
        y_gp = self.fc22(y_gp)

        y_gp = x_gp + y_gp
        x_gp = self.fc31(x_gp)
        y_gp = self.fc32(y_gp)

        TE = TE.squeeze(0).long()
        TE = F.one_hot(TE, num_classes=T).float()
        TE = self.fc4(TE).unsqueeze(0)
        TE = TE.repeat(N_target, 1, 1) 

        y_gp = y_gp.squeeze(2)
        x_gp = x_gp.squeeze(2)

        g1_gp = F.relu(self.fc5(x_gp))
        g1_gp = 1 / torch.exp(g1_gp)
        y_gp = g1_gp * y_gp
        y_gp = torch.cat([y_gp, TE], dim=-1)

        pred_gp = []
        b = y_gp.shape[1] // self.horizon
        gru_gp = rearrange(y_gp, 'n (b t) d -> (n b) t d', b=b)

        state = torch.zeros((N_target*b, y_gp.shape[-1])).to(x_gp.device)
        for i in range(self.horizon):
            g2_gp = F.relu(self.gru_ff(state))
            g2_gp = 1 / torch.exp(g2_gp)
            state_gp = g2_gp * state
            state_gp = self.gru(gru_gp[:, i], state_gp)
            pred_gp.append(state_gp)
        
        pred_gp = torch.stack(pred_gp, dim=1)
        pred_gp = rearrange(pred_gp, '(n b) t d -> n (b t) d', b=b)

        pred = pred_gp.unsqueeze(2)

        # Attention
        a = self.fc62(self.fc61(pred))
        a = F.softmax(torch.permute(a, (0, 1, 3, 2)))
        x = torch.matmul(a, pred).squeeze(2)

        out = F.relu(self.fc71(x))
        out = self.fc72(out)
        out = rearrange(out, 'n (b t) d -> b t n d', b=b)
        return out

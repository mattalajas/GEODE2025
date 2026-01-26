from einops import rearrange
from torch import Tensor, nn
import torch
from torch_geometric.typing import Adj, OptTensor

from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.blocks.encoders import DCRNN, ConditionalBlock
from tsl.nn.models.base_model import BaseModel
from tsl.nn.blocks.encoders.mlp import MLP


class DCRNNModel(BaseModel):
    r"""The Diffusion Convolutional Recurrent Neural Network from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Differently from the original implementation, the recurrent decoder is
    substituted with a fixed-length nonlinear readout.

    Args:
        input_size (int): Number of features of the input sample.
        output_size (int): Number of output channels.
        horizon (int): Number of future time steps to forecast.
        exog_size (int): Number of features of the input covariate,
            if any. (default: :obj:`0`)
        hidden_size (int): Number of hidden units.
            (default: :obj:`32`)
        kernel_size (int): Order of the spatial diffusion process.
            (default: :obj:`2`)
        ff_size (int): Number of units in the nonlinear readout.
            (default: :obj:`256`)
        n_layers (int): Number of DCRNN cells.
            (default: :obj:`1`)
        dropout (float): Dropout probability.
            (default: :obj:`0`)
        activation (str): Activation function in the readout.
            (default: :obj:`'relu'`)
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 hidden_size: int = 32,
                 kernel_size: int = 2,
                 ff_size: int = 256,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 activation: str = 'relu'):
        super(DCRNNModel, self).__init__()
        self.input_encoder = nn.Linear(input_size, hidden_size)

        self.dcrnn = DCRNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           k=kernel_size,
                           return_only_last_state=True)

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=output_size,
                                  horizon=horizon,
                                  activation=activation,
                                  dropout=dropout)

    def forward(self,
                transform,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        x = self.input_encoder(x)   

        out = self.dcrnn(x,
                         edge_index,
                         edge_weight)
        return self.readout(out)

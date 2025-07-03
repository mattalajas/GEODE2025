from torch import Tensor, nn
import torch

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import MLP
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog

from torch import nn

from tsl.nn.layers.recurrent import EvolveGCNHCell, EvolveGCNOCell


class EvolveGCNMod(nn.Module):
    r"""EvolveGCN encoder from the paper `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ (Pereja et al., AAAI 2020).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        n_layers (int): Number of layers in the encoder.
        asymmetric_norm (bool): Whether to consider the input graph as directed.
        variant (str): Variant of EvolveGCN to use (options: 'H' or 'O')
        root_weight (bool): Whether to add a parametrized skip connection.
        cached (bool): Whether to cache normalized edge_weights.
        activation (str): Activation after each GCN layer.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 norm,
                 variant='H',
                 root_weight=False,
                 cached=False,
                 activation='relu'):
        super(EvolveGCNMod, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_cells = nn.ModuleList()
        if variant == 'H':
            cell = EvolveGCNHCell
        elif variant == 'O':
            cell = EvolveGCNOCell
        else:
            raise NotImplementedError

        for i in range(self.n_layers):
            self.rnn_cells.append(
                cell(in_size=self.input_size if i == 0 else self.hidden_size,
                     out_size=self.hidden_size,
                     norm=norm,
                     activation=activation,
                     root_weight=root_weight,
                     cached=cached))

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # x : b t n f
        steps = x.size(1)
        h = [None] * len(self.rnn_cells)
        outs = []
        for t in range(steps):
            out = x[:, t]
            for c, cell in enumerate(self.rnn_cells):
                out, h[c] = cell(out, h[c], edge_index, edge_weight)
            
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs


class EvolveGCNModel(BaseModel):
    r"""The EvolveGCN model from the paper `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ (Pereja et al., AAAI 2020).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        output_size (int): Size of the output.
        horizon (int): Forecasting steps.
        exog_size (int): Size of the optional exogenous variables.
        n_layers (int): Number of layers in the encoder.
        asymmetric_norm (bool): Whether to consider the input graph as directed.
        root_weight (bool): Whether to add a parametrized skip connection.
        cached (bool): Whether to cache normalized edge_weights.
        variant (str): Variant of EvolveGCN to use (options: 'H' or 'O')
        activation (str): Activation after each GCN layer.
    """

    return_type = Tensor

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 n_layers,
                 horizon,
                 norm,
                 root_weight,
                 variant='H',
                 activation='relu'):
        super(EvolveGCNModel, self).__init__()
        self.input_encoder = nn.Sequential(nn.Linear(input_size,
                                                     hidden_size), )

        self.encoder = EvolveGCNMod(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 n_layers=n_layers,
                                 norm=norm,
                                 variant=variant,
                                 activation=activation,
                                 root_weight=root_weight)

        self.readout = MLP(input_size=hidden_size,
                           hidden_size=hidden_size,
                           output_size=output_size,
                           activation='relu')

    def forward(self, x, edge_index, transform, edge_weight=None):
        """"""
        # x: [batches steps nodes features]
        x = self.input_encoder(x)
        x = self.encoder(x, edge_index, edge_weight)
        out = self.readout(x)
        return out

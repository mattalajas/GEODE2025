from torch import Tensor, nn

from tsl.nn.blocks.decoders import LinearReadout
from tsl.nn.blocks.encoders import EvolveGCN
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog


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

        self.encoder = EvolveGCN(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 n_layers=n_layers,
                                 norm=norm,
                                 variant=variant,
                                 activation=activation,
                                 root_weight=root_weight)

        self.readout = LinearReadout(hidden_size, output_size, horizon)

    def forward(self, x, edge_index, transform, edge_weight=None):
        """"""
        # x: [batches steps nodes features]
        x = self.input_encoder(x)
        x = self.encoder(x, edge_index, edge_weight)
        out = self.readout(x)
        return out

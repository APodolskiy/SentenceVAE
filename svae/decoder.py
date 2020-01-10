from typing import Optional, Iterable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from svae import RNN_TYPES
from svae.utils.modules import ResidualBlock


class RNNDecoder(nn.Module):
    def __init__(self,
                 rnn_type: str = 'gru',
                 input_size: int = 128,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0,
                 pad_value: int = 0):
        super().__init__()
        assert rnn_type in RNN_TYPES
        self.type = rnn_type
        self.num_layers = num_layers
        self.pad_value = pad_value
        rnn_class = RNN_TYPES[self.type]
        self.rnn = rnn_class(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=self.num_layers,
                             dropout=dropout,
                             bidirectional=False)

    # TODO: try out different conditioning schemes
    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor,
                init_state: Optional[torch.Tensor] = None):
        x_packed = pack_padded_sequence(x, lengths)
        out, h_n = self.rnn(x_packed, init_state)
        out, _ = pad_packed_sequence(out, padding_value=self.pad_value)
        return out, h_n


class ConvDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 1024,
                 residual_size: int = 512,
                 kernel_size: int = 3,
                 dilation_sizes: Iterable[int] = (1, 2, 4),
                 dropout: float = 0.1):
        super().__init__()
        self.modules = nn.ModuleList()
        for dilation_size in dilation_sizes:
            self.modules.append(ResidualBlock(
                in_channels=input_size,
                residual_channels=residual_size,
                kernel_size=kernel_size,
                dilation=dilation_size
            ))

    def forward(self, x: torch.Tensor):
        # (s, b, u) -> (b, u, s)
        x = x.permute(1, 2, 0)
        for module in self.modules:
            x = module(x)
        # (b, u, s) -> (s, b, u)
        x = x.permute(2, 0, 1)
        return x

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNDecoder(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0, pad_value: int = 0):
        super().__init__()
        self.num_layers = num_layers
        self.pad_value = pad_value
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=self.num_layers,
                          dropout=dropout,
                          bidirectional=False)

    # TODO: try out different conditioning schemes
    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor,
                init_state: Optional[torch.Tensor] = None):
        init_state = init_state.expand((self.num_layers, *init_state.size())).contiguous()
        x_packed = pack_padded_sequence(x, lengths)
        out, h_n = self.rnn(x_packed, init_state)
        out, _ = pad_packed_sequence(out, padding_value=self.pad_value)
        return out

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, input_size: int = 128, hidden_size: int = 128,
                 num_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0, pad_value: int = 0):
        super().__init__()
        self.pad_value = pad_value
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          bidirectional=self.bidirectional,
                          num_layers=self.num_layers,
                          dropout=dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, bs, _ = x.size()
        x = pack_padded_sequence(x, lengths)
        out, h_n = self.rnn(x)
        h_n, lengths = pad_packed_sequence(h_n, padding_value=self.pad_value)
        if self.bidirectional or self.num_layers > 1:
            h_n_transposed = h_n.transpose(0, 1).contiguous()
            h_n = h_n_transposed.view(bs, -1).contiguous()
        else:
            h_n = h_n.squeeze(0)
        return h_n

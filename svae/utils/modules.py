import torch
import torch.nn as nn


# TODO: add batch normalization or layer normalization?
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1024,
                 residual_channels: int = 512,
                 dilation: int = 1,
                 kernel_size: int = 3,
                 drop_prob: float = 0.1):
        super(ResidualBlock, self).__init__()
        self._pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=residual_channels,
                               kernel_size=1,
                               padding=0)
        self.drop_layer_1 = nn.Dropout(p=drop_prob)
        self.conv2 = nn.Conv1d(in_channels=residual_channels,
                               out_channels=residual_channels,
                               dilation=dilation,
                               kernel_size=kernel_size,
                               padding=self._pad)
        self.drop_layer_2 = nn.Dropout(p=drop_prob)
        self.conv3 = nn.Conv1d(in_channels=residual_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               padding=0)

    def forward(self, x: torch.Tensor):
        x_ = self.conv1(x).relu()
        x_ = self.drop_layer_1(x_)
        x_ = self.conv2(x_)
        if self._pad != 0:
            x_ = x_[:, :, :-self._pad]
        x_ = x_.relu()
        x_ = self.drop_layer_2(x_)
        x_ = self.conv3(x_)
        repr_sum = x + x_
        res = repr_sum.relu()
        return res

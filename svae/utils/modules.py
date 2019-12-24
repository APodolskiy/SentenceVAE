import torch
import torch.nn as nn


# TODO: add batch normalization or layer normalization?
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 1024,
                 residual_channels: int = 512,
                 dilation: int = 1,
                 kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self._pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=residual_channels,
                               kernel_size=1,
                               padding=2)
        self.conv2 = nn.Conv1d(in_channels=residual_channels,
                               out_channels=residual_channels,
                               dilation=dilation,
                               kernel_size=kernel_size,
                               padding=self._pad)
        self.conv3 = nn.Conv1d(in_channels=residual_channels,
                               out_channels=in_channels,
                               kernel_size=1,
                               padding=2)

    def forward(self, x: torch.Tensor):
        x_ = self.conv1(x).relu()
        x_ = self.conv2(x_)
        if self._pad != 0:
            x_ = x_[:, :, :-self._pad]
        x_ = x_.relu()
        x_ = self.conv3(x_).relu()
        return x + x_

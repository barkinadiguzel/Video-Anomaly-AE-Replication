import torch
import torch.nn as nn
from layers.convlstm_layer import ConvLSTMLayer

class TemporalEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.convlstm = ConvLSTMLayer(
            in_channels=in_channels,
            hidden_channels=in_channels
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        return self.convlstm(x)

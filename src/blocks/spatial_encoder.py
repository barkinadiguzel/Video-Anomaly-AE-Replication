import torch
import torch.nn as nn
from layers.conv_block import ConvBlock

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 128, kernel_size=3, padding=1, pool=True),
            ConvBlock(128, 256, kernel_size=3, padding=1, pool=True)
        )

    def forward(self, x):
        return self.encoder(x)

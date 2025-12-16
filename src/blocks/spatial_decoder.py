import torch
import torch.nn as nn
from layers.deconv_block import DeconvBlock

class SpatialDecoder(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.decoder = nn.Sequential(
            DeconvBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            DeconvBlock(128, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

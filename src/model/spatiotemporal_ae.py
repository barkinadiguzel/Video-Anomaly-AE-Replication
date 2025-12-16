import torch
import torch.nn as nn

from blocks.spatial_encoder import SpatialEncoder
from blocks.spatial_decoder import SpatialDecoder
from blocks.temporal_encoder import TemporalEncoder
from blocks.temporal_decoder import TemporalDecoder


class SpatioTemporalAutoEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.spatial_encoder = SpatialEncoder(in_channels)
        self.temporal_encoder = TemporalEncoder(in_channels=256)
        self.temporal_decoder = TemporalDecoder(in_channels=256)
        self.spatial_decoder = SpatialDecoder(out_channels=in_channels)

    def forward(self, x):

        B, T, C, H, W = x.size()

        # ---- Spatial encoding ----
        spatial_feats = []
        for t in range(T):
            feat = self.spatial_encoder(x[:, t])
            spatial_feats.append(feat.unsqueeze(1))

        spatial_feats = torch.cat(spatial_feats, dim=1)

        # ---- Temporal encoding ----
        encoded = self.temporal_encoder(spatial_feats)

        # ---- Temporal decoding ----
        decoded = self.temporal_decoder(encoded)

        # ---- Spatial decoding ----
        recon_frames = []
        for t in range(T):
            frame = self.spatial_decoder(decoded[:, t])
            recon_frames.append(frame.unsqueeze(1))

        recon = torch.cat(recon_frames, dim=1)
        return recon

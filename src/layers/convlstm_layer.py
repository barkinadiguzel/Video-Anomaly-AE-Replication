import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels)

    def forward(self, x):
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)

        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1)

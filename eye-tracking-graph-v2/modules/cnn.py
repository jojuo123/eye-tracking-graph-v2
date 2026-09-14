"""Basic CNN building blocks."""

import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class ConvBlock(nn.Module):
    """Conv2d -> (BatchNorm2d) -> Activation, the basic convolutional building block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=True, activation="relu"):
        super().__init__()
        act_layer = _ACTIVATIONS[activation]
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act_layer())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleCNNEncoder(nn.Module):
    """A small stack of stride-2 `ConvBlock`s followed by global average pooling,
    producing a flat feature vector (optionally projected to `out_dim`)."""

    def __init__(self, in_channels=3, channels=(32, 64, 128), out_dim=None):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out, stride=2))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim if out_dim is not None else c_in
        self.proj = nn.Linear(c_in, out_dim) if out_dim is not None else nn.Identity()

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

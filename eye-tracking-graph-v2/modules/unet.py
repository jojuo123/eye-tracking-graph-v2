"""U-Net encoder/decoder (Ronneberger et al., 2015), supporting both 2D images
and 3D volumes/video via `spatial_dims`.
"""

import torch.nn as nn

from modules.nd import get_nd_layers, match_and_concat


class DoubleConv(nn.Module):
    """(Conv -> Norm -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_channels, out_channels, spatial_dims=2):
        super().__init__()
        Conv, _, Norm, _, _ = get_nd_layers(spatial_dims)
        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True),
            Conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    """Downsampling path: `DoubleConv` then max-pool at each level (no pool
    after the last level, which is the bottleneck). Returns the list of
    pre-pool feature maps -- skip connections for a `UNetDecoder`, with the
    bottleneck as the last element.
    """

    def __init__(self, in_channels=3, channels=(64, 128, 256, 512), spatial_dims=2):
        super().__init__()
        _, _, _, Pool, _ = get_nd_layers(spatial_dims)
        self.pool = Pool(kernel_size=2, stride=2)

        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channels:
            self.blocks.append(DoubleConv(in_ch, out_ch, spatial_dims=spatial_dims))
            in_ch = out_ch
        self.out_channels = list(channels)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if i < len(self.blocks) - 1:
                x = self.pool(x)
        return skips


class UNetDecoder(nn.Module):
    """Upsampling path: transposed-conv upsample + concat skip + `DoubleConv`
    at each level, mirroring a `UNetEncoder`'s `channels`. Consumes the skip
    list produced by `UNetEncoder.forward` (bottleneck last)."""

    def __init__(self, channels=(64, 128, 256, 512), spatial_dims=2):
        super().__init__()
        _, ConvTranspose, _, _, _ = get_nd_layers(spatial_dims)
        rev_channels = list(reversed(channels))

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for in_ch, out_ch in zip(rev_channels[:-1], rev_channels[1:]):
            self.upsamples.append(ConvTranspose(in_ch, out_ch, kernel_size=2, stride=2))
            self.blocks.append(DoubleConv(out_ch * 2, out_ch, spatial_dims=spatial_dims))

    def forward(self, skips):
        x = skips[-1]
        for i, (up, block) in enumerate(zip(self.upsamples, self.blocks)):
            x = up(x)
            skip = skips[-(i + 2)]
            x = match_and_concat(x, skip)
            x = block(x)
        return x


class UNet(nn.Module):
    """Full U-Net: `UNetEncoder` + `UNetDecoder` + a final 1x1(x1) conv to
    `out_channels`. Works for 2D images (`spatial_dims=2`) or 3D volumes/video
    (`spatial_dims=3`). Input spatial size should be divisible by
    `2 ** (len(channels) - 1)`.
    """

    def __init__(self, in_channels=3, out_channels=1, channels=(64, 128, 256, 512), spatial_dims=2):
        super().__init__()
        Conv, _, _, _, _ = get_nd_layers(spatial_dims)
        self.encoder = UNetEncoder(in_channels, channels, spatial_dims=spatial_dims)
        self.decoder = UNetDecoder(channels, spatial_dims=spatial_dims)
        self.head = Conv(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = self.encoder(x)
        x = self.decoder(skips)
        return self.head(x)

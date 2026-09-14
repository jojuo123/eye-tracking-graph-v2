"""ResNet-backbone U-Net: a `ResNetEncoder` for feature extraction, paired
with a U-Net-style decoder that upsamples back to the input resolution using
the encoder's multi-scale features as skip connections. Supports 2D images or
3D volumes/video via `spatial_dims`.
"""

import torch.nn as nn

from modules.nd import get_nd_layers, match_and_concat
from modules.resnet import ResNetEncoder
from modules.unet import DoubleConv


class ResNetUNet(nn.Module):
    """Encoder: `ResNetEncoder(return_intermediate=True)` (stem + 4 stages ->
    5 feature maps at strides 2/4/8/16/32 of the input). Decoder: 4
    upsample+concat+`DoubleConv` stages mirroring the encoder back up to
    stride 2, then a final upsample to the input resolution and a 1x1(x1) conv
    to `out_channels`.
    """

    def __init__(self, in_channels=3, out_channels=1, base_channels=64, layers=(2, 2, 2, 2), spatial_dims=2):
        super().__init__()
        Conv, ConvTranspose, _, _, _ = get_nd_layers(spatial_dims)

        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            layers=layers,
            spatial_dims=spatial_dims,
            return_intermediate=True,
        )
        enc_channels = [base_channels] + self.encoder.out_channels  # [stem, s1, s2, s3, s4]

        rev = list(reversed(enc_channels))  # [s4, s3, s2, s1, stem]
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for in_ch, skip_ch in zip(rev[:-1], rev[1:]):
            self.upsamples.append(ConvTranspose(in_ch, skip_ch, kernel_size=2, stride=2))
            self.blocks.append(DoubleConv(skip_ch * 2, skip_ch, spatial_dims=spatial_dims))

        self.final_upsample = ConvTranspose(enc_channels[0], enc_channels[0], kernel_size=2, stride=2)
        self.head = Conv(enc_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)          # [stem, s1, s2, s3, s4]
        rev_feats = list(reversed(feats))  # [s4, s3, s2, s1, stem]

        h = rev_feats[0]
        for i, (up, block) in enumerate(zip(self.upsamples, self.blocks)):
            h = up(h)
            skip = rev_feats[i + 1]
            h = match_and_concat(h, skip)
            h = block(h)

        h = self.final_upsample(h)
        return self.head(h)

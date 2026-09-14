"""ResNet-style convolutional encoder (BasicBlock / ResNet-18/34 style),
supporting both 2D images and 3D volumes/video via `spatial_dims`.
"""

import torch.nn as nn

from modules.nd import get_nd_layers


class BasicBlock(nn.Module):
    """Conv-Norm-ReLU-Conv-Norm + residual (He et al., 2016). The 1x1(x1)
    `downsample` projection is only added when the shortcut needs to change
    shape (stride != 1 or channel count changes)."""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, spatial_dims=2):
        super().__init__()
        Conv, _, Norm, _, _ = get_nd_layers(spatial_dims)

        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                Conv(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                Norm(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetEncoder(nn.Module):
    """Stem (7x7(x7) conv, stride 2 -> 3x3(x3) max-pool, stride 2) followed by
    `len(layers)` stages of `BasicBlock`s, each stage (after the first)
    downsampling by 2.

    With `return_intermediate=True`, `forward` returns the feature map after
    the stem and after each stage (a multi-scale pyramid) instead of just the
    final one -- used as skip connections by `modules/resnet_unet.py`.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        layers=(2, 2, 2, 2),
        spatial_dims=2,
        return_intermediate=False,
    ):
        super().__init__()
        Conv, _, Norm, Pool, _ = get_nd_layers(spatial_dims)
        self.return_intermediate = return_intermediate

        self.stem = nn.Sequential(
            Conv(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            Norm(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = Pool(kernel_size=3, stride=2, padding=1)

        channels = [base_channels * (2 ** i) for i in range(len(layers))]
        self.stages = nn.ModuleList()
        in_ch = base_channels
        for i, (out_ch, n_blocks) in enumerate(zip(channels, layers)):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock(in_ch, out_ch, stride=stride, spatial_dims=spatial_dims)]
            blocks += [BasicBlock(out_ch, out_ch, spatial_dims=spatial_dims) for _ in range(n_blocks - 1)]
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.out_channels = channels  # per-stage output channel counts

    def forward(self, x):
        feats = []
        x = self.stem(x)
        feats.append(x)
        x = self.stem_pool(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats if self.return_intermediate else x

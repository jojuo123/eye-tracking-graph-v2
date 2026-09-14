"""Small helpers for writing conv nets that work in both 2D and 3D, so a
single module implementation (ResNet, U-Net, diffusion U-Net, ...) can support
images or volumes/video via one `spatial_dims` argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_nd_layers(spatial_dims: int):
    """Return `(Conv, ConvTranspose, BatchNorm, MaxPool, AdaptiveAvgPool)`
    classes for 2D (`spatial_dims=2`) or 3D (`spatial_dims=3`)."""
    if spatial_dims == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    if spatial_dims == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


def match_and_concat(x, skip):
    """Zero-pad `x` to `skip`'s spatial size (handles off-by-one sizes that
    arise from odd input dimensions going through stride-2 pooling) then
    concatenate along the channel dimension. Used to join a decoder's
    upsampled features with an encoder skip connection."""
    if x.shape[2:] != skip.shape[2:]:
        diffs = [s - xs for s, xs in zip(skip.shape[2:], x.shape[2:])]
        pad = []
        for d in reversed(diffs):
            pad += [d // 2, d - d // 2]
        x = F.pad(x, pad)
    return torch.cat([x, skip], dim=1)

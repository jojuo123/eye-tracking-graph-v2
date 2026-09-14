"""Registry of interchangeable image/patch encoders, each reducing a batch of
2D images `(B, C, H, W)` to flat feature vectors `(B, out_dim)`.

Exists so a model that embeds a batch of images/patches (e.g.
`SinkhornPatchSorter` in `models/sinkhorn_sort_model.py`) can swap its backbone
-- a small CNN, a ResNet, a ViT-style patch+attention encoder -- through
`encoder_cfg` alone, the same way losses/metrics are swapped via `LOSSES`/
`METRICS`. Add a new backbone by subclassing `nn.Module`, implementing
`forward(x) -> (B, out_dim)`, and decorating with `@ENCODERS.register("name")`.
"""

import torch.nn as nn

from modules.cnn import SimpleCNNEncoder
from modules.nd import get_nd_layers
from modules.resnet import ResNetEncoder
from modules.transformer import TransformerEncoder
from modules.vision_transformer import PatchEmbedding
from utils.registry import Registry

ENCODERS = Registry("encoders")


def build_encoder(cfg):
    """Build an image encoder from a config dict, e.g. {"type": "resnet", "out_dim": 128}."""
    return ENCODERS.build(cfg)


@ENCODERS.register("simple_cnn")
class SimpleCNNPatchEncoder(nn.Module):
    """Thin wrapper around `modules.cnn.SimpleCNNEncoder` -- a small stack of
    stride-2 conv blocks + global average pool. Cheapest option, good for
    small patches."""

    def __init__(self, in_channels=3, out_dim=128, channels=(32, 64, 128)):
        super().__init__()
        self.net = SimpleCNNEncoder(in_channels, channels, out_dim=out_dim)

    def forward(self, x):
        return self.net(x)


@ENCODERS.register("resnet")
class ResNetPatchEncoder(nn.Module):
    """`modules.resnet.ResNetEncoder` backbone (final-stage feature map) +
    global average pool + linear projection to `out_dim`. More capacity than
    `simple_cnn`, still works on small patches since `ResNetEncoder`'s stem
    only downsamples by 4x total."""

    def __init__(self, in_channels=3, out_dim=128, base_channels=64, layers=(2, 2, 2, 2), spatial_dims=2):
        super().__init__()
        self.backbone = ResNetEncoder(
            in_channels, base_channels, layers, spatial_dims=spatial_dims, return_intermediate=False
        )
        _, _, _, _, AdaptivePool = get_nd_layers(spatial_dims)
        self.pool = AdaptivePool(1)
        self.proj = nn.Linear(self.backbone.out_channels[-1], out_dim)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)


@ENCODERS.register("vit")
class ViTPatchEncoder(nn.Module):
    """ViT-style encoder without a classification head: splits each (already
    small) patch into `sub_patch_size` sub-patches via `PatchEmbedding`, runs a
    `TransformerEncoder` over them, and mean-pools the resulting tokens into a
    flat `(B, out_dim)` vector. See `modules.vision_transformer.VisionTransformer`
    for the full ViT (`[CLS]` token + head) this is adapted from."""

    def __init__(self, in_channels=3, out_dim=128, sub_patch_size=4, depth=2, n_heads=4, spatial_dims=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, out_dim, sub_patch_size, spatial_dims=spatial_dims)
        self.encoder = TransformerEncoder(out_dim, n_heads, depth)

    def forward(self, x):
        tokens = self.patch_embed(x)  # (B, num_sub_patches, out_dim)
        tokens = self.encoder(tokens)
        return tokens.mean(dim=1)

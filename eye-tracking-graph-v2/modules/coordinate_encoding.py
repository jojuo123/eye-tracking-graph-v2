"""2D coordinate encodings for spatial positions (a fixation's `(x, y)` location, or a grid
of an image feature map's token locations).

Unlike `modules.transformer.PositionalEncoding` (which encodes a token's *index* in a
sequence), these encode a token's coordinate *value* -- so a token's encoding says nothing
about where it sits in whatever list it was passed in, only where it is spatially. That
distinction matters for permutation learning: the fixation order is exactly what a downstream
model has to recover, so nothing here may leak it (see `modules.multiscale_fixation_encoder`).
"""

import math

import torch
import torch.nn as nn

from utils.registry import Registry

COORD_ENCODINGS = Registry("coordinate_encodings")


def build_coordinate_encoding(cfg):
    """Build a coordinate encoding from a config dict, e.g. {"type": "fourier", "embed_dim": 128}."""
    return COORD_ENCODINGS.build(cfg)


@COORD_ENCODINGS.register("sinusoidal")
class SinusoidalCoordinateEncoding(nn.Module):
    """2D analogue of the Transformer's sinusoidal positional encoding (Vaswani et al.,
    2017): each axis gets its own bank of geometrically-spaced sin/cos frequencies, evaluated
    at the coordinate's continuous value (assumed normalized to [0, 1], e.g.
    `SingleH5Dataset`'s `*_norm` fixation columns) rather than an integer sequence index --
    so two tokens get similar encodings whenever they are spatially close, regardless of
    their order in a batch's coordinate list.
    """

    def __init__(self, embed_dim, coord_dim=2, max_freq=10000.0):
        super().__init__()
        if embed_dim % (2 * coord_dim) != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by 2 * coord_dim ({2 * coord_dim})")
        num_freqs = embed_dim // (2 * coord_dim)
        freqs = torch.exp(torch.arange(num_freqs).float() * (-math.log(max_freq) / num_freqs))
        self.register_buffer("freqs", freqs)  # (num_freqs,)

    def forward(self, coords):
        # coords: (..., coord_dim) -> (..., embed_dim)
        angles = coords.unsqueeze(-1) * self.freqs  # (..., coord_dim, num_freqs)
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., coord_dim, 2 * num_freqs)
        return enc.flatten(-2)  # (..., coord_dim * 2 * num_freqs) == (..., embed_dim)


@COORD_ENCODINGS.register("fourier")
class FourierFeatureEncoding(nn.Module):
    """Random Fourier features (Tancik et al., 2020, "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains"): each coordinate is projected
    through a Gaussian random matrix before taking sin/cos, then a learned linear layer maps
    the result to `embed_dim`. Compared to `SinusoidalCoordinateEncoding`'s fixed geometric
    frequency bank, the random frequencies here (scaled by `sigma`) give the encoding a
    tunable, isotropic frequency spectrum -- higher `sigma` resolves finer spatial detail at
    the cost of smoothness. The random projection itself is fixed (a buffer, not a parameter)
    unless `learnable=True`.
    """

    def __init__(self, embed_dim, coord_dim=2, num_frequencies=64, sigma=10.0, learnable=False):
        super().__init__()
        b_matrix = torch.randn(coord_dim, num_frequencies) * sigma
        if learnable:
            self.b_matrix = nn.Parameter(b_matrix)
        else:
            self.register_buffer("b_matrix", b_matrix)
        self.proj = nn.Linear(2 * num_frequencies, embed_dim)

    def forward(self, coords):
        # coords: (..., coord_dim) -> (..., embed_dim)
        angles = 2 * math.pi * (coords @ self.b_matrix)  # (..., num_frequencies)
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(features)

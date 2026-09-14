"""Minimal DDPM-style diffusion building blocks (Ho et al., 2020): a
time-conditioned U-Net denoiser plus the Gaussian forward-process/training-loss
machinery. Supports 2D images or 3D volumes/video via `spatial_dims`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.nd import get_nd_layers, match_and_concat


class SinusoidalTimeEmbedding(nn.Module):
    """Embeds a scalar diffusion timestep into a `dim`-dimensional vector using
    fixed sinusoidal frequencies -- the same idea as Transformer positional
    encoding, applied to one scalar per sample instead of a sequence position.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConditionedResBlock(nn.Module):
    """Conv -> Norm -> SiLU -> (+ time embedding, broadcast per-channel) ->
    Conv -> Norm -> SiLU, with a residual connection (1x1(x1) conv on the
    shortcut if channel count changes)."""

    def __init__(self, in_channels, out_channels, time_dim, spatial_dims=2):
        super().__init__()
        Conv, _, Norm, _, _ = get_nd_layers(spatial_dims)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = Norm(out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = Norm(out_channels)
        self.act = nn.SiLU()
        self.skip = nn.Identity() if in_channels == out_channels else Conv(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        bias_shape = (t_emb.size(0), -1) + (1,) * (h.dim() - 2)
        h = h + self.time_proj(t_emb).view(*bias_shape)
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """Time-conditioned U-Net that predicts the noise added to a sample at
    timestep `t` -- the standard DDPM denoising network. Works for 2D images
    (`spatial_dims=2`) or 3D volumes/video (`spatial_dims=3`). Input spatial
    size should be divisible by `2 ** (len(channel_mults) - 1)`.

    Usage: `model(x_t, t)` where `x_t` is `(B, C, *spatial)` and `t` is a
    `(B,)` integer tensor of per-sample timesteps.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=256,
        spatial_dims=2,
    ):
        super().__init__()
        Conv, ConvTranspose, _, Pool, _ = get_nd_layers(spatial_dims)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]
        self.pool = Pool(kernel_size=2, stride=2)
        self.stem = Conv(in_channels, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.down_blocks.append(TimeConditionedResBlock(in_ch, out_ch, time_dim, spatial_dims))
            in_ch = out_ch

        rev_channels = list(reversed(channels))
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(rev_channels[:-1], rev_channels[1:]):
            self.upsamples.append(ConvTranspose(in_ch, out_ch, kernel_size=2, stride=2))
            self.up_blocks.append(TimeConditionedResBlock(out_ch * 2, out_ch, time_dim, spatial_dims))

        self.head = Conv(channels[0], out_channels, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.stem(x)

        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, t_emb)
            skips.append(x)
            if i < len(self.down_blocks) - 1:
                x = self.pool(x)

        for i, (up, block) in enumerate(zip(self.upsamples, self.up_blocks)):
            x = up(x)
            skip = skips[-(i + 2)]
            x = match_and_concat(x, skip)
            x = block(x, t_emb)

        return self.head(x)


class GaussianDiffusion:
    """DDPM forward process + training loss (Ho et al., 2020). Wraps a
    noise-prediction network (e.g. `DiffusionUNet`) with:
      - a linear beta schedule over `num_timesteps` steps,
      - `q_sample`: sample `x_t` from `x_0` and noise in closed form,
      - `training_loss`: MSE between predicted and actual noise at a random `t`.

    Usage:
        diffusion = GaussianDiffusion(num_timesteps=1000)
        loss = diffusion.training_loss(model, x0)   # x0: (B, C, *spatial) clean data
        loss.backward()
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Sample `x_t ~ q(x_t | x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) noise`."""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alphas_cumprod.to(x0.device)[t]
        shape = (x0.size(0),) + (1,) * (x0.dim() - 1)
        sqrt_alpha_bar = alpha_bar.sqrt().view(*shape)
        sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt().view(*shape)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise, noise

    def training_loss(self, model, x0, t=None):
        """Sample a random `t` (if not given) and noise, then return the MSE
        between the model's predicted noise and the actual noise -- the
        standard DDPM training objective."""
        if t is None:
            t = torch.randint(0, self.num_timesteps, (x0.size(0),), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)
